"""Native Megatron Core ModelConfig/ModelBuilder for the CURRENT IQ runtime.

This implements the documented builder-backed model API for the single-
pipeline-stage, TP=1 IQ model. It does not fake an upstream GPTModel spec:
IQ's Mamba-3 MIMO / CSA / Block AttnRes operators remain their exact own
modules. Expert parallelism uses the real MCore ProcessGroupCollection.ep
with differentiable all-to-all, and gradient sync/checkpoints use IQ's
existing explicit EP adapters.

Unsupported TP, PP, VPP, FSDP and automatic MCore DDP wrapping are
rejected before model allocation. This is not yet an AutoBridge model
conversion or a complete native Megatron checkpoint training recipe.

NVIDIA:
https://docs.nvidia.com/nemo/megatron-bridge/nightly/bridge-guide.html
https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html
https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.base import ModelBuilder, ModelConfig

from iq_model.hybrid import IQHybridConfig, IQHybridForCausalLM
from iq_transfer.complete_transplant import nvidia_parallel_78_config
from iq_training.pipeline_contract import partition_iq_78


class IQBridgeBuilderError(RuntimeError):
    pass


@dataclass
class IQMegatronModelConfig(ModelConfig):
    """Serializable IQ model config with an actual importable builder."""

    builder: ClassVar[str] = "iq_training.mcore_builder.IQMegatronModelBuilder"
    iq_hybrid: dict[str, Any] = field(
        default_factory=lambda: nvidia_parallel_78_config().to_dict()
    )
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    expert_model_parallel_size: int = 1
    params_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        hybrid = self.validated_iq_config()
        if self.tensor_model_parallel_size != 1:
            raise IQBridgeBuilderError(
                "IQ Mamba-3/CSA tensor sharding has not been implemented; TP must equal 1"
            )
        if self.pipeline_model_parallel_size != 1:
            # Validate dependencies without silently dropping AttnRes or
            # a DSA source; current IQ builds one complete stack per rank.
            raise IQBridgeBuilderError(
                "IQ has no Megatron pipeline activation/state handoff; PP must equal 1"
            )
        if self.virtual_pipeline_model_parallel_size is not None:
            raise IQBridgeBuilderError("virtual pipeline partition is not implemented")
        if self.params_dtype != "bfloat16":
            raise IQBridgeBuilderError(
                "IQ Mamba-3 GPU runtime currently uses BF16 precision"
            )
        if self.expert_model_parallel_size != hybrid.expert_parallel_size:
            raise IQBridgeBuilderError(
                "Megatron expert parallel size contradicts IQ's expert_parallel_size"
            )
        partition_iq_78(hybrid)

    def validated_iq_config(self) -> IQHybridConfig:
        return IQHybridConfig.from_dict(self.iq_hybrid)


class IQMegatronModule(MegatronModule):
    """Actual MegatronModule target, containing IQ's genuine 78-block graph."""

    def __init__(
        self,
        config: TransformerConfig,
        iq_config: IQHybridConfig,
        *,
        ep_group: dist.ProcessGroup | None,
    ) -> None:
        super().__init__(config)
        if not torch.cuda.is_available():
            raise IQBridgeBuilderError(
                "actual native Mamba-3 MIMO model requires an NVIDIA CUDA GPU"
            )
        self.iq = IQHybridForCausalLM(
            iq_config, dtype=torch.bfloat16,
            device=torch.device("cuda", torch.cuda.current_device()),
            expert_parallel_group=ep_group,
        )
        self.iq_config_fingerprint = iq_config.fingerprint

    def forward(self, *args: Any, **kwargs: Any):
        return self.iq(*args, **kwargs)

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        raise IQBridgeBuilderError(
            "pipeline handoff cannot be used until native PP stage execution is implemented"
        )


def iq_mcore_transformer_metadata(iq: IQHybridConfig) -> TransformerConfig:
    """Real MCore 0.19 config validation for IQ EP-only model metadata.

    TransformerConfig describes widths/parallel groups; it does not mean the
    model executes the vanilla GPT TransformerLayer.
    """
    return TransformerConfig(
        num_layers=78,
        hidden_size=iq.model.hidden_size,
        num_attention_heads=iq.model.num_attention_heads,
        ffn_hidden_size=iq.model.intermediate_size,
        num_moe_experts=iq.stable_moe.num_experts if iq.stable_moe else None,
        moe_router_topk=iq.stable_moe.top_k if iq.stable_moe else 2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=iq.expert_parallel_size,
        bf16=True,
    )


class IQMegatronModelBuilder(ModelBuilder[IQMegatronModule, IQMegatronModelConfig]):
    """Documented NVIDIA builder for a real, EP-only IQ model graph."""

    def __init__(self, model_config: IQMegatronModelConfig) -> None:
        if not isinstance(model_config, IQMegatronModelConfig):
            raise TypeError("IQMegatronModelBuilder requires IQMegatronModelConfig")
        super().__init__(model_config)

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> IQMegatronModule:
        spec = self._model_config
        spec.__post_init__()  # also catch post-construction modifications
        if pre_process is False or post_process is False or vp_stage is not None:
            raise IQBridgeBuilderError(
                "IQ builder only supports an unsplit complete first/last pipeline stage"
            )
        if not dist.is_initialized():
            raise IQBridgeBuilderError(
                "initialize real torch.distributed and Megatron process groups first"
            )
        iq = spec.validated_iq_config()
        ep = getattr(pg_collection, "ep", None)
        if iq.expert_parallel_size > 1:
            if ep is None:
                raise IQBridgeBuilderError("missing real Megatron EP process group")
            if dist.get_world_size(ep) != iq.expert_parallel_size:
                raise IQBridgeBuilderError("Megatron EP group size differs from IQ config")
        else:
            ep = None
        if not torch.cuda.is_available():
            raise IQBridgeBuilderError(
                "constructing native IQ Mamba-3 requires an NVIDIA GPU"
            )
        native = iq_mcore_transformer_metadata(iq)
        return IQMegatronModule(native, iq, ep_group=ep)

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: Any | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper: Any | None = None,
        model_type: Any = None,
        use_layer_wise_distributed_optimizer: bool = False,
        use_layer_wise_param_layout: bool = True,
    ) -> list[IQMegatronModule]:
        if (
            wrap_with_ddp or use_megatron_fsdp or use_torch_fsdp2
            or overlap_param_gather_with_optimizer_step or data_parallel_random_init
            or use_layer_wise_distributed_optimizer
            or mixed_precision_wrapper is not None
        ):
            raise IQBridgeBuilderError(
                "IQ's distributed path is currently explicit EP all-to-all "
                "plus replica-gradient sync; native MCore DDP/FSDP/optimizer "
                "wrapping is not implemented"
            )
        return [self.build_model(pg_collection)]
