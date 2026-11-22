"""CPU mirror of the production IQ hybrid backbone for local streaming.

Composes the 235-layer schedule exactly as production `IQHybridForCausalLM`
(iq_model/hybrid.py:702-833): mamba layers wrap `Mamba3CPU` inside the
production residual-layer pattern (iq_model/hybrid.py:587-628), CSA/HCA and
dense-attention layers instantiate the production classes directly
(iq_model/hybrid.py:631-699), and MoE layers use a functional variant of the
production Stable LatentMoE (iq_model/mlp/latent_moe.py:91-309).

Deviations from production, all deliberate:

- Routed experts are not `nn.Module`s. `FunctionalStableLatentMoE.forward`
  fetches them through `expert_fn(stage, expert_idx) -> (gate, up, down)`
  weight tensors, mirroring the streaming weight provider contract; model
  state therefore never contains `*.routed_experts.*` keys. Every other
  module, attribute name, and state-dict key matches production exactly
  (router, latent_down, latent_up, latent_norm, shared_experts,
  routing_bias).
- `expert_bias` (routed expert linear biases) is unsupported in the
  functional contract and fails loudly; the IQ checkpoint runs
  `expert_bias: false`.
- `moe_variant` is restricted to `stable_latent` (the IQ checkpoint
  variant); the swiglu variant would require a second functional port.
- Default device is `cpu` and `device="meta"` is a first-class option:
  the full production schedule allocates roughly 40+ GB of non-expert
  weights, which exceeds this machine's RAM, so state-dict key parity is
  verified on meta tensors (shape-only, no storage).
- `forward(input_ids, attention_mask=None, expert_fn=None) -> logits` is
  the plain non-streaming parity path: it mirrors the production
  composition (embedding, AttnRes block residual wiring
  (iq_model/hybrid.py:980-1041), reasoning recurrence/injection
  (iq_model/hybrid.py:1043-1065), final norm, lm_head) but returns logits
  only and takes no labels/loss. MoE layers require `expert_fn`.
  `attention_mask` must be None or all-ones; padded batches are
  unsupported in the CPU mirror — production varlen-packs mamba on the
  mask (iq_model/hybrid.py:995-1000) while this port's batched CPU mamba
  prefill is unmasked, so any non-trivial mask is rejected loudly and
  all-ones masks are passed downstream as None.
- Mamba residual layers run the batched CPU prefill path directly
  (`norm -> Mamba3CPU.prefill -> residual`); the production varlen
  pack/unpack and `cu_seqlens` contract is a CUDA-kernel runtime detail
  and is rejected at layer level when `attention_mask`/`document_ids` are
  supplied directly.
"""

from __future__ import annotations

import torch
from torch import nn

from iq_model.architecture import HybridLayerType
from iq_model.hybrid import (
    CompressedContextResidualLayer,
    DenseContextResidualLayer,
    HybridModelError,
    _reasoning_masks,
)
from iq_model.mlp import (
    SiTUAndMul,
    SiTUExpert,
    StableLatentMoEConfig,
    StableLatentMoEError,
    StableLatentMoEOutput,
)
from iq_model.norm import RMSNorm
from iq_model.reasoning import ReasoningRecurrence, ReasoningStateInjector
from iq_model.residual import BlockAttentionResidual
from iq_model.state.mamba3_cpu import CPUConfig, Mamba3CPU


class FunctionalStableLatentMoE(nn.Module):
    """Production `StableLatentMoE` (iq_model/mlp/latent_moe.py:91-288) with
    routed experts computed functionally through `expert_fn`.

    `_route`, the shared-expert path, the routing-bias buffer, and the output
    dataclass are mirrored verbatim so routing and aggregation are bit-for-bit
    comparable with production on the same weights.
    """

    def __init__(
        self,
        config: StableLatentMoEConfig,
        *,
        stage: int = 0,
    ) -> None:
        super().__init__()
        if config.expert_bias:
            raise HybridModelError(
                "FunctionalStableLatentMoE requires expert_bias=false: the "
                "expert_fn contract returns (gate, up, down) weight tensors "
                "without biases"
            )
        self.config = config
        self._stage = stage
        self.router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=config.router_bias,
        )
        self.latent_down = nn.Linear(
            config.hidden_size,
            config.latent_size,
            bias=False,
        )
        self.latent_up = nn.Linear(
            config.latent_size,
            config.hidden_size,
            bias=False,
        )
        self.latent_norm = RMSNorm(
            config.latent_size,
            config.rms_norm_eps,
        )
        self.shared_experts = nn.ModuleList(
            SiTUExpert(
                config.hidden_size,
                config.expert_intermediate_size,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
                bias=config.expert_bias,
            )
            for _ in range(config.num_shared_experts)
        )
        self._activation = SiTUAndMul(
            beta=config.situ_beta,
            linear_beta=config.situ_linear_beta,
        )
        self.register_buffer(
            "routing_bias",
            torch.zeros(config.num_experts),
            persistent=True,
        )

    def _route(
        self,
        flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.sigmoid(self.router(flat).float())
        choice_scores = raw + self.routing_bias.float().unsqueeze(0)
        _, selected = torch.topk(
            choice_scores,
            k=self.config.top_k,
            dim=-1,
            largest=True,
            sorted=False,
        )
        weights = raw.gather(1, selected)
        weights = weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(weights.dtype).tiny)
        weights = weights * self.config.routed_scaling_factor
        return raw, selected, weights

    def _run_expert(
        self,
        expert_fn,
        expert_index: int,
        expert_input: torch.Tensor,
    ) -> torch.Tensor:
        gate_weight, up_weight, down_weight = expert_fn(self._stage, expert_index)
        activated = self._activation(
            nn.functional.linear(expert_input, gate_weight),
            nn.functional.linear(expert_input, up_weight),
        )
        return nn.functional.linear(activated, down_weight)

    def forward(
        self,
        x: torch.Tensor,
        expert_fn,
    ) -> StableLatentMoEOutput:
        if expert_fn is None:
            raise HybridModelError(
                "FunctionalStableLatentMoE.forward requires expert_fn("
                "stage, expert_index) -> (gate, up, down) weight tensors"
            )
        if x.ndim < 2 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Stable LatentMoE input must end in hidden_size="
                f"{self.config.hidden_size}"
            )
        original_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        if flat.shape[0] == 0:
            raise ValueError("Stable LatentMoE requires at least one token")

        raw_scores, selected, weights = self._route(flat)
        latent = self.latent_down(flat)

        routed = torch.zeros_like(latent)
        for expert_index in range(self.config.num_experts):
            token_rows, slots = torch.where(selected == expert_index)
            if token_rows.numel() == 0:
                continue
            expert_input = latent.index_select(0, token_rows)
            expert_output = self._run_expert(expert_fn, expert_index, expert_input)
            token_weights = weights[
                token_rows,
                slots,
            ].to(expert_output.dtype)
            routed.index_add_(
                0,
                token_rows,
                expert_output * token_weights.unsqueeze(-1),
            )

        routed = self.latent_up(self.latent_norm(routed))

        shared = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared = shared + expert(flat)

        hidden = (routed + shared).reshape(original_shape)
        if not bool(torch.isfinite(hidden).all()):
            raise StableLatentMoEError(
                "Stable LatentMoE produced non-finite hidden states"
            )
        counts = torch.bincount(
            selected.reshape(-1),
            minlength=self.config.num_experts,
        )
        return StableLatentMoEOutput(
            hidden_states=hidden,
            raw_router_scores=raw_scores.reshape(
                *original_shape[:-1],
                self.config.num_experts,
            ),
            selected_experts=selected.reshape(
                *original_shape[:-1],
                self.config.top_k,
            ),
            selected_weights=weights.reshape(
                *original_shape[:-1],
                self.config.top_k,
            ),
            expert_counts=counts,
            routing_bias=self.routing_bias.detach().clone(),
        )


class FunctionalStableLatentMoELayer(nn.Module):
    """Production `StableLatentMoELayer` (iq_model/mlp/latent_moe.py:291-309)
    wrapping the functional MoE; same attribute names, same residual pattern."""

    def __init__(
        self,
        config: StableLatentMoEConfig,
        *,
        stage: int = 0,
        norm_eps: float = 1e-5,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= residual_dropout < 1.0):
            raise ValueError("residual_dropout must be in [0, 1)")
        self.norm = RMSNorm(config.hidden_size, norm_eps)
        self.moe = FunctionalStableLatentMoE(config, stage=stage)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        expert_fn,
    ) -> StableLatentMoEOutput:
        output = self.moe(self.norm(x), expert_fn)
        output.hidden_states = x + self.residual_dropout(output.hidden_states)
        return output


class CPUMamba3ResidualLayer(nn.Module):
    """Production `Mamba3ResidualLayer` (iq_model/hybrid.py:587-628) with the
    CUDA `Mamba3MIMOState` replaced by `Mamba3CPU`; identical attribute names
    (norm / mamba / residual_dropout) and state-dict keys
    (`layers.<i>.mamba.core.*`)."""

    def __init__(
        self,
        model_config,
        mamba_config,
        *,
        mamba_layer_idx: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(
            model_config.hidden_size,
            model_config.rms_norm_eps,
        ).to(device=device, dtype=dtype)
        cpu_config = CPUConfig(
            d_model=mamba_config.d_model,
            headdim=mamba_config.headdim,
            d_state=mamba_config.d_state,
            mimo_rank=mamba_config.mimo_rank,
            num_layers=mamba_config.num_layers,
            expand=mamba_config.expand,
            chunk_size=mamba_config.chunk_size,
            rope_fraction=mamba_config.rope_fraction,
            outproj_norm=mamba_config.outproj_norm,
        )
        self.mamba = Mamba3CPU(
            cpu_config,
            layer_idx=mamba_layer_idx,
            dtype=dtype,
            device=device,
        )
        self.residual_dropout = nn.Dropout(model_config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None or document_ids is not None:
            raise HybridModelError(
                "CPUMamba3ResidualLayer runs the batched CPU prefill path; "
                "varlen packing (attention_mask/document_ids) is a CUDA "
                "runtime contract not supported by the CPU mirror"
            )
        mixed = self.mamba.prefill(self.norm(x))[0]
        return x + self.residual_dropout(mixed)


class CPUHybridForCausalLM(nn.Module):
    """CPU mirror of `IQHybridForCausalLM` (iq_model/hybrid.py:702-833).

    Schedule composition, attribute names, initialization, and state-dict keys
    mirror production; only the CUDA gate is removed (device defaults to cpu)
    and routed MoE experts are functional instead of resident parameters.
    """

    def __init__(
        self,
        config,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.model_config = config.model
        device_obj = torch.device(device)
        if config.moe_variant != "stable_latent":
            raise HybridModelError(
                "CPUHybridForCausalLM supports the stable_latent MoE variant "
                f"only, got {config.moe_variant}"
            )
        if config.multimodal is not None:
            raise HybridModelError(
                "CPUHybridForCausalLM does not support the multimodal pathway"
            )

        self.embed_tokens = nn.Embedding(
            config.model.vocab_size,
            config.model.hidden_size,
            device=device_obj,
            dtype=dtype,
        )
        self.layers = nn.ModuleList()
        mamba_layer_idx = 0
        moe_stage_idx = 0
        for layer_type in config.schedule.layers:
            if layer_type is HybridLayerType.MAMBA3:
                layer = CPUMamba3ResidualLayer(
                    config.model,
                    config.mamba3,
                    mamba_layer_idx=mamba_layer_idx,
                    dtype=dtype,
                    device=device_obj,
                )
                mamba_layer_idx += 1
            elif layer_type is HybridLayerType.MOE:
                assert config.stable_moe is not None
                layer = FunctionalStableLatentMoELayer(
                    config.stable_moe,
                    stage=moe_stage_idx,
                    norm_eps=config.model.rms_norm_eps,
                    residual_dropout=config.model.residual_dropout,
                ).to(device=device_obj, dtype=dtype)
                moe_stage_idx += 1
            elif layer_type is HybridLayerType.DENSE_ATTENTION:
                layer = DenseContextResidualLayer(
                    config.model,
                ).to(device=device_obj, dtype=dtype)
            elif layer_type in {HybridLayerType.CSA, HybridLayerType.HCA}:
                assert config.compressed_context is not None
                layer = CompressedContextResidualLayer(
                    config.model,
                    config.compressed_context,
                    mode=layer_type,
                ).to(device=device_obj, dtype=dtype)
            else:
                raise HybridModelError(
                    f"unsupported hybrid layer type: {layer_type.value}"
                )
            self.layers.append(layer)

        self.attnres = (
            BlockAttentionResidual(config.attnres).to(
                device=device_obj,
                dtype=dtype,
            )
            if config.attnres is not None
            else None
        )
        self.norm = RMSNorm(
            config.model.hidden_size,
            config.model.rms_norm_eps,
        ).to(device=device_obj, dtype=dtype)
        self.reasoning = (
            ReasoningRecurrence(config.reasoning).to(device=device_obj, dtype=dtype)
            if config.reasoning is not None
            else None
        )
        self.reasoning_injector = (
            ReasoningStateInjector(
                config.model.hidden_size,
                config.reasoning.state_dim,
            ).to(device=device_obj, dtype=dtype)
            if config.reasoning is not None
            else None
        )

        self.lm_head = nn.Linear(
            config.model.hidden_size,
            config.model.vocab_size,
            bias=False,
            device=device_obj,
            dtype=dtype,
        )
        nn.init.normal_(
            self.embed_tokens.weight,
            mean=0.0,
            std=config.model.initializer_range,
        )
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=config.model.initializer_range,
        )
        if config.model.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        expert_fn=None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer token ids")
        if input_ids.numel() and (
            int(input_ids.min()) < 0
            or int(input_ids.max()) >= self.model_config.vocab_size
        ):
            raise ValueError("input_ids contain token ids outside vocabulary")
        if attention_mask is not None and not bool((attention_mask == 1).all()):
            raise ValueError(
                "CPUHybridForCausalLM.forward supports attention_mask=None or "
                "all-ones only: the CPU mamba prefill is unmasked, so padded "
                "batches would let pad tokens corrupt mamba states "
                "(production varlen-packs mamba on the mask instead; use "
                "unpadded batches for the CPU mirror)"
            )

        x = self.embed_tokens(input_ids)
        attnres_state = self.attnres.init_state(x) if self.attnres is not None else None
        for layer_type, layer in zip(
            self.config.schedule.layers,
            self.layers,
            strict=True,
        ):
            layer_input = (
                self.attnres.read(attnres_state)
                if self.attnres is not None and attnres_state is not None
                else x
            )
            if layer_type is HybridLayerType.MAMBA3:
                layer_output = layer(
                    layer_input,
                    attention_mask=None,
                    document_ids=None,
                )
            elif layer_type is HybridLayerType.MOE:
                layer_output = layer(layer_input, expert_fn).hidden_states
            elif layer_type in {
                HybridLayerType.DENSE_ATTENTION,
                HybridLayerType.CSA,
                HybridLayerType.HCA,
            }:
                layer_output = layer(
                    layer_input,
                    position_ids=None,
                    attention_mask=attention_mask,
                    document_ids=None,
                )
            else:
                raise HybridModelError(
                    f"unsupported runtime layer type: {layer_type.value}"
                )
            if self.attnres is not None and attnres_state is not None:
                layer_delta = layer_output - layer_input
                attnres_state = self.attnres.advance(
                    attnres_state,
                    layer_delta,
                )
            x = layer_output

        if self.attnres is not None:
            assert attnres_state is not None
            x = self.attnres.finalize(attnres_state)

        hidden = self.norm(x)
        if self.reasoning is not None:
            reasoning_context_mask, reasoning_injection_mask = _reasoning_masks(
                input_ids,
                attention_mask,
                None,
                labels_present=False,
            )
            reasoning_output = self.reasoning(
                hidden,
                attention_mask=attention_mask,
                reasoning_context_mask=reasoning_context_mask,
                document_ids=None,
                verifier=None,
                require_verifier_stability=self.config.require_verifier_stability,
            )
            hidden = self.reasoning_injector(
                hidden,
                reasoning_output.state,
                token_mask=reasoning_injection_mask,
            )
        return self.lm_head(hidden)
