"""GLM-founded, no-learning serial Mamba-3 MIMO reference integration.

Preserves the *internal* 24-block pretrained Mamba donor ordering, including
mixer + gated MLP per block, pre-norm residual staging, and terminal norm.
The GLM causal LM remains entirely unchanged unless the new residual branch
is explicitly enabled. A nonzero gate still requires representation alignment
and quality proof; this file deliberately does NOT advertise weight-only
functional equivalence or accelerated replacement of GLM attention.

Only the official donor's non-fused-add-norm configuration is supported here.
Generation with enabled Mamba is prohibited until cache and checkpoint
integrity are validated; a full uncached forward can be used for controlled
comparisons and training with adequate compute.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from .glm_first_mamba import (
    GLMFirstIntegrationError,
    GLMFirstMambaContract,
    GLMFirstMambaHybrid,
    validate_complete_mamba_block,
)


def validate_serial_donor(mamba_model: nn.Module, contract: GLMFirstMambaContract) -> nn.Module:
    config = getattr(mamba_model, "config", None)
    backbone = getattr(mamba_model, "backbone", None)
    layers = getattr(backbone, "layers", None)
    norm_f = getattr(backbone, "norm_f", None)
    if config is None or not isinstance(layers, nn.ModuleList) or not isinstance(norm_f, nn.Module):
        raise GLMFirstIntegrationError("pretrained serial donor must expose backbone.layers and norm_f")
    expected = (
        ("n_layer", contract.mamba_layers),
        ("d_model", contract.mamba_hidden),
    )
    for name, value in expected:
        if getattr(config, name, None) != value:
            raise GLMFirstIntegrationError(f"Mamba serial donor {name} differs from contract")
    if getattr(config, "fused_add_norm", False):
        raise GLMFirstIntegrationError("unverified fused donor residual path is unsupported")
    # Explicitly match the official released Mamba-3 MIMO config when using
    # its full-sized donor; synthetic tests use their own compatible config.
    if contract == GLMFirstMambaContract():
        if (
            getattr(config, "d_intermediate", None) != 3824
            or getattr(config, "tie_embeddings", None) is not True
            or getattr(config, "residual_in_fp32", None) is not True
            or getattr(config, "rms_norm", None) is not True
        ):
            raise GLMFirstIntegrationError("released Mamba-3 block config differs")
        ssm = getattr(config, "ssm_cfg", None)
        if not isinstance(ssm, dict) or (
            ssm.get("layer") != "Mamba3"
            or ssm.get("is_mimo") is not True
            or ssm.get("mimo_rank") != contract.mimo_rank
            or ssm.get("d_state") != contract.state_size
            or ssm.get("chunk_size") != 16
        ):
            raise GLMFirstIntegrationError("released Mamba-3 recurrent config differs")
    if len(layers) != contract.mamba_layers:
        raise GLMFirstIntegrationError("serial Mamba block count mismatch")
    for block in layers:
        validate_complete_mamba_block(block, contract)
    norm_weight = getattr(norm_f, "weight", None)
    if norm_weight is None or tuple(norm_weight.shape) != (contract.mamba_hidden,):
        raise GLMFirstIntegrationError("Mamba final normalization dimension mismatch")
    return backbone


class SerialMamba3DonorPath(nn.Module):
    """Original donor recurrence chain with GLM-width input/output boundaries."""

    def __init__(
        self,
        mamba_model: nn.Module,
        contract: GLMFirstMambaContract,
    ) -> None:
        super().__init__()
        self.backbone = validate_serial_donor(mamba_model, contract)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        weight = self.backbone.norm_f.weight
        factory = {"device": weight.device, "dtype": weight.dtype}
        self.in_adapter = nn.Linear(
            contract.glm_hidden, contract.mamba_hidden, bias=False, **factory
        )
        self.out_adapter = nn.Linear(
            contract.mamba_hidden, contract.glm_hidden, bias=False, **factory
        )
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32, device=weight.device))
        self.enabled = False
        self.contract = contract

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _serial_donor_forward(self, projected: torch.Tensor) -> torch.Tensor:
        hidden_states = projected
        residual: torch.Tensor | None = None
        for layer in self.backbone.layers:
            hidden_states, residual = layer(
                hidden_states, residual=residual, inference_params=None
            )
            if not isinstance(hidden_states, torch.Tensor) or not isinstance(residual, torch.Tensor):
                raise GLMFirstIntegrationError("Mamba donor block residual interface changed")
        # Exactly as the non-fused upstream MixerModel.forward:
        residual = hidden_states + residual if residual is not None else hidden_states
        norm = self.backbone.norm_f
        final = norm(residual.to(dtype=norm.weight.dtype))
        if final.shape != projected.shape:
            raise GLMFirstIntegrationError("Mamba donor final output shape changed")
        if not bool(torch.isfinite(final).all()):
            raise GLMFirstIntegrationError("Mamba donor produced NaN/Inf")
        return final

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.contract.glm_hidden:
            raise GLMFirstIntegrationError("unexpected GLM residual dimensions")
        if not self.enabled:
            if bool(torch.count_nonzero(self.gate.detach())):
                raise GLMFirstIntegrationError("disabled serial donor retains nonzero gate")
            return hidden
        if hidden.device != self.in_adapter.weight.device:
            raise GLMFirstIntegrationError(
                "serial Mamba donor and GLM output must share device; no implicit transfers"
            )
        projected = self.in_adapter(hidden.to(self.in_adapter.weight.dtype))
        transformed = self._serial_donor_forward(projected)
        delta = self.out_adapter(transformed.to(self.out_adapter.weight.dtype))
        return hidden + self.gate.to(hidden.dtype) * delta.to(hidden.dtype)


class GLMFirstSerialMambaHybrid(nn.Module):
    """Connects the complete ordered donor stack after one original GLM layer.

    With the gate disabled, forward and generate call the original GLM model
    without altering any decoder output. In enabled mode only non-cached
    full-prefix forward is permitted.
    """

    def __init__(
        self,
        glm_model: nn.Module,
        mamba_model: nn.Module,
        *,
        contract: GLMFirstMambaContract = GLMFirstMambaContract(),
        insertion_layer: int | None = None,
    ) -> None:
        super().__init__()
        config = getattr(glm_model, "config", None)
        decoder = getattr(glm_model, "model", None)
        layers = getattr(decoder, "layers", None)
        if config is None or not isinstance(layers, nn.ModuleList):
            raise GLMFirstIntegrationError("original GLM decoder not found")
        for key, value in (
            ("vocab_size", contract.glm_vocab),
            ("hidden_size", contract.glm_hidden),
            ("num_hidden_layers", contract.glm_layers),
            ("tie_word_embeddings", False),
        ):
            if getattr(config, key, None) != value:
                raise GLMFirstIntegrationError(f"original GLM config {key} mismatch")
        if len(layers) != contract.glm_layers:
            raise GLMFirstIntegrationError("GLM pretrained layer count mismatch")
        embedding = getattr(decoder, "embed_tokens", None)
        head = getattr(glm_model, "lm_head", None)
        if not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear):
            raise GLMFirstIntegrationError("original GLM lexical tensors missing")
        expected_shape = (contract.glm_vocab, contract.glm_hidden)
        if embedding.weight.shape != expected_shape or head.weight.shape != expected_shape:
            raise GLMFirstIntegrationError("GLM lexical tensor dimensions changed")
        if contract == GLMFirstMambaContract():
            # Strict production ownership; custom small contract is for
            # deterministic CPU verification only.
            semantic_requirements = {
                "model_type": "glm_moe_dsa",
                "qk_head_dim": 256,
                "qk_nope_head_dim": 192,
                "qk_rope_head_dim": 64,
                "v_head_dim": 256,
                "q_lora_rank": 2048,
                "kv_lora_rank": 512,
                "rms_norm_eps": 1e-5,
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
                "routed_scaling_factor": 2.5,
                "n_shared_experts": 1,
            }
            for key, val in semantic_requirements.items():
                if getattr(config, key, None) != val:
                    raise GLMFirstIntegrationError(
                        f"GLM pretrained semantics {key} changed"
                    )
            if list(getattr(config, "mlp_layer_types", ()) or ()) != (
                ["dense"] * 3 + ["sparse"] * 75
            ):
                raise GLMFirstIntegrationError("GLM original dense/MoE schedule changed")
            eos = getattr(config, "eos_token_id", None)
            if eos != [154820, 154827, 154829]:
                raise GLMFirstIntegrationError("GLM special token contract mismatch")
            for key, val in (("n_routed_experts", 256),
                             ("num_experts_per_tok", 8),
                             ("hidden_act", "silu"),
                             ("num_attention_heads", 64)):
                if getattr(config, key, None) != val:
                    raise GLMFirstIntegrationError(f"GLM source semantics {key} mismatch")

        placement = contract.glm_layers - 1 if insertion_layer is None else insertion_layer
        if not isinstance(placement, int) or not (0 <= placement < contract.glm_layers):
            raise GLMFirstIntegrationError("serial Mamba insertion index out of range")
        self.glm_model = glm_model
        self.glm_model.requires_grad_(False)
        self.glm_model.eval()
        self.path = SerialMamba3DonorPath(mamba_model, contract)
        self.insertion_layer = placement
        self.contract = contract
        self._in_forward = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.glm_model.eval()
        self.path.backbone.eval()
        return self

    def enable_branch(self, enabled: bool = True) -> None:
        self.path.enabled = bool(enabled)

    @contextmanager
    def _hook(self) -> Iterator[None]:
        if self._in_forward:
            raise GLMFirstIntegrationError("nested hybrid execution unsupported")
        self._in_forward = True

        def apply(_layer, _args, _kwargs, output):
            if isinstance(output, torch.Tensor):
                return self.path(output)
            if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                return (self.path(output[0]), *output[1:])
            raise GLMFirstIntegrationError("GLM decoder output contract changed")

        handle = self.glm_model.model.layers[self.insertion_layer].register_forward_hook(
            apply, with_kwargs=True
        )
        try:
            yield
        finally:
            handle.remove()
            self._in_forward = False

    def forward(self, *args: Any, **kwargs: Any):
        if self.path.enabled:
            # Fail closed on opaque positional params; the official GLM
            # CausalLM accepts input_ids positionally but other args may
            # encode caches, positions and attention masks.
            if len(args) > 1:
                raise GLMFirstIntegrationError("enabled hybrid accepts only positional input_ids")
            if kwargs.get("inputs_embeds") is not None:
                raise GLMFirstIntegrationError("enabled hybrid requires GLM token IDs")
            if not args and kwargs.get("input_ids") is None:
                raise GLMFirstIntegrationError("enabled hybrid requires GLM input_ids")
            GLMFirstMambaHybrid._noncached_input_check(args, kwargs)
        with self._hook():
            return self.glm_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        if self.path.enabled:
            raise GLMFirstIntegrationError(
                "nonzero serial-donor generation is unvalidated; "
                "full Mamba recurrent/cache parity and task quality are required"
            )
        # Nothing is installed when the Mamba branch is disabled.
        return self.glm_model.generate(*args, **kwargs)



def wrap_preloaded_official_serial_models(
    glm_model: nn.Module,
    mamba_model: nn.Module,
    *,
    tokenizer: Any,
    expected_glm_tokenizer_sha256: str,
    contract: GLMFirstMambaContract = GLMFirstMambaContract(),
    insertion_layer: int | None = None,
) -> GLMFirstSerialMambaHybrid:
    """Production construction requires the pinned GLM lexical fingerprint.

    This verifies model configuration and full tokenizer ID mapping, but
    does NOT independently verify pretrained tensor bytes or prove that the
    Mamba adapter is calibrated. Callers must validate source weight hashes
    before loading the actual pretrained checkpoints on approved hardware.
    """
    if contract == GLMFirstMambaContract():
        from iq_transfer.glm_first_tokenizer import verify_glm_tokenizer

        verify_glm_tokenizer(
            tokenizer,
            glm_model,
            expected_mapping_sha256=expected_glm_tokenizer_sha256,
        )
    return GLMFirstSerialMambaHybrid(
        glm_model, mamba_model,
        contract=contract, insertion_layer=insertion_layer,
    )
