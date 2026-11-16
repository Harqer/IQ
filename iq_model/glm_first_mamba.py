"""GLM-5.3-first hybrid with complete pretrained Mamba-3 MIMO donor blocks.

This is the first *implementable*, function-neutral insertion path. It does NOT
convert GLM attention weights into Mamba state weights, does NOT replace GLM's
tokenizer or lexical weights, and does NOT claim learned hybrid quality.

The original GLM modules remain registered under glm_model. Forward hooks
temporarily intercept selected decoder-layer outputs and preserve their other
return values (including DSA routing indices). Inactive zero-gated branches are
never invoked: baseline forward/generate delegates exactly to original GLM.

Active recurrent branches currently support non-cached full-sequence forward
only. Incremental Mamba state, padded/packed batches and caching MUST be
validated before accepting nonzero-gate generation or deployment.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch import nn


class GLMFirstIntegrationError(RuntimeError):
    """Refuse a structurally or semantically incompatible hybrid."""


@dataclass(frozen=True)
class GLMFirstMambaContract:
    glm_vocab: int = 154880
    glm_hidden: int = 6144
    glm_layers: int = 78
    mamba_hidden: int = 2048
    mamba_layers: int = 24
    mamba_mlp_hidden: int = 3840  # 3824 rounded to the upstream MLP multiple-of-128
    mimo_rank: int = 4
    state_size: int = 128

    def __post_init__(self) -> None:
        if min(
            self.glm_vocab, self.glm_hidden, self.glm_layers,
            self.mamba_hidden, self.mamba_layers, self.mamba_mlp_hidden,
            self.mimo_rank, self.state_size,
        ) <= 0:
            raise ValueError("GLM/Mamba contract dimensions must be positive")


def evenly_spaced_insertions(glm_layers: int, mamba_layers: int) -> tuple[int, ...]:
    """Deterministic *provisional* schedule; not a trained/validated ratio."""
    if not 1 <= mamba_layers <= glm_layers:
        raise GLMFirstIntegrationError("Mamba block count must be <= GLM block count")
    if mamba_layers == 1:
        return (glm_layers - 1,)
    result = tuple(
        round(i * (glm_layers - 1) / (mamba_layers - 1))
        for i in range(mamba_layers)
    )
    if len(set(result)) != len(result):
        raise GLMFirstIntegrationError("Mamba placement collides")
    return result


def _required_submodule(block: nn.Module, path: str) -> nn.Module:
    target: Any = block
    for part in path.split("."):
        target = getattr(target, part, None)
    if not isinstance(target, nn.Module):
        raise GLMFirstIntegrationError(f"donor block lacks required module {path}")
    return target


def validate_complete_mamba_block(
    block: nn.Module, contract: GLMFirstMambaContract,
) -> None:
    """Validate both donor sublayers, not just a rank-4 MIMO mixer."""
    norm = _required_submodule(block, "norm")
    norm2 = _required_submodule(block, "norm2")
    mixer = _required_submodule(block, "mixer")
    fc1 = _required_submodule(block, "mlp.fc1")
    fc2 = _required_submodule(block, "mlp.fc2")
    if getattr(mixer, "is_mimo", False) is not True:
        raise GLMFirstIntegrationError("donor Mamba mixer is not MIMO")
    if getattr(mixer, "mimo_rank", None) != contract.mimo_rank:
        raise GLMFirstIntegrationError("donor MIMO rank differs from contract")
    if getattr(mixer, "d_state", None) != contract.state_size:
        raise GLMFirstIntegrationError("donor MIMO state size differs from contract")
    for name, module in (("norm", norm), ("norm2", norm2)):
        weight = getattr(module, "weight", None)
        if weight is None or tuple(weight.shape) != (contract.mamba_hidden,):
            raise GLMFirstIntegrationError(f"donor {name} has wrong hidden width")
    if tuple(fc1.weight.shape) != (
        2 * contract.mamba_mlp_hidden, contract.mamba_hidden
    ):
        raise GLMFirstIntegrationError("donor gated MLP fc1 tensor shape mismatch")
    if tuple(fc2.weight.shape) != (
        contract.mamba_hidden, contract.mamba_mlp_hidden
    ):
        raise GLMFirstIntegrationError("donor gated MLP fc2 tensor shape mismatch")


class FullMamba3DonorBranch(nn.Module):
    """Injects one *complete* pretrained Mamba donor block into GLM width.

    The donor block includes its mixer, its *own* gated MLP, and both
    pretrained normalizations. Its original weight tensors are not widened or
    reinterpreted. Only the GLM<->Mamba adapters are new/learnable.
    """

    def __init__(
        self,
        donor_block: nn.Module,
        contract: GLMFirstMambaContract,
    ) -> None:
        super().__init__()
        validate_complete_mamba_block(donor_block, contract)
        self.donor_block = donor_block
        # Match the actual pretrained donor block's parameter placement and
        # precision; avoid accidental CPU FP32 adapters on a CUDA BF16 donor.
        donor_norm_weight = donor_block.norm.weight
        factory = {"device": donor_norm_weight.device, "dtype": donor_norm_weight.dtype}
        self.in_adapter = nn.Linear(
            contract.glm_hidden, contract.mamba_hidden, bias=False, **factory
        )
        self.out_adapter = nn.Linear(
            contract.mamba_hidden, contract.glm_hidden, bias=False, **factory
        )
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.enabled = False

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.in_adapter.in_features:
            raise GLMFirstIntegrationError("GLM hidden-state shape is incompatible")
        if not self.enabled:
            if bool(torch.count_nonzero(self.gate.detach())):
                raise GLMFirstIntegrationError("nonzero gate on a disabled branch")
            return hidden  # Exact GLM baseline: do not even invoke Mamba.
        if hidden.device != self.in_adapter.weight.device:
            raise GLMFirstIntegrationError(
                "Mamba donor and adapters must be colocated with the selected GLM layer; "
                "distribute/place all blocks before enabling the hybrid"
            )
        if hidden.device.type != "cuda" and type(self.donor_block).__module__.startswith("mamba_ssm"):
            raise GLMFirstIntegrationError("official Mamba-3 kernel needs validated CUDA")
        # Match adapter parameter dtype without changing GLM's residual dtype.
        projected = self.in_adapter(hidden.to(self.in_adapter.weight.dtype))
        # Official upstream Block.forward returns (hidden_states, residual).
        # The next donor block adds these before its first normalization.
        # To use a donor block as an isolated residual function, restore that
        # pending add after both pretrained mixer and gated MLP operations.
        result = self.donor_block(projected, residual=None, inference_params=None)
        if not (
            isinstance(result, tuple) and len(result) == 2
            and isinstance(result[0], torch.Tensor)
            and isinstance(result[1], torch.Tensor)
        ):
            raise GLMFirstIntegrationError("Mamba donor block return contract changed")
        donor_hidden = result[0] + result[1]
        if donor_hidden.shape != projected.shape:
            raise GLMFirstIntegrationError("Mamba donor block changed hidden dimensions")
        if not bool(torch.isfinite(donor_hidden).all()):
            raise GLMFirstIntegrationError("Mamba donor block produced nonfinite states")
        delta = self.out_adapter(donor_hidden.to(self.out_adapter.weight.dtype))
        delta = delta.to(hidden.dtype)
        return hidden + self.gate.to(hidden.dtype) * delta


class GLMFirstMambaHybrid(nn.Module):
    """Non-destructive GLM layer overlay with 24 complete Mamba donor blocks."""

    def __init__(
        self,
        glm_model: nn.Module,
        donor_blocks: Sequence[nn.Module],
        *,
        contract: GLMFirstMambaContract = GLMFirstMambaContract(),
        placements: Sequence[int] | None = None,
        freeze_glm: bool = True,
    ) -> None:
        super().__init__()
        config = getattr(glm_model, "config", None)
        if config is None:
            raise GLMFirstIntegrationError("original GLM model must expose config")
        if contract == GLMFirstMambaContract():
            # Production: preserve GLM's forward semantics, not just matrix sizes.
            official_semantics = {
                "model_type": "glm_moe_dsa",
                "num_attention_heads": 64,
                "qk_head_dim": 256,
                "hidden_act": "silu",
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
                "num_experts_per_tok": 8,
                "n_routed_experts": 256,
                "n_shared_experts": 1,
                "rms_norm_eps": 1e-5,
                "routed_scaling_factor": 2.5,
            }
            for attr, value in official_semantics.items():
                if getattr(config, attr, None) != value:
                    raise GLMFirstIntegrationError(
                        f"source GLM semantics {attr} differ from official contract"
                    )
            if list(getattr(config, "eos_token_id", [])) != [154820, 154827, 154829]:
                raise GLMFirstIntegrationError("GLM EOS IDs have changed")
            kinds = getattr(config, "mlp_layer_types", None)
            if list(kinds or []) != ["dense"] * 3 + ["sparse"] * 75:
                raise GLMFirstIntegrationError("GLM dense/MoE layer pattern changed")
        for key, expected in (
            ("vocab_size", contract.glm_vocab),
            ("hidden_size", contract.glm_hidden),
            ("num_hidden_layers", contract.glm_layers),
        ):
            if getattr(config, key, None) != expected:
                raise GLMFirstIntegrationError(f"original GLM config {key} mismatch")
        if getattr(config, "tie_word_embeddings", False):
            raise GLMFirstIntegrationError("GLM lexical weights must remain untied")
        decoder = getattr(glm_model, "model", None)
        layers = getattr(decoder, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) != contract.glm_layers:
            raise GLMFirstIntegrationError("original GLM decoder layers unavailable")
        embedding = getattr(decoder, "embed_tokens", None)
        head = getattr(glm_model, "lm_head", None)
        if not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear):
            raise GLMFirstIntegrationError("original GLM embedding/output head unavailable")
        if tuple(embedding.weight.shape) != (contract.glm_vocab, contract.glm_hidden):
            raise GLMFirstIntegrationError("original GLM embeddings have wrong shape")
        if tuple(head.weight.shape) != (contract.glm_vocab, contract.glm_hidden):
            raise GLMFirstIntegrationError("original GLM LM head has wrong shape")

        if len(donor_blocks) != contract.mamba_layers:
            raise GLMFirstIntegrationError("must supply ALL complete Mamba donor blocks")
        target_positions = tuple(placements) if placements is not None else (
            evenly_spaced_insertions(contract.glm_layers, contract.mamba_layers)
        )
        if (
            len(target_positions) != contract.mamba_layers
            or len(set(target_positions)) != len(target_positions)
            or tuple(sorted(target_positions)) != target_positions
            or any(p < 0 or p >= contract.glm_layers for p in target_positions)
        ):
            raise GLMFirstIntegrationError("invalid ordered Mamba placement schedule")

        self.glm_model = glm_model
        self.contract = contract
        self.placements = target_positions
        self.branches = nn.ModuleList(
            [FullMamba3DonorBranch(block, contract) for block in donor_blocks]
        )
        if freeze_glm:
            self.glm_model.requires_grad_(False)
        # Never auto-enable branches or declare the resulting model trained.
        self._in_forward = False

    def enable_branches(self, enabled: bool = True) -> None:
        for branch in self.branches:
            branch.enabled = bool(enabled)

    def _active(self) -> bool:
        return any(branch.enabled for branch in self.branches)

    @staticmethod
    def _noncached_input_check(args: tuple, kwargs: dict) -> None:
        if kwargs.get("use_cache", True) is not False:
            raise GLMFirstIntegrationError(
                "active Mamba branch requires explicit use_cache=False "
                "until stateful GLM KV + Mamba SSM decode is validated"
            )
        if kwargs.get("cache") is not None or kwargs.get("past_key_values") is not None:
            raise GLMFirstIntegrationError("active branch cannot reuse cached GLM states")
        mask = kwargs.get("attention_mask")
        if isinstance(mask, torch.Tensor) and not bool(torch.all(mask != 0)):
            raise GLMFirstIntegrationError(
                "active branch needs unpadded sequences until Mamba masking is verified"
            )
        pos = kwargs.get("position_ids")
        if pos is not None:
            if not isinstance(pos, torch.Tensor) or pos.ndim != 2:
                raise GLMFirstIntegrationError("unsupported position_ids format")
            expected = torch.arange(pos.shape[-1], device=pos.device).expand_as(pos)
            if not torch.equal(pos, expected):
                raise GLMFirstIntegrationError("active Mamba requires uncached contiguous positions")

    def _hook(self, block_index: int):
        branch = self.branches[block_index]

        def add_branch(_module, _args, _kwargs, output):
            if isinstance(output, torch.Tensor):
                return branch(output)
            if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                return (branch(output[0]), *output[1:])
            raise GLMFirstIntegrationError(
                "GLM decoder return must be Tensor or tuple[Tensor, ...]"
            )

        return add_branch

    @contextmanager
    def _installed_hooks(self) -> Iterator[None]:
        if self._in_forward:
            raise GLMFirstIntegrationError("nested hybrid execution is unsupported")
        self._in_forward = True
        handles = []
        try:
            for index, physical in enumerate(self.placements):
                handles.append(self.glm_model.model.layers[physical].register_forward_hook(
                    self._hook(index), with_kwargs=True
                ))
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._in_forward = False

    def forward(self, *args: Any, **kwargs: Any):
        if self._active():
            self._noncached_input_check(args, kwargs)
        with self._installed_hooks():
            return self.glm_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        """Native GLM generation only while all new Mamba branches are disabled."""
        if self._active():
            raise GLMFirstIntegrationError(
                "hybrid generation is blocked until trained Mamba recurrence, "
                "KV/SSM cache integration, and decoding parity are validated"
            )
        generate = getattr(self.glm_model, "generate", None)
        if not callable(generate):
            raise GLMFirstIntegrationError("GLM model lacks native generate()")
        with self._installed_hooks():
            return generate(*args, **kwargs)


def wrap_preloaded_official_models(
    glm_model: nn.Module,
    mamba_model: nn.Module,
    *,
    contract: GLMFirstMambaContract = GLMFirstMambaContract(),
    placements: Sequence[int] | None = None,
) -> GLMFirstMambaHybrid:
    """Reuse the EXACT pretrained Mamba block objects, not just mixer kernels.

    Callers are responsible for loading/checksumming the immutable official
    source revisions, and for placing weights on a supported device. This
    function does not allocate a teacher, download 753B weights, or run a job.
    """
    if contract == GLMFirstMambaContract():
        donor_config = getattr(mamba_model, "config", None)
        if donor_config is None:
            raise GLMFirstIntegrationError("official Mamba donor config unavailable")
        for attr, expected in (
            ("n_layer", 24), ("d_model", 2048),
            ("d_intermediate", 3824), ("tie_embeddings", True),
        ):
            if getattr(donor_config, attr, None) != expected:
                raise GLMFirstIntegrationError(
                    f"official Mamba donor config {attr} mismatch"
                )
        ssm = getattr(donor_config, "ssm_cfg", None)
        if not isinstance(ssm, dict) or (
            ssm.get("is_mimo") is not True
            or ssm.get("mimo_rank") != 4
            or ssm.get("d_state") != 128
            or ssm.get("layer") != "Mamba3"
        ):
            raise GLMFirstIntegrationError(
                "official Mamba donor configuration is not rank-4 MIMO"
            )
    backbone = getattr(mamba_model, "backbone", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise GLMFirstIntegrationError("official Mamba backbone.layers unavailable")
    return GLMFirstMambaHybrid(
        glm_model, list(layers),
        contract=contract, placements=placements,
    )
