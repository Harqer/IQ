"""Stateful inference for the complete IQ 235-layer hybrid backbone.

Single-sequence inference contract:
- prefill() executes the real full hybrid forward ONCE, with native MIMO final
  states, simultaneously building every context layer's inference cache.
- decode_token() executes the entire 235-layer schedule on ONE new token,
  using Mamba-3 MIMO step state, native CSA/HCA forward_step, dense KV caches,
  true MoE top-k routing and model-owned AttnRes/reasoning modules.
- sync_prompt() safely reuses a processed token prefix across chat turns. If
  retokenization changes the prefix, it deliberately rebuilds state.
- All checkpoint weights, route scores, tensor geometry and attention
  definitions are unchanged. No CPU/SISO/emulator inference fallback.

This class requires the pinned CUDA Mamba-3 MIMO runtime to generate tokens.
Do not silently substitute the full-prefix model as a purported decode cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from iq_model.architecture import HybridLayerType
from iq_model.position import apply_rotary


@dataclass
class IQInferenceState:
    max_context_tokens: int
    mamba: Any = field(default_factory=lambda: SimpleNamespace(
        seqlen_offset=0, key_value_memory_dict={},
    ))
    context: dict[int, Any] = field(default_factory=dict)
    dense_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    processed_ids: torch.Tensor | None = None
    reasoning_hidden: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None
    prefill_count: int = 0
    decode_count: int = 0

    @property
    def length(self) -> int:
        return 0 if self.processed_ids is None else self.processed_ids.shape[1]


class IQHybridInferenceEngine:
    """Single-request-at-a-time state manager for genuine hybrid IQ inference."""

    def __init__(self, model: nn.Module, *, max_context_tokens: int = 2048):
        if model.training:
            raise ValueError("IQ inference engine requires model.eval()")
        if model.multimodal is not None:
            raise ValueError("Stateful text inference does not support multimodal input")
        if not 1 <= max_context_tokens <= model.model_config.max_position_embeddings:
            raise ValueError("Invalid IQ positional context limit")
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.state = IQInferenceState(max_context_tokens)
        if len(model.layers) != len(model.config.schedule.layers):
            raise RuntimeError("IQ physical schedule/layers mismatch")
        for kind, layer in zip(model.config.schedule.layers, model.layers, strict=True):
            if kind is HybridLayerType.MAMBA3:
                if not hasattr(layer.mamba.core, "step"):
                    raise RuntimeError("Pinned Mamba-3 MIMO recurrent decode is missing")
                if getattr(layer.mamba.core, "mimo_rank", -1) != model.config.mamba3.mimo_rank:
                    raise RuntimeError("Mamba-3 MIMO rank mismatch")
            elif kind in {HybridLayerType.CSA, HybridLayerType.HCA}:
                if not callable(getattr(layer.attention, "forward_step", None)):
                    raise RuntimeError("Missing native compressed-context decode step")
            elif kind is HybridLayerType.DENSE_ATTENTION:
                if not hasattr(layer.attention, "k_proj"):
                    raise RuntimeError("Missing dense-context KV projection")
            elif kind is not HybridLayerType.MOE:
                raise RuntimeError(f"Unknown hybrid inference layer kind: {kind}")

    def reset(self) -> None:
        """Invalidate all token-dependent states without changing model weights."""
        self.state = IQInferenceState(self.max_context_tokens)

    def _validate_ids(self, ids: torch.Tensor, *, single: bool) -> None:
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("IQ inference accepts [1, sequence] token IDs")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("IQ token IDs must be int32/int64")
        n = ids.shape[1]
        if not (n == 1 if single else 1 <= n <= self.max_context_tokens):
            raise ValueError("Invalid IQ incremental token count")
        if ids.device != self.model.embed_tokens.weight.device:
            raise ValueError("IQ token IDs must reside on the model device")
        if bool(((ids < 0) | (ids >= self.model.model_config.vocab_size)).any()):
            raise ValueError("IQ token ID outside model vocabulary")

    def _dense_project(self, attention: nn.Module, hidden: torch.Tensor,
                       position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = attention.q_norm(attention._shape_q(attention.q_proj(hidden)))
        k = attention.k_norm(attention._shape_kv(attention.k_proj(hidden)))
        v = attention._shape_kv(attention.v_proj(hidden))
        cos, sin = attention.rotary.cos_sin(
            position, dtype=q.dtype, device=q.device,
        )
        q, k = apply_rotary(
            q, k, cos, sin, attention.config.rotary_dim,
        )
        return q, k, v

    @torch.inference_mode()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Build ALL layer caches from the same real full78 model forward."""
        self._validate_ids(ids, single=False)
        self.reset()
        state = self.state
        handles = []
        positions = torch.arange(
            ids.shape[1], device=ids.device, dtype=torch.long,
        ).unsqueeze(0)

        for physical, (kind, layer) in enumerate(zip(
            self.model.config.schedule.layers, self.model.layers, strict=True,
        )):
            if kind in {HybridLayerType.CSA, HybridLayerType.HCA}:
                def capture_compressed(module, args, index=physical):
                    hidden = args[0]
                    cache = module.new_cache(
                        device=hidden.device, dtype=hidden.dtype,
                    )
                    for token in range(hidden.shape[1]):
                        module.forward_step(
                            hidden[:, token:token + 1], cache, position=token,
                        )
                    state.context[index] = cache
                handles.append(layer.attention.register_forward_pre_hook(
                    capture_compressed,
                ))
            elif kind is HybridLayerType.DENSE_ATTENTION:
                def capture_dense(module, args, index=physical):
                    _, k, v = self._dense_project(module, args[0], positions)
                    state.dense_kv[index] = (k, v)
                handles.append(layer.attention.register_forward_pre_hook(
                    capture_dense,
                ))

        if self.model.reasoning is not None:
            def capture_reasoning(_module, args):
                state.reasoning_hidden = args[0].detach()
            handles.append(self.model.reasoning.register_forward_pre_hook(
                capture_reasoning,
            ))

        try:
            result = self.model(
                ids, mamba_inference_params=state.mamba,
                return_hidden_states=True,
            )
        except Exception:
            self.reset()
            raise
        finally:
            for hook in handles:
                hook.remove()

        expected_mamba = sum(
            1 for k in self.model.config.schedule.layers
            if k is HybridLayerType.MAMBA3
        )
        if len(state.mamba.key_value_memory_dict) != expected_mamba:
            self.reset()
            raise RuntimeError(
                "Full78 Mamba-3 MIMO prefill did not populate every state"
            )
        for kind, physical in (
            (kind, physical)
            for physical, kind in enumerate(self.model.config.schedule.layers)
        ):
            if kind in {HybridLayerType.CSA, HybridLayerType.HCA}:
                if physical not in state.context or state.context[physical].length != ids.shape[1]:
                    self.reset()
                    raise RuntimeError("Missing compressed-context prefill state")
            elif kind is HybridLayerType.DENSE_ATTENTION:
                if physical not in state.dense_kv:
                    self.reset()
                    raise RuntimeError("Missing dense attention KV state")
        if self.model.reasoning is not None and state.reasoning_hidden is None:
            self.reset()
            raise RuntimeError("Missing reasoning context during prefill")
        state.processed_ids = ids.detach().clone()
        state.mamba.seqlen_offset = ids.shape[1]
        state.last_logits = result.logits[:, -1:, :]
        state.prefill_count += 1
        print(
            f"IQ_STATEFUL_PREFILL_PASS=tokens={ids.shape[1]} "
            f"mamba_layers={expected_mamba} "
            f"attention_layers={len(state.context)+len(state.dense_kv)}",
            flush=True,
        )
        return state.last_logits

    def _dense_decode(self, physical: int, layer: nn.Module,
                      normalized: torch.Tensor, position: int) -> torch.Tensor:
        att = layer.attention
        positions = torch.full(
            (1, 1), position, device=normalized.device, dtype=torch.long,
        )
        q, k, v = self._dense_project(att, normalized, positions)
        old_k, old_v = self.state.dense_kv[physical]
        new_k = torch.cat((old_k, k), dim=2)
        new_v = torch.cat((old_v, v), dim=2)
        if new_k.shape[2] != position + 1:
            raise RuntimeError("Dense-context cache length mismatch")
        self.state.dense_kv[physical] = (new_k, new_v)
        if att.config.kv_repeat != 1:
            new_k = new_k.repeat_interleave(att.config.kv_repeat, dim=1)
            new_v = new_v.repeat_interleave(att.config.kv_repeat, dim=1)
        # The new query is the last token: every key in the cache is causal.
        output = F.scaled_dot_product_attention(
            q, new_k, new_v, dropout_p=0.0, is_causal=False,
        )
        return att.o_proj(
            output.transpose(1, 2).contiguous().view(1, 1, -1)
        )

    def _mamba_decode(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        core = layer.mamba.core
        idx = core.layer_idx
        states = self.state.mamba.key_value_memory_dict.get(idx)
        if states is None or len(states) != 4:
            raise RuntimeError(f"Missing native MIMO recurrent state for layer {idx}")
        normalized = layer.norm(hidden)
        result = core.step(normalized[:, 0, :], *states)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("Native Mamba-3 MIMO step returned invalid state tuple")
        mixed = result[0]
        if mixed.shape != normalized[:, 0, :].shape:
            raise RuntimeError("Native Mamba-3 MIMO token output has wrong shape")
        if not bool(torch.isfinite(mixed).all()):
            raise RuntimeError("Native Mamba-3 MIMO step produced invalid values")
        return hidden + layer.residual_dropout(mixed.unsqueeze(1))

    @torch.inference_mode()
    def decode_token(self, token_id: torch.Tensor) -> torch.Tensor:
        """Advance MIMO, context, MoE, AttnRes and reasoning one token exactly."""
        self._validate_ids(token_id, single=True)
        state = self.state
        if state.processed_ids is None or state.last_logits is None:
            raise RuntimeError("IQ requires prefill before decoding")
        position = state.length
        if position >= state.max_context_tokens:
            raise ValueError("IQ context budget exhausted; cannot decode")
        if state.mamba.seqlen_offset != position:
            raise RuntimeError("Mamba-3 state offset does not match IQ token history")
        x = self.model.embed_tokens(token_id)
        attnres = self.model.attnres
        attn_state = attnres.init_state(x) if attnres is not None else None

        for physical, (kind, layer) in enumerate(zip(
            self.model.config.schedule.layers, self.model.layers, strict=True,
        )):
            layer_input = (
                attnres.read(attn_state) if attn_state is not None else x
            )
            if kind is HybridLayerType.MAMBA3:
                output = self._mamba_decode(layer, layer_input)
            elif kind is HybridLayerType.MOE:
                output = layer(layer_input).hidden_states
            elif kind is HybridLayerType.DENSE_ATTENTION:
                normal = layer.norm(layer_input)
                output = layer_input + layer.residual_dropout(
                    self._dense_decode(physical, layer, normal, position)
                )
            elif kind in {HybridLayerType.CSA, HybridLayerType.HCA}:
                normal = layer.norm(layer_input)
                if physical not in state.context:
                    raise RuntimeError("Missing compressed-context layer cache")
                context = state.context[physical]
                if context.length != position:
                    raise RuntimeError("Compressed-context cache offset mismatch")
                output = layer_input + layer.residual_dropout(
                    layer.attention.forward_step(
                        normal, context, position=position,
                    )
                )
            else:
                raise RuntimeError("Unhandled IQ inference layer")
            if attn_state is not None:
                attn_state = attnres.advance(
                    attn_state, output - layer_input,
                )
            x = output

        if attn_state is not None:
            x = attnres.finalize(attn_state)
        hidden = self.model.norm(x)
        if self.model.reasoning is not None:
            if state.reasoning_hidden is None:
                raise RuntimeError("Missing IQ reasoning history")
            state.reasoning_hidden = torch.cat(
                (state.reasoning_hidden, hidden), dim=1,
            )
            result = self.model.reasoning(
                state.reasoning_hidden,
                verifier=self.model.reasoning_verifier,
                require_verifier_stability=self.model.config.require_verifier_stability,
            )
            hidden = self.model.reasoning_injector(hidden, result.state)

        logits = self.model.lm_head(hidden)
        if logits.shape != (1, 1, self.model.model_config.vocab_size):
            raise RuntimeError("IQ recurrent decode returned incorrect logits")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("IQ recurrent decode produced non-finite logits")
        state.last_logits = logits
        state.processed_ids = torch.cat(
            (state.processed_ids, token_id), dim=1,
        )
        state.mamba.seqlen_offset = position + 1
        state.decode_count += 1
        print(f"IQ_STATEFUL_DECODE_PASS=position={position} "
              f"count={state.decode_count}", flush=True)
        return logits

    @torch.inference_mode()
    def sync_prompt(self, ids: torch.Tensor) -> torch.Tensor:
        """Reuse matching complete prefix; rebuild on tokenization changes."""
        self._validate_ids(ids, single=False)
        processed = self.state.processed_ids
        if processed is None or ids.shape[1] < processed.shape[1] or not torch.equal(
            processed, ids[:, :processed.shape[1]],
        ):
            if processed is not None:
                print("IQ_STATEFUL_PROMPT_RESET=token_prefix_changed", flush=True)
            return self.prefill(ids)
        for i in range(processed.shape[1], ids.shape[1]):
            self.decode_token(ids[:, i:i+1])
        assert self.state.last_logits is not None
        return self.state.last_logits
