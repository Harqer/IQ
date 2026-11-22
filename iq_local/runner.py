"""Greedy streaming decode runner for CPUHybridForCausalLM.

Drives the schedule layer-by-layer (not `model.forward`), mirroring the
production forward composition (iq_model/hybrid.py:855-1070):

- Mamba layers keep one `CPUMamba3State` per layer across decode steps
  (`prefill` for the prompt, `step` per token; bf16 activations, fp32
  internals per Task 7).
- Production attention layers are stateless per call and derive positions
  internally (iq_model/attention/context_dense.py:51,
  iq_model/attention/compressed.py:1009); the v1 streaming approach
  recomputes the full prefix through each attention layer per decode step,
  caching that layer's per-token inputs, and takes the last position. This
  reproduces the production semantics exactly (causal attention) at O(T)
  attention work per layer per step.
- MoE layers call the Task 8 functional layer forward with an
  `expert_fn(stage, expert)` closure over `StreamingWeightProvider`, so the
  runner and parity tests share one code path. Routing is the production
  `_route` math itself (`FunctionalStableLatentMoE._route`), guaranteeing
  identical results to `StableLatentMoE._route` on the same weights.
- AttnRes block residuals are per-position independent
  (iq_model/residual/attnres.py:101-169): each decode step rebuilds the
  state from the token embedding, which equals the full-sequence state
  sliced at that position.
- Reasoning recurrence pools context over the sequence
  (iq_model/reasoning/recurrence.py:381); each decode step recomputes it
  over the cached per-token post-norm hiddens plus the current token,
  reproducing `model.forward(prefix)` at the last position.

v1 limits, documented: no padding (attention_mask=None paths), batch size 1,
greedy argmax only. `reset()` clears all streaming state; there is no other
mutable global.
"""

from __future__ import annotations

import torch

from iq_local.cpu_hybrid import (
    CompressedContextResidualLayer,
    CPUMamba3ResidualLayer,
    DenseContextResidualLayer,
    FunctionalStableLatentMoELayer,
)
from iq_model.hybrid import _reasoning_masks


class HybridRunner:
    def __init__(self, model, provider) -> None:
        self.model = model
        self.model.eval()
        self.provider = provider
        self._layers = list(model.layers)
        self._schedule = list(model.config.schedule.layers)
        if len(self._schedule) != len(self._layers):
            raise ValueError("schedule length must match model layers")
        self._moe_stage_of = {}
        stage = 0
        for position, layer in enumerate(self._layers):
            if isinstance(layer, FunctionalStableLatentMoELayer):
                self._moe_stage_of[position] = stage
                stage += 1
        self.reset()

    def reset(self) -> None:
        self._mamba_states: dict[int, object] = {}
        self._attn_cache: dict[int, list[torch.Tensor]] = {}
        self._hidden_cache: list[torch.Tensor] = []
        self._token_ids: list[int] = []

    def _expert_fn(self, stage: int, expert: int):
        return self.provider.load_expert(stage, int(expert))

    def _moe_layer_for_stage(self, stage: int) -> FunctionalStableLatentMoELayer:
        for position, moe_stage in self._moe_stage_of.items():
            if moe_stage == stage:
                return self._layers[position]
        raise ValueError(f"no MoE layer for stage {stage}")

    def select_experts(self, x: torch.Tensor, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Production router mirror: returns (selected_weights, selected_experts).

        Delegates to the functional MoE layer's `_route`, which is a verbatim
        mirror of production `StableLatentMoE._route` (iq_model/mlp/
        latent_moe.py:155-174), so indices and fp32 weights match exactly.
        """
        layer = self._moe_layer_for_stage(stage)
        hidden = layer.moe.config.hidden_size
        if x.ndim < 2 or x.shape[-1] != hidden:
            raise ValueError(
                f"select_experts input must end in hidden_size={hidden}"
            )
        flat = x.reshape(-1, hidden)
        if flat.shape[0] == 0:
            raise ValueError("select_experts requires at least one token")
        _, selected, weights = layer.moe._route(flat)
        return weights, selected

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        stop_tokens: set[int] | None = None,
        progress_callback=None,
    ) -> list[int]:
        if not prompt_ids:
            raise ValueError("prompt_ids must contain at least one token")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(token < 0 or token >= self.model.model_config.vocab_size for token in prompt_ids):
            raise ValueError("prompt token outside vocabulary")
        self.reset()
        logits = self._prefill(prompt_ids, progress_callback=progress_callback)
        next_token = int(logits[0, -1].argmax())
        generated = [next_token]
        if progress_callback is not None:
            progress_callback({"phase": "token", "token_index": 1, "token_id": next_token})
        if stop_tokens and next_token in stop_tokens:
            return generated
        for _ in range(max_new_tokens - 1):
            logits = self._decode_step(next_token, progress_callback=progress_callback)
            next_token = int(logits[0, -1].argmax())
            generated.append(next_token)
            if progress_callback is not None:
                progress_callback({
                    "phase": "token",
                    "token_index": len(generated),
                    "token_id": next_token,
                })
            if stop_tokens and next_token in stop_tokens:
                break
        return generated

    def _prefill(self, prompt_ids: list[int], *, progress_callback=None) -> torch.Tensor:
        self._token_ids = list(prompt_ids)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        x = self.model.embed_tokens(input_ids)
        attnres_state = (
            self.model.attnres.init_state(x)
            if self.model.attnres is not None
            else None
        )
        for position, layer in enumerate(self._layers):
            layer_input = (
                self.model.attnres.read(attnres_state)
                if attnres_state is not None
                else x
            )
            layer_output = self._run_layer(position, layer, layer_input, prefill=True)
            if attnres_state is not None:
                attnres_state = self.model.attnres.advance(
                    attnres_state,
                    layer_output - layer_input,
                )
            x = layer_output
            self._notify_layer(progress_callback, "prefill", position)
        if self.model.attnres is not None:
            assert attnres_state is not None
            x = self.model.attnres.finalize(attnres_state)
        hidden = self.model.norm(x)
        self._hidden_cache = [row for row in hidden[0]]
        final_hidden = self._apply_reasoning(hidden)
        return self.model.lm_head(final_hidden)

    def _decode_step(self, token_id: int, *, progress_callback=None) -> torch.Tensor:
        self._token_ids.append(token_id)
        input_ids = torch.tensor([[token_id]], dtype=torch.long)
        x = self.model.embed_tokens(input_ids)
        attnres_state = (
            self.model.attnres.init_state(x)
            if self.model.attnres is not None
            else None
        )
        for position, layer in enumerate(self._layers):
            layer_input = (
                self.model.attnres.read(attnres_state)
                if attnres_state is not None
                else x
            )
            layer_output = self._run_layer(position, layer, layer_input, prefill=False)
            if attnres_state is not None:
                attnres_state = self.model.attnres.advance(
                    attnres_state,
                    layer_output - layer_input,
                )
            x = layer_output
            self._notify_layer(progress_callback, "decode", position)
        if self.model.attnres is not None:
            assert attnres_state is not None
            x = self.model.attnres.finalize(attnres_state)
        hidden = self.model.norm(x)
        self._hidden_cache.append(hidden[0, 0])
        history = torch.stack(self._hidden_cache, dim=0).unsqueeze(0)
        final_hidden = self._apply_reasoning(history)
        logits = self.model.lm_head(final_hidden)
        return logits[:, -1:]

    def _notify_layer(self, progress_callback, phase: str, position: int) -> None:
        # Callbacks run only after real layer math returned, never at dispatch.
        done = position + 1
        if progress_callback is not None and (done % 10 == 0 or done == len(self._layers)):
            progress_callback({
                "phase": phase,
                "completed_layers": done,
                "total_layers": len(self._layers),
            })

    def _run_layer(
        self,
        position: int,
        layer,
        layer_input: torch.Tensor,
        *,
        prefill: bool,
    ) -> torch.Tensor:
        if isinstance(layer, CPUMamba3ResidualLayer):
            normed = layer.norm(layer_input)
            if prefill:
                mixed, state = layer.mamba.prefill(normed)
                self._mamba_states[position] = state
                return layer_input + layer.residual_dropout(mixed)
            y, state = layer.mamba.step(
                normed.reshape(1, -1),
                self._mamba_states[position],
            )
            self._mamba_states[position] = state
            return layer_input + layer.residual_dropout(y.unsqueeze(1))
        if isinstance(
            layer,
            (DenseContextResidualLayer, CompressedContextResidualLayer),
        ):
            if prefill:
                layer_output = layer(
                    layer_input,
                    position_ids=None,
                    attention_mask=None,
                    document_ids=None,
                )
                self._attn_cache[position] = [row for row in layer_input[0]]
                return layer_output
            cached = torch.stack(self._attn_cache[position], dim=0)
            full = torch.cat([cached, layer_input[0]], dim=0).unsqueeze(0)
            layer_output = layer(
                full,
                position_ids=None,
                attention_mask=None,
                document_ids=None,
            )
            self._attn_cache[position].append(layer_input[0, 0])
            return layer_output[:, -1:]
        if isinstance(layer, FunctionalStableLatentMoELayer):
            output = layer(layer_input, self._expert_fn)
            return output.hidden_states
        raise ValueError(
            f"unsupported layer type at position {position}: {type(layer).__name__}"
        )

    def _apply_reasoning(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.model.reasoning is None:
            return hidden
        input_ids = torch.tensor([self._token_ids], dtype=torch.long)
        context_mask, injection_mask = _reasoning_masks(
            input_ids,
            None,
            None,
            labels_present=False,
        )
        output = self.model.reasoning(
            hidden,
            attention_mask=None,
            reasoning_context_mask=context_mask,
            document_ids=None,
            verifier=None,
            require_verifier_stability=self.model.config.require_verifier_stability,
        )
        return self.model.reasoning_injector(
            hidden,
            output.state,
            token_mask=injection_mask,
        )
