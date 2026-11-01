"""CPU orchestration tests: correct native MIMO token/state semantics.

This does NOT claim CUDA kernel validation. H200 parity runs separately.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from iq_model.state.mamba3 import (
    Mamba3MIMORuntimeError, native_mimo_aligned_prefill,
)


class FakeNativeMIMO:
    layer_idx = 39

    def __init__(self):
        self.prefix_calls = []
        self.step_calls = 0

    def _get_states_from_cache(self, params, batch):
        if self.layer_idx not in params.key_value_memory_dict:
            params.key_value_memory_dict[self.layer_idx] = tuple(
                torch.zeros((batch, 4), dtype=torch.float32) for _ in range(4)
            )
        return params.key_value_memory_dict[self.layer_idx]

    def __call__(self, tokens, *, inference_params):
        self.prefix_calls.append(tokens.shape[1])
        state = list(self._get_states_from_cache(inference_params, tokens.shape[0]))
        outputs = []
        for pos in range(tokens.shape[1]):
            current = state[0] + tokens[:, pos].float()
            outputs.append(current)
            state[0] = current
        inference_params.key_value_memory_dict[self.layer_idx] = tuple(state)
        return torch.stack(outputs, dim=1)

    def step(self, token, *states):
        self.step_calls += 1
        next_val = states[0] + token.float()
        return next_val, next_val, states[1], states[2], states[3]


class NativeChunkAlignedTests(unittest.TestCase):
    def test_18_token_16_plus_2_native_steps_preserves_outputs_and_cache(self):
        for length in (13, 17, 18, 31, 33):
            with self.subTest(length=length):
                core = FakeNativeMIMO()
                params = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})
                inputs = torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4)
                actual = native_mimo_aligned_prefill(core, inputs, params, chunk_size=16)
                expected = torch.cumsum(inputs, dim=1)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(core.prefix_calls, [length // 16 * 16] if length >= 16 else [])
                self.assertEqual(core.step_calls, length % 16)
                self.assertEqual(len(params.key_value_memory_dict[39]), 4)
                torch.testing.assert_close(
                    params.key_value_memory_dict[39][0], expected[:, -1], atol=0, rtol=0
                )

    def test_missing_or_stale_native_state_fails_closed(self):
        core = FakeNativeMIMO()
        inputs = torch.zeros((1, 18, 4))
        stale = SimpleNamespace(seqlen_offset=1, key_value_memory_dict={})
        with self.assertRaisesRegex(Mamba3MIMORuntimeError, "fresh state"):
            native_mimo_aligned_prefill(core, inputs, stale, chunk_size=16)
        prepopulated = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={39: ("bad",)})
        with self.assertRaisesRegex(Mamba3MIMORuntimeError, "unexpectedly populated"):
            native_mimo_aligned_prefill(core, inputs, prepopulated, chunk_size=16)

    def test_bad_step_state_tuple_is_rejected(self):
        class BrokenStep(FakeNativeMIMO):
            def step(self, token, *states):
                return (torch.zeros_like(token),)
        params = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})
        with self.assertRaisesRegex(Mamba3MIMORuntimeError, "wrong state tuple"):
            native_mimo_aligned_prefill(
                BrokenStep(), torch.zeros((1, 18, 4)), params, chunk_size=16
            )


if __name__ == "__main__":
    unittest.main()
