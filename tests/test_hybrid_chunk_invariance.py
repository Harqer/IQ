from __future__ import annotations

import unittest

import torch

from iq_model import (
    BlockAttentionResidual,
    BlockAttnResConfig,
    StableLatentMoEConfig,
    StableLatentMoELayer,
)


class HybridTokenLocalComponentsTests(unittest.TestCase):
    """Real IQ components; no mocks or replacement Mamba-3 runtime."""

    def test_canonical_latent_moe_and_block_attnres_have_chunk_invariant_tokens(self):
        torch.manual_seed(93)
        moe = StableLatentMoELayer(
            StableLatentMoEConfig(
                hidden_size=16,
                latent_size=8,
                expert_intermediate_size=24,
                num_experts=4,
                top_k=2,
            ),
            residual_dropout=0.0,
        ).eval()
        attnres = BlockAttentionResidual(
            BlockAttnResConfig(
                hidden_size=16,
                num_layers=4,
                block_size=2,
            )
        ).eval()
        x = torch.randn(2, 9, 16)

        def run(sequence: torch.Tensor) -> torch.Tensor:
            state = attnres.init_state(sequence)
            for _ in range(4):
                layer_input = attnres.read(state)
                layer_output = moe(layer_input).hidden_states
                state = attnres.advance(state, layer_output - layer_input)
            return attnres.finalize(state)

        with torch.no_grad():
            full = run(x)
            for chunks in ((9,), (1,) * 9, (3, 2, 4)):
                outputs = []
                first = 0
                for size in chunks:
                    outputs.append(run(x[:, first:first + size]))
                    first += size
                torch.testing.assert_close(
                    torch.cat(outputs, dim=1),
                    full,
                    atol=1e-5,
                    rtol=1e-5,
                )


if __name__ == "__main__":
    unittest.main()
