from __future__ import annotations

import unittest

import numpy as np
import torch

from iq_transfer.mamba3_direct import (
    Mamba3DonorConfig,
    expand_mamba3_foundation_globals,
)
from iq_transfer.transport import CoordinateMap
from iq_transfer.warm import (
    align_to_mamba_replication_frame,
    orthogonality_error,
)


class MambaFoundationTests(unittest.TestCase):
    def config(self) -> Mamba3DonorConfig:
        return Mamba3DonorConfig(
            d_model=2,
            d_intermediate=3,
            n_layer=1,
            vocab_size=5,
            d_state=2,
            expand=2.0,
            headdim=1,
            ngroups=1,
            rope_fraction=0.5,
            chunk_size=1,
            is_mimo=True,
            mimo_rank=2,
            is_outproj_norm=False,
            rms_norm=True,
            tie_embeddings=True,
        )

    def test_exact_lexical_widening_preserves_logits(self):
        config = self.config()
        embedding = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        lm_head = torch.tensor(
            [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0], [4.5, 5.0]]
        )
        state = {
            "backbone.embedding.weight": embedding,
            "backbone.norm_f.weight": torch.tensor([1.25, 0.75]),
            "lm_head.weight": lm_head,
        }
        widened = expand_mamba3_foundation_globals(
            source_config=config,
            state=state,
            target_hidden_size=4,
        )
        self.assertTrue(torch.equal(
            widened["embed_tokens.weight"],
            torch.cat((embedding, embedding), dim=1),
        ))
        hidden = torch.tensor([[2.0, -1.0], [0.25, 3.0]])
        widened_hidden = torch.cat((hidden, hidden), dim=1)
        donor_logits = hidden @ lm_head.T
        target_logits = widened_hidden @ widened["lm_head.weight"].T
        self.assertTrue(torch.allclose(donor_logits, target_logits, atol=1e-7))
        self.assertTrue(torch.equal(
            widened["norm.weight"],
            torch.tensor([1.25, 0.75, 1.25, 0.75]),
        ))

    def test_warm_target_rotation_is_anchored_to_mamba_replication_frame(self):
        base = CoordinateMap(
            matrix=np.eye(4, dtype=np.float64),
            ridge=1e-12,
            source_space="glm",
            target_space="abstract_iq",
            diagnostics=None,
        )
        aligned = align_to_mamba_replication_frame(
            base,
            foundation_features=2,
        ).matrix
        scale = 1.0 / np.sqrt(2.0)
        self.assertTrue(np.allclose(aligned[0], [scale, 0.0, scale, 0.0]))
        self.assertTrue(np.allclose(aligned[1], [0.0, scale, 0.0, scale]))
        self.assertTrue(np.allclose(aligned[2], [scale, 0.0, -scale, 0.0]))
        self.assertTrue(np.allclose(aligned[3], [0.0, scale, 0.0, -scale]))
        self.assertLess(orthogonality_error(aligned), 1e-10)


if __name__ == "__main__":
    unittest.main()
