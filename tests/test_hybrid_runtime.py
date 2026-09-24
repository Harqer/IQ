from __future__ import annotations

import unittest

import torch

from iq_model import (
    HybridLayerType,
    HybridModelError,
    HybridSchedule,
    IQHybridConfig,
    IQModelConfig,
    Mamba3MIMOConfig,
    MTPConfig,
    RoutedMoEConfig,
    pack_mamba_varlen,
    unpack_mamba_varlen,
)


class HeterogeneousHybridRuntimeTests(unittest.TestCase):
    def language_config(self) -> IQModelConfig:
        return IQModelConfig(
            vocab_size=97,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )

    def hybrid_config(self) -> IQHybridConfig:
        schedule = HybridSchedule.parse("M E M A E M")
        return IQHybridConfig(
            model=self.language_config(),
            schedule=schedule,
            mamba3=Mamba3MIMOConfig(
                d_model=16,
                num_layers=3,
                d_state=8,
                headdim=8,
                mimo_rank=4,
                expand=2.0,
                rope_fraction=0.5,
                chunk_size=16,
            ),
            moe=RoutedMoEConfig(
                hidden_size=16,
                expert_intermediate_size=24,
                num_experts=4,
                top_k=2,
                shared_expert_intermediate_size=16,
            ),
        )

    def test_hybrid_config_matches_schedule_and_fingerprints(self):
        config = self.hybrid_config()
        self.assertEqual(
            config.schedule.count(HybridLayerType.MAMBA3),
            3,
        )
        self.assertEqual(
            config.schedule.count(HybridLayerType.MOE),
            2,
        )
        self.assertEqual(len(config.fingerprint), 64)
        self.assertEqual(config.fingerprint, self.hybrid_config().fingerprint)

        with self.assertRaises(HybridModelError):
            IQHybridConfig(
                model=self.language_config(),
                schedule=config.schedule,
                mamba3=Mamba3MIMOConfig(
                    d_model=16,
                    num_layers=2,
                    d_state=8,
                    headdim=8,
                    mimo_rank=4,
                    expand=2.0,
                    rope_fraction=0.5,
                    chunk_size=16,
                ),
                moe=config.moe,
            )

    def test_varlen_pack_resets_at_rows_and_document_boundaries(self):
        hidden = torch.arange(
            2 * 5 * 3,
            dtype=torch.float32,
        ).reshape(2, 5, 3)
        mask = torch.tensor(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0],
            ]
        )
        docs = torch.tensor(
            [
                [0, 0, 99, 1, 1],
                [3, 3, 4, 99, 99],
            ]
        )
        layout = pack_mamba_varlen(
            hidden,
            attention_mask=mask,
            document_ids=docs,
        )
        self.assertTrue(layout.packed)
        self.assertEqual(
            layout.cu_seqlens.tolist(),
            [0, 2, 4, 6, 7],
        )
        self.assertEqual(
            tuple(layout.packed_hidden_states.shape),
            (1, 7, 3),
        )

        transformed = layout.packed_hidden_states + 1.0
        unpacked = unpack_mamba_varlen(transformed, layout)
        valid = mask.to(dtype=torch.bool)
        self.assertTrue(
            torch.equal(
                unpacked[valid],
                hidden[valid] + 1.0,
            )
        )
        self.assertTrue(
            torch.equal(
                unpacked[~valid],
                torch.zeros_like(unpacked[~valid]),
            )
        )

    def test_fully_valid_batch_uses_native_batched_mamba_path(self):
        hidden = torch.randn(3, 5, 8)
        layout = pack_mamba_varlen(hidden)
        self.assertFalse(layout.packed)
        self.assertIsNone(layout.cu_seqlens)
        self.assertTrue(
            torch.equal(
                unpack_mamba_varlen(hidden, layout),
                hidden,
            )
        )

    def test_noncontiguous_document_reuse_fails_closed(self):
        hidden = torch.randn(1, 5, 4)
        docs = torch.tensor([[0, 0, 1, 0, 0]])
        with self.assertRaises(HybridModelError):
            pack_mamba_varlen(hidden, document_ids=docs)


if __name__ == "__main__":
    unittest.main()
