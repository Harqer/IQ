from __future__ import annotations

import unittest

import torch

from iq_model.hybrid import _reasoning_masks
from iq_model import (
    BlockAttnResConfig,
    CompressedContextConfig,
    HybridLayerType,
    HybridModelError,
    HybridSchedule,
    IQHybridConfig,
    IQModelConfig,
    Mamba3MIMOConfig,
    ReasoningEnergyCriticConfig,
    ReasoningRecurrenceConfig,
    RoutedMoEConfig,
    StableLatentMoEConfig,
    pack_mamba_varlen,
    validate_canonical_hybrid_backbone,
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
            reasoning=ReasoningRecurrenceConfig(
                hidden_size=16,
                state_dim=8,
                transition_hidden_dim=24,
                max_steps=4,
                min_steps=1,
            ),
            energy_critic=ReasoningEnergyCriticConfig(
                state_dim=8,
                context_dim=16,
                hidden_dim=12,
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
        restored = IQHybridConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.fingerprint, config.fingerprint)

        self.assertIsNotNone(restored.reasoning)
        self.assertIsNotNone(restored.energy_critic)

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

    def test_canonical_backbone_requires_attnres_stable_moe_and_compressed_context(self):
        base = self.hybrid_config()
        canonical = IQHybridConfig(
            model=base.model,
            schedule=HybridSchedule.parse("M E M C E M"),
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
            moe=base.moe,
            moe_variant="stable_latent",
            stable_moe=StableLatentMoEConfig(
                hidden_size=16,
                latent_size=8,
                expert_intermediate_size=24,
                num_experts=4,
                top_k=2,
            ),
            attnres=BlockAttnResConfig(
                hidden_size=16,
                num_layers=6,
                block_size=2,
            ),
            compressed_context=CompressedContextConfig(
                hidden_size=16,
                num_attention_heads=4,
                head_dim=8,
                q_lora_rank=8,
                partial_rotary_dim=4,
                max_position_embeddings=64,
                sliding_window=3,
                csa_compress_rate=2,
                hca_compress_rate=4,
                o_groups=2,
                o_lora_rank=4,
                index_n_heads=2,
                index_head_dim=4,
                index_topk=2,
                compress_rope_theta=10000.0,
            ),
        )
        self.assertIs(validate_canonical_hybrid_backbone(canonical), canonical)

        with self.assertRaisesRegex(
            HybridModelError,
            "requires Block AttnRes",
        ):
            validate_canonical_hybrid_backbone(
                IQHybridConfig(
                    model=canonical.model,
                    schedule=canonical.schedule,
                    mamba3=canonical.mamba3,
                    moe=canonical.moe,
                    moe_variant=canonical.moe_variant,
                    stable_moe=canonical.stable_moe,
                    compressed_context=canonical.compressed_context,
                )
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

    def test_reasoning_masks_require_explicit_training_boundary(self):
        input_ids = torch.tensor(
            [
                [10, 11, 12, 13, 14, 0],
                [20, 21, 22, 23, 0, 0],
            ]
        )
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
            ]
        )

        with self.assertRaisesRegex(
            HybridModelError,
            "required when labels are provided",
        ):
            _reasoning_masks(
                input_ids,
                mask,
                None,
                labels_present=True,
            )

        context, injection = _reasoning_masks(
            input_ids,
            mask,
            torch.tensor([3, 2]),
            labels_present=True,
        )
        self.assertTrue(
            torch.equal(
                context,
                torch.tensor(
                    [
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                    ],
                    dtype=torch.bool,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                injection,
                torch.tensor(
                    [
                        [0, 0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0, 0],
                    ],
                    dtype=torch.bool,
                ),
            )
        )

    def test_reasoning_inference_uses_full_visible_prefix(self):
        input_ids = torch.tensor([[4, 5, 6, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        context, injection = _reasoning_masks(
            input_ids,
            mask,
            None,
            labels_present=False,
        )
        self.assertTrue(
            torch.equal(
                context,
                torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
            )
        )
        self.assertTrue(
            torch.equal(
                injection,
                torch.tensor([[0, 0, 1, 0]], dtype=torch.bool),
            )
        )

    def test_noncontiguous_document_reuse_fails_closed(self):
        hidden = torch.randn(1, 5, 4)
        docs = torch.tensor([[0, 0, 1, 0, 0]])
        with self.assertRaises(HybridModelError):
            pack_mamba_varlen(hidden, document_ids=docs)


if __name__ == "__main__":
    unittest.main()
