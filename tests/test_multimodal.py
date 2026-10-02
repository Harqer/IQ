import unittest

import torch

from iq_model.multimodal import (
    IQMultimodalConfig,
    MultimodalError,
    CrossModalFusion,
    TransVTransfer,
    VisualMemory,
    _tome_merge,
)


class MultimodalTests(unittest.TestCase):
    def test_config_roundtrip_and_transv_contract(self):
        config = IQMultimodalConfig(
            vision_model_name="example/vision",
            fusion_layers=(2, 5, 8),
            transv_layers=(5, 8),
            transv_shallow_keep_ratio=0.5,
            transv_deep_keep_ratio=0.5,
            min_visual_tokens=2,
        )
        self.assertEqual(IQMultimodalConfig.from_dict(config.to_dict()), config)
        self.assertEqual(config.temporal_dilations, (1, 2, 4))

        hidden = torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.bool)
        frames = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 2, 3, 0, 0, 0, 0],
        ])
        out = TransVTransfer(config)(
            VisualMemory(hidden, mask, frames, "video"),
            relevance_scores=None,
            deep=False,
        )
        self.assertEqual(out.attention_mask.sum(dim=1).tolist(), [4, 2])
        self.assertEqual(tuple(out.hidden_states.shape), (2, 4, 4))
        self.assertTrue(bool(torch.isfinite(out.hidden_states).all()))

    def test_transv_rejects_invalid_layer_contract(self):
        with self.assertRaisesRegex(ValueError, "TransV"):
            IQMultimodalConfig(
                vision_model_name="example/vision",
                fusion_layers=(2,),
                transv_layers=(3,),
            )

    def test_config_rejects_unsorted_layers(self):
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            IQMultimodalConfig(
                vision_model_name="example/vision",
                fusion_layers=(4, 2),
                transv_layers=(),
            )

    def test_cross_modal_fusion_is_function_neutral_at_init(self):
        fusion = CrossModalFusion(
            hidden_size=8,
            heads=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        text = torch.randn(2, 4, 8)
        memory = VisualMemory(
            torch.randn(2, 6, 8),
            torch.ones(2, 6, dtype=torch.bool),
            torch.zeros(2, 6, dtype=torch.long),
            "video",
        )
        output = fusion(text, memory, text_mask=None)
        self.assertTrue(torch.equal(output, text))

    def test_deep_transv_requires_attention_scores(self):
        config = IQMultimodalConfig(
            vision_model_name="example/vision",
            fusion_layers=(2,),
            transv_layers=(2,),
            min_visual_tokens=1,
        )
        memory = VisualMemory(
            torch.randn(1, 8, 4),
            torch.ones(1, 8, dtype=torch.bool),
            torch.zeros(1, 8, dtype=torch.long),
            "video",
        )
        with self.assertRaisesRegex(MultimodalError, "attention"):
            TransVTransfer(config)(
                memory,
                relevance_scores=None,
                deep=True,
            )

    def test_config_rejects_invalid_temporal_dilations(self):
        with self.assertRaisesRegex(ValueError, "temporal_dilations"):
            IQMultimodalConfig(
                vision_model_name="example/vision",
                fusion_layers=(2,),
                transv_layers=(),
                temporal_dilations=(1, 0, 4),
            )

    def test_query_projector_token_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "query_projector_tokens"):
            IQMultimodalConfig(
                vision_model_name="example/vision",
                fusion_layers=(2,),
                transv_layers=(),
                query_projector_tokens=0,
            )

    def test_tome_merges_to_exact_target(self):
        tokens = torch.randn(37, 8)
        merged = _tome_merge(tokens, 16)
        self.assertEqual(tuple(merged.shape), (16, 8))
        self.assertTrue(bool(torch.isfinite(merged).all()))


if __name__ == "__main__":
    unittest.main()
