import torch
import pytest

from iq_model.multimodal import IQMultimodalConfig, TransVTransfer, VisualMemory


def test_multimodal_config_roundtrip_and_transv_contract():
    config = IQMultimodalConfig(
        vision_model_name="example/vision",
        fusion_layers=(2, 5, 8),
        transv_layers=(5, 8),
        transv_shallow_keep_ratio=0.5,
        transv_deep_keep_ratio=0.5,
        min_visual_tokens=2,
    )
    assert IQMultimodalConfig.from_dict(config.to_dict()) == config

    hidden = torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4)
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ], dtype=torch.bool)
    frames = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 1, 2, 3, 0, 0, 0, 0],
    ])
    text = torch.randn(2, 3, 4)
    out = TransVTransfer(config)(
        VisualMemory(hidden, mask, frames, "video"),
        text,
        text_mask=None,
        deep=False,
    )
    assert out.attention_mask.sum(dim=1).tolist() == [4, 2]
    assert out.hidden_states.shape == (2, 4, 4)
    assert torch.isfinite(out.hidden_states).all()


def test_transv_rejects_invalid_layer_contract():
    with pytest.raises(ValueError, match="TransV"):
        IQMultimodalConfig(
            vision_model_name="example/vision",
            fusion_layers=(2,),
            transv_layers=(3,),
        )


def test_multimodal_config_rejects_duplicate_or_unsorted_layers():
    with pytest.raises(ValueError, match="sorted and unique"):
        IQMultimodalConfig(
            vision_model_name="example/vision",
            fusion_layers=(4, 2),
            transv_layers=(),
        )
