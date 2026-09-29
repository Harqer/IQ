from __future__ import annotations

import unittest

import torch

from iq_transfer.mamba3_direct import (
    Mamba3DirectTransferError,
    Mamba3DonorConfig,
    evenly_spaced_layer_placements,
    expand_mamba3_layer,
    validate_official_mamba3_mimo_15b_config,
)
from iq_transfer.mamba3_init import Mamba3Layout


class Mamba3DirectTransferTests(unittest.TestCase):
    def tiny_config(self) -> Mamba3DonorConfig:
        return Mamba3DonorConfig(
            d_model=8,
            d_intermediate=12,
            n_layer=2,
            vocab_size=32,
            d_state=4,
            expand=2.0,
            headdim=4,
            ngroups=1,
            rope_fraction=0.5,
            chunk_size=2,
            is_mimo=True,
            mimo_rank=2,
            is_outproj_norm=False,
            rms_norm=True,
            tie_embeddings=True,
        )

    def tiny_layer_state(self, config: Mamba3DonorConfig, layer: int = 0):
        layout = config.layout
        p = f"backbone.layers.{layer}"
        state = {
            f"{p}.norm.weight": torch.arange(
                config.d_model, dtype=torch.float32
            ) + 1,
            f"{p}.mixer.in_proj.weight": torch.arange(
                layout.in_proj_shape[0] * layout.in_proj_shape[1],
                dtype=torch.float32,
            ).reshape(layout.in_proj_shape),
            f"{p}.mixer.dt_bias": torch.arange(
                layout.nheads, dtype=torch.float32
            ),
            f"{p}.mixer.B_bias": torch.arange(
                layout.nheads * layout.effective_mimo_rank * layout.d_state,
                dtype=torch.float32,
            ).reshape(
                layout.nheads,
                layout.effective_mimo_rank,
                layout.d_state,
            ),
            f"{p}.mixer.C_bias": torch.arange(
                layout.nheads * layout.effective_mimo_rank * layout.d_state,
                dtype=torch.float32,
            ).reshape(
                layout.nheads,
                layout.effective_mimo_rank,
                layout.d_state,
            ) + 100,
            f"{p}.mixer.B_norm.weight": torch.arange(
                layout.d_state, dtype=torch.float32
            ) + 1,
            f"{p}.mixer.C_norm.weight": torch.arange(
                layout.d_state, dtype=torch.float32
            ) + 2,
            f"{p}.mixer.mimo_x": torch.arange(
                layout.nheads * layout.effective_mimo_rank * layout.headdim,
                dtype=torch.float32,
            ).reshape(
                layout.nheads,
                layout.effective_mimo_rank,
                layout.headdim,
            ),
            f"{p}.mixer.mimo_z": torch.arange(
                layout.nheads * layout.effective_mimo_rank * layout.headdim,
                dtype=torch.float32,
            ).reshape(
                layout.nheads,
                layout.effective_mimo_rank,
                layout.headdim,
            ) + 10,
            f"{p}.mixer.mimo_o": torch.arange(
                layout.nheads * layout.effective_mimo_rank * layout.headdim,
                dtype=torch.float32,
            ).reshape(
                layout.nheads,
                layout.effective_mimo_rank,
                layout.headdim,
            ) + 20,
            f"{p}.mixer.D": torch.arange(layout.nheads, dtype=torch.float32) + 3,
            f"{p}.mixer.out_proj.weight": torch.arange(
                layout.out_proj_shape[0] * layout.out_proj_shape[1],
                dtype=torch.float32,
            ).reshape(layout.out_proj_shape),
        }
        return state

    def test_depth_expansion_uses_monotonic_unique_slots_and_identity_gaps(self):
        positions = evenly_spaced_layer_placements(24, 32)
        self.assertEqual(len(positions), 24)
        self.assertEqual(len(set(positions)), 24)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 31)
        self.assertEqual(len(set(range(32)) - set(positions)), 8)
        self.assertEqual(tuple(sorted(positions)), positions)

    def test_width_expansion_preserves_each_packed_semantic_slice(self):
        config = self.tiny_config()
        state = self.tiny_layer_state(config)
        target = Mamba3Layout(
            d_model=16,
            d_state=4,
            expand=2.0,
            headdim=4,
            ngroups=1,
            rope_fraction=0.5,
            is_mimo=True,
            mimo_rank=2,
        )
        result = expand_mamba3_layer(
            source_config=config,
            target_layout=target,
            source_layer=0,
            target_physical_layer=3,
            state=state,
        )
        source_layout = config.layout
        source_in = state["backbone.layers.0.mixer.in_proj.weight"]
        target_in = result["layers.3.mamba.core.in_proj.weight"]
        for name in ("z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angle"):
            s = source_layout.slices()[name]
            t = target.slices()[name]
            rows = s.stop - s.start
            self.assertTrue(
                torch.equal(
                    target_in[t.start : t.start + rows, : config.d_model],
                    source_in[s, :],
                )
            )
            self.assertTrue(
                torch.count_nonzero(
                    target_in[t.start : t.start + rows, config.d_model :]
                ).item()
                == 0
            )

    def test_output_projection_is_prefix_subspace_embedding(self):
        config = self.tiny_config()
        state = self.tiny_layer_state(config)
        target = Mamba3Layout(
            d_model=16,
            d_state=4,
            expand=2.0,
            headdim=4,
            ngroups=1,
            rope_fraction=0.5,
            is_mimo=True,
            mimo_rank=2,
        )
        result = expand_mamba3_layer(
            source_config=config,
            target_layout=target,
            source_layer=0,
            target_physical_layer=1,
            state=state,
        )
        source = state["backbone.layers.0.mixer.out_proj.weight"]
        mapped = result["layers.1.mamba.core.out_proj.weight"]
        self.assertTrue(
            torch.equal(
                mapped[: config.d_model, : config.layout.d_inner],
                source,
            )
        )
        self.assertEqual(
            torch.count_nonzero(mapped[config.d_model :, :]).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(mapped[:, config.layout.d_inner :]).item(),
            0,
        )

    def test_head_parameters_preserve_donor_and_use_neutral_extra_heads(self):
        config = self.tiny_config()
        state = self.tiny_layer_state(config)
        target = Mamba3Layout(
            d_model=16,
            d_state=4,
            expand=2.0,
            headdim=4,
            ngroups=1,
            rope_fraction=0.5,
            is_mimo=True,
            mimo_rank=2,
        )
        result = expand_mamba3_layer(
            source_config=config,
            target_layout=target,
            source_layer=0,
            target_physical_layer=2,
            state=state,
        )
        d = result["layers.2.mamba.core.D"]
        self.assertTrue(
            torch.equal(
                d[: config.layout.nheads],
                state["backbone.layers.0.mixer.D"],
            )
        )
        self.assertTrue(
            torch.equal(
                d[config.layout.nheads :],
                torch.ones(target.nheads - config.layout.nheads),
            )
        )
        norm = result["layers.2.norm.weight"]
        self.assertTrue(
            torch.equal(
                norm[: config.d_model],
                state["backbone.layers.0.norm.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                norm[config.d_model :],
                torch.ones(target.d_model - config.d_model),
            )
        )

    def test_recurrent_semantic_mismatch_fails_closed(self):
        config = self.tiny_config()
        state = self.tiny_layer_state(config)
        target = Mamba3Layout(
            d_model=16,
            d_state=8,
            expand=2.0,
            headdim=4,
            ngroups=1,
            rope_fraction=0.5,
            is_mimo=True,
            mimo_rank=2,
        )
        with self.assertRaisesRegex(
            Mamba3DirectTransferError,
            "recurrent semantics",
        ):
            expand_mamba3_layer(
                source_config=config,
                target_layout=target,
                source_layer=0,
                target_physical_layer=0,
                state=state,
            )

    def test_pinned_15b_config_validation_is_strict(self):
        official = Mamba3DonorConfig(
            d_model=2048,
            d_intermediate=3824,
            n_layer=24,
            vocab_size=128256,
            d_state=128,
            expand=2.0,
            headdim=64,
            ngroups=1,
            rope_fraction=0.5,
            chunk_size=16,
            is_mimo=True,
            mimo_rank=4,
            is_outproj_norm=False,
            rms_norm=True,
            tie_embeddings=True,
        )
        validate_official_mamba3_mimo_15b_config(official)
        bad = Mamba3DonorConfig(
            **{**official.__dict__, "d_model": 1536}
        )
        with self.assertRaisesRegex(
            Mamba3DirectTransferError,
            "not the pinned official",
        ):
            validate_official_mamba3_mimo_15b_config(bad)


if __name__ == "__main__":
    unittest.main()
