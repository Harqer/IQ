"""CPU-only source lexical and observability safety checks; no HF reads."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from iq_cloud.full235_weight_loader import (
    VerifiedBodyLoaderError, install_verified_shard,
    verify_source_lexical_relation,
)
from iq_cloud.inference_diagnostics import (
    IQ235NumericError, emit_progress, examine_layer_output, span_scope,
)
from iq_cloud.remote_shard_loader import verify_safetensors_header


class FakeSpan:
    def __init__(self):
        self.events = []
        self.attributes = {}

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes))

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FakeTracer:
    def __init__(self):
        self.calls = []

    def start_as_current_span(self, name, attributes=None):
        from contextlib import nullcontext
        self.calls.append((name, attributes))
        return nullcontext(FakeSpan())


class IQ235SourceInstrumentationTests(unittest.TestCase):
    def test_source_mode_guard_denies_gpu_before_manifest_pin(self):
        from iq_cloud.full_iq235 import load_pretrained_source, FullIQ235Error
        with patch.dict("os.environ", {"IQ_PHASE": "pretrained_source"}):
            with self.assertRaisesRegex(FullIQ235Error, "no GPU or checkpoint"):
                load_pretrained_source()

    def test_stage_contains_real_verified_source_loader_and_telemetry(self):
        from scripts.stage_iq235_baseten import selected_sources
        names = {str(p) for p in selected_sources()}
        for required in ("iq_cloud/full235_weight_loader.py",
                         "iq_cloud/remote_shard_loader.py",
                         "iq_cloud/inference_diagnostics.py",
                         "iq_cloud/lexical_warmstart.py"):
            self.assertIn(required, names)

    def test_source_mode_can_install_only_original_lexical_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            embed = torch.tensor([[2., 4., 2., 4.], [6., 8., 6., 8.]])
            head = embed / 2
            save_file({
                "embed_tokens.weight": embed,
                "lm_head.weight": head,
                "norm.weight": torch.tensor([3., 5., 7., 9.]),
            }, path)
            with path.open("rb") as stream:
                header = verify_safetensors_header(
                    stream, file_size=path.stat().st_size,
                )
            targets = {
                "embed_tokens.weight": torch.zeros_like(embed),
                "lm_head.weight": torch.zeros_like(head),
                "norm.weight": torch.zeros(4),
            }
            written = install_verified_shard(
                path, header, set(header), targets,
                original_lexical_shape=(2, 4), require_original_bf16=False,
                install_source_lexical=True,
            )
            self.assertEqual(written, set(targets))
            self.assertEqual(
                verify_source_lexical_relation(targets, rows=(0, 1)), 2
            )
            targets["lm_head.weight"][1, 0] += 1
            with self.assertRaisesRegex(VerifiedBodyLoaderError, "row 1"):
                verify_source_lexical_relation(targets, rows=(0, 1))

    def test_layer_nonfinite_is_localized(self):
        span = FakeSpan()
        maximum, rms = examine_layer_output(
            torch, torch.tensor([1., -3.]), index=14,
            layer_type="mamba3", span=span,
        )
        self.assertEqual(maximum, 3.0)
        self.assertGreater(rms, 0)
        self.assertEqual(span.events[0][0], "iq.layer.numeric")
        with self.assertRaisesRegex(IQ235NumericError, "layer 14"):
            examine_layer_output(
                torch, torch.tensor([float("nan")]), index=14,
                layer_type="mamba3", span=span,
            )
        self.assertEqual(span.events[-1][0], "iq.layer.nonfinite")

    def test_progress_events_are_bounded(self):
        span = FakeSpan()
        for shard in range(1, 503):
            emit_progress(span, shard, 502, shard + 2)
        self.assertEqual([event[1]["iq.shards.verified"]
                          for event in span.events], [1, 250, 500, 502])

    def test_optional_tracer_does_not_require_otel_installation(self):
        with span_scope(None, "iq235.checkpoint.stream_verified_shards") as span:
            self.assertIsNone(span)
        tracer = FakeTracer()
        with span_scope(tracer, "iq235.inference", {"iq.phase": "source"}):
            pass
        self.assertEqual(tracer.calls[0][0], "iq235.inference")


if __name__ == "__main__":
    unittest.main()
