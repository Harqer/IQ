from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iq_transfer.glm53_stream_capture import HubSafetensorsSource


class GLM53StreamingCaptureTests(unittest.TestCase):
    def test_shard_eviction_waits_for_last_referenced_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.a": "shared.safetensors",
                            "model.layers.1.b": "shared.safetensors",
                            "model.layers.0.c": "layer0.safetensors",
                            "model.embed_tokens.weight": "globals.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            source = HubSafetensorsSource(
                root,
                repo_id="unused/test",
                revision="deadbeef",
            )
            shared = root / "shared.safetensors"
            layer0 = root / "layer0.safetensors"
            globals_ = root / "globals.safetensors"
            for path in (shared, layer0, globals_):
                path.write_bytes(b"x")
                source._ephemeral_shards.add(path)

            source.clear_ephemeral_shards(completed_layer=0)
            self.assertTrue(shared.exists())
            self.assertFalse(layer0.exists())
            self.assertFalse(globals_.exists())

            source.clear_ephemeral_shards(completed_layer=1)
            self.assertFalse(shared.exists())
            self.assertEqual(source._ephemeral_shards, set())


if __name__ == "__main__":
    unittest.main()
