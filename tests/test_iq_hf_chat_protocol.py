"""CPU-only HTTP contract tests for Hugging Face InferenceClient against IQ."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from huggingface_hub import InferenceClient

# Avoid importing the GPU runtime or allocating CUDA during protocol tests.
fake_torch = types.ModuleType("torch")
fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoTokenizer = object
fake_transfer = types.ModuleType("iq_transfer")
fake_transfer.load_complete_iq_checkpoint = lambda *args, **kwargs: None

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "serve_iq_hf.py"
spec = importlib.util.spec_from_file_location("iq_server_protocol_test", SOURCE)
assert spec and spec.loader
with patch.dict(sys.modules, {
    "torch": fake_torch,
    "transformers": fake_transformers,
    "iq_transfer": fake_transfer,
}):
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)


class TestIQChatProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(
            ("127.0.0.1", 0),
            server_module.create_handler(tokenizer=None, model=None),
        )
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def test_health(self):
        with urlopen(self.base + "/health", timeout=5) as r:
            self.assertEqual(json.load(r), {"status": "ready"})

    def test_hf_inference_client_typed_chat(self):
        with patch.object(
            server_module, "generate",
            return_value=("4", "stop", 5, 1),
        ):
            client = InferenceClient(base_url=self.base, token="fake", timeout=5)
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "2 + 2?"}],
                max_tokens=8, stream=False
            )
            self.assertEqual(result.choices[0].message.content, "4")
            self.assertEqual(result.usage.prompt_tokens, 5)
            self.assertEqual(result.usage.total_tokens, 6)

    def test_raw_completion_probe_uses_separate_prompt_format(self):
        with patch.object(
            server_module, "generate",
            return_value=("4", "length", 5, 1),
        ) as generate:
            request = Request(
                self.base + "/chat",
                data=json.dumps({
                    "prompt": "2 + 2 =",
                    "prompt_format": "completion",
                    "max_new_tokens": 4,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response)["response"], "4")
            generate.assert_called_once_with(
                None, None,
                [{"role": "user", "content": "2 + 2 ="}],
                4, "completion",
            )

    def test_streaming_is_rejected_not_silently_ignored(self):
        request = Request(
            self.base + "/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "stream": True,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as cm:
            urlopen(request, timeout=5)
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
