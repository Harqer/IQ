"""Local-only first-party IQ harness HTTP integration.

Checks that the diagram's harness/model-adapter edge works for text and
fails closed for unavailable function-tool calls. No outside AI, HF jobs or
GPU activity.
"""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from iq_harness import (
    AgentRuntime,
    AgentSpec,
    IQHTTPModelBackend,
    IQHTTPModelError,
    ModelRequest,
    Message,
    ToolDescriptor,
)


class LocalIQServer(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        self.requests.append({
            "path": self.path,
            "request": request,
            "authorization": self.headers.get("Authorization"),
        })
        data = json.dumps({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "IQ says 4"},
                "finish_reason": "stop",
            }]
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class IQHTTPBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), LocalIQServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        LocalIQServer.requests.clear()

    def test_harness_calls_iq_directly_with_original_user_messages(self):
        backend = IQHTTPModelBackend(self.base, token="secret-token", max_tokens=8)
        runtime = AgentRuntime(
            backend,
            [AgentSpec("main", "You are IQ.", skills=())],
        )
        result = runtime.run("main", "What is 2 + 2?")
        self.assertEqual(result.output, "IQ says 4")
        self.assertEqual(result.turns, 1)
        self.assertEqual(len(LocalIQServer.requests), 1)
        call = LocalIQServer.requests[0]
        self.assertEqual(call["path"], "/v1/chat/completions")
        self.assertEqual(call["authorization"], "Bearer secret-token")
        self.assertEqual(call["request"]["messages"], [
            {"role": "system", "content": "You are IQ."},
            {"role": "user", "content": "What is 2 + 2?"},
        ])
        self.assertFalse(call["request"]["stream"])

    def test_tool_calling_not_silently_discarded(self):
        backend = IQHTTPModelBackend(self.base)
        with self.assertRaisesRegex(IQHTTPModelError, "does not yet implement"):
            backend.generate(ModelRequest(
                system_prompt="IQ",
                messages=(Message("user", "run"),),
                tools=(ToolDescriptor("math.add", "add", {"type": "object"}),),
            ))
        self.assertEqual(LocalIQServer.requests, [])

    def test_tool_history_cannot_be_encoded_as_plain_chat(self):
        backend = IQHTTPModelBackend(self.base)
        with self.assertRaisesRegex(IQHTTPModelError, "cannot encode role"):
            backend.generate(ModelRequest(
                system_prompt="IQ",
                messages=(Message("tool", "secret result"),),
                tools=(),
            ))

    def test_nonlocal_http_cannot_receive_a_bearer_token(self):
        with self.assertRaisesRegex(ValueError, "restricted to localhost"):
            IQHTTPModelBackend("http://example.com", token="secret")

    def test_token_not_embedded_in_url_and_tokens_bounded(self):
        with self.assertRaisesRegex(ValueError, "embed credentials"):
            IQHTTPModelBackend("https://user:pass@example.com")
        with self.assertRaisesRegex(ValueError, "1..64"):
            IQHTTPModelBackend(self.base, max_tokens=65)


if __name__ == "__main__":
    unittest.main()
