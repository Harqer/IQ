"""OTLP/HTTP protobuf integration check using loopback-only CPU telemetry.

Uses the real OpenTelemetry Python exporter + SDK, no GPU, model, Baseten calls
or external collector. Asserts the exported protobuf contains actual spans and
events, and JSONL carries the same trace ID.
"""
from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from iq_observability.full_trace import IQTrace


class OtlpReceiver(BaseHTTPRequestHandler):
    incoming: queue.Queue

    def do_POST(self):
        payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.incoming.put((self.path, self.headers.get("Content-Type"), payload))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args):
        pass


class OtlpLoopbackIntegrationTest(unittest.TestCase):
    def test_real_otlp_http_protobuf_export_with_trace_correlation(self):
        receiver = ThreadingHTTPServer(("127.0.0.1", 0), OtlpReceiver)
        receiver.RequestHandlerClass.incoming = queue.Queue()
        worker = threading.Thread(target=receiver.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(receiver.server_close)
        self.addCleanup(worker.join, timeout=5)
        self.addCleanup(receiver.shutdown)
        with tempfile.TemporaryDirectory() as folder:
            endpoint = f"http://127.0.0.1:{receiver.server_port}/v1/traces"
            with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint}):
                trace = IQTrace(folder)
                try:
                    with trace.stage("checkpoint.integrity"):
                        trace.event("checkpoint.integrity.complete",
                                    shards=82, tensors=100, sha256_verified=True)
                    name = trace.path
                finally:
                    trace.close()
            path, content_type, payload = receiver.RequestHandlerClass.incoming.get(timeout=10)
            self.assertEqual(path, "/v1/traces")
            self.assertEqual(content_type, "application/x-protobuf")
            request = ExportTraceServiceRequest()
            request.ParseFromString(payload)
            spans = [s for resource in request.resource_spans
                     for scope in resource.scope_spans for s in scope.spans]
            self.assertTrue(spans, "OTLP export contained no real spans")
            root_span = next(s for s in spans if s.name == "checkpoint.integrity")
            self.assertTrue(root_span.trace_id)
            self.assertIn("checkpoint.integrity.complete", [e.name for e in root_span.events])
            records = [json.loads(line) for line in name.read_text().splitlines()]
            completed = next(x for x in records if x["event"] == "checkpoint.integrity.complete")
            self.assertEqual(completed["trace_id"], root_span.trace_id.hex())


if __name__ == "__main__":
    unittest.main()
