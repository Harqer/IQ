"""Test an existing Baseten deployment once, then deactivate that exact deployment."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class BasetenProbeError(RuntimeError):
    pass


def api_json(url, key, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=1200 if "/predict" in url else 30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise BasetenProbeError(f"HTTP {error.code}: {error.read(250).decode(errors='replace')}") from None
    except (URLError, OSError, ValueError) as error:
        raise BasetenProbeError(f"request failed: {error}") from None


def stop_deployment(model_id, deployment_id, token):
    base = f"https://api.baseten.co/v1/models/{model_id}/deployments/{deployment_id}"
    result = api_json(base + "/deactivate", token, "POST", {})
    if not result.get("success"):
        raise BasetenProbeError(f"deactivation refused: {result}")
    for index in range(24):
        status_data = api_json(base, token)
        status = str(status_data.get("deployment", status_data).get("status", "UNKNOWN")).upper()
        if status == "INACTIVE":
            return status
        if index < 23:
            time.sleep(3)
    raise BasetenProbeError("deactivation accepted, but INACTIVE was not confirmed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.getenv("BASETEN_MODEL_ID"))
    parser.add_argument("--deployment-id", default=os.getenv("BASETEN_DEPLOYMENT_ID"))
    parser.add_argument("--expect-phase", choices=["architecture_only", "injected"],
                        default="architecture_only")
    parser.add_argument("--prompt", default="The next integer after 1 2 3 is")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/iq-cloud-probes"))
    args = parser.parse_args()
    token = os.getenv("BASETEN_API_KEY")
    if not token or not args.model_id or not args.deployment_id:
        parser.error("BASETEN_API_KEY, BASETEN_MODEL_ID, BASETEN_DEPLOYMENT_ID required")
    for value in (args.model_id, args.deployment_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{4,128}", value):
            parser.error("invalid model/deployment ID")
    url = (f"https://model-{args.model_id}.api.baseten.co/"
           f"deployment/{args.deployment_id}/predict")
    error = None
    try:
        response = api_json(url, token, "POST", {
            "prompt": args.prompt, "max_new_tokens": args.max_new_tokens})
        if response.get("phase") != args.expect_phase:
            raise BasetenProbeError("wrong deployed phase")
        if response.get("weights_injected") != (args.expect_phase == "injected"):
            raise BasetenProbeError("unexpected injected-weight flag")
        if (response.get("completed_layers_per_token") != 24
                or not response.get("generated_ids")
                or not response.get("schema_sha256")
                or not response.get("tokenizer_sha256")):
            raise BasetenProbeError("complete inference evidence missing")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        file = args.output_dir / (args.expect_phase + ".json")
        if file.exists():
            raise BasetenProbeError("refusing to overwrite prior test evidence")
        file.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        print("PASS: full model generated", len(response["generated_ids"]), "tokens:", file)
    except Exception as exc:
        error = str(exc)
        print("INFERENCE FAILED:", error)
    finally:
        try:
            status = stop_deployment(args.model_id, args.deployment_id, token)
            print("DEPLOYMENT STOPPED:", status)
        except Exception as exc:
            print("URGENT: SHUTDOWN NOT VERIFIED:", exc)
            raise SystemExit(3)
    if error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
