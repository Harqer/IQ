"""Execute one full-IQ deployment prediction and deactivate that deployment.

Separate from the *push* command. The script cannot stop a deployment
whose IDs were never captured; recover its IDs from Baseten and use --stop-only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BasetenFullIQError(RuntimeError):
    pass


def request(url, key, *, method="GET", body=None, timeout=45):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req = Request(url, method=method, data=data, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise BasetenFullIQError(
            f"Baseten HTTP {exc.code}: {exc.read(400).decode(errors='replace')}"
        ) from None
    except (OSError, URLError, ValueError) as exc:
        raise BasetenFullIQError(f"Baseten request failed: {exc}") from None


def deploy_url(model_id, deployment_id):
    return f"https://api.baseten.co/v1/models/{model_id}/deployments/{deployment_id}"


def ensure_exact_deployment(model_id, deployment_id, key):
    record = request(deploy_url(model_id, deployment_id), key)
    if not isinstance(record, dict) or record.get("id") != deployment_id:
        raise BasetenFullIQError("deployment ID mismatch")
    if record.get("model_id") != model_id:
        raise BasetenFullIQError("model ID mismatch")
    return record


def deactivate(model_id, deployment_id, key):
    target = deploy_url(model_id, deployment_id)
    for attempt in range(3):
        try:
            record = ensure_exact_deployment(model_id, deployment_id, key)
            if record.get("status") == "INACTIVE" and record.get("active_replica_count") == 0:
                return record
            response = request(target + "/deactivate", key, method="POST", body={})
            if response.get("success") is not True:
                raise BasetenFullIQError(f"deactivation rejected: {response!r}")
            break
        except BasetenFullIQError:
            if attempt == 2:
                raise
            time.sleep(2)
    for _ in range(40):
        record = ensure_exact_deployment(model_id, deployment_id, key)
        if record.get("status") == "INACTIVE" and record.get("active_replica_count") == 0:
            return record
        time.sleep(3)
    raise BasetenFullIQError("deactivation not confirmed: inspect Baseten manually")


def verify_response(data):
    if not isinstance(data, dict):
        raise BasetenFullIQError("prediction response is not an object")
    expected = {
        "phase": "architecture_only",
        "checkpoint_weights_injected": False,
        "full_model": True,
        "physical_layers": 235,
        "mamba3_layers": 79,
        "context_layers": 78,
        "moe_layers": 78,
        "experts_per_moe_layer": 256,
        "glm_tokenizer_vocab": 154880,
        "all_layers_executed": True,
        "gpu_count": 8,
    }
    unexpected = {
        field: data.get(field) for field, target in expected.items()
        if type(data.get(field)) is not type(target) or data.get(field) != target
    }
    if unexpected:
        raise BasetenFullIQError(f"full IQ proof fields missing or incorrect: {unexpected}")
    if (not isinstance(data.get("generated_ids"), list)
            or not data["generated_ids"]
            or not all(type(x) is int and 0 <= x < 154880
                       for x in data["generated_ids"])):
        raise BasetenFullIQError("invalid generated GLM token IDs")
    for fingerprint in ("schema_sha256", "tokenizer_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", str(data.get(fingerprint, ""))):
            raise BasetenFullIQError(f"invalid {fingerprint}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.getenv("BASETEN_MODEL_ID"))
    parser.add_argument("--deployment-id", default=os.getenv("BASETEN_DEPLOYMENT_ID"))
    parser.add_argument("--stop-only", action="store_true",
                        help="Emergency deactivation without requesting inference")
    parser.add_argument("--prompt", default="1 2 3")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/iq235-baseten"))
    args = parser.parse_args()
    key = os.getenv("BASETEN_API_KEY")
    if not key or not args.model_id or not args.deployment_id:
        parser.error("BASETEN_API_KEY, BASETEN_MODEL_ID and BASETEN_DEPLOYMENT_ID must be set")
    for identifier in (args.model_id, args.deployment_id):
        if not re.fullmatch("[A-Za-z0-9_-]{4,128}", identifier):
            parser.error("unexpected deployment/model ID format")

    inference_error = None
    try:
        if not args.stop_only:
            record = ensure_exact_deployment(args.model_id, args.deployment_id, key)
            if record["status"] != "ACTIVE":
                raise BasetenFullIQError(f"deployment not ACTIVE: {record['status']}")
            url = (f"https://model-{args.model_id}.api.baseten.co/"
                   f"deployment/{args.deployment_id}/predict")
            response = request(url, key, method="POST",
                               body={"prompt": args.prompt, "max_new_tokens": 1},
                               timeout=900)
            verify_response(response)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output = args.output_dir / "architecture_only.json"
            if output.exists():
                raise BasetenFullIQError(f"refusing to overwrite prior result: {output}")
            output.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
            print(f"FULL_IQ235_FORWARD_PASS: {output}", flush=True)
    except Exception as exc:
        inference_error = str(exc)
        print(f"INFERENCE_FAILED: {inference_error}", file=sys.stderr, flush=True)
    finally:
        try:
            terminal = deactivate(args.model_id, args.deployment_id, key)
            print(f"DEPLOYMENT_STOPPED: {terminal['status']}, replicas={terminal['active_replica_count']}",
                  flush=True)
        except Exception as exc:
            print(f"CRITICAL: BAS​ETEN DEACTIVATION UNVERIFIED: {exc}",
                  file=sys.stderr, flush=True)
            raise SystemExit(3)
    if inference_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
