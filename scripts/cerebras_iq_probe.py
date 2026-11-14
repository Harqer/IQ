"""Cerebras connection, IQ endpoint guard, and one-shot traced inference probe.

Account model discovery is NOT a proof that IQ weights were deployed. The
custom IQ request is opt-in and requires a separate dedicated endpoint that
the user has verified with Cerebras. No automatic upload, paid job, or retries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from iq_telemetry import configure_telemetry, trace_stage

IQ_CHECKPOINT = "kofecheckmate/iq-glm53-all256"
IQ_ARCHITECTURE = "iq-mamba3-mimo-csa-hca-stable-latent-moe-attnres"


def _safe_error(exc: Exception) -> dict[str, Any]:
    """Never include exception text, HTTP request body, or secrets in reports."""
    status = getattr(exc, "status_code", None)
    result: dict[str, Any] = {"type": type(exc).__name__}
    if isinstance(status, int):
        result["status_code"] = status
    return result


def run_probe(
    client: Any,
    *,
    endpoint_id: str | None = None,
    execute_iq: bool = False,
    confirm_deployed: bool = False,
    allow_billed_request: bool = False,
) -> dict[str, Any]:
    """Check account access; optionally call an already-deployed custom IQ endpoint.

    A successful response proves only endpoint reachability, not internal tensor
    identity, exact weight provenance, or preservation of GLM capabilities.
    """
    report: dict[str, Any] = {
        "checkpoint": IQ_CHECKPOINT,
        "architecture": IQ_ARCHITECTURE,
        "api_authenticated": False,
        "available_model_ids": [],
        "iq_deployment_verified": False,
        "iq_inference_executed": False,
        "checkpoint_weights_verified_by_cerebras": False,
        "status": "unstarted",
    }
    try:
        with trace_stage("iq.cerebras.models_list"):
            listing = client.models.list()
            raw_models = getattr(listing, "data", None)
            if raw_models is None:
                raise ValueError("Cerebras models response had no data")
            models = [getattr(item, "id", None) for item in raw_models]
            if any(not isinstance(model, str) for model in models):
                raise ValueError("Malformed Cerebras models response")
            report["available_model_ids"] = sorted(models)
            report["api_authenticated"] = True
            report["status"] = "api_connected"
    except Exception as exc:
        report["status"] = "api_error"
        report["error"] = _safe_error(exc)
        return report

    if not execute_iq:
        report["next_step"] = "Dedicated IQ deployment approval and endpoint required"
        return report

    if not endpoint_id or not endpoint_id.strip():
        report["status"] = "blocked_no_iq_endpoint"
        return report
    if endpoint_id.strip() in report["available_model_ids"] and not confirm_deployed:
        report["status"] = "blocked_unverified_endpoint"
        report["next_step"] = "An API-visible model ID alone does not prove IQ deployment"
        return report
    if not confirm_deployed:
        report["status"] = "blocked_unverified_endpoint"
        report["next_step"] = "Get explicit confirmation of IQ checkpoint and architecture deployment"
        return report
    if not allow_billed_request:
        report["status"] = "blocked_billing_not_approved"
        return report

    try:
        with trace_stage("iq.cerebras.dedicated_inference"):
            response = client.chat.completions.create(
                model=endpoint_id.strip(),
                messages=[{"role": "user", "content": "Reply with exactly IQ_OK"}],
                max_completion_tokens=8,
            )
        report["iq_inference_executed"] = True
        report["response_model"] = getattr(response, "model", None)
        report["status"] = "dedicated_endpoint_responded"
        report["warning"] = (
            "An API response alone does not prove that the server used IQ weights; "
            "confirm the Cerebras deployment version and checkpoint fingerprint."
        )
    except Exception as exc:
        report["status"] = "dedicated_inference_error"
        report["error"] = _safe_error(exc)
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", action="store_true", help="Check local dependencies without contacting Cerebras")
    parser.add_argument("--run-iq", action="store_true", help="Send ONE real request to a verified IQ dedicated endpoint")
    parser.add_argument("--endpoint-id", default=os.getenv("CEREBRAS_IQ_ENDPOINT_ID"))
    parser.add_argument("--confirm-deployed", action="store_true", help="Confirm Cerebras has deployed the actual IQ architecture and checkpoint")
    parser.add_argument("--allow-billed-request", action="store_true", help="Authorize one chargeable inference request; default is no inference")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    configure_telemetry()
    result: dict[str, Any]
    if args.doctor:
        try:
            import cerebras.cloud.sdk  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        result = {
            "sdk_present": sdk_present,
            "key_present": bool(os.getenv("CEREBRAS_API_KEY")),
            "iq_endpoint_configured": bool(args.endpoint_id),
            "remote_inference_executed": False,
        }
        print(json.dumps(result, indent=2))
        return 0 if sdk_present else 2

    token = os.getenv("CEREBRAS_API_KEY")
    if not token:
        print(json.dumps({"status": "missing_key", "next_step": "Set CEREBRAS_API_KEY (e.g. Colab Secrets)", "iq_inference_executed": False}, indent=2))
        return 2

    try:
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=token)
    except Exception as exc:
        print(json.dumps({"status": "client_initialization_error", "error": _safe_error(exc)}, indent=2))
        return 2
    result = run_probe(
        client,
        endpoint_id=args.endpoint_id,
        execute_iq=args.run_iq,
        confirm_deployed=args.confirm_deployed,
        allow_billed_request=args.allow_billed_request,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("api_connected", "dedicated_endpoint_responded") else 3


if __name__ == "__main__":
    sys.exit(main())
