#!/usr/bin/env python3
"""CPU-only BULK benchmark of actual generated answers: original Mamba-3 vs Phase 5e.

Exercises the same runner.generate() method used by the inference HTTP server.
Never updates pretrained weights or adapter weights. Responses, failures,
timings and paired original/swapped evidence are saved incrementally.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import urllib.request
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iq_local.mamba3_bulk_audit import grade_response, summarize
from iq_local.mamba3_csa_hca import Mamba3CSAHCARunner, PHASE5E_SHA256, MODEL_ID

EXPERIMENT_DIR = ROOT / "experiments" / "mamba3-csa-hca" / "iq_mamba3_csa_hca_phase5e"
TOKENIZER_REPO = "unsloth/Meta-Llama-3.1-8B"
AUDIT_VERSION = 1
# Separate from Phase 5e's 8 trained + 4 dev + 8 locked keys.
NEW_UNSEEN_KEYS = [
    "cypress", "tundra", "pavilion", "marigold", "magnolia",
    "redwood", "sequoia", "compass", "bluejay", "moonstone",
    "pinecrest", "starling", "coralreef", "glacier", "wildwood",
    "evergreen", "fernwood", "silverlake", "sunbeam", "honeycomb",
]
GENERAL = [
    ("capital", "The capital of France is", ["Paris"]),
    ("planet", "The largest planet in the solar system is", ["Jupiter"]),
    ("arithmetic", "Question: What is 3 plus 4? Answer:", ["7", "seven"]),
    ("opposite", "The opposite of hot is", ["cold"]),
    ("addition", "Two plus two equals", ["4", "four"]),
    ("freeform", "Continue this story: The traveler opened the door and found", None),
]


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    group: str
    variant: str
    pair_id: str | None
    expected: tuple[str, ...] | None
    ids: tuple[int, ...]
    prompt: str


def _phase_modules():
    """Import the exact case construction and source-swap used for Phase 5e."""
    if not EXPERIMENT_DIR.is_dir():
        raise FileNotFoundError(f"Phase 5e sources unavailable: {EXPERIMENT_DIR}")
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    records = importlib.import_module("iq_retrieval_phase5b")
    phase5e = importlib.import_module("iq_answer_phase5e")
    phase5e.ensure_fresh_keys()
    if len(records.PROJECTS) == 36:
        records.PROJECTS.extend(NEW_UNSEEN_KEYS)
    if records.PROJECTS[36:56] != NEW_UNSEEN_KEYS:
        raise RuntimeError("Fresh benchmark key registry is not stable")
    return records, phase5e


def _add_pair(cases, tok, records, phase5e, index, group, paraphrase=False):
    original = records.make_record(tok, index, group)
    swapped = phase5e.swap_context_answer(original, tok)
    if int((original.ids != swapped.ids).sum().item()) != 1:
        raise AssertionError("Swapped evidence must differ by exactly one token")
    for variant, ex in (("original", original), ("swapped", swapped)):
        ids = ex.ids[0].tolist()
        if paraphrase:
            ids = (ids[:270] + tok.encode(
                f" Note: Locate the archived record for Project {ex.key}. "
                "Which code is assigned to it? Reply with only that code:",
                add_special_tokens=False).ids)
        if not (270 < len(ids) < 400):
            raise ValueError(f"{group}/{ex.key}: context length outside expected range")
        cases.append(AuditCase(
            case_id=f"{group}:{ex.key}:{variant}",
            group=group, variant=variant, pair_id=ex.key,
            expected=(ex.answer,), ids=tuple(ids),
            prompt=tok.decode(ids, skip_special_tokens=False),
        ))


def make_cases(tok, *, fresh_pairs=8, regression_pairs=4,
               paraphrase_pairs=4, general_prompts=6):
    if not (0 <= regression_pairs <= 8 and 0 <= general_prompts <= len(GENERAL)):
        raise ValueError("regression_pairs must be 0..8; general_prompts 0..6")
    if fresh_pairs < 0 or paraphrase_pairs < 0 or fresh_pairs + paraphrase_pairs > len(NEW_UNSEEN_KEYS):
        raise ValueError("fresh_pairs + paraphrase_pairs must be 0..20")
    records, phase5e = _phase_modules()
    cases = []
    # Previously evaluated Phase 5e heldouts: regression only, NOT clean tests.
    for index in range(24, 24 + regression_pairs):
        _add_pair(cases, tok, records, phase5e, index, "regression")
    # Newly held-out project identities never used in Phase 5e training.
    for index in range(36, 36 + fresh_pairs):
        _add_pair(cases, tok, records, phase5e, index, "fresh")
    # Disjoint names and altered question wording probe template dependence.
    for index in range(36 + fresh_pairs, 36 + fresh_pairs + paraphrase_pairs):
        _add_pair(cases, tok, records, phase5e, index, "paraphrase", paraphrase=True)
    for name, prompt, answers in GENERAL[:general_prompts]:
        cases.append(AuditCase(
            case_id=f"general:{name}", group="general", variant="single",
            pair_id=None, expected=tuple(answers) if answers else None,
            ids=tuple(tok.encode(prompt, add_special_tokens=True).ids),
            prompt=prompt,
        ))
    identifiers = [c.case_id for c in cases]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Duplicate benchmark case IDs")
    return cases


def _save_json(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _resume_rows(path: Path, run_id: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Corrupt JSONL record {number}; recover before resume") from exc
        if row.get("run_id") != run_id:
            raise RuntimeError("Refusing to combine results from different benchmark settings")
        rows.append(row)
    return rows


def _run_case(runner, tok, case: AuditCase, mode: str, new_tokens: int,
              stops: set[int], run_id: str) -> dict:
    started = time.perf_counter()
    generated = []
    response = ""
    error = None
    try:
        runner.mode = mode
        generated = runner.generate(list(case.ids), new_tokens, stop_tokens=stops)
        response = tok.decode(generated, skip_special_tokens=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    score = grade_response(response, list(case.expected) if case.expected else None)
    return {
        "run_id": run_id, "case_id": case.case_id, "group": case.group,
        "variant": case.variant, "pair_id": case.pair_id, "mode": mode,
        "expected": list(case.expected) if case.expected else None,
        "prompt": case.prompt, "prompt_tokens": len(case.ids),
        "response": response, "generated_token_ids": generated,
        "seconds": round(time.perf_counter() - started, 3),
        "error": error, **score,
    }


class RemoteMamba3Runner:
    """One remote exact-model deployment; never a provider's generic model API.

    A single benchmark command sends all case/mode generations and journals
    completed decoded responses. Remote GPU workers must load the *same*
    original Mamba-3 foundation and pinned Phase 5e checkpoint.
    """
    def __init__(self, predict_url: str, api_key: str):
        from urllib.parse import urlparse
        parsed = urlparse(predict_url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username
                or parsed.password or not parsed.hostname
                or not parsed.hostname.endswith(".api.baseten.co")):
            raise ValueError("Remote inference requires an HTTPS Baseten predict URL")
        if not api_key:
            raise ValueError("BASETEN_API_KEY must be set for --predict-url")
        self.url = predict_url
        self.api_key = api_key
        self.mode = "hybrid"

    def generate(self, prompt_ids, max_new_tokens, *, stop_tokens=None):
        payload = json.dumps({
            "prompt_ids": prompt_ids,
            "max_new_tokens": max_new_tokens,
            "stop_tokens": sorted(stop_tokens) if stop_tokens else [],
            "mode": self.mode,
        }).encode()
        request = urllib.request.Request(
            self.url, data=payload, method="POST",
            headers={"Authorization": "Api-Key " + self.api_key,
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        if (not isinstance(result, dict)
                or result.get("source_model") != MODEL_ID
                or result.get("checkpoint_sha256") != PHASE5E_SHA256
                or result.get("mode") != self.mode):
            raise RuntimeError("Remote response identity mismatch: not the exact Phase 5e model/mode")
        generated = result.get("generated_token_ids")
        if (not isinstance(generated, list) or not generated
                or len(generated) > max_new_tokens
                or any(type(token) is not int or not 0 <= token < 128256 for token in generated)):
            raise RuntimeError("Remote endpoint returned invalid generated token IDs")
        return generated


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attention-checkpoint", type=Path, default=ROOT / "checkpoints" /
                   "mamba3-csa-hca" / "iq_phase5e_attention_candidate.pt")
    p.add_argument("--foundation-checkpoint", type=Path, default=None)
    p.add_argument("--tokenizer-json", type=Path, default=None,
                   help="Optional local tokenizer.json, otherwise uses cached HF tokenizer")
    p.add_argument("--output-dir", type=Path, default=Path.home() / "iq-phase5e-bulk-audit")
    p.add_argument("--fresh-pairs", type=int, default=8)
    p.add_argument("--regression-pairs", type=int, default=4)
    p.add_argument("--paraphrase-pairs", type=int, default=4)
    p.add_argument("--general-prompts", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--resume", action="store_true",
                   help="Continue a matching interrupted run without redoing completed responses")
    p.add_argument("--predict-url", default=None,
                   help="Exact Phase 5e Truss prediction endpoint (Baseten custom deployment)")
    p.add_argument("--strict", action="store_true",
                   help="Return nonzero on runtime errors or baseline-over-hybrid accuracy regressions")
    args = p.parse_args()
    if args.max_new_tokens < 2 or args.max_new_tokens > 16:
        p.error("max-new-tokens must be 2..16")
    if args.threads < 1:
        p.error("threads must be >=1")
    torch.set_num_threads(args.threads)
    tok_path = args.tokenizer_json or Path(
        hf_hub_download(TOKENIZER_REPO, filename="tokenizer.json"))
    tok = Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size(True) != 128256 or tok.token_to_id("<|begin_of_text|>") != 128000:
        raise RuntimeError("Original model/tokenizer identity mismatch")
    cases = make_cases(tok, fresh_pairs=args.fresh_pairs,
                       regression_pairs=args.regression_pairs,
                       paraphrase_pairs=args.paraphrase_pairs,
                       general_prompts=args.general_prompts)
    if not cases:
        p.error("At least one benchmark case required")
    if any(len(c.ids) + args.max_new_tokens - 1 > 2048 for c in cases):
        raise ValueError("Benchmark prompt exceeds validated model context limit")

    config = {
        "audit_version": AUDIT_VERSION, "checkpoint_sha256": PHASE5E_SHA256,
        "fresh_pairs": args.fresh_pairs, "regression_pairs": args.regression_pairs,
        "paraphrase_pairs": args.paraphrase_pairs,
        "general_prompts": args.general_prompts,
        "max_new_tokens": args.max_new_tokens, "tokenizer_repo": TOKENIZER_REPO,
        "execution_backend": args.predict_url or "local",
    }
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    root = args.output_dir.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    manifest, journal, report = (root / "manifest.json",
                                 root / "responses.jsonl",
                                 root / "summary.json")
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if not args.resume or prior.get("run_id") != run_id:
            p.error(f"Existing run in {root}; pass --resume for identical settings "
                    "or choose another --output-dir")
    elif journal.exists():
        p.error(f"Found result journal without manifest: {journal}")
    else:
        _save_json(manifest, {"run_id": run_id, "config": config,
                              "planned_generations": len(cases) * 2})
    rows = _resume_rows(journal, run_id)
    completed = {(r["case_id"], r["mode"]) for r in rows if r.get("error") is None}
    # If an interrupted case is retried, the latest record supersedes the error.
    print(f"IQ Phase 5e BULK generation: {len(cases)} prompts x 2 modes; "
          f"already complete={len(completed)}", flush=True)
    if args.predict_url:
        print("Using remote exact-checkpoint Mamba-3 inference; no local model loading.", flush=True)
        runner = RemoteMamba3Runner(args.predict_url, os.environ.get("BASETEN_API_KEY", ""))
    else:
        print("Loading original frozen Mamba-3 and verified Phase 5e adapter...", flush=True)
        runner = Mamba3CSAHCARunner.from_checkpoints(
            args.attention_checkpoint, foundation_path=args.foundation_checkpoint,
            expected_sha256=PHASE5E_SHA256, mode="hybrid")
    stops = {tok.token_to_id(s) for s in ("<|eot_id|>", "<|end_of_text|>")}
    stops.discard(None)
    count = len(cases) * 2
    for index, case in enumerate(cases):
        for mode_pos, mode in enumerate(("baseline", "hybrid")):
            slot = index * 2 + mode_pos + 1
            if (case.case_id, mode) in completed:
                continue
            row = _run_case(runner, tok, case, mode, args.max_new_tokens, stops, run_id)
            with journal.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
            correct = row.get("correct")
            grade = "N/A" if correct is None else ("OK" if correct else "MISS")
            print(f"[{slot}/{count}] {mode:<8} {case.case_id:<34} "
                  f"answer={grade} generated={row['response']!r} "
                  f"time={row['seconds']}s error={row['error']}", flush=True)
            if row["error"] and args.predict_url:
                raise RuntimeError(f"Remote inference failed for {case.case_id}/{mode}; "
                                   f"preserved journal for --resume: {row[\'error\']}")
            completed.add((case.case_id, mode))

    # A resumed case may have an earlier failed attempt; use the final attempt.
    latest = {}
    for item in _resume_rows(journal, run_id):
        latest[(item["case_id"], item["mode"])] = item
    final_rows = list(latest.values())
    summary = summarize(final_rows)
    _save_json(report, {"run_id": run_id, "config": config, "summary": summary,
                        "results": final_rows})
    print(json.dumps({
        "event": "phase5e_bulk_generated_answer_audit",
        "generations": len(final_rows),
        "group_metrics": summary["group_metrics"],
        "paired_evidence": summary["pair_metrics"],
        "head_to_head": summary["head_to_head"],
        "finding_counts": summary["finding_counts"],
        "promotion_status": summary["promotion_status"],
        "full_report": str(report),
        "all_responses": str(journal),
    }, indent=2), flush=True)
    if args.strict and (summary["head_to_head"]["runtime_errors"] or
                        summary["head_to_head"]["baseline_wins"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
