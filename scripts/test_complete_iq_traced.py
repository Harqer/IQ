#!/usr/bin/env python3
"""Real 82-shard IQ checkpoint, full 80-layer CUDA inference, per-token tracing."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from iq_observability.full_trace import IQTrace
from iq_transfer import load_complete_iq_checkpoint
from iq_transfer.checkpoint_preflight import preflight_iq_checkpoint, verify_iq_tokenizer


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--tokenizer-dir", type=Path, default=None)
    p.add_argument("--prompt", default="Write a Python function that checks whether a string is a palindrome.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--trace-dir", type=Path, default=Path(os.getenv("IQ_TRACE_DIR", "/tmp/iq-traces")))
    p.add_argument("--profile-cuda", action="store_true", help="Record CUDA operator profiling during first REAL full-model forward")
    return p


def run(args):
    # No execution with a partial/unverified compiled checkpoint.
    os.environ["IQ_REQUIRE_CHECKPOINT_SHA256"] = "1"
    os.environ["IQ_EXPECTED_INDEX_SHA256"] = "0154a99903ea86daea357f855ee24ae10e430e2adda022f2a6843476eaaa6128"
    os.environ["IQ_EXPECTED_SHA256SUMS_SHA256"] = "705b9127a8575acc5aba9e48dafa58ddfbb6d7616917711bffcd4f52478e25e2"
    if not torch.cuda.is_available():
        raise RuntimeError("Full-weight IQ inference requires CUDA; refusing CPU fallback")
    if not 1 <= args.max_new_tokens <= 64:
        raise ValueError("max-new-tokens must be 1..64")
    tokenizer_dir = args.tokenizer_dir or Path(os.getenv("IQ_TOKENIZER_DIR") or args.checkpoint)
    recorder = IQTrace(args.trace_dir)
    try:
        with recorder.stage("checkpoint.preflight"):
            evidence = preflight_iq_checkpoint(args.checkpoint, tokenizer_dir=tokenizer_dir)
            recorder.event("checkpoint.metadata",
                           shards=evidence.shard_count, weight_tensors=evidence.weight_count,
                           calibration_mode=evidence.manifest.get("calibration_mode") if evidence.manifest else "unknown",
                           checksum_verified=False)
        with recorder.stage("tokenizer.load"):
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
            lexical = verify_iq_tokenizer(tokenizer, evidence.config, evidence.manifest)
            recorder.event("tokenizer.verified", count=len(tokenizer), lexical_source=lexical)
        with recorder.stage("checkpoint.weight_load"):
            model = load_complete_iq_checkpoint(
                args.checkpoint, device=torch.device("cuda:0"), dtype=torch.bfloat16, telemetry=recorder
            ).eval()
        recorder.attach(model)
        recorder.event("checkpoint.loaded", physical_layers=len(model.layers),
                       source="compiled_safetensors", randomized_weights=False)
        try:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True, return_tensors="pt")
            tokens = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
        except (KeyError, TypeError, ValueError, AttributeError):
            tokens = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
            recorder.event("tokenizer.chat_template_unavailable", fallback="raw_completion")
        generated = tokens.to(model.embed_tokens.weight.device)
        eos = tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos_set = set(eos)
        else:
            eos_set = {eos} if isinstance(eos, int) else set()
        start = time.perf_counter()
        next_ids = []
        with recorder.stage("inference.full_model", prompt_tokens=int(tokens.shape[1]),
                            max_new_tokens=args.max_new_tokens):
            with torch.inference_mode():
                for step in range(args.max_new_tokens):
                    recorder.round_start()
                    with recorder.stage("inference.token", step=step, prefix_len=int(generated.shape[1])):
                        if step == 0 and args.profile_cuda:
                            with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=True,
                            ) as profiler:
                                result = model(generated, return_hidden_states=True)
                                torch.cuda.synchronize()
                            profiler.export_chrome_trace(str(args.trace_dir / (recorder.run_id + "-cuda.json")))
                        else:
                            result = model(generated, return_hidden_states=True)
                        recorder.assert_complete()
                        logits = result.logits[:, -1, :].float()
                        if logits.shape != (1, model.config.model.vocab_size):
                            raise RuntimeError(f"Unexpected logits: {tuple(logits.shape)}")
                        if not bool(torch.isfinite(logits).all().item()):
                            raise FloatingPointError(f"Non-finite logits at generated token {step}")
                        probs = torch.softmax(logits, dim=-1)
                        top_probs, top_ids = probs.topk(5, dim=-1)
                        recorder.event("logits.checked", step=step,
                                       max_logit=float(logits.max().item()),
                                       min_logit=float(logits.min().item()),
                                       entropy=float((-(probs * probs.clamp_min(1e-30).log()).sum()).item()),
                                       top_ids=top_ids[0].tolist(), top_probabilities=top_probs[0].tolist(),
                                       reasoning_steps=result.reasoning_steps_executed,
                                       moe_layers=len(result.expert_counts))
                        if len(result.expert_counts) != 24:
                            raise RuntimeError("Mandatory 24 MoE outputs absent")
                        next_token = logits.argmax(dim=-1, keepdim=True)
                        token_id = int(next_token[0, 0].item())
                        next_ids.append(token_id)
                        generated = torch.cat((generated, next_token), dim=1)
                    if token_id in eos_set:
                        break
        response = tokenizer.decode(next_ids, skip_special_tokens=True)
        if not response.strip():
            raise RuntimeError("Full model returned an empty response")
        seconds = time.perf_counter() - start
        recorder.event("inference.completed", new_tokens=len(next_ids),
                       elapsed_seconds=round(seconds, 3),
                       tokens_per_second=round(len(next_ids)/max(seconds, 1e-9), 3),
                       output_nonempty=True)
        result = {"run_id": recorder.run_id, "checkpoint": str(args.checkpoint),
                  "trace_file": str(recorder.path), "generated_tokens": len(next_ids),
                  "text": response, "status": "execution_passed_quality_not_validated"}
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return result
    except BaseException as exc:
        recorder.event("inference.failed", error_type=type(exc).__name__, detail=str(exc)[:350])
        raise
    finally:
        recorder.close()


if __name__ == "__main__":
    run(make_parser().parse_args())
