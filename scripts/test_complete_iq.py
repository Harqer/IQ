from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from iq_telemetry import configure_telemetry, trace_stage

from iq_transfer import load_complete_iq_checkpoint


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load the complete transplanted IQ checkpoint and run generation."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--prompt",
        default="Write a Python function that checks whether a string is a palindrome.",
    )
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--numeric-only", action="store_true", help="Run real checkpoint inference using in-vocabulary token IDs; no tokenizer")
    p.add_argument("--token-ids", default="1,2,3,4", help="Comma-separated numerical IDs for --numeric-only")
    p.add_argument("--device-map", default=os.getenv("IQ_DEVICE_MAP"), help="Metadata-only hybrid stage placement JSON")
    return p


def main() -> int:
    args = parser().parse_args()
    configure_telemetry()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not torch.cuda.is_available():
        raise SystemExit("complete IQ inference requires CUDA")

    root = Path(args.checkpoint)
    config = json.loads((root / "iq_config.json").read_text(encoding="utf-8"))
    expected_vocab = config["model"]["vocab_size"]
    tokenizer = None
    if args.numeric_only:
        token_ids = [int(part.strip()) for part in args.token_ids.split(",")]
        if not token_ids or any(token_id < 0 or token_id >= expected_vocab for token_id in token_ids):
            raise ValueError(f"Numerical token IDs must be within [0, {expected_vocab})")
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        print(f"IQ: real-checkpoint numerical input: shape={tuple(input_ids.shape)}, vocab={expected_vocab}", flush=True)
    else:
        tokenizer_root = Path(os.getenv("IQ_TOKENIZER_DIR") or root)
        from transformers import AutoTokenizer
        print(f"IQ: loading tokenizer from {tokenizer_root}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
        if len(tokenizer) != expected_vocab:
            raise RuntimeError(
                f"Checkpoint vocabulary {expected_vocab} does not match "
                f"tokenizer size {len(tokenizer)}"
            )
        messages = [{"role": "user", "content": args.prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = (
                encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
            )
        except (ValueError, AttributeError, KeyError):
            input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    print(f"IQ: loading checkpoint from {root}", flush=True)
    with trace_stage("iq.checkpoint.load", vocab_size=int(expected_vocab)):
        model = load_complete_iq_checkpoint(
            root,
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
            device_map_path=args.device_map,
        )

    # GeeksforGeeks-style parameter accounting; diagnostic only, no training.
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(
        f"IQ: parameters={total_params:,}, trainable={trainable_params:,}; "
        f"inference layers={len(model.layers)}",
        flush=True,
    )
    print("IQ: checkpoint loaded; running real forward pass", flush=True)
    input_ids = input_ids.to("cuda:0")
    with trace_stage("iq.first_forward", batch_size=int(input_ids.shape[0]), sequence_length=int(input_ids.shape[1]), vocab_size=int(expected_vocab)):
        with torch.inference_mode():
            first = model(input_ids)
    expected_shape = (input_ids.shape[0], input_ids.shape[1], expected_vocab)
    if tuple(first.logits.shape) != expected_shape:
        raise AssertionError(
            f"IQ logits shape mismatch: expected={expected_shape}, got={tuple(first.logits.shape)}"
        )
    if not bool(torch.isfinite(first.logits).all()):
        raise AssertionError("complete IQ checkpoint produced non-finite logits")
    print(
        f"IQ_FIRST_REAL_FORWARD_PASS: logits_shape={tuple(first.logits.shape)} "
        f"finite=True device={first.logits.device}",
        flush=True,
    )

    generated = input_ids
    for generation_step in range(args.max_new_tokens):
        with trace_stage("iq.generate.step", generation_step=generation_step, sequence_length=int(generated.shape[1])):
            with torch.inference_mode():
                output = model(generated)
        logits = output.logits[:, -1, :].float()
        if args.temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        # The LM head can reside on the final GPU; generated token IDs stay on GPU 0.
        next_token = next_token.to(generated.device)
        generated = torch.cat((generated, next_token), dim=1)
        if tokenizer is not None and tokenizer.eos_token_id is not None and bool(
            (next_token == tokenizer.eos_token_id).all()
        ):
            break

    if tokenizer is not None:
        print(tokenizer.decode(generated[0], skip_special_tokens=False))
    else:
        print(f"IQ_NUMERICAL_GENERATION_PASS token_ids={generated[0].tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
