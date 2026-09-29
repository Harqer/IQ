from __future__ import annotations

import argparse
from pathlib import Path

import torch

from iq_transfer import load_complete_iq_checkpoint


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load the complete transplanted IQ checkpoint and run generation."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="Write a Python function that checks whether a string is a palindrome.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not torch.cuda.is_available():
        raise SystemExit("complete IQ inference requires CUDA")

    root = Path(args.checkpoint)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
    )
    model = load_complete_iq_checkpoint(
        root,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )

    messages = [{"role": "user", "content": args.prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    except (ValueError, AttributeError, KeyError):
        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]

    input_ids = input_ids.to("cuda:0")
    with torch.inference_mode():
        first = model(input_ids)
    if not bool(torch.isfinite(first.logits).all()):
        raise AssertionError("complete IQ checkpoint produced non-finite logits")

    generated = input_ids
    for _ in range(args.max_new_tokens):
        with torch.inference_mode():
            output = model(generated)
        logits = output.logits[:, -1, :].float()
        if args.temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if tokenizer.eos_token_id is not None and bool(
            (next_token == tokenizer.eos_token_id).all()
        ):
            break

    print(
        tokenizer.decode(
            generated[0],
            skip_special_tokens=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
