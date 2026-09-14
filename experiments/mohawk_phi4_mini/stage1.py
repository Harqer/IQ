from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iq_transfer.linear_mixer import IQLinearAttentionMixer
from iq_transfer.mohawk import normalized_frobenius_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOHAWK Stage 1: Phi-4-mini -> IQ linear mixer")
    p.add_argument("--teacher", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--corpus-root", type=Path, default=Path("."))
    p.add_argument("--extensions", nargs="+", default=[".py", ".mojo"])
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--min-tokens", type=int, default=64)
    p.add_argument("--max-files", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=64)
    p.add_argument("--eval-chunks", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("artifacts/mohawk_phi4_mini_stage1.pt"))
    return p.parse_args()


def _split_is_eval(path: Path) -> bool:
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).digest()
    return digest[0] % 5 == 0


def collect_files(root: Path, extensions: list[str], max_files: int) -> tuple[list[Path], list[Path]]:
    allowed = set(extensions)
    paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in allowed:
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__", "artifacts"} for part in path.parts):
            continue
        paths.append(path)
        if len(paths) >= max_files:
            break
    train = [p for p in paths if not _split_is_eval(p)]
    eval_ = [p for p in paths if _split_is_eval(p)]
    if not train or not eval_:
        raise RuntimeError("curated corpus must contain both deterministic train and held-out files")
    return train, eval_


def token_chunks(
    paths: Iterable[Path],
    tokenizer,
    *,
    seq_len: int,
    min_tokens: int,
) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="strict")
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        for start in range(0, ids.numel(), seq_len):
            chunk = ids[start : start + seq_len]
            if chunk.numel() >= min_tokens:
                chunks.append(chunk.contiguous())
    if not chunks:
        raise RuntimeError("no token chunks survived corpus filtering")
    return chunks


def teacher_targets(teacher, layer_idx: int, input_ids: torch.Tensor):
    with torch.no_grad():
        outputs = teacher(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.attentions is None or outputs.attentions[layer_idx] is None:
            raise RuntimeError("teacher did not return attention matrices; load it with eager attention")
        hidden = outputs.hidden_states[layer_idx]
        layer = teacher.model.layers[layer_idx]
        mixer_input = layer.input_layernorm(hidden)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_embeddings = teacher.model.rotary_emb(mixer_input, position_ids=positions)
        return mixer_input, position_embeddings, outputs.attentions[layer_idx]


@torch.no_grad()
def evaluate(teacher, student, chunks, *, layer_idx: int, device: str, limit: int) -> float:
    student.eval()
    losses = []
    for chunk in chunks[:limit]:
        input_ids = chunk.unsqueeze(0).to(device)
        hidden, pos, teacher_matrix = teacher_targets(teacher, layer_idx, input_ids)
        student_matrix = student.mixing_matrix(hidden, position_embeddings=pos)
        losses.append(float(normalized_frobenius_loss(student_matrix.float(), teacher_matrix.float()).item()))
    student.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    teacher.eval().requires_grad_(False)

    cfg = teacher.config
    if not (0 <= args.layer < cfg.num_hidden_layers):
        raise ValueError(f"layer must be in [0, {cfg.num_hidden_layers - 1}]")

    train_files, eval_files = collect_files(args.corpus_root, args.extensions, args.max_files)
    train_chunks = token_chunks(train_files, tokenizer, seq_len=args.seq_len, min_tokens=args.min_tokens)
    eval_chunks = token_chunks(eval_files, tokenizer, seq_len=args.seq_len, min_tokens=args.min_tokens)

    student = IQLinearAttentionMixer(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
    ).to(device=args.device, dtype=torch.bfloat16)

    teacher_attn = teacher.model.layers[args.layer].self_attn
    student.initialize_from_phi4(teacher_attn.qkv_proj.weight, teacher_attn.o_proj.weight)

    # Stage 1 only needs to orient Q/K because the loss is defined on the mixer matrix.
    for p in student.parameters():
        p.requires_grad_(False)
    student.q_proj.weight.requires_grad_(True)
    student.k_proj.weight.requires_grad_(True)
    optimizer = torch.optim.AdamW([student.q_proj.weight, student.k_proj.weight], lr=args.lr, weight_decay=0.0)

    initial_val = evaluate(
        teacher, student, eval_chunks, layer_idx=args.layer, device=args.device, limit=args.eval_chunks
    )
    print(f"initial held-out normalized Frobenius error: {initial_val:.6f}")

    student.train()
    for step in range(args.train_steps):
        chunk = train_chunks[step % len(train_chunks)]
        input_ids = chunk.unsqueeze(0).to(args.device)
        hidden, pos, teacher_matrix = teacher_targets(teacher, args.layer, input_ids)

        optimizer.zero_grad(set_to_none=True)
        student_matrix = student.mixing_matrix(hidden, position_embeddings=pos)
        loss = normalized_frobenius_loss(student_matrix.float(), teacher_matrix.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([student.q_proj.weight, student.k_proj.weight], 1.0)
        optimizer.step()

        if step == 0 or (step + 1) % 8 == 0:
            print(f"step={step + 1:04d} stage1_loss={loss.item():.6f}")

    final_val = evaluate(
        teacher, student, eval_chunks, layer_idx=args.layer, device=args.device, limit=args.eval_chunks
    )
    improvement = (initial_val - final_val) / max(initial_val, 1e-12)
    print(f"final held-out normalized Frobenius error: {final_val:.6f}")
    print(f"relative held-out improvement: {improvement:.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "teacher": args.teacher,
        "layer": args.layer,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "initial_val_normalized_frobenius": initial_val,
        "final_val_normalized_frobenius": final_val,
        "relative_improvement": improvement,
        "train_files": [str(p) for p in train_files],
        "eval_files": [str(p) for p in eval_files],
    }
    torch.save({"state_dict": student.state_dict(), "metadata": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
