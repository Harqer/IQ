"""CPU-only GLM tokenizer/lexical tensor identity check.

A 154880-size vocabulary does NOT prove token-ID compatibility. Pin the exact
full token->ID mapping hash from the same immutable GLM source revision as the
model weights. Do not substitute Llama/Mamba tokenizer in a GLM-first hybrid.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


class GLMTokenizerContractError(ValueError):
    pass


def mapping_fingerprint(vocab: dict[str, int]) -> str:
    if not vocab or not all(isinstance(k, str) and isinstance(v, int)
                            for k, v in vocab.items()):
        raise GLMTokenizerContractError("Tokenizer mapping must be nonempty token->integer ID")
    serialized = json.dumps(
        vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


def verify_glm_tokenizer(
    tokenizer: Any,
    model: Any,
    *,
    expected_mapping_sha256: str,
    expected_vocab_size: int = 154880,
    expected_eos_ids: tuple[int, ...] = (154820, 154827, 154829),
) -> dict:
    """Fail closed unless token IDs are tied to a pinned GLM lexical mapping."""
    expected_hash = expected_mapping_sha256.lower().strip()
    if len(expected_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_hash):
        raise GLMTokenizerContractError(
            "A real 64-character reference mapping SHA-256 is required"
        )
    config = getattr(model, "config", None)
    if config is None or getattr(config, "vocab_size", None) != expected_vocab_size:
        raise GLMTokenizerContractError("GLM configuration vocabulary mismatch")
    embed = model.get_input_embeddings()
    output = model.get_output_embeddings()
    for label, layer in (("embedding", embed), ("lm_head", output)):
        weight = getattr(layer, "weight", None)
        if weight is None or weight.ndim != 2 or weight.shape[0] != expected_vocab_size:
            raise GLMTokenizerContractError(f"GLM {label} row count mismatch")
    if embed.weight.shape[1] != output.weight.shape[1]:
        raise GLMTokenizerContractError("GLM embedding/LM-head hidden dimensions differ")
    if getattr(config, "hidden_size", None) != embed.weight.shape[1]:
        raise GLMTokenizerContractError("GLM residual width differs from lexical weights")

    observed_eos = getattr(config, "eos_token_id", None)
    if isinstance(observed_eos, int):
        observed_eos = [observed_eos]
    if tuple(observed_eos or ()) != expected_eos_ids:
        raise GLMTokenizerContractError("GLM model special EOS IDs mismatch")
    if getattr(tokenizer, "eos_token_id", None) not in expected_eos_ids:
        raise GLMTokenizerContractError("tokenizer eos_token_id not in official GLM EOS set")
    pad = getattr(config, "pad_token_id", None)
    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    if pad != tokenizer_pad:
        raise GLMTokenizerContractError("GLM tokenizer/model pad-token IDs mismatch")

    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise GLMTokenizerContractError("tokenizer does not expose a vocabulary")
    ids = list(vocab.values())
    if len(ids) != len(set(ids)):
        raise GLMTokenizerContractError("duplicate tokenizer ID assignments")
    if any(type(v) is not int or not (0 <= v < expected_vocab_size) for v in ids):
        raise GLMTokenizerContractError("tokenizer ID outside embedding range")
    for eid in expected_eos_ids:
        token = tokenizer.convert_ids_to_tokens(eid)
        if not isinstance(token, str) or not token:
            raise GLMTokenizerContractError(f"missing GLM control token ID: {eid}")
        if tokenizer.convert_tokens_to_ids(token) != eid:
            raise GLMTokenizerContractError(f"GLM control token round-trip failed: {eid}")
    digest = mapping_fingerprint(vocab)
    if digest != expected_hash:
        raise GLMTokenizerContractError(
            "full token-to-ID map does not match pinned GLM source fingerprint"
        )
    return {
        "status": "verified_mapping_only",
        "token_to_id_sha256": digest,
        "model_vocab_size": expected_vocab_size,
        "tokenizer_entries": len(vocab),
        "model_hidden_size": embed.weight.shape[1],
        "eos_token_ids": list(expected_eos_ids),
        "note": "Full ID mapping and lexical tensor geometry matched a pinned source; "
                "model generation parity and chat-template correctness still require tests.",
    }
