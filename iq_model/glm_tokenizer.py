"""GLM-5.3 tokenizer-only contract for the existing IQ hybrid model.

No donor neural weights, new inference engine, or replacement Mamba backbone.
The IQ lexical embedding and LM-head must be trained for the GLM token IDs.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Any

from .config import IQModelConfig
from .hybrid import IQHybridConfig

GLM53_TOKENIZER_REPO = "zai-org/GLM-5.3-BF16"
GLM53_TOKENIZER_REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
GLM53_VOCAB_SIZE = 154880
GLM53_PAD_TOKEN_ID = 154820
GLM53_EOS_TOKEN_IDS = (154820, 154827, 154829)


class GLMTokenizerError(ValueError):
    """GLM lexical identity and IQ model configuration disagree."""


def with_glm_tokenizer_config(config: IQHybridConfig) -> IQHybridConfig:
    """Change only lexical vocabulary width; preserve all IQ layers and schedules."""
    return replace(
        config,
        model=replace(config.model, vocab_size=GLM53_VOCAB_SIZE),
    )


def tokenizer_mapping_sha256(vocab: dict[str, int]) -> str:
    if not isinstance(vocab, dict) or not vocab:
        raise GLMTokenizerError("tokenizer must expose a nonempty token-to-ID mapping")
    if any(type(token) is not str or type(idx) is not int for token, idx in vocab.items()):
        raise GLMTokenizerError("tokenizer mapping must contain string tokens and integer IDs")
    if len(set(vocab.values())) != len(vocab):
        raise GLMTokenizerError("duplicate token IDs in tokenizer mapping")
    encoded = json.dumps(vocab, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def verify_glm_tokenizer_for_iq(
    tokenizer: Any,
    source_config: dict[str, Any],
    model_config: IQModelConfig | None = None,
    *,
    expected_mapping_sha256: str | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Check exact token-space geometry; do not claim neural-weight compatibility."""
    source_eos = source_config.get("eos_token_id")
    if (
        source_config.get("model_type") != "glm_moe_dsa"
        or source_config.get("vocab_size") != GLM53_VOCAB_SIZE
        or source_config.get("pad_token_id") != GLM53_PAD_TOKEN_ID
        or not isinstance(source_eos, (tuple, list))
        or tuple(source_eos) != GLM53_EOS_TOKEN_IDS
    ):
        raise GLMTokenizerError("GLM-5.3 tokenizer source config does not match official IDs")
    if model_config is not None and model_config.vocab_size != GLM53_VOCAB_SIZE:
        raise GLMTokenizerError(
            f"IQ vocab_size={model_config.vocab_size} differs from GLM {GLM53_VOCAB_SIZE}; "
            "create a new IQ config with with_glm_tokenizer_config"
        )
    if model is not None and model_config is None:
        raise GLMTokenizerError("checking IQ model weights requires an IQ model config")
    if (
        getattr(tokenizer, "pad_token_id", None) != GLM53_PAD_TOKEN_ID
        or getattr(tokenizer, "eos_token_id", None) not in GLM53_EOS_TOKEN_IDS
    ):
        raise GLMTokenizerError("GLM tokenizer EOS/PAD IDs disagree with official config")
    vocab = tokenizer.get_vocab()
    digest = tokenizer_mapping_sha256(vocab)
    if any(identifier < 0 or identifier >= GLM53_VOCAB_SIZE for identifier in vocab.values()):
        raise GLMTokenizerError("tokenizer ID exceeds IQ embedding/output vocabulary")
    if not (0 < len(tokenizer) <= GLM53_VOCAB_SIZE):
        raise GLMTokenizerError("tokenizer length exceeds IQ vocabulary")
    for identifier in GLM53_EOS_TOKEN_IDS:
        token = tokenizer.convert_ids_to_tokens(identifier)
        if not isinstance(token, str) or not token:
            raise GLMTokenizerError(f"missing GLM special token ID {identifier}")
        if tokenizer.convert_tokens_to_ids(token) != identifier:
            raise GLMTokenizerError(f"GLM special token roundtrip failed for {identifier}")
    if expected_mapping_sha256 is not None:
        pinned = expected_mapping_sha256.lower().strip()
        if len(pinned) != 64 or any(ch not in "0123456789abcdef" for ch in pinned):
            raise GLMTokenizerError("expected_mapping_sha256 must be 64 hex characters")
        if digest != pinned:
            raise GLMTokenizerError("GLM token-to-ID mapping differs from pinned hash")
    if model is not None:
        for label, path in (("embedding", "embed_tokens"), ("LM head", "lm_head")):
            module = getattr(model, path, None)
            weight = getattr(module, "weight", None)
            expected_shape = (GLM53_VOCAB_SIZE, model_config.hidden_size)
            if weight is None or tuple(weight.shape) != expected_shape:
                raise GLMTokenizerError(f"IQ {label} must have shape {expected_shape}")
    return {
        "status": "tokenizer_contract_verified",
        "tokenizer_repo": GLM53_TOKENIZER_REPO,
        "tokenizer_revision": GLM53_TOKENIZER_REVISION,
        "tokenizer_sha256": digest,
        "vocab_size": GLM53_VOCAB_SIZE,
        "pad_token_id": GLM53_PAD_TOKEN_ID,
        "eos_token_ids": list(GLM53_EOS_TOKEN_IDS),
        "iq_config_checked": model_config is not None,
        "iq_hidden_size": model_config.hidden_size if model_config is not None else None,
        "iq_config_fingerprint": model_config.fingerprint if model_config is not None else None,
        "weight_alignment_verified": False,
    }


def encode_glm_prompt(
    tokenizer: Any, text: str, model_config: IQModelConfig | None = None,
) -> list[int]:
    """Validate token IDs; also check IQ context when a config is supplied."""
    if not text:
        raise GLMTokenizerError("prompt must be nonempty")
    ids = tokenizer.encode(text, add_special_tokens=True)
    if not ids:
        raise GLMTokenizerError("GLM tokenizer produced an empty prompt")
    if model_config is not None and len(ids) > model_config.max_position_embeddings:
        raise GLMTokenizerError("GLM prompt exceeds IQ configured context")
    max_vocab = model_config.vocab_size if model_config is not None else GLM53_VOCAB_SIZE
    if any(type(token_id) is not int or not (0 <= token_id < max_vocab)
           for token_id in ids):
        raise GLMTokenizerError("encoded GLM token ID outside IQ vocabulary")
    return ids
