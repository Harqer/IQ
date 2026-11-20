"""Read-only GLM-5.3 lexical identity and IQ235 compatibility contracts.

This module never downloads checkpoint weights, alters tensors, or promotes
the 128256-token compiled checkpoint to a GLM-tokenized trained checkpoint.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any, Mapping

GLM_REPO = "zai-org/GLM-5.3-BF16"
GLM_REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
GLM_PINNED_MAPPING_SHA256 = "e3ed8f39509e41ae3b307ed3b3d8e7237a2e9b9bd13ee4d06da1905f9768639e"  # Verified official token map in CI
GLM_VOCAB = 154880
GLM_PAD_ID = 154820
GLM_EOS_IDS = (154820, 154827, 154829)
GLM_WIDTH = 6144
IQ_WIDTH = 4096
SOURCE_IQ_VOCAB = 128256
PHYSICAL_LAYERS = {"mamba3": 79, "csa": 21, "hca": 57, "moe": 78}


class GLM53LexicalError(ValueError):
    """A tokenizer, model, or provenance boundary is incompatible."""


def inspect_lexical_boundary(
    source_iq: Mapping[str, Any], target_iq: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the *configuration* boundary; never imply trained weights."""
    source = json.loads(json.dumps(source_iq))
    target = json.loads(json.dumps(target_iq))
    if source.get("model", {}).get("vocab_size") != SOURCE_IQ_VOCAB:
        raise GLM53LexicalError("original IQ checkpoint is not 128256-token source")
    if target.get("model", {}).get("vocab_size") != GLM_VOCAB:
        raise GLM53LexicalError("target must use GLM-5.3's 154880-token ID space")
    expected = json.loads(json.dumps(source))
    expected["model"]["vocab_size"] = GLM_VOCAB
    if target != expected:
        raise GLM53LexicalError("GLM profile changed architecture beyond vocabulary")
    layers = target["schedule"]["layers"]
    if (len(layers) != 235 or dict(Counter(layers)) != PHYSICAL_LAYERS
            or target["model"]["hidden_size"] != IQ_WIDTH
            or target["mamba3"]["mimo_rank"] != 4
            or target["mamba3"]["num_layers"] != 79
            or target["moe"]["num_experts"] != 256
            or target["stable_moe"]["num_experts"] != 256):
        raise GLM53LexicalError("full235 structure or expert topology changed")
    return {
        "status": "architecture_compatible_only",
        "source_vocab": SOURCE_IQ_VOCAB,
        "glm_target_vocab": GLM_VOCAB,
        "physical_layers": len(layers),
        "target_embedding_shape": [GLM_VOCAB, IQ_WIDTH],
        "target_lm_head_shape": [GLM_VOCAB, IQ_WIDTH],
        "native_glm_lexical_shape": [GLM_VOCAB, GLM_WIDTH],
        "source_lexical_weights_reusable_as_is": False,
        "trained_glm_lexical_alignment_verified": False,
        "full_pretrained_inference_ready": False,
    }


def mapping_sha256(vocab: Mapping[str, int]) -> str:
    if not isinstance(vocab, dict) or not 0 < len(vocab) <= GLM_VOCAB:
        raise GLM53LexicalError("invalid GLM token-to-ID mapping size")
    if any(type(k) is not str or type(v) is not int for k, v in vocab.items()):
        raise GLM53LexicalError("token mapping must contain string tokens and integer IDs")
    values = list(vocab.values())
    if (len(set(values)) != len(values)
            or min(values) < 0 or max(values) >= GLM_VOCAB):
        raise GLM53LexicalError("GLM tokenizer IDs must be unique and in range")
    blob = json.dumps(vocab, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


def verify_glm53_tokenizer(
    tokenizer: Any,
    official_config: Mapping[str, Any],
    *,
    expected_mapping_sha256: str | None = None,
) -> dict[str, Any]:
    """Check the pinned official tokenizer and full ID mapping, with no weights."""
    if (official_config.get("model_type") != "glm_moe_dsa"
            or official_config.get("vocab_size") != GLM_VOCAB
            or official_config.get("hidden_size") != GLM_WIDTH
            or official_config.get("tie_word_embeddings") is not False
            or official_config.get("pad_token_id") != GLM_PAD_ID
            or official_config.get("eos_token_id") != list(GLM_EOS_IDS)):
        raise GLM53LexicalError("official GLM-5.3 source config mismatch")
    if not 0 < len(tokenizer) <= GLM_VOCAB:
        raise GLM53LexicalError("GLM tokenizer length outside configured ID space")
    if getattr(tokenizer, "pad_token_id", None) != GLM_PAD_ID:
        raise GLM53LexicalError("GLM tokenizer padding ID mismatch")
    if getattr(tokenizer, "eos_token_id", None) not in GLM_EOS_IDS:
        raise GLM53LexicalError("GLM tokenizer primary EOS ID mismatch")
    vocab = tokenizer.get_vocab()
    digest = mapping_sha256(vocab)
    if len(tokenizer) != len(vocab):
        raise GLM53LexicalError("reported tokenizer length and mapping disagree")
    for identifier in GLM_EOS_IDS:
        token = tokenizer.convert_ids_to_tokens(identifier)
        if not isinstance(token, str) or not token:
            raise GLM53LexicalError(f"missing GLM control ID {identifier}")
        if tokenizer.convert_tokens_to_ids(token) != identifier:
            raise GLM53LexicalError(f"GLM control ID failed roundtrip {identifier}")
    if expected_mapping_sha256 is not None:
        expected = expected_mapping_sha256.lower().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or expected != digest:
            raise GLM53LexicalError("independently pinned GLM token map digest mismatch")
    return {
        "status": "glm53_tokenizer_contract_verified",
        "repo": GLM_REPO,
        "revision": GLM_REVISION,
        "mapping_sha256": digest,
        "mapping_independently_pinned": expected_mapping_sha256 is not None,
        "vocab_size": GLM_VOCAB,
        "defined_token_ids": len(vocab),
        "reserved_unmapped_rows": GLM_VOCAB - len(vocab),
        "highest_defined_id": max(vocab.values()),
        "eos_ids": list(GLM_EOS_IDS),
        "pad_id": GLM_PAD_ID,
        "neural_weight_alignment_verified": False,
    }


def require_trained_glm53_lexical_weights(
    source_iq: Mapping[str, Any], target_iq: Mapping[str, Any]
) -> None:
    """Fail closed: config/tokenizer parity is not pretrained weight alignment.

    A future loader must validate separate provenance, source checkpoint and
    exact payload digests, lexical tensor shapes, and numerical alignment.
    """
    inspect_lexical_boundary(source_iq, target_iq)
    raise GLM53LexicalError(
        "IQ235's 128256-token pretrained lexical tensors are not aligned to "
        "GLM-5.3's 154880 IDs; a separately trained/validated 154880x4096 "
        "embedding and LM head overlay is required before pretrained inference"
    )
