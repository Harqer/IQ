from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .transport import CoordinateMap, TransportError, transport_linear


class GLM53DirectTransformError(RuntimeError):
    pass


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise GLM53DirectTransformError("source tensor contains non-finite values")
    return result


@dataclass(frozen=True)
class SubcloningMap:
    coordinate_map: CoordinateMap
    source_indices: tuple[int, ...]


def fit_importance_subcloning_map(
    activations: Any,
    *,
    target_features: int,
) -> SubcloningMap:
    """Select the highest-energy donor residual channels as an exact sub-basis.

    The initial GLM->IQ skeleton needs a recipient basis before paired target
    activations exist. A one-hot subcloning map preserves selected donor
    coordinates exactly and is therefore safer for bootstrap than fitting
    against a random recipient basis.
    """
    x = _array(activations)
    if x.ndim != 2 or x.shape[0] == 0:
        raise GLM53DirectTransformError(
            "subcloning activations must be a non-empty rank-2 matrix"
        )
    if target_features <= 0 or target_features > x.shape[1]:
        raise GLM53DirectTransformError(
            "target_features must be in [1, donor_feature_count]"
        )
    energy = np.mean(x * x, axis=0)
    # Descending importance, with source index as deterministic tie-breaker.
    order = np.lexsort((np.arange(x.shape[1]), -energy))
    selected = order[:target_features]
    matrix = np.zeros((x.shape[1], target_features), dtype=np.float64)
    matrix[selected, np.arange(target_features)] = 1.0
    return SubcloningMap(
        coordinate_map=CoordinateMap(
            matrix=matrix,
            ridge=1e-12,
            source_space="glm53.residual",
            target_space="iq.bootstrap_residual",
            diagnostics=None,
        ),
        source_indices=tuple(int(x) for x in selected),
    )


def transport_embedding_and_lm_head(
    embedding_weight: Any,
    lm_head_weight: Any,
    residual_map: CoordinateMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lexical input/output weights into the IQ residual basis."""
    embedding = _array(embedding_weight)
    lm_head = _array(lm_head_weight)
    if embedding.ndim != 2 or lm_head.ndim != 2:
        raise GLM53DirectTransformError(
            "embedding and LM-head weights must be rank-2"
        )
    if embedding.shape != lm_head.shape:
        raise GLM53DirectTransformError(
            "GLM embedding and LM-head shapes must match for lexical transport"
        )
    if embedding.shape[1] != residual_map.matrix.shape[0]:
        raise GLM53DirectTransformError(
            "residual map source width does not match lexical weights"
        )
    # For x_t = x_s P, embedding rows use the same forward basis map. The
    # output head uses pinv(P)^T so logits approximate the donor function.
    target_embedding = embedding @ residual_map.matrix
    target_lm = lm_head @ np.linalg.pinv(residual_map.matrix).T
    if not np.isfinite(target_embedding).all() or not np.isfinite(target_lm).all():
        raise GLM53DirectTransformError(
            "lexical transport produced non-finite weights"
        )
    return target_embedding, target_lm


def fit_orthogonal_subspace(
    activations: Any,
    *,
    target_features: int,
) -> CoordinateMap:
    """Compress a donor feature space into its dominant orthogonal subspace.

    This is donor-only basis construction: it does not require a randomly
    initialized recipient feature basis. The returned matrix has orthonormal
    columns, so the same basis can be applied to compatible query/key dual
    coordinates.
    """
    x = _array(activations)
    if x.ndim != 2 or x.shape[0] < 2:
        raise GLM53DirectTransformError(
            "subspace activations must be rank-2 with at least two samples"
        )
    if target_features <= 0 or target_features > x.shape[1]:
        raise GLM53DirectTransformError(
            "target_features must be in [1, donor_feature_count]"
        )
    centered = x - x.mean(axis=0, keepdims=True)
    # Feature-space covariance is avoided for wide donor activations. The
    # right singular vectors are the exact principal feature directions.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < target_features:
        raise GLM53DirectTransformError(
            "calibration sample rank is too small for requested subspace"
        )
    matrix = vh[:target_features].T.copy()
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(target_features), atol=1e-8):
        raise GLM53DirectTransformError("PCA subspace is not orthonormal")
    return CoordinateMap(
        matrix=matrix,
        ridge=1e-12,
        source_space="glm53.compressed_kv",
        target_space="iq.compressed_candidate",
        diagnostics=None,
    )


def fit_mla_compressed_subspace(
    compressed_kv_activations: Any,
    *,
    kv_lora_rank: int,
    rope_dim: int,
    target_latent_dim: int,
) -> CoordinateMap:
    """Compress only MLA's non-positional latent and preserve RoPE coordinates.

    The returned basis is block diagonal:
      [ PCA(kv_latent)      0 ]
      [      0          I_rope ]

    This is important because arbitrary mixing of rotary and non-rotary
    channels would destroy the positional operator that the recipient applies.
    """
    x = _array(compressed_kv_activations)
    expected = kv_lora_rank + rope_dim
    if x.ndim != 2 or x.shape[1] != expected:
        raise GLM53DirectTransformError(
            f"compressed KV activations must have shape [samples, {expected}]"
        )
    if kv_lora_rank <= 0 or rope_dim < 0 or target_latent_dim <= 0:
        raise GLM53DirectTransformError("MLA subspace dimensions are invalid")
    if target_latent_dim > kv_lora_rank:
        raise GLM53DirectTransformError(
            "target_latent_dim cannot exceed donor kv_lora_rank"
        )
    latent = fit_orthogonal_subspace(
        x[:, :kv_lora_rank],
        target_features=target_latent_dim,
    ).matrix
    matrix = np.zeros(
        (expected, target_latent_dim + rope_dim),
        dtype=np.float64,
    )
    matrix[:kv_lora_rank, :target_latent_dim] = latent
    if rope_dim:
        matrix[
            kv_lora_rank:,
            target_latent_dim:,
        ] = np.eye(rope_dim, dtype=np.float64)
    return CoordinateMap(
        matrix=matrix,
        ridge=1e-12,
        source_space="glm53.compressed_kv",
        target_space="iq.compressed_candidate",
        diagnostics=None,
    )


@dataclass(frozen=True)
class GLM53MLALayout:
    source_hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    target_hidden_size: int
    target_head_dim: int
    output_groups: int
    output_rank: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def compressed_kv_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def target_attention_width(self) -> int:
        return self.num_heads * self.target_head_dim

    def __post_init__(self) -> None:
        values = (
            self.source_hidden_size,
            self.num_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.v_head_dim,
            self.target_hidden_size,
            self.target_head_dim,
            self.output_groups,
            self.output_rank,
        )
        if any(int(x) <= 0 for x in values):
            raise GLM53DirectTransformError("MLA dimensions must be positive")
        if self.qk_rope_head_dim < 0:
            raise GLM53DirectTransformError("qk_rope_head_dim cannot be negative")
        if self.target_attention_width % self.output_groups:
            raise GLM53DirectTransformError(
                "target attention width must be divisible by output groups"
            )
        if self.output_rank != self.target_attention_width // self.output_groups:
            raise GLM53DirectTransformError(
                "lossless grouped-output identity requires rank == input width per group"
            )


@dataclass(frozen=True)
class GLM53MLATransform:
    q_b_weight: np.ndarray
    kv_weight: np.ndarray
    output_group_weight: np.ndarray
    output_weight: np.ndarray


def grouped_output_identity(layout: GLM53MLALayout) -> np.ndarray:
    width = layout.target_attention_width // layout.output_groups
    identity = np.eye(width, dtype=np.float64)
    return np.stack([identity.copy() for _ in range(layout.output_groups)], axis=0)


def transform_glm53_mla(
    *,
    q_b_weight: Any,
    kv_a_weight: Any,
    kv_b_weight: Any,
    o_weight: Any,
    residual_input_map: CoordinateMap,
    residual_output_map: CoordinateMap,
    compressed_kv_map: CoordinateMap,
    layout: GLM53MLALayout,
) -> GLM53MLATransform:
    """Directly refactor GLM MLA weights into IQ compressed-context weights.

    GLM's compressed latent is [kv_lora, rotary_key]. For each attention head,
    its non-RoPE query is analytically pulled back through that head's K
    expansion so query and cached compressed KV live in dual coordinates:

        q_latent = [W_k^T W_q_nope, W_q_rope]

    A shared orthogonal compression then maps both sides into IQ's candidate
    width. V expansion followed by GLM O is folded into IQ's output projection.
    No model training, logit imitation, or Transformer->Mamba recurrence mapping
    occurs here.
    """
    q_b = _array(q_b_weight)
    kv_a = _array(kv_a_weight)
    kv_b = _array(kv_b_weight)
    o = _array(o_weight)
    expected = {
        "q_b": (layout.num_heads * layout.qk_head_dim, layout.q_lora_rank),
        "kv_a": (layout.compressed_kv_dim, layout.source_hidden_size),
        "kv_b": (
            layout.num_heads * (layout.qk_nope_head_dim + layout.v_head_dim),
            layout.kv_lora_rank,
        ),
        "o": (layout.source_hidden_size, layout.num_heads * layout.v_head_dim),
    }
    actual = {
        "q_b": q_b.shape,
        "kv_a": kv_a.shape,
        "kv_b": kv_b.shape,
        "o": o.shape,
    }
    mismatches = [
        f"{name}: got {actual[name]}, expected {shape}"
        for name, shape in expected.items()
        if actual[name] != shape
    ]
    if mismatches:
        raise GLM53DirectTransformError(
            "GLM MLA source shape mismatch: " + "; ".join(mismatches)
        )
    projection = np.asarray(compressed_kv_map.matrix, dtype=np.float64)
    if projection.shape != (
        layout.compressed_kv_dim,
        layout.target_head_dim,
    ):
        raise GLM53DirectTransformError(
            "compressed_kv_map shape does not match MLA layout"
        )
    if not np.allclose(
        projection.T @ projection,
        np.eye(layout.target_head_dim),
        atol=1e-6,
    ):
        raise GLM53DirectTransformError(
            "compressed_kv_map must have orthonormal columns"
        )
    if residual_input_map.matrix.shape != (
        layout.source_hidden_size,
        layout.target_hidden_size,
    ):
        raise GLM53DirectTransformError("residual_input_map shape mismatch")
    if residual_output_map.matrix.shape != (
        layout.source_hidden_size,
        layout.target_hidden_size,
    ):
        raise GLM53DirectTransformError("residual_output_map shape mismatch")

    q_heads: list[np.ndarray] = []
    output_heads: list[np.ndarray] = []
    kv_stride = layout.qk_nope_head_dim + layout.v_head_dim
    for head in range(layout.num_heads):
        q_start = head * layout.qk_head_dim
        q_nope = q_b[
            q_start : q_start + layout.qk_nope_head_dim
        ]
        q_rope = q_b[
            q_start + layout.qk_nope_head_dim :
            q_start + layout.qk_head_dim
        ]
        kv_start = head * kv_stride
        k_expand = kv_b[
            kv_start : kv_start + layout.qk_nope_head_dim
        ]
        v_expand = kv_b[
            kv_start + layout.qk_nope_head_dim :
            kv_start + kv_stride
        ]

        effective_query = np.concatenate(
            (k_expand.T @ q_nope, q_rope),
            axis=0,
        )
        q_heads.append(projection.T @ effective_query)

        o_start = head * layout.v_head_dim
        o_head = o[:, o_start : o_start + layout.v_head_dim]
        effective_value_output = np.zeros(
            (layout.source_hidden_size, layout.compressed_kv_dim),
            dtype=np.float64,
        )
        effective_value_output[:, : layout.kv_lora_rank] = (
            o_head @ v_expand
        )
        # z_target = z_source @ projection. Decode only the represented
        # subspace before applying the donor V+O composite.
        decoded_output = (
            effective_value_output @ np.linalg.pinv(projection).T
        )
        output_heads.append(decoded_output)

    target_q_b = np.concatenate(q_heads, axis=0)
    latent_map = CoordinateMap(
        matrix=projection,
        ridge=compressed_kv_map.ridge,
        source_space=compressed_kv_map.source_space,
        target_space=compressed_kv_map.target_space,
    )
    try:
        target_kv = transport_linear(
            kv_a,
            residual_input_map,
            latent_map,
        )
    except TransportError as exc:
        raise GLM53DirectTransformError(
            f"failed to transport compressed KV projection: {exc}"
        ) from exc

    source_output = np.concatenate(output_heads, axis=1)
    target_output = residual_output_map.matrix.T @ source_output
    expected_q = (
        layout.target_attention_width,
        layout.q_lora_rank,
    )
    expected_kv = (layout.target_head_dim, layout.target_hidden_size)
    expected_o = (
        layout.target_hidden_size,
        layout.target_attention_width,
    )
    if target_q_b.shape != expected_q:
        raise GLM53DirectTransformError(
            f"target q_b shape {target_q_b.shape} != {expected_q}"
        )
    if target_kv.shape != expected_kv:
        raise GLM53DirectTransformError(
            f"target kv shape {target_kv.shape} != {expected_kv}"
        )
    if target_output.shape != expected_o:
        raise GLM53DirectTransformError(
            f"target output shape {target_output.shape} != {expected_o}"
        )
    return GLM53MLATransform(
        q_b_weight=target_q_b,
        kv_weight=target_kv,
        output_group_weight=grouped_output_identity(layout),
        output_weight=target_output,
    )
