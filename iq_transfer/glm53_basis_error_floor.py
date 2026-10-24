"""Read-only geometry and input-projection error budget for one GLM routed expert.

All values use identical donor-space output coordinates and are normalized
by ||uncompressed donor output||_F. They are NOT model-quality metrics.
"""
from __future__ import annotations

from math import sqrt

import torch

from .glm53_expert_fidelity import expert_function


class GLM53ErrorBudgetError(ValueError):
    pass


def _squared(tensor: torch.Tensor) -> float:
    return float(tensor.to(torch.float64).square().sum())


@torch.no_grad()
def output_error_budget(
    *,
    donor: dict[str, torch.Tensor],
    input_activations: torch.Tensor,
    reference: torch.Tensor,
    estimate: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, float]:
    """Separate unreachable output energy from loss inside the chosen basis.

    The output floor is an exact lower bound for any estimate constrained to
    the *given* output column-space. It is not the universal IQ error floor.
    The oracle recomputes the original donor FFN on x P P^T and projects
    its output with P P^T, using unrounded float32 donor weights.
    """
    if (
        reference.ndim != 2 or estimate.shape != reference.shape
        or input_activations.ndim != 2 or basis.ndim != 2
        or reference.shape[0] < 1
        or input_activations.shape != reference.shape
        or basis.shape[0] != reference.shape[1]
        or basis.shape[1] >= basis.shape[0]
    ):
        raise GLM53ErrorBudgetError("incompatible donor-space input/output geometry")
    if not all(bool(torch.isfinite(t).all()) for t in (
        reference, estimate, input_activations, basis,
    )):
        raise GLM53ErrorBudgetError("nonfinite error-budget tensor")
    for name in ("gate", "up", "down"):
        if name not in donor or not bool(torch.isfinite(donor[name]).all()):
            raise GLM53ErrorBudgetError(f"missing or nonfinite donor tensor: {name}")
    p = basis.float()
    x = input_activations.float()
    ref = reference.float()
    predicted = estimate.float()
    y_coordinates = ref @ p
    y_projection = y_coordinates @ p.T
    output_unreachable = ref - y_projection

    total_ref_squared = _squared(ref)
    if total_ref_squared <= 1e-20:
        raise GLM53ErrorBudgetError("zero-energy donor reference has no useful RMSE scale")
    total_error_squared = _squared(predicted - ref)
    floor_squared = _squared(output_unreachable)
    reachable_error_squared = _squared((predicted @ p) - y_coordinates)

    # Check the Pythagorean decomposition instead of assuming P is orthonormal.
    identity_gap = abs(
        total_error_squared - floor_squared - reachable_error_squared
    ) / total_ref_squared
    if identity_gap > 5e-4:
        raise GLM53ErrorBudgetError(
            "reference/prediction error does not decompose into fixed WARM basis"
        )

    input_projection = (x @ p) @ p.T
    input_squared = _squared(x)
    if input_squared <= 1e-20:
        raise GLM53ErrorBudgetError("zero-energy inputs cannot estimate projection loss")
    input_removed_fraction = _squared(x - input_projection) / input_squared

    # Recompute from original BF16 donor weights converted to float32, with
    # truncated input/output but WITHOUT quantizing projected IQ weights.
    oracle_donor_output = expert_function(input_projection, **donor)
    ideal_in_space = (oracle_donor_output @ p) @ p.T
    oracle_squared = _squared(ref - ideal_in_space)
    rounding_gap_squared = _squared(predicted - ideal_in_space)
    oracle_input_gap_squared = _squared(ideal_in_space - y_projection)

    return {
        "output_subspace_floor_relative_rmse": sqrt(floor_squared / total_ref_squared),
        "actual_relative_rmse": sqrt(total_error_squared / total_ref_squared),
        "within_subspace_relative_rmse": sqrt(
            reachable_error_squared / total_ref_squared
        ),
        "output_floor_share_of_actual_squared_error": (
            floor_squared / total_error_squared if total_error_squared > 1e-20 else 0.0
        ),
        "output_reachable_energy_fraction": max(
            0.0, 1.0 - floor_squared / total_ref_squared
        ),
        "actual_minus_output_floor_relative_rmse": (
            sqrt(total_error_squared / total_ref_squared)
            - sqrt(floor_squared / total_ref_squared)
        ),
        "input_activation_energy_removed_fraction": input_removed_fraction,
        "full_precision_input_projected_oracle_relative_rmse": sqrt(
            oracle_squared / total_ref_squared
        ),
        "input_projection_output_delta_relative_rmse": sqrt(
            oracle_input_gap_squared / total_ref_squared
        ),
        "projected_weights_vs_float32_oracle_relative_rmse": sqrt(
            rounding_gap_squared / total_ref_squared
        ),
        "squared_error_decomposition_abs_gap": identity_gap,
    }
