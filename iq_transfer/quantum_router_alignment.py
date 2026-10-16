"""Experimental full-width GLM-5.3 -> IQ expert identity alignment.

Classical full 256-expert alignment plus a constrained QAOA neighborhood.
NOT a quantum-advantage claim and NOT an automatic checkpoint transplant.

The local move variables exchange destination IDs between disjoint donor
expert pairs. All 256 expert identities remain represented at all times.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment


N_EXPERTS = 256  # GLM-5.3, IQ nvidia_parallel_78_config(), all routed experts


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class RouterProblem:
    linear_cost: np.ndarray
    donor_affinity: np.ndarray
    iq_affinity: np.ndarray
    interaction_weight: float

    def __post_init__(self) -> None:
        for name in ("linear_cost", "donor_affinity", "iq_affinity"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (N_EXPERTS, N_EXPERTS) or not np.isfinite(value).all():
                raise AlignmentError(f"{name} must be finite {N_EXPERTS}x{N_EXPERTS}")
            if name != "linear_cost" and not np.allclose(value, value.T, rtol=0, atol=1e-10):
                raise AlignmentError(f"{name} must be symmetric")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.interaction_weight) or self.interaction_weight < 0:
            raise AlignmentError("interaction_weight must be finite and nonnegative")

    def score(self, permutation: Sequence[int] | np.ndarray) -> float:
        p = np.asarray(permutation, dtype=np.int64)
        if p.shape != (N_EXPERTS,) or not np.array_equal(np.sort(p), np.arange(N_EXPERTS)):
            raise AlignmentError("mapping must be a bijection of all 256 routed experts")
        total = float(self.linear_cost[np.arange(N_EXPERTS), p].sum())
        if self.interaction_weight:
            residual = self.donor_affinity - self.iq_affinity[np.ix_(p, p)]
            # Off-diagonal Frobenius loss, each undirected pair counted once.
            total += self.interaction_weight * float(np.square(residual).sum() - np.square(np.diag(residual)).sum()) / 2.0
        return total

    def linear_assignment(self) -> np.ndarray:
        row, col = linear_sum_assignment(self.linear_cost)
        p = np.empty(N_EXPERTS, dtype=np.int64)
        p[row] = col
        return p


def angular_matrices(donor: np.ndarray, recipient: np.ndarray, coordinate_map: np.ndarray) -> RouterProblem:
    """Use complete router matrices and the existing learned WARM coordinate map.

    Cost matches transferred donor router directions to recipient router rows.
    Donor/recipient affinity measure router-weight cosine structure; they are
    NOT activation co-selection frequencies. Both require subsequent validation.
    """
    d = np.asarray(donor, dtype=np.float64)
    r = np.asarray(recipient, dtype=np.float64)
    w = np.asarray(coordinate_map, dtype=np.float64)
    if d.shape != (N_EXPERTS, 6144) or r.shape != (N_EXPERTS, 4096) or w.shape != (6144, 4096):
        raise AlignmentError("require donor 256x6144, IQ 256x4096, WARM 6144x4096; no dimension reduction")
    if not all(np.isfinite(x).all() for x in (d, r, w)):
        raise AlignmentError("router or coordinate map contains non-finite values")
    transported = d @ w
    norms_d = np.linalg.norm(transported, axis=1)
    norms_r = np.linalg.norm(r, axis=1)
    if np.any(norms_d <= 1e-12) or np.any(norms_r <= 1e-12):
        raise AlignmentError("cannot compute identity costs for zero-norm router rows")
    nd = transported / norms_d[:, None]
    nr = r / norms_r[:, None]
    cross = np.clip(nd @ nr.T, -1., 1.)
    a = np.clip(nd @ nd.T, -1., 1.)
    b = np.clip(nr @ nr.T, -1., 1.)
    np.fill_diagonal(a, 0.)
    np.fill_diagonal(b, 0.)
    return RouterProblem(1. - cross, a, b, 0.)


def choose_disjoint_pairs(seed: int, count: int = 8) -> tuple[tuple[int, int], ...]:
    if not 1 <= count <= N_EXPERTS // 2:
        raise AlignmentError("pair count outside [1,128]")
    rng = np.random.default_rng(seed)
    sampled = rng.permutation(N_EXPERTS)[:2 * count]
    return tuple((int(sampled[2*i]), int(sampled[2*i+1])) for i in range(count))


def apply_swaps(permutation: np.ndarray, pairs: Sequence[tuple[int, int]], state: int) -> np.ndarray:
    result = np.asarray(permutation, dtype=np.int64).copy()
    used: set[int] = set()
    for i, (a,b) in enumerate(pairs):
        if a == b or min(a,b) < 0 or max(a,b) >= N_EXPERTS or a in used or b in used:
            raise AlignmentError("swap pairs must be disjoint valid donor indices")
        used.update((a,b))
        if state & (1 << i):
            result[a], result[b] = result[b], result[a]
    return result


def exact_neighborhood(problem: RouterProblem, base: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    if len(pairs) > 20:
        raise AlignmentError("statevector neighborhood supports at most 20 swap variables")
    # This enumerates all states of the *local move circuit*, not a smaller model.
    return np.array([problem.score(apply_swaps(base, pairs, state)) for state in range(1 << len(pairs))])


def fit_qubo(energies: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    e = np.asarray(energies, dtype=np.float64)
    n = int(np.log2(e.size)) if e.size else -1
    if e.ndim != 1 or e.size != 1 << n or n < 1:
        raise AlignmentError("energy table must cover all 2**q swap states")
    constant = float(e[0]); linear = np.zeros(n); quadratic = np.zeros((n,n))
    for i in range(n):
        linear[i] = e[1 << i] - constant
        for j in range(i):
            quadratic[i,j] = quadratic[j,i] = e[(1<<i)|(1<<j)] - constant - linear[i] - (e[1<<j] - constant)
    indices = np.arange(e.size)
    reconstructed = np.full(e.size, constant)
    for i in range(n):
        bit_i = ((indices >> i) & 1).astype(float)
        reconstructed += linear[i] * bit_i
        for j in range(i):
            reconstructed += quadratic[i,j] * bit_i * ((indices >> j) & 1)
    if not np.allclose(e, reconstructed, atol=1e-6, rtol=1e-10):
        raise AlignmentError("local permutation objective is not representable by a quadratic Ising Hamiltonian")
    return constant, linear, quadratic


def qaoa_probabilities(energies: np.ndarray, gamma: float, beta: float) -> np.ndarray:
    """Exact noiseless p=1 QAOA statevector (not measured QPU runtime)."""
    n = (len(energies) - 1).bit_length()
    if len(energies) != (1<<n):
        raise AlignmentError("QAOA requires a full power-of-two energy table")
    e = np.asarray(energies, dtype=np.float64)
    psi = np.exp(-1j * gamma * e) / np.sqrt(len(e))
    indexes = np.arange(len(e))
    for bit in range(n):
        psi = np.cos(beta) * psi - 1j * np.sin(beta) * psi[indexes ^ (1 << bit)]
    return np.square(np.abs(psi)).real


def cvar_expectation(energies: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    if not (0 < alpha <= 1):
        raise AlignmentError("CVaR alpha must be in (0,1]")
    order = np.argsort(energies, kind="stable")
    e, p = energies[order], probabilities[order]
    weights = np.minimum(p, np.maximum(alpha - np.cumsum(p) + p, 0))
    return float(np.dot(weights, e) / alpha)


def optimize_qaoa(energies: np.ndarray, *, seed: int, alpha: float = 1., maxiter: int = 24, shots: int = 1024) -> dict:
    t0 = perf_counter()
    table = np.asarray(energies, dtype=np.float64)
    spread = float(np.ptp(table))
    if spread <= 1e-14:
        raise AlignmentError("flat quantum objective cannot improve mapping")
    scaled = (table - table.min()) / spread
    def objective(theta: np.ndarray) -> float:
        return cvar_expectation(scaled, qaoa_probabilities(scaled, float(theta[0]), float(theta[1])), alpha)
    fit = differential_evolution(objective, [(0., 2*np.pi), (0., np.pi)], seed=seed, maxiter=maxiter, popsize=8, polish=True, tol=1e-6)
    distribution = qaoa_probabilities(scaled, float(fit.x[0]), float(fit.x[1]))
    if shots <= 0:
        raise AlignmentError("shots must be positive")
    draws = np.random.default_rng(seed + 1).choice(len(table), size=shots, p=distribution)
    best_state = int(draws[np.argmin(table[draws])])
    return {
        "alpha": float(alpha), "gamma": float(fit.x[0]), "beta": float(fit.x[1]),
        "best_state": best_state, "best_energy": float(table[best_state]),
        "exact_local_energy": float(table.min()), "exact_local_state": int(np.argmin(table)),
        "best_probability": float(distribution[np.argmin(table)]),
        "expectation": float(np.dot(table, distribution)),
        "simulator_s": perf_counter() - t0, "shots": shots, "method": "noiseless_statevector_p1"
    }


def qasm3_p1(energies: np.ndarray, *, gamma: float, beta: float) -> str:
    """Gate-based diagonal Ising encoding, tested against exact QAOA amplitudes."""
    _, l, q = fit_qubo(energies)
    n = len(l)
    fields = -(l / 2. + q.sum(axis=1) / 4.)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;", f"bit[{n}] c;"]
    lines.extend(f"h q[{i}];" for i in range(n))
    for i in range(n):
        if abs(fields[i]) > 1e-13:
            lines.append(f"rz({2.*gamma*fields[i]:.16g}) q[{i}];")
    for i in range(n):
        for j in range(i):
            if abs(q[i,j]) > 1e-13:
                lines.extend([f"cx q[{i}], q[{j}];", f"rz({gamma*q[i,j]/2.:.16g}) q[{j}];", f"cx q[{i}], q[{j}];"])
    lines.extend(f"rx({2.*beta:.16g}) q[{i}];" for i in range(n))
    lines.append("c = measure q;")
    return "\n".join(lines) + "\n"