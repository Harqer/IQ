"""Run an opt-in 256-to-256 GLM-5.3/IQ router permutation experiment.

No checkpoints are modified. Input is real BF16 router weights or a previously
prepared full 256x256 cost bundle. All 256 experts remain in the assignment.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

from iq_transfer.quantum_router_alignment import (
    N_EXPERTS, RouterProblem, angular_matrices, apply_swaps,
    choose_disjoint_pairs, exact_neighborhood, optimize_qaoa, qasm3_p1,
)


def sha_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path, help='full 256x256 matrices in .npz; not a synthetic model')
    source.add_argument('--glm-checkpoint', type=Path, help='real local BF16 GLM safetensors root')
    parser.add_argument('--iq-checkpoint', type=Path)
    parser.add_argument('--glm-router-key')
    parser.add_argument('--iq-router-key')
    parser.add_argument('--warm-map', type=Path, help='directory of saved WARM CoordinateMap')
    parser.add_argument('--export-bundle', type=Path, help='save actual geometry arrays for reproducible reuse')
    parser.add_argument('--interaction-weight', type=float, default=0., help='weight of router-affinity matching (0 is linear assignment)')
    parser.add_argument('--swap-pairs', type=int, default=8, help='number of binary permutation moves, not model expert count')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--maxiter', type=int, default=24)
    parser.add_argument('--cvar-alpha', type=float, default=0.25)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qasm-out', type=Path)
    return parser.parse_args()


def run():
    a = parse_args()
    started = perf_counter()
    if a.bundle:
        with np.load(a.bundle, allow_pickle=False) as data:
            arrays = [np.asarray(data[key], dtype=np.float64) for key in ('linear_cost', 'donor_affinity', 'iq_affinity')]
        source_identity = {'input_bundle': str(a.bundle), 'sha256': sha_file(a.bundle)}
    else:
        if not all((a.iq_checkpoint, a.glm_router_key, a.iq_router_key, a.warm_map)):
            raise ValueError('real-checkpoint mode requires --iq-checkpoint, --glm-router-key, --iq-router-key and --warm-map')
        from iq_transfer.checkpoint import SafetensorsSource
        from iq_transfer.transport import load_coordinate_map
        donor = SafetensorsSource(a.glm_checkpoint).get(a.glm_router_key)
        student = SafetensorsSource(a.iq_checkpoint).get(a.iq_router_key)
        # SafetensorsSource loads raw tensors, never pretend raw FP8 is BF16.
        import torch
        if donor.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise ValueError(f'donor router must be explicitly dequantized or BF16: {donor.dtype}')
        if student.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise ValueError(f'recipient router must be a floating-point tensor: {student.dtype}')
        warm = load_coordinate_map(a.warm_map).matrix
        geometry = angular_matrices(donor.float().numpy(), student.float().numpy(), warm)
        arrays = [geometry.linear_cost, geometry.donor_affinity, geometry.iq_affinity]
        source_identity = {
            'glm_checkpoint': str(a.glm_checkpoint), 'glm_router_key': a.glm_router_key,
            'iq_checkpoint': str(a.iq_checkpoint), 'iq_router_key': a.iq_router_key,
            'warm_map': str(a.warm_map), 'source': 'real_tensor_geometry',
        }
        if a.export_bundle:
            a.export_bundle.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.export_bundle, linear_cost=arrays[0], donor_affinity=arrays[1], iq_affinity=arrays[2])
            source_identity['export_bundle_sha256'] = sha_file(a.export_bundle)
    problem = RouterProblem(*arrays, a.interaction_weight)
    t_classic = perf_counter()
    baseline = problem.linear_assignment()
    base_cost = problem.score(baseline)
    classical_s = perf_counter() - t_classic
    pairs = choose_disjoint_pairs(a.seed, a.swap_pairs)
    t_encode = perf_counter()
    energies = exact_neighborhood(problem, baseline, pairs)
    encode_s = perf_counter() - t_encode
    if np.ptp(energies) <= 1e-14:
        raise ValueError('flat local objective: no QAOA optimization experiment is meaningful')
    results = {}
    for label, alpha in [('qaoa',1.),('cvar_qaoa',a.cvar_alpha)]:
        result = optimize_qaoa(energies, seed=a.seed, alpha=alpha, maxiter=a.maxiter, shots=a.shots)
        chosen = apply_swaps(baseline, pairs, result['best_state'])
        result['full_assignment_integrity'] = bool(np.array_equal(np.sort(chosen), np.arange(N_EXPERTS)))
        result['improvement_vs_hungarian'] = base_cost - result['best_energy']
        result['preserve_baseline_if_not_better'] = result['best_energy'] >= base_cost
        results[label] = result
    if a.qasm_out:
        best = results['cvar_qaoa']
        scaled = (energies - energies.min())/np.ptp(energies)
        a.qasm_out.parent.mkdir(parents=True, exist_ok=True)
        a.qasm_out.write_text(qasm3_p1(scaled, gamma=best['gamma'], beta=best['beta']), encoding='utf-8')
    output = {
        'provenance': source_identity, 'donor_experts': N_EXPERTS, 'recipient_experts': N_EXPERTS,
        'quantum_move_bits': len(pairs), 'local_move_pairs': [list(p) for p in pairs],
        'interaction_weight': a.interaction_weight, 'seed': a.seed,
        'baseline': {'method':'scipy_hungarian_linear_assignment', 'objective':base_cost,'seconds':classical_s},
        'exact_local': {'method':'enumeration', 'objective':float(energies.min()), 'state':int(np.argmin(energies))},
        'encoding_seconds': encode_s, 'quantum': results,
        'total_seconds':perf_counter() - started,
        'quantum_hardware_executed':False,
        'qualification':'These are NOISELESS simulator results; no quantum speedup or model-quality improvement is claimed.',
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    run()