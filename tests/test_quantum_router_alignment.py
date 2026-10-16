from __future__ import annotations
import unittest
import numpy as np
from iq_transfer.quantum_router_alignment import (
    RouterProblem, AlignmentError, choose_disjoint_pairs, apply_swaps,
    exact_neighborhood, fit_qubo, qaoa_probabilities, cvar_expectation,
    qasm3_p1, optimize_qaoa,
)


class QuantumRouterAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test-only known-answer geometry with full 256 expert identities.
        # This is NOT an experiment using pretrained GLM or IQ checkpoints.
        rng = np.random.default_rng(19)
        mat = rng.normal(size=(256,256)) * 0.05
        a = rng.normal(size=(256,256))
        b = rng.normal(size=(256,256))
        a = (a+a.T)/2; b=(b+b.T)/2
        np.fill_diagonal(a, 0); np.fill_diagonal(b, 0)
        cls.problem = RouterProblem(mat,a,b,0.1)
        cls.base = cls.problem.linear_assignment()
        cls.pairs = choose_disjoint_pairs(23,8)
        cls.table = exact_neighborhood(cls.problem,cls.base,cls.pairs)

    def test_global_integrity_256_experts(self):
        for state in (0,1,45,255):
            p = apply_swaps(self.base, self.pairs, state)
            np.testing.assert_array_equal(np.sort(p), np.arange(256))
            self.assertEqual(p.shape, (256,))

    def test_qubo_is_exact_for_global_objective(self):
        constant,linear,pair = fit_qubo(self.table)
        for state in range(len(self.table)):
            x = np.array([(state>>i)&1 for i in range(8)])
            energy = constant + np.dot(linear,x) + 0.5 * x @ pair @ x
            self.assertAlmostEqual(energy, self.table[state], places=6)

    def test_ideal_qaoa_distribution_is_normalized(self):
        p=qaoa_probabilities(self.table,0.02,0.32)
        self.assertAlmostEqual(p.sum(),1,places=12)
        self.assertTrue(np.all(p>=0))
        self.assertTrue(np.isfinite(p).all())
        self.assertLessEqual(cvar_expectation(self.table,p,0.25),np.dot(self.table,p)+1e-8)

    def test_can_run_full_width_noiseless_quantum_benchmark(self):
        outcome=optimize_qaoa(self.table,seed=11,alpha=0.25,maxiter=2,shots=128)
        self.assertGreaterEqual(outcome['best_state'],0)
        self.assertLess(outcome['best_state'],256)
        self.assertGreaterEqual(outcome['best_energy'],outcome['exact_local_energy']-1e-7)

    def test_qasm_output(self):
        table=(self.table-self.table.min())/np.ptp(self.table)
        qasm=qasm3_p1(table,gamma=0.5,beta=0.3)
        self.assertIn('qubit[8] q;',qasm)
        self.assertIn('OPENQASM 3.0;',qasm)
        self.assertIn('c = measure q;',qasm)

    def test_invalid_geometry_fails(self):
        with self.assertRaisesRegex(AlignmentError,'256x256'):
            RouterProblem(np.ones((32,32)),np.eye(256),np.eye(256),0.1)
        with self.assertRaisesRegex(AlignmentError,'bijection'):
            self.problem.score(np.zeros(256,dtype=int))


if __name__ == '__main__':
    unittest.main()