import math
import unittest
from unittest import mock

import torch

from agd import agd_convex


def box_prox(value, coeff):
    del coeff
    return value.clamp(-1.0, 1.0)


def box_oracle(coefficients):
    return [-a.sign() for a in coefficients]


class CertifiedAGDTests(unittest.TestCase):
    def solve(self, phi, initial=None, lip=1.0, epsilon=1e-5, cap=None, **kwargs):
        if initial is None:
            initial = [torch.zeros(2, dtype=torch.float64)]
        return agd_convex(
            initial, phi, [box_prox] * len(initial), lip, epsilon,
            2.0 * math.sqrt(sum(v.numel() for v in initial)),
            max_iter=cap, linear_minimizer=box_oracle, **kwargs,
        )

    def test_constant_objective_normalizes_weights(self):
        for shift in (1.0, -3.0, 0.0):
            with self.subTest(shift=shift):
                _, stats = self.solve(lambda vs: vs[0].new_tensor(shift))
                self.assertEqual(stats['num_iters'], 1)
                self.assertAlmostEqual(stats['lower_bound'], shift)
                self.assertEqual(stats['certificate_gap'], 0.0)
                self.assertTrue(stats['terminated'])
                self.assertFalse(stats['capped'])

    def test_affine_zero_smoothness_uses_oracle(self):
        coefficients = torch.tensor([2.0, -3.0], dtype=torch.float64)
        result, stats = self.solve(lambda vs: coefficients @ vs[0] + 7.0, lip=0.0)
        torch.testing.assert_close(result[0], torch.tensor([-1.0, 1.0], dtype=torch.float64))
        self.assertEqual(stats['objective'], 2.0)
        self.assertEqual(stats['certificate_gap'], 0.0)
        self.assertEqual(stats['num_iters'], 1)
        self.assertEqual(stats['target_num_iters'], 0)

    def test_zero_cap_certifies_without_moving(self):
        seed = [torch.tensor([0.2, -0.1], dtype=torch.float64)]
        for lip in (0.0, 1.0):
            with self.subTest(lip=lip):
                result, stats = self.solve(lambda vs: vs[0].sum(), seed, lip=lip, cap=0)
                torch.testing.assert_close(result[0], seed[0])
                self.assertEqual(stats['num_iters'], 0)
                self.assertAlmostEqual(stats['certificate_gap'], 2.1)
                self.assertTrue(stats['capped'])
                self.assertFalse(stats['terminated'])
        _, stats = self.solve(lambda vs: vs[0].new_tensor(2.0), cap=0)
        self.assertTrue(stats['terminated'])
        self.assertFalse(stats['capped'])

    def test_quadratic_certificate_bounds_true_error_at_each_prefix(self):
        center = torch.tensor([0.3, 2.0], dtype=torch.float64)
        curvature = torch.tensor([0.2, 1.0], dtype=torch.float64)

        def phi(vs):
            return 0.5 * (curvature * (vs[0] - center).square()).sum()

        optimum = phi([center.clamp(-1, 1)]).item()
        for cap in (1, 2, 3, 10, 30):
            with self.subTest(cap=cap):
                result, stats = self.solve(phi, cap=cap)
                error = phi(result).item() - optimum
                self.assertGreaterEqual(stats['certificate_gap'] + 1e-12, error)
                self.assertLessEqual(stats['lower_bound'], optimum + 1e-12)
                self.assertEqual(stats['terminated'], stats['certificate_gap'] <= stats['epsilon'])
        result, stats = self.solve(phi, epsilon=1e-3)
        self.assertTrue(stats['terminated'])
        self.assertLessEqual(phi(result).item() - optimum, 1e-3)
        self.assertTrue(torch.all(result[0].abs() <= 1.0))
        self.assertFalse(result[0].requires_grad)

    def test_affine_model_is_invariant_to_objective_shift(self):
        records = []
        for shift in (0.0, 5.0, -5.0):
            result, stats = self.solve(
                lambda vs: 0.1 * vs[0].sum() + shift, lip=1.0, cap=3, epsilon=1e-12,
            )
            records.append((result, stats))
            self.assertTrue(stats['capped'])
            self.assertFalse(stats['terminated'])
            self.assertEqual(stats['stopping_reason'], 'max_iter')
            self.assertAlmostEqual(stats['lower_bound'], shift - 0.2)
        for result, stats in records[1:]:
            torch.testing.assert_close(result[0], records[0][0][0])
            self.assertAlmostEqual(stats['certificate_gap'], records[0][1]['certificate_gap'])

    def test_nominal_iteration_estimate_is_not_a_stop(self):
        with mock.patch('agd._iteration_bound', return_value=1):
            _, stats = self.solve(lambda vs: 0.1 * vs[0].sum(), epsilon=1e-6)
        self.assertTrue(stats['terminated'])
        self.assertGreater(stats['num_iters'], stats['target_num_iters'])

    def test_multiple_tensor_blocks_and_reporting_objective(self):
        initial = [torch.zeros(1, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]
        result, stats = self.solve(
            lambda vs: vs[0].sum() - vs[1].sum(), initial, lip=0.0,
            objective_func=lambda vs: vs[0].sum() + vs[1].sum(),
        )
        self.assertEqual(stats['objective'], -3.0)
        self.assertEqual(stats['final_objective'], 1.0)
        for value in result:
            self.assertFalse(value.requires_grad)

    def test_nonfinite_evaluations_and_bad_oracle_are_rejected(self):
        for value in (math.nan, math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Nonfinite'):
                self.solve(lambda vs: vs[0].sum() * value)
        with self.assertRaisesRegex(ValueError, 'Nonfinite AGD gradient'):
            self.solve(lambda vs: vs[0].sqrt().sum())
        with self.assertRaisesRegex(ValueError, 'Negative AGD certificate gap'):
            agd_convex(
                [torch.zeros(1, dtype=torch.float64)], lambda vs: vs[0].sum(),
                [box_prox], 1.0, 1e-3, 2.0,
                linear_minimizer=lambda coeffs: [torch.ones_like(a) for a in coeffs],
            )

    def test_small_negative_roundoff_is_clamped(self):
        from agd import _certificate_stats
        value = torch.zeros(1, dtype=torch.float64)
        stats = _certificate_stats(
            [value], lambda vs: vs[0].new_tensor(1.0), [value],
            value.new_tensor(1.0 + 1e-15), box_oracle,
        )
        self.assertEqual(stats['certificate_gap'], 0.0)

    def test_non_oracle_callers_keep_fixed_iteration_behavior(self):
        for cap in (None, 2, 0):
            with self.subTest(cap=cap):
                _, stats = agd_convex(
                    [torch.zeros(1, dtype=torch.float64)], lambda vs: 0.0 * vs[0].sum(),
                    [box_prox], 1.0, 0.01, 2.0, max_iter=cap,
                )
                target = math.ceil(2.0 * math.sqrt(200.0))
                self.assertEqual(stats['num_iters'], target if cap is None else cap)
                self.assertEqual(stats['terminated'], cap is None)
                self.assertEqual(stats['capped'], cap is not None)
                self.assertNotIn('certificate_gap', stats)


if __name__ == '__main__':
    unittest.main()
