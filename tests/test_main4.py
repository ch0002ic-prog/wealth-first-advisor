from __future__ import annotations

import unittest

from wealth_first.main4 import Main4Config, _evaluate_gate_checks, _resolve_validation_threshold


class Main4GateTests(unittest.TestCase):
    def test_evaluate_gate_checks_fails_when_robust_min_below_threshold(self) -> None:
        cfg = Main4Config(
            min_validation_threshold=0.001,
            min_robust_min_relative=0.0,
            min_active_fraction=0.5,
        )

        checks = _evaluate_gate_checks(
            cfg=cfg,
            mean_validation_relative=0.004,
            robust_min_test_relative=-0.001,
            active_fraction=1.0,
        )

        self.assertTrue(checks["validation_threshold"]["passed"])
        self.assertFalse(checks["robust_min_threshold"]["passed"])
        self.assertTrue(checks["active_fraction_threshold"]["passed"])
        self.assertFalse(checks["overall_passed"])

    def test_resolve_validation_threshold_uses_explicit_override(self) -> None:
        threshold, source = _resolve_validation_threshold(
            gate="020",
            min_validation_threshold=0.006,
            gate_scale="bps",
        )
        self.assertAlmostEqual(threshold, 0.006)
        self.assertEqual(source, "explicit")

    def test_resolve_validation_threshold_bps_scale(self) -> None:
        threshold, source = _resolve_validation_threshold(
            gate="009",
            min_validation_threshold=None,
            gate_scale="bps",
        )
        self.assertAlmostEqual(threshold, 0.0009)
        self.assertEqual(source, "gate_bps")

    def test_resolve_validation_threshold_legacy_scale(self) -> None:
        threshold, source = _resolve_validation_threshold(
            gate="009",
            min_validation_threshold=None,
            gate_scale="legacy",
        )
        self.assertAlmostEqual(threshold, 0.009)
        self.assertEqual(source, "gate_legacy")

    def test_evaluate_gate_checks_passes_when_all_thresholds_met(self) -> None:
        cfg = Main4Config(
            min_validation_threshold=0.001,
            min_robust_min_relative=-0.01,
            min_active_fraction=0.5,
        )

        checks = _evaluate_gate_checks(
            cfg=cfg,
            mean_validation_relative=0.002,
            robust_min_test_relative=-0.005,
            active_fraction=1.0,
        )

        self.assertTrue(checks["validation_threshold"]["passed"])
        self.assertTrue(checks["robust_min_threshold"]["passed"])
        self.assertTrue(checks["active_fraction_threshold"]["passed"])
        self.assertTrue(checks["overall_passed"])

    def test_evaluate_gate_checks_fails_when_validation_below_threshold(self) -> None:
        cfg = Main4Config(
            min_validation_threshold=0.009,
            min_robust_min_relative=-0.01,
            min_active_fraction=0.5,
        )

        checks = _evaluate_gate_checks(
            cfg=cfg,
            mean_validation_relative=0.004,
            robust_min_test_relative=0.001,
            active_fraction=1.0,
        )

        self.assertFalse(checks["validation_threshold"]["passed"])
        self.assertTrue(checks["robust_min_threshold"]["passed"])
        self.assertTrue(checks["active_fraction_threshold"]["passed"])
        self.assertFalse(checks["overall_passed"])


if __name__ == "__main__":
    unittest.main()
