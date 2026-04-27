from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from wealth_first.medium_capacity import MediumCapacityConfig, MediumCapacityParams, simulate_medium_capacity_policy


class MediumCapacityDiagnosticsTests(unittest.TestCase):
    def test_no_trade_band_is_applied_in_target_space_not_smoothed_space(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        features = pd.DataFrame(
            {
                "ret_21": [0.0, 0.03],
                "ret_63": [0.0, 0.0],
                "dd_63": [0.0, 0.0],
                "vol_21": [0.0, 0.0],
            },
            index=dates,
        )
        spy_returns = pd.Series([0.0, 0.0], index=dates, dtype=float)
        cfg = MediumCapacityConfig(
            min_spy_weight=0.8,
            max_spy_weight=1.05,
            initial_spy_weight=1.0,
            action_smoothing=0.5,
            no_trade_band=0.02,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
        params = MediumCapacityParams(
            signal_weights=np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
            signal_scale=1.0,
            feature_mu=np.zeros(4, dtype=float),
            feature_std=np.ones(4, dtype=float),
        )

        _, _, _, _, diagnostics = simulate_medium_capacity_policy(
            features=features,
            spy_returns=spy_returns,
            start_index=0,
            end_index=1,
            cfg=cfg,
            params=params,
        )

        # raw_target_delta = 0.03 > no_trade_band, so this step must execute
        # even though the smoothed step is only 0.015.
        self.assertEqual(diagnostics["gate_suppressed_step_count"], 0)
        self.assertEqual(diagnostics["executed_step_count"], 1)

    def test_execution_gate_tolerance_suppresses_near_boundary_flip(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        features = pd.DataFrame(
            {
                "ret_21": [0.0, 0.02000000001],
                "ret_63": [0.0, 0.0],
                "dd_63": [0.0, 0.0],
                "vol_21": [0.0, 0.0],
            },
            index=dates,
        )
        spy_returns = pd.Series([0.0, 0.0], index=dates, dtype=float)
        params = MediumCapacityParams(
            signal_weights=np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
            signal_scale=1.0,
            feature_mu=np.zeros(4, dtype=float),
            feature_std=np.ones(4, dtype=float),
        )

        cfg_no_tol = MediumCapacityConfig(
            min_spy_weight=0.8,
            max_spy_weight=1.05,
            initial_spy_weight=1.0,
            action_smoothing=1.0,
            no_trade_band=0.02,
            execution_gate_tolerance=0.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
        _, _, _, _, diag_no_tol = simulate_medium_capacity_policy(
            features=features,
            spy_returns=spy_returns,
            start_index=0,
            end_index=1,
            cfg=cfg_no_tol,
            params=params,
        )

        cfg_tol = MediumCapacityConfig(
            min_spy_weight=0.8,
            max_spy_weight=1.05,
            initial_spy_weight=1.0,
            action_smoothing=1.0,
            no_trade_band=0.02,
            execution_gate_tolerance=1e-9,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
        _, _, _, _, diag_tol = simulate_medium_capacity_policy(
            features=features,
            spy_returns=spy_returns,
            start_index=0,
            end_index=1,
            cfg=cfg_tol,
            params=params,
        )

        self.assertEqual(diag_no_tol["executed_step_count"], 1)
        self.assertEqual(diag_no_tol["gate_suppressed_step_count"], 0)
        self.assertEqual(diag_tol["executed_step_count"], 0)
        self.assertEqual(diag_tol["gate_suppressed_step_count"], 1)

    def test_simulate_medium_capacity_policy_reports_gate_suppression_diagnostics(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        features = pd.DataFrame(
            {
                "ret_21": [0.0, 0.01, 0.015, -0.05],
                "ret_63": [0.0, 0.0, 0.0, 0.0],
                "dd_63": [0.0, 0.0, 0.0, 0.0],
                "vol_21": [0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        spy_returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates, dtype=float)
        cfg = MediumCapacityConfig(
            min_spy_weight=0.8,
            max_spy_weight=1.05,
            initial_spy_weight=1.0,
            action_smoothing=0.5,
            no_trade_band=0.02,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
        params = MediumCapacityParams(
            signal_weights=np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
            signal_scale=1.0,
            feature_mu=np.zeros(4, dtype=float),
            feature_std=np.ones(4, dtype=float),
        )

        _, _, _, _, diagnostics = simulate_medium_capacity_policy(
            features=features,
            spy_returns=spy_returns,
            start_index=0,
            end_index=3,
            cfg=cfg,
            params=params,
        )

        self.assertEqual(diagnostics["proposed_steps_over_band"], 1)
        self.assertEqual(diagnostics["executed_step_count"], 1)
        self.assertEqual(diagnostics["gate_suppressed_step_count"], 2)
        self.assertAlmostEqual(diagnostics["gate_suppression_rate"], 2.0 / 3.0)
        self.assertGreater(diagnostics["signal_abs_p95"], 0.0)


if __name__ == "__main__":
    unittest.main()