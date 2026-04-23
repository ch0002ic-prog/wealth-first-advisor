from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from wealth_first import rl as rl_module
from wealth_first.optimizer import WealthFirstConfig


class WealthFirstEnvTests(unittest.TestCase):
    def test_env_smoke_or_dependency_error_is_explicit(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.005, 0.012, -0.003, 0.011, 0.004],
            },
            index=dates,
        )
        benchmark = pd.Series([0.008, -0.004, 0.009, -0.002, 0.007, 0.003], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(returns, lookback=2)
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(
                include_cash=True,
                max_weight_overrides={"SPY": 0.4},
                min_weight_overrides={"CASH": 0.2},
                transaction_cost_bps=5.0,
                slippage_bps=5.0,
                benchmark_loss_penalty=4.0,
            ),
        )

        observation, info = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertAlmostEqual(sum(info["current_weights"].values()), 1.0)

        next_observation, reward, terminated, truncated, step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertEqual(next_observation.shape, env.observation_space.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(truncated)
        self.assertLessEqual(step_info["target_weights"]["SPY"], 0.4 + 1e-8)
        self.assertGreaterEqual(step_info["target_weights"]["CASH"], 0.2 - 1e-8)
        self.assertIn("reward_core_component", step_info)
        self.assertIn("benchmark_reward_component", step_info)
        self.assertIn("execution_cost_reward_drag", step_info)
        self.assertIn("friction_reward_drag", step_info)
        self.assertIsInstance(terminated, bool)

    def test_env_benchmark_relative_observations_require_benchmark_and_append_relative_window(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015, 0.0, 0.01, -0.005],
            },
            index=dates,
        )
        benchmark = pd.Series([0.005, -0.01, 0.006, -0.002, 0.004, -0.001], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    benchmark_relative_observations=True,
                )
            return

        with self.assertRaisesRegex(ValueError, "benchmark_relative_observations requires benchmark returns"):
            rl_module.WealthFirstEnv(
                returns,
                lookback=2,
                config=WealthFirstConfig(include_cash=True),
                benchmark_relative_observations=True,
            )

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            benchmark_relative_observations=True,
            config=WealthFirstConfig(include_cash=True),
        )

        observation, info = env.reset(seed=7)

        self.assertTrue(info["benchmark_relative_observations"])
        self.assertEqual(observation.shape, (14,))
        relative_slice = observation[6:10]
        expected_relative = np.array([0.005, -0.005, -0.01, 0.01], dtype=np.float32)
        np.testing.assert_allclose(relative_slice, expected_relative)

    def test_env_benchmark_regime_observations_require_benchmark_and_append_summary_features(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015, 0.0, 0.01, -0.005],
            },
            index=dates,
        )
        benchmark = pd.Series([0.005, -0.01, 0.006, -0.002, 0.004, -0.001], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    benchmark_regime_observations=True,
                )
            return

        with self.assertRaisesRegex(ValueError, "benchmark_regime_observations requires benchmark returns"):
            rl_module.WealthFirstEnv(
                returns,
                lookback=2,
                config=WealthFirstConfig(include_cash=True),
                benchmark_regime_observations=True,
            )

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            benchmark_regime_observations=True,
            config=WealthFirstConfig(include_cash=True),
        )

        observation, info = env.reset(seed=7)

        self.assertTrue(info["benchmark_regime_observations"])
        self.assertEqual(observation.shape, (15,))
        regime_slice = observation[6:11]
        expected_regime = np.array([-0.00505, 0.0075, -0.01, -0.00515, 0.00505], dtype=np.float32)
        np.testing.assert_allclose(regime_slice, expected_regime, atol=1e-6)

    def test_env_benchmark_regime_summary_only_observations_append_only_summary_features(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015, 0.0, 0.01, -0.005],
            },
            index=dates,
        )
        benchmark = pd.Series([0.005, -0.01, 0.006, -0.002, 0.004, -0.001], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            benchmark_regime_summary_observations=True,
            config=WealthFirstConfig(include_cash=True),
        )

        observation, info = env.reset(seed=7)

        self.assertTrue(info["benchmark_regime_observations"])
        self.assertTrue(info["benchmark_regime_summary_observations"])
        self.assertFalse(info["benchmark_regime_relative_cumulative_observations"])
        self.assertEqual(observation.shape, (13,))
        summary_slice = observation[6:9]
        expected_summary = np.array([-0.00505, 0.0075, -0.01], dtype=np.float32)
        np.testing.assert_allclose(summary_slice, expected_summary, atol=1e-6)

    def test_env_benchmark_regime_relative_only_observations_append_only_relative_cumulative_features(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015, 0.0, 0.01, -0.005],
            },
            index=dates,
        )
        benchmark = pd.Series([0.005, -0.01, 0.006, -0.002, 0.004, -0.001], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            benchmark_regime_relative_cumulative_observations=True,
            config=WealthFirstConfig(include_cash=True),
        )

        observation, info = env.reset(seed=7)

        self.assertTrue(info["benchmark_regime_observations"])
        self.assertFalse(info["benchmark_regime_summary_observations"])
        self.assertTrue(info["benchmark_regime_relative_cumulative_observations"])
        self.assertEqual(observation.shape, (12,))
        relative_slice = observation[6:8]
        expected_relative = np.array([-0.00515, 0.00505], dtype=np.float32)
        np.testing.assert_allclose(relative_slice, expected_relative, atol=1e-6)

    def test_env_state_trend_preservation_guard_caps_cash_raising_trend_reduction(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    state_trend_preservation_symbol="SPY",
                    state_trend_preservation_cash_max_threshold=0.55,
                    state_trend_preservation_symbol_min_weight=0.45,
                    state_trend_preservation_max_symbol_reduction=0.02,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            state_trend_preservation_symbol="SPY",
            state_trend_preservation_cash_max_threshold=0.55,
            state_trend_preservation_symbol_min_weight=0.45,
            state_trend_preservation_max_symbol_reduction=0.02,
        )

        _, info = env.reset(seed=7)
        self.assertEqual(info["state_trend_preservation_symbol"], "SPY")

        _, _, _, _, step_info = env.step(np.array([0.0, 1.0], dtype=np.float32))

        self.assertTrue(step_info["state_trend_preservation_window_active"])
        self.assertTrue(step_info["state_trend_preservation_condition_met"])
        self.assertTrue(step_info["state_trend_preservation_guard_applied"])
        self.assertAlmostEqual(step_info["target_weights"]["SPY"], 0.48, places=6)
        self.assertAlmostEqual(step_info["target_weights"]["CASH"], 0.52, places=6)
        self.assertAlmostEqual(step_info["turnover"], 0.02, places=6)

    def test_env_state_trend_preservation_guard_stays_inactive_when_cash_is_already_high(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            state_trend_preservation_symbol="SPY",
            state_trend_preservation_cash_max_threshold=0.45,
            state_trend_preservation_symbol_min_weight=0.45,
            state_trend_preservation_max_symbol_reduction=0.02,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.0, 1.0], dtype=np.float32))

        self.assertFalse(step_info["state_trend_preservation_window_active"])
        self.assertFalse(step_info["state_trend_preservation_condition_met"])
        self.assertFalse(step_info["state_trend_preservation_guard_applied"])
        self.assertAlmostEqual(step_info["target_weights"]["SPY"], 0.0)
        self.assertAlmostEqual(step_info["target_weights"]["CASH"], 1.0)

    def test_env_action_smoothing_and_no_trade_band_reduce_small_rebalances(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.005, 0.012, -0.003, 0.011, 0.004],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    action_smoothing=0.5,
                    no_trade_band=0.3,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            action_smoothing=0.5,
            no_trade_band=0.3,
        )

        _, info = env.reset(seed=7)
        self.assertAlmostEqual(info["current_weights"]["SPY"], 0.5)
        _, _, _, _, step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertAlmostEqual(step_info["proposed_weights"]["SPY"], 0.75)
        self.assertAlmostEqual(step_info["proposed_turnover"], 0.25)
        self.assertTrue(step_info["trade_suppressed"])
        self.assertAlmostEqual(step_info["turnover"], 0.0)
        self.assertAlmostEqual(step_info["target_weights"]["SPY"], 0.5)

    def test_env_max_executed_rebalances_suppresses_later_trades(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    max_executed_rebalances=1,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            max_executed_rebalances=1,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.5)
        self.assertEqual(first_step_info["executed_rebalances"], 1)
        self.assertFalse(first_step_info["rebalance_budget_exhausted"])
        self.assertTrue(second_step_info["trade_suppressed"])
        self.assertTrue(second_step_info["rebalance_budget_exhausted"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.0)
        self.assertEqual(second_step_info["executed_rebalances"], 1)
        self.assertEqual(second_step_info["max_executed_rebalances"], 1)

    def test_env_rebalance_cooldown_suppresses_immediate_follow_up_trade(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    rebalance_cooldown_steps=1,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            rebalance_cooldown_steps=1,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        _, _, _, _, third_step_info = env.step(np.array([0.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.5)
        self.assertEqual(first_step_info["rebalance_cooldown_remaining"], 1)
        self.assertFalse(first_step_info["rebalance_cooldown_active"])
        self.assertTrue(second_step_info["trade_suppressed"])
        self.assertTrue(second_step_info["rebalance_cooldown_active"])
        self.assertTrue(second_step_info["rebalance_cooldown_blocked"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.0)
        self.assertEqual(second_step_info["rebalance_cooldown_remaining"], 0)
        self.assertFalse(third_step_info["trade_suppressed"])
        self.assertFalse(third_step_info["rebalance_cooldown_active"])
        self.assertAlmostEqual(third_step_info["turnover"], 1.0)
        self.assertEqual(third_step_info["rebalance_cooldown_remaining"], 1)

    def test_env_early_rebalance_risk_penalty_applies_only_within_configured_window(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    early_rebalance_risk_penalty=0.05,
                    early_rebalance_risk_penalty_before=1,
                    early_rebalance_risk_penalty_cash_max_threshold=0.6,
                    early_rebalance_risk_penalty_symbol="SPY",
                    early_rebalance_risk_penalty_symbol_min_weight=0.4,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_penalty=0.05,
            early_rebalance_risk_penalty_before=1,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, first_reward, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, second_reward, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.25)
        self.assertTrue(first_step_info["early_rebalance_risk_window_active"])
        self.assertTrue(first_step_info["early_rebalance_risk_condition_met"])
        self.assertTrue(first_step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(first_step_info["early_rebalance_risk_penalty_component"], -0.05)
        self.assertAlmostEqual(first_reward, -0.05)
        self.assertEqual(first_step_info["early_rebalance_risk_penalty_before"], 1)

        self.assertAlmostEqual(second_step_info["turnover"], 0.25)
        self.assertFalse(second_step_info["early_rebalance_risk_window_active"])
        self.assertFalse(second_step_info["early_rebalance_risk_condition_met"])
        self.assertFalse(second_step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_penalty_component"], 0.0)
        self.assertAlmostEqual(second_reward, 0.0)

    def test_env_early_rebalance_risk_turnover_cap_limits_executed_trade(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    early_rebalance_risk_turnover_cap=0.15,
                    early_rebalance_risk_penalty_before=1,
                    early_rebalance_risk_penalty_cash_max_threshold=0.6,
                    early_rebalance_risk_penalty_symbol="SPY",
                    early_rebalance_risk_penalty_symbol_min_weight=0.4,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_penalty_before=1,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertTrue(first_step_info["early_rebalance_risk_window_active"])
        self.assertTrue(first_step_info["early_rebalance_risk_condition_met"])
        self.assertTrue(first_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(first_step_info["turnover"], 0.15)
        self.assertAlmostEqual(first_step_info["target_weights"]["SPY"], 0.65)
        self.assertAlmostEqual(first_step_info["early_rebalance_risk_turnover_cap"], 0.15)

        self.assertFalse(second_step_info["early_rebalance_risk_window_active"])
        self.assertFalse(second_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.35)

    def test_env_early_rebalance_risk_turnover_cap_max_applications_releases_later_trade(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_max_applications=1,
            early_rebalance_risk_penalty_before=3,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertTrue(first_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertEqual(first_step_info["early_rebalance_risk_turnover_cap_applications"], 1)
        self.assertTrue(first_step_info["early_rebalance_risk_turnover_cap_max_applications_reached"])
        self.assertEqual(first_step_info["early_rebalance_risk_turnover_cap_max_applications"], 1)
        self.assertAlmostEqual(first_step_info["turnover"], 0.15)

        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(second_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertEqual(second_step_info["early_rebalance_risk_turnover_cap_applications"], 1)
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_max_applications_reached"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.35)

    def test_env_early_rebalance_risk_turnover_cap_secondary_cap_relaxes_later_applications(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_secondary_cap=0.25,
            early_rebalance_risk_turnover_cap_secondary_after_applications=1,
            early_rebalance_risk_penalty_before=3,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertTrue(first_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertFalse(first_step_info["early_rebalance_risk_turnover_cap_secondary_active"])
        self.assertAlmostEqual(first_step_info["early_rebalance_risk_turnover_cap_effective_cap"], 0.15)
        self.assertEqual(first_step_info["early_rebalance_risk_turnover_cap_applications"], 1)
        self.assertAlmostEqual(first_step_info["turnover"], 0.15)

        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_secondary_active"])
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_turnover_cap_effective_cap"], 0.25)
        self.assertEqual(second_step_info["early_rebalance_risk_turnover_cap_applications"], 2)
        self.assertAlmostEqual(second_step_info["target_weights"]["SPY"], 0.9)
        self.assertAlmostEqual(second_step_info["turnover"], 0.25)

    def test_env_early_rebalance_risk_turnover_cap_secondary_cap_can_require_catchup_state(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_secondary_cap=0.25,
            early_rebalance_risk_turnover_cap_secondary_after_applications=1,
            early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold=0.02,
            early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio=0.9995,
            early_rebalance_risk_penalty_before=3,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertTrue(first_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertFalse(first_step_info["early_rebalance_risk_turnover_cap_secondary_after_applications_reached"])
        self.assertFalse(first_step_info["early_rebalance_risk_turnover_cap_secondary_active"])
        self.assertAlmostEqual(first_step_info["early_rebalance_risk_turnover_cap_effective_cap"], 0.15)

        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_secondary_after_applications_reached"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_secondary_state_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_secondary_active"])
        self.assertAlmostEqual(second_step_info["benchmark_regime_cumulative_return"], 0.02)
        self.assertLess(second_step_info["pre_trade_relative_wealth_ratio"], 0.9995)
        self.assertAlmostEqual(
            second_step_info[
                "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold"
            ],
            0.02,
        )
        self.assertAlmostEqual(
            second_step_info[
                "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio"
            ],
            0.9995,
        )
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_turnover_cap_effective_cap"], 0.25)
        self.assertAlmostEqual(second_step_info["turnover"], 0.25)

    def test_env_reset_clears_early_rebalance_risk_turnover_cap_applications(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_secondary_cap=0.25,
            early_rebalance_risk_turnover_cap_secondary_after_applications=1,
            early_rebalance_risk_penalty_before=3,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertEqual(first_step_info["early_rebalance_risk_turnover_cap_applications"], 1)
        self.assertEqual(second_step_info["early_rebalance_risk_turnover_cap_applications"], 2)
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_secondary_after_applications_reached"])

        env.reset(seed=7)
        _, _, _, _, reset_first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertEqual(reset_first_step_info["early_rebalance_risk_turnover_cap_applications"], 1)
        self.assertFalse(
            reset_first_step_info["early_rebalance_risk_turnover_cap_secondary_after_applications_reached"]
        )
        self.assertFalse(reset_first_step_info["early_rebalance_risk_turnover_cap_secondary_active"])

    def test_env_early_rebalance_risk_penalty_after_skips_first_rebalance(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_penalty=0.05,
            early_rebalance_risk_penalty_after=1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, first_reward, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, second_reward, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, third_reward, _, _, third_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertFalse(first_step_info["early_rebalance_risk_window_active"])
        self.assertFalse(first_step_info["early_rebalance_risk_condition_met"])
        self.assertFalse(first_step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(first_reward, 0.0)
        self.assertEqual(first_step_info["early_rebalance_risk_penalty_after"], 1)

        self.assertTrue(second_step_info["early_rebalance_risk_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(second_reward, -0.05)

        self.assertFalse(third_step_info["early_rebalance_risk_window_active"])
        self.assertFalse(third_step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(third_reward, 0.0)

    def test_env_early_rebalance_risk_symbol_max_weight_targets_low_trend_states(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        low_trend_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=np.array([0.55, 0.45], dtype=np.float32),
            early_rebalance_risk_penalty=0.05,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_max_weight=0.56,
        )

        low_trend_env.reset(seed=7)
        _, low_trend_reward, _, _, low_trend_info = low_trend_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(low_trend_info["early_rebalance_risk_window_active"])
        self.assertTrue(low_trend_info["early_rebalance_risk_condition_met"])
        self.assertTrue(low_trend_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(low_trend_reward, -0.05)
        self.assertAlmostEqual(low_trend_info["early_rebalance_risk_penalty_symbol_max_weight"], 0.56)

        high_trend_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=np.array([0.62, 0.38], dtype=np.float32),
            early_rebalance_risk_penalty=0.05,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_max_weight=0.56,
        )

        high_trend_env.reset(seed=7)
        _, high_trend_reward, _, _, high_trend_info = high_trend_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(high_trend_info["early_rebalance_risk_window_active"])
        self.assertFalse(high_trend_info["early_rebalance_risk_condition_met"])
        self.assertFalse(high_trend_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(high_trend_reward, 0.0)

    def test_env_early_rebalance_risk_benchmark_filters_require_benchmark_returns(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        with self.assertRaisesRegex(ValueError, "benchmark returns"):
            rl_module.WealthFirstEnv(
                returns,
                lookback=2,
                config=WealthFirstConfig(include_cash=True),
                early_rebalance_risk_penalty=0.05,
                early_rebalance_risk_penalty_before=2,
                early_rebalance_risk_penalty_cash_max_threshold=0.6,
                early_rebalance_risk_penalty_symbol="SPY",
                early_rebalance_risk_penalty_symbol_min_weight=0.4,
                early_rebalance_risk_penalty_benchmark_drawdown_min_threshold=-0.01,
            )

        with self.assertRaisesRegex(ValueError, "benchmark returns"):
            rl_module.WealthFirstEnv(
                returns,
                lookback=2,
                config=WealthFirstConfig(include_cash=True),
                early_rebalance_risk_penalty=0.05,
                early_rebalance_risk_penalty_before=2,
                early_rebalance_risk_penalty_cash_max_threshold=0.6,
                early_rebalance_risk_penalty_symbol="SPY",
                early_rebalance_risk_penalty_symbol_min_weight=0.4,
                early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio=1.0,
            )

    def test_env_early_rebalance_risk_benchmark_drawdown_threshold_blocks_penalty_not_turnover_cap(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_penalty=0.05,
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_penalty_benchmark_drawdown_min_threshold=-0.01,
        )

        env.reset(seed=7)
        _, reward, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertFalse(step_info["early_rebalance_risk_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_drawdown"], -0.02)
        self.assertAlmostEqual(step_info["early_rebalance_risk_penalty_benchmark_drawdown_min_threshold"], -0.01)
        self.assertAlmostEqual(step_info["turnover"], 0.15)
        self.assertAlmostEqual(reward, 0.0)

    def test_env_early_rebalance_risk_turnover_cap_benchmark_drawdown_threshold_blocks_cap_when_unmet(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold=-0.01,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_drawdown"], -0.02)
        self.assertAlmostEqual(step_info["early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold"], -0.01)
        self.assertAlmostEqual(step_info["turnover"], 0.25)

    def test_env_early_rebalance_risk_turnover_cap_benchmark_drawdown_threshold_allows_cap_when_met(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold=-0.03,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold"], -0.03)
        self.assertAlmostEqual(step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_benchmark_cumulative_return_threshold_blocks_cap_when_unmet(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold=0.01,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_cumulative_return"], 0.02)
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.01,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.25)

    def test_env_early_rebalance_risk_turnover_cap_benchmark_cumulative_return_threshold_allows_cap_when_met(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold=0.03,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.03,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_penalty_benchmark_drawdown_max_threshold_blocks_penalty_when_too_shallow(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_penalty=0.1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_penalty_benchmark_drawdown_max_threshold=-0.03,
        )

        env.reset(seed=7)
        _, reward, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_drawdown"], -0.02)
        self.assertAlmostEqual(step_info["early_rebalance_risk_penalty_benchmark_drawdown_max_threshold"], -0.03)
        self.assertAlmostEqual(step_info["turnover"], 0.25)
        self.assertGreater(reward, -0.1)

    def test_env_early_rebalance_risk_penalty_benchmark_drawdown_max_threshold_applies_penalty_when_met(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.04, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_penalty=0.1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_penalty_benchmark_drawdown_max_threshold=-0.03,
        )

        env.reset(seed=7)
        _, reward, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_drawdown"], -0.04)
        self.assertAlmostEqual(step_info["early_rebalance_risk_penalty_benchmark_drawdown_max_threshold"], -0.03)
        self.assertAlmostEqual(step_info["turnover"], 0.25)
        self.assertLess(reward, -0.099)

    def test_env_early_rebalance_risk_deep_drawdown_turnover_cap_blocks_when_drawdown_too_shallow(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_deep_drawdown_turnover_cap=0.15,
            early_rebalance_risk_deep_drawdown_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold=-0.03,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_deep_drawdown_turnover_cap_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_deep_drawdown_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_deep_drawdown_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_drawdown"], -0.02)
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold"],
            -0.03,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.25)

    def test_env_early_rebalance_risk_deep_drawdown_turnover_cap_applies_when_met(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.04, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_deep_drawdown_turnover_cap=0.15,
            early_rebalance_risk_deep_drawdown_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold=-0.03,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_deep_drawdown_turnover_cap_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_deep_drawdown_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_deep_drawdown_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_drawdown"], -0.04)
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold"],
            -0.03,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_shallow_drawdown_turnover_cap_stacks_on_primary_cap(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_shallow_drawdown_turnover_cap=0.05,
            early_rebalance_risk_shallow_drawdown_turnover_cap_before=2,
            early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold=0.6,
            early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold=-0.03,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold"], 0.6)
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold"],
            -0.03,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.05)

    def test_env_early_rebalance_risk_shallow_drawdown_turnover_cap_uses_own_cash_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, -0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_shallow_drawdown_turnover_cap=0.05,
            early_rebalance_risk_shallow_drawdown_turnover_cap_before=2,
            early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold=0.4,
            early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold=-0.03,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_shallow_drawdown_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_mean_reversion_turnover_cap_stacks_on_primary_cap(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "MEAN_REVERSION", "CASH"]),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_mean_reversion_turnover_cap=0.05,
            early_rebalance_risk_mean_reversion_turnover_cap_before=2,
            early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold=0.18,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.65, 0.24, 0.11], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_applied"])
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold"],
            0.18,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.05)

    def test_env_early_rebalance_risk_mean_reversion_turnover_cap_respects_delta_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "MEAN_REVERSION", "CASH"]),
            early_rebalance_risk_turnover_cap=0.30,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_mean_reversion_turnover_cap=0.05,
            early_rebalance_risk_mean_reversion_turnover_cap_before=2,
            early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold=0.20,
            early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold=-0.02,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.05, 0.20], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_mean_reversion_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_mean_reversion_turnover_cap_applied"])
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold"],
            0.20,
        )
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold"],
            -0.02,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.30)

    def test_env_early_rebalance_risk_mean_reversion_turnover_cap_uses_own_benchmark_cumulative_return_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "MEAN_REVERSION", "CASH"]),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_mean_reversion_turnover_cap=0.05,
            early_rebalance_risk_mean_reversion_turnover_cap_before=2,
            early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold=0.18,
            early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold=0.01,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.65, 0.24, 0.11], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_mean_reversion_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_mean_reversion_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_cumulative_return"], 0.02)
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.01,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_mean_reversion_turnover_cap_applies_when_own_benchmark_cumulative_return_threshold_is_met(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "MEAN_REVERSION", "CASH"]),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_mean_reversion_turnover_cap=0.05,
            early_rebalance_risk_mean_reversion_turnover_cap_before=2,
            early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold=0.18,
            early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold=0.03,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.65, 0.24, 0.11], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_applied"])
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.03,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.05)

    def test_env_early_rebalance_risk_mean_reversion_action_smoothing_applies_without_overlay_cap(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "MEAN_REVERSION", "CASH"]),
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_mean_reversion_action_smoothing=0.25,
            early_rebalance_risk_mean_reversion_turnover_cap_before=2,
            early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold=0.18,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.65, 0.24, 0.11], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_mean_reversion_action_smoothing_applied"])
        self.assertFalse(step_info["early_rebalance_risk_mean_reversion_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["early_rebalance_risk_mean_reversion_action_smoothing"], 0.25)
        self.assertAlmostEqual(step_info["turnover"], 0.3075, places=6)
        self.assertAlmostEqual(step_info["target_weights"]["SPY"], 0.55, places=6)
        self.assertAlmostEqual(step_info["target_weights"]["MEAN_REVERSION"], 0.2425, places=6)
        self.assertAlmostEqual(step_info["target_weights"]["CASH"], 0.2075, places=6)

    def test_env_early_rebalance_risk_trend_turnover_cap_stacks_on_primary_cap(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "TREND_FOLLOWING", "CASH"]),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_trend_turnover_cap=0.05,
            early_rebalance_risk_trend_turnover_cap_before=2,
            early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold=0.18,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.65, 0.24, 0.11], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_applied"])
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold"],
            0.18,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.05)

    def test_env_early_rebalance_risk_trend_turnover_cap_respects_delta_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "TREND_FOLLOWING", "CASH"]),
            early_rebalance_risk_turnover_cap=0.30,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_trend_turnover_cap=0.05,
            early_rebalance_risk_trend_turnover_cap_before=2,
            early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold=0.20,
            early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold=0.0,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.05, 0.20], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_trend_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_trend_turnover_cap_applied"])
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold"],
            0.20,
        )
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold"],
            0.0,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.30)

    def test_env_early_rebalance_risk_trend_turnover_cap_uses_own_benchmark_cumulative_return_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "TREND_FOLLOWING", "CASH"]),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_trend_turnover_cap=0.05,
            early_rebalance_risk_trend_turnover_cap_before=2,
            early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold=0.18,
            early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold=0.01,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.65, 0.24, 0.11], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_window_active"])
        self.assertFalse(step_info["early_rebalance_risk_trend_turnover_cap_condition_met"])
        self.assertFalse(step_info["early_rebalance_risk_trend_turnover_cap_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_cumulative_return"], 0.02)
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.01,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_trend_turnover_cap_applies_when_own_benchmark_cumulative_return_threshold_is_met(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=pd.Series([0.25, 0.25, 0.50], index=["SPY", "TREND_FOLLOWING", "CASH"]),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.2,
            early_rebalance_risk_trend_turnover_cap=0.05,
            early_rebalance_risk_trend_turnover_cap_before=2,
            early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold=0.18,
            early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold=0.03,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.65, 0.24, 0.11], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_trend_turnover_cap_applied"])
        self.assertAlmostEqual(
            step_info["early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.03,
        )
        self.assertAlmostEqual(step_info["turnover"], 0.05)

    def test_env_early_rebalance_risk_turnover_cap_can_use_separate_window(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_penalty=0.1,
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_after=1,
            early_rebalance_risk_turnover_cap_before=2,
            early_rebalance_risk_penalty_before=1,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))
        self.assertTrue(first_step_info["early_rebalance_risk_window_active"])
        self.assertFalse(first_step_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(first_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(first_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(first_step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(first_step_info["turnover"], 0.25)

        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertFalse(second_step_info["early_rebalance_risk_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertFalse(second_step_info["early_rebalance_risk_penalty_applied"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_relative_wealth_ratio_threshold_blocks_penalty_not_turnover_cap(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_penalty_after=1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio=1.0,
        )

        env.reset(seed=7)
        env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertTrue(second_step_info["early_rebalance_risk_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(second_step_info["early_rebalance_risk_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertLess(second_step_info["pre_trade_relative_wealth_ratio"], 1.0)
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio"], 1.0)
        self.assertAlmostEqual(second_step_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_relative_wealth_ratio_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio=1.0,
            early_rebalance_risk_penalty_after=1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertTrue(second_step_info["early_rebalance_risk_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(second_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(second_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertLess(second_step_info["pre_trade_relative_wealth_ratio"], 1.0)
        self.assertAlmostEqual(
            second_step_info["early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio"],
            1.0,
        )
        self.assertAlmostEqual(second_step_info["turnover"], 0.25)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_relative_wealth_ratio_max_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio=1.0,
            early_rebalance_risk_penalty_after=1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        env.step(np.array([0.75, 0.25], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertTrue(second_step_info["early_rebalance_risk_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(second_step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(second_step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertGreater(second_step_info["pre_trade_relative_wealth_ratio"], 1.0)
        self.assertAlmostEqual(
            second_step_info["early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio"],
            1.0,
        )
        self.assertAlmostEqual(second_step_info["turnover"], 0.25)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_cash_weight_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight=0.6,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight"], 0.6)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight=0.4,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight"], 0.4)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_target_cash_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_cash_min_threshold=0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_target_cash_min_threshold"], 0.3)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_cash_min_threshold=0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_target_cash_min_threshold"], 0.2)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_target_cash_max_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_cash_max_threshold=0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_target_cash_max_threshold"], 0.2)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_cash_max_threshold=0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_target_cash_max_threshold"], 0.3)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_target_trend_max_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_trend_max_threshold=0.7,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_target_trend_max_threshold"], 0.7)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_trend_max_threshold=0.8,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_target_trend_max_threshold"], 0.8)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_target_mean_reversion_max_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        initial_weights = pd.Series(
            [0.5, 0.0, 0.5],
            index=["TREND_FOLLOWING", "MEAN_REVERSION", "CASH"],
            dtype=float,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=initial_weights,
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold=0.1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="TREND_FOLLOWING",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.65, 0.15, 0.20], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(
            blocked_info["early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold"],
            0.1,
        )
        self.assertGreater(blocked_info["turnover"], 0.15)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=initial_weights,
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold=0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="TREND_FOLLOWING",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.65, 0.15, 0.20], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(
            allowed_info["early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold"],
            0.2,
        )
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_delta_cash_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_cash_min_threshold=-0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_delta_cash_min_threshold"], -0.2)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_cash_min_threshold=-0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_delta_cash_min_threshold"], -0.3)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_delta_cash_max_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_cash_max_threshold=-0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_delta_cash_max_threshold"], -0.3)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_cash_max_threshold=-0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_delta_cash_max_threshold"], -0.2)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_delta_trend_min_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_trend_min_threshold=0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_delta_trend_min_threshold"], 0.3)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_trend_min_threshold=0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_delta_trend_min_threshold"], 0.2)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_allow_nonincreasing_risk_symbol(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_trend_min_threshold=-0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.25, 0.75], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_trend_min_threshold=-0.3,
            early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol=True,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.25, 0.75], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_delta_trend_min_threshold"], -0.3)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_delta_mean_reversion_min_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        initial_weights = pd.Series(
            [0.5, 0.0, 0.5],
            index=["TREND_FOLLOWING", "MEAN_REVERSION", "CASH"],
            dtype=float,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=initial_weights,
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold=0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="TREND_FOLLOWING",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.65, 0.15, 0.20], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(
            blocked_info["early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold"],
            0.2,
        )
        self.assertGreater(blocked_info["turnover"], 0.15)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            initial_weights=initial_weights,
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold=0.1,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="TREND_FOLLOWING",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.65, 0.15, 0.20], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(
            allowed_info["early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold"],
            0.1,
        )
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_proposed_turnover_min_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold=0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold"], 0.3)
        self.assertAlmostEqual(blocked_info["proposed_turnover"], 0.25)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold=0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold"], 0.2)
        self.assertAlmostEqual(allowed_info["proposed_turnover"], 0.25)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_direct_proposed_turnover_max_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold=0.2,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(blocked_info["early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold"], 0.2)
        self.assertAlmostEqual(blocked_info["proposed_turnover"], 0.25)
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold=0.3,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=1.0,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertAlmostEqual(allowed_info["early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold"], 0.3)
        self.assertAlmostEqual(allowed_info["proposed_turnover"], 0.25)
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_reuse_penalty_state_filters(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_use_penalty_state_filters=True,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_penalty_symbol_max_weight=0.45,
        )

        env.reset(seed=7)
        _, _, _, _, blocked_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_use_penalty_state_filters"])
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_use_penalty_state_filters=True,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
            early_rebalance_risk_penalty_symbol_max_weight=0.55,
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_use_penalty_state_filters"])
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_turnover_cap_can_use_cash_only_penalty_state_filter(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        blocked_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_use_penalty_state_filters=True,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.4,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        blocked_env.reset(seed=7)
        _, _, _, _, blocked_info = blocked_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(blocked_info["early_rebalance_risk_window_active"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertFalse(blocked_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(blocked_info["early_rebalance_risk_turnover_cap_use_penalty_state_filters"])
        self.assertAlmostEqual(blocked_info["turnover"], 0.25)

        allowing_env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.15,
            early_rebalance_risk_turnover_cap_use_penalty_state_filters=True,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
        )

        allowing_env.reset(seed=7)
        _, _, _, _, allowed_info = allowing_env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(allowed_info["early_rebalance_risk_turnover_cap_use_penalty_state_filters"])
        self.assertAlmostEqual(allowed_info["turnover"], 0.15)

    def test_env_early_rebalance_risk_zero_turnover_cap_blocks_trade(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_turnover_cap=0.0,
            early_rebalance_risk_penalty_before=2,
            early_rebalance_risk_penalty_cash_max_threshold=0.6,
            early_rebalance_risk_penalty_symbol="SPY",
            early_rebalance_risk_penalty_symbol_min_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_rebalance_risk_window_active"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_condition_met"])
        self.assertTrue(step_info["early_rebalance_risk_turnover_cap_applied"])
        self.assertTrue(step_info["trade_suppressed"])
        self.assertFalse(step_info["early_rebalance_risk_condition_met"])
        self.assertAlmostEqual(step_info["turnover"], 0.0)

    def test_env_repeat_early_risk_turnover_cap_uses_previous_executed_trade_shape(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_repeat_turnover_cap=0.05,
            early_rebalance_risk_repeat_turnover_cap_after=1,
            early_rebalance_risk_repeat_turnover_cap_before=3,
            early_rebalance_risk_repeat_symbol="SPY",
            early_rebalance_risk_repeat_previous_cash_reduction_min=0.15,
            early_rebalance_risk_repeat_previous_symbol_increase_min=0.15,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.7, 0.3], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.9, 0.1], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.2, places=6)
        self.assertFalse(first_step_info["early_rebalance_risk_repeat_turnover_cap_window_active"])
        self.assertAlmostEqual(first_step_info["early_rebalance_risk_repeat_cash_reduction"], 0.2, places=6)
        self.assertAlmostEqual(first_step_info["early_rebalance_risk_repeat_symbol_increase"], 0.2, places=6)

        self.assertTrue(second_step_info["early_rebalance_risk_repeat_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_repeat_turnover_cap_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_repeat_turnover_cap_applied"])
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_repeat_previous_cash_reduction"], 0.2, places=6)
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_repeat_previous_symbol_increase"], 0.2, places=6)
        self.assertAlmostEqual(second_step_info["turnover"], 0.05, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["SPY"], 0.75, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["CASH"], 0.25, places=6)

    def test_env_repeat_early_risk_action_smoothing_uses_previous_executed_trade_shape(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_repeat_action_smoothing=0.5,
            early_rebalance_risk_repeat_turnover_cap_after=1,
            early_rebalance_risk_repeat_turnover_cap_before=3,
            early_rebalance_risk_repeat_symbol="SPY",
            early_rebalance_risk_repeat_previous_cash_reduction_min=0.15,
            early_rebalance_risk_repeat_previous_symbol_increase_min=0.15,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.7, 0.3], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.9, 0.1], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.2, places=6)
        self.assertFalse(first_step_info["early_rebalance_risk_repeat_turnover_cap_window_active"])

        self.assertTrue(second_step_info["early_rebalance_risk_repeat_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_repeat_turnover_cap_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_repeat_action_smoothing_applied"])
        self.assertFalse(second_step_info["early_rebalance_risk_repeat_turnover_cap_applied"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.1, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["SPY"], 0.8, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["CASH"], 0.2, places=6)

    def test_env_repeat_unrecovered_early_risk_turnover_cap_requires_relative_wealth_recovery(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.0, 0.02, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_repeat_unrecovered_turnover_cap=0.05,
            early_rebalance_risk_repeat_unrecovered_turnover_cap_after=1,
            early_rebalance_risk_repeat_unrecovered_turnover_cap_before=3,
            early_rebalance_risk_repeat_unrecovered_symbol="SPY",
            early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min=0.15,
            early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min=0.15,
            early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery=0.01,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.7, 0.3], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.9, 0.1], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.2, places=6)
        self.assertFalse(first_step_info["early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active"])
        self.assertAlmostEqual(first_step_info["pre_trade_relative_wealth_ratio"], 1.0, places=6)
        self.assertAlmostEqual(first_step_info["last_executed_rebalance_pre_trade_relative_wealth_ratio"], 1.0, places=6)

        self.assertTrue(second_step_info["early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_repeat_unrecovered_turnover_cap_applied"])
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction"], 0.2, places=6)
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase"], 0.2, places=6)
        self.assertAlmostEqual(
            second_step_info["early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(second_step_info["pre_trade_relative_wealth_ratio"], 1.0 / 1.02, places=6)
        self.assertLess(second_step_info["early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery"], 0.01)
        self.assertAlmostEqual(second_step_info["turnover"], 0.05, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["SPY"], 0.75, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["CASH"], 0.25, places=6)

    def test_env_cumulative_early_risk_turnover_cap_uses_projected_budget_breach(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            early_rebalance_risk_cumulative_turnover_cap=0.05,
            early_rebalance_risk_cumulative_turnover_cap_after=1,
            early_rebalance_risk_cumulative_turnover_cap_before=3,
            early_rebalance_risk_cumulative_symbol="SPY",
            early_rebalance_risk_cumulative_cash_reduction_budget=0.35,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.7, 0.3], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.9, 0.1], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.2, places=6)
        self.assertFalse(first_step_info["early_rebalance_risk_cumulative_turnover_cap_window_active"])
        self.assertAlmostEqual(first_step_info["cumulative_executed_rebalance_cash_reduction"], 0.2, places=6)

        self.assertTrue(second_step_info["early_rebalance_risk_cumulative_turnover_cap_window_active"])
        self.assertTrue(second_step_info["early_rebalance_risk_cumulative_turnover_cap_condition_met"])
        self.assertTrue(second_step_info["early_rebalance_risk_cumulative_turnover_cap_applied"])
        self.assertAlmostEqual(second_step_info["early_rebalance_risk_cumulative_prior_cash_reduction"], 0.2, places=6)
        self.assertAlmostEqual(second_step_info["turnover"], 0.05, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["SPY"], 0.75, places=6)
        self.assertAlmostEqual(second_step_info["target_weights"]["CASH"], 0.25, places=6)
        self.assertAlmostEqual(second_step_info["cumulative_executed_rebalance_cash_reduction"], 0.25, places=6)

    def test_env_early_benchmark_euphoria_turnover_cap_applies_in_shallow_drawdown_state(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    benchmark_returns=benchmark,
                    config=WealthFirstConfig(include_cash=True),
                    early_benchmark_euphoria_turnover_cap=0.10,
                    early_benchmark_euphoria_before=2,
                    early_benchmark_euphoria_benchmark_drawdown_min_threshold=-0.01,
                    early_benchmark_euphoria_symbol="SPY",
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
            early_benchmark_euphoria_turnover_cap=0.10,
            early_benchmark_euphoria_before=2,
            early_benchmark_euphoria_benchmark_drawdown_min_threshold=-0.01,
            early_benchmark_euphoria_symbol="SPY",
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertTrue(step_info["early_benchmark_euphoria_window_active"])
        self.assertTrue(step_info["early_benchmark_euphoria_condition_met"])
        self.assertTrue(step_info["early_benchmark_euphoria_turnover_cap_applied"])
        self.assertFalse(step_info["early_benchmark_euphoria_penalty_applied"])
        self.assertAlmostEqual(step_info["benchmark_regime_drawdown"], 0.0)
        self.assertAlmostEqual(step_info["turnover"], 0.10)
        self.assertAlmostEqual(step_info["target_weights"]["SPY"], 0.60)
        self.assertAlmostEqual(step_info["target_weights"]["CASH"], 0.40)
        self.assertEqual(step_info["early_benchmark_euphoria_before"], 2)

    def test_env_exposes_non_leaky_pre_trade_relative_wealth_ratio(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.0, 0.0, 0.01, 0.0, 0.0, 0.0], index=dates, name="SPY_BENCHMARK")

        if not rl_module.GYMNASIUM_AVAILABLE:
            self.skipTest("gymnasium is not installed")

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            benchmark_returns=benchmark,
            config=WealthFirstConfig(include_cash=True),
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.5, 0.5], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.5, 0.5], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["pre_trade_relative_wealth_ratio"], 1.0)
        self.assertAlmostEqual(second_step_info["pre_trade_relative_wealth_ratio"], 1.0 / 1.01)

    def test_env_late_rebalance_penalty_applies_after_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    late_rebalance_penalty=0.05,
                    late_rebalance_penalty_after=1,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            late_rebalance_penalty=0.05,
            late_rebalance_penalty_after=1,
        )

        env.reset(seed=7)
        _, first_reward, _, _, first_step_info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, second_reward, _, _, second_step_info = env.step(np.array([0.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(first_reward, 0.0)
        self.assertAlmostEqual(first_step_info["late_rebalance_penalty_component"], 0.0)
        self.assertFalse(first_step_info["late_rebalance_penalty_applied"])
        self.assertFalse(first_step_info["late_rebalance_threshold_reached"])
        self.assertAlmostEqual(second_reward, -0.05)
        self.assertAlmostEqual(second_step_info["late_rebalance_penalty_component"], -0.05)
        self.assertTrue(second_step_info["late_rebalance_penalty_applied"])
        self.assertTrue(second_step_info["late_rebalance_threshold_reached"])
        self.assertEqual(second_step_info["late_rebalance_penalty_after"], 1)

    def test_env_late_rebalance_gate_blocks_non_restoring_trade_and_allows_restoring_trade(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    late_rebalance_gate_after=1,
                    late_rebalance_gate_cash_threshold=0.6,
                    late_rebalance_gate_symbol="SPY",
                    late_rebalance_gate_symbol_max_weight=0.4,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            late_rebalance_gate_after=1,
            late_rebalance_gate_cash_threshold=0.6,
            late_rebalance_gate_symbol="SPY",
            late_rebalance_gate_symbol_max_weight=0.4,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.25, 0.75], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.20, 0.80], dtype=np.float32))
        _, _, _, _, third_step_info = env.step(np.array([0.55, 0.45], dtype=np.float32))

        self.assertAlmostEqual(first_step_info["turnover"], 0.25)
        self.assertEqual(first_step_info["executed_rebalances"], 1)
        self.assertFalse(first_step_info["late_rebalance_gate_active"])

        self.assertTrue(second_step_info["late_rebalance_gate_active"])
        self.assertFalse(second_step_info["late_rebalance_gate_condition_met"])
        self.assertTrue(second_step_info["late_rebalance_gate_blocked"])
        self.assertTrue(second_step_info["late_rebalance_gate_threshold_reached"])
        self.assertTrue(second_step_info["trade_suppressed"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.0)
        self.assertEqual(second_step_info["executed_rebalances"], 1)
        self.assertEqual(second_step_info["late_rebalance_gate_symbol"], "SPY")

        self.assertTrue(third_step_info["late_rebalance_gate_active"])
        self.assertTrue(third_step_info["late_rebalance_gate_condition_met"])
        self.assertFalse(third_step_info["late_rebalance_gate_blocked"])
        self.assertFalse(third_step_info["trade_suppressed"])
        self.assertAlmostEqual(third_step_info["turnover"], 0.30)
        self.assertEqual(third_step_info["executed_rebalances"], 2)

    def test_env_late_rebalance_gate_refinement_blocks_aggressive_cash_reducing_trade(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    late_rebalance_gate_after=1,
                    late_rebalance_gate_cash_threshold=0.2,
                    late_rebalance_gate_symbol="SPY",
                    late_rebalance_gate_symbol_max_weight=0.8,
                    late_rebalance_gate_cash_reduction_max=0.05,
                    late_rebalance_gate_symbol_increase_max=0.05,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            late_rebalance_gate_after=1,
            late_rebalance_gate_cash_threshold=0.2,
            late_rebalance_gate_symbol="SPY",
            late_rebalance_gate_symbol_max_weight=0.8,
            late_rebalance_gate_cash_reduction_max=0.05,
            late_rebalance_gate_symbol_increase_max=0.05,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.25, 0.75], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.35, 0.65], dtype=np.float32))
        _, _, _, _, third_step_info = env.step(np.array([0.28, 0.72], dtype=np.float32))

        self.assertFalse(first_step_info["late_rebalance_gate_active"])

        self.assertTrue(second_step_info["late_rebalance_gate_active"])
        self.assertFalse(second_step_info["late_rebalance_gate_refinement_condition_met"])
        self.assertTrue(second_step_info["late_rebalance_gate_blocked"])
        self.assertTrue(second_step_info["trade_suppressed"])
        self.assertAlmostEqual(second_step_info["late_rebalance_gate_cash_reduction"], 0.10)
        self.assertAlmostEqual(second_step_info["late_rebalance_gate_symbol_increase"], 0.10)
        self.assertAlmostEqual(second_step_info["turnover"], 0.0)

        self.assertTrue(third_step_info["late_rebalance_gate_active"])
        self.assertTrue(third_step_info["late_rebalance_gate_refinement_condition_met"])
        self.assertTrue(third_step_info["late_rebalance_gate_condition_met"])
        self.assertFalse(third_step_info["late_rebalance_gate_blocked"])
        self.assertFalse(third_step_info["trade_suppressed"])
        self.assertAlmostEqual(third_step_info["late_rebalance_gate_cash_reduction"], 0.03)
        self.assertAlmostEqual(third_step_info["late_rebalance_gate_symbol_increase"], 0.03)
        self.assertAlmostEqual(third_step_info["turnover"], 0.03)

    def test_env_late_rebalance_gate_target_cash_floor_blocks_trade_below_floor(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    late_rebalance_gate_after=1,
                    late_rebalance_gate_cash_threshold=0.2,
                    late_rebalance_gate_target_cash_min_threshold=0.25,
                    late_rebalance_gate_symbol="SPY",
                    late_rebalance_gate_symbol_max_weight=0.8,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            late_rebalance_gate_after=1,
            late_rebalance_gate_cash_threshold=0.2,
            late_rebalance_gate_target_cash_min_threshold=0.25,
            late_rebalance_gate_symbol="SPY",
            late_rebalance_gate_symbol_max_weight=0.8,
        )

        env.reset(seed=7)
        _, _, _, _, first_step_info = env.step(np.array([0.25, 0.75], dtype=np.float32))
        _, _, _, _, second_step_info = env.step(np.array([0.80, 0.20], dtype=np.float32))
        _, _, _, _, third_step_info = env.step(np.array([0.75, 0.25], dtype=np.float32))

        self.assertFalse(first_step_info["late_rebalance_gate_active"])

        self.assertTrue(second_step_info["late_rebalance_gate_active"])
        self.assertFalse(second_step_info["late_rebalance_gate_condition_met"])
        self.assertTrue(second_step_info["late_rebalance_gate_blocked"])
        self.assertTrue(second_step_info["trade_suppressed"])
        self.assertAlmostEqual(second_step_info["turnover"], 0.0)

        self.assertTrue(third_step_info["late_rebalance_gate_active"])
        self.assertTrue(third_step_info["late_rebalance_gate_condition_met"])
        self.assertFalse(third_step_info["late_rebalance_gate_blocked"])
        self.assertFalse(third_step_info["trade_suppressed"])
        self.assertAlmostEqual(third_step_info["turnover"], 0.50)

    def test_env_late_defensive_posture_penalty_applies_after_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    late_defensive_posture_penalty=0.05,
                    late_defensive_posture_penalty_after=1,
                    late_defensive_posture_penalty_cash_min_threshold=0.6,
                    late_defensive_posture_penalty_symbol="SPY",
                    late_defensive_posture_penalty_symbol_max_weight=0.4,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            late_defensive_posture_penalty=0.05,
            late_defensive_posture_penalty_after=1,
            late_defensive_posture_penalty_cash_min_threshold=0.6,
            late_defensive_posture_penalty_symbol="SPY",
            late_defensive_posture_penalty_symbol_max_weight=0.4,
        )

        env.reset(seed=7)
        _, first_reward, _, _, first_step_info = env.step(np.array([0.25, 0.75], dtype=np.float32))
        _, second_reward, _, _, second_step_info = env.step(np.array([0.25, 0.75], dtype=np.float32))

        self.assertAlmostEqual(first_reward, 0.0)
        self.assertFalse(first_step_info["late_defensive_posture_window_active"])
        self.assertFalse(first_step_info["late_defensive_posture_condition_met"])
        self.assertFalse(first_step_info["late_defensive_posture_penalty_applied"])
        self.assertAlmostEqual(first_step_info["late_defensive_posture_penalty_component"], 0.0)

        self.assertTrue(second_step_info["late_defensive_posture_window_active"])
        self.assertTrue(second_step_info["late_defensive_posture_condition_met"])
        self.assertTrue(second_step_info["late_defensive_posture_penalty_applied"])
        self.assertAlmostEqual(second_step_info["late_defensive_posture_penalty_component"], -0.05)
        self.assertAlmostEqual(second_reward, -0.05)
        self.assertEqual(second_step_info["late_defensive_posture_penalty_after"], 1)
        self.assertEqual(second_step_info["late_defensive_posture_penalty_symbol"], "SPY")

    def test_env_late_trend_mean_reversion_conflict_penalty_applies_after_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "MEAN_REVERSION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    late_trend_mean_reversion_conflict_penalty=0.05,
                    late_trend_mean_reversion_conflict_penalty_after=1,
                    late_trend_mean_reversion_conflict_trend_symbol="TREND_FOLLOWING",
                    late_trend_mean_reversion_conflict_trend_min_weight=0.66,
                    late_trend_mean_reversion_conflict_mean_reversion_symbol="MEAN_REVERSION",
                    late_trend_mean_reversion_conflict_mean_reversion_min_weight=0.13,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            late_trend_mean_reversion_conflict_penalty=0.05,
            late_trend_mean_reversion_conflict_penalty_after=1,
            late_trend_mean_reversion_conflict_trend_symbol="TREND_FOLLOWING",
            late_trend_mean_reversion_conflict_trend_min_weight=0.66,
            late_trend_mean_reversion_conflict_mean_reversion_symbol="MEAN_REVERSION",
            late_trend_mean_reversion_conflict_mean_reversion_min_weight=0.13,
        )

        env.reset(seed=7)
        _, first_reward, _, _, first_step_info = env.step(np.array([0.55, 0.15, 0.30], dtype=np.float32))
        _, second_reward, _, _, second_step_info = env.step(np.array([0.67, 0.15, 0.18], dtype=np.float32))

        self.assertAlmostEqual(first_reward, 0.0)
        self.assertFalse(first_step_info["late_trend_mean_reversion_conflict_window_active"])
        self.assertFalse(first_step_info["late_trend_mean_reversion_conflict_condition_met"])
        self.assertFalse(first_step_info["late_trend_mean_reversion_conflict_penalty_applied"])
        self.assertAlmostEqual(first_step_info["late_trend_mean_reversion_conflict_penalty_component"], 0.0)

        self.assertTrue(second_step_info["late_trend_mean_reversion_conflict_window_active"])
        self.assertTrue(second_step_info["late_trend_mean_reversion_conflict_condition_met"])
        self.assertTrue(second_step_info["late_trend_mean_reversion_conflict_penalty_applied"])
        self.assertAlmostEqual(second_step_info["late_trend_mean_reversion_conflict_penalty_component"], -0.05)
        self.assertAlmostEqual(second_reward, -0.05)
        self.assertEqual(second_step_info["late_trend_mean_reversion_conflict_penalty_after"], 1)
        self.assertEqual(second_step_info["late_trend_mean_reversion_conflict_trend_symbol"], "TREND_FOLLOWING")
        self.assertEqual(second_step_info["late_trend_mean_reversion_conflict_mean_reversion_symbol"], "MEAN_REVERSION")

    def test_env_reports_cash_penalty_and_sleeve_contributions(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.005, 0.012, -0.003, 0.011, 0.004],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(
                    returns,
                    lookback=2,
                    config=WealthFirstConfig(include_cash=True),
                    cash_weight_penalty=0.5,
                    cash_target_weight=0.1,
                )
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            config=WealthFirstConfig(include_cash=True),
            cash_weight_penalty=0.5,
            cash_target_weight=0.1,
        )

        env.reset(seed=7)
        _, _, _, _, step_info = env.step(np.array([0.25, 0.75], dtype=np.float32))

        self.assertAlmostEqual(step_info["cash_weight"], 0.75)
        self.assertAlmostEqual(step_info["cash_target_weight"], 0.1)
        self.assertAlmostEqual(step_info["excess_cash_weight"], 0.65)
        self.assertAlmostEqual(step_info["cash_weight_penalty_component"], -0.325)
        self.assertAlmostEqual(step_info["gross_return_contributions"]["SPY"], 0.25 * returns.iloc[2]["SPY"])
        self.assertIn("CASH", step_info["asset_returns"])
        self.assertIn("SPY", step_info["gross_return_contributions"])

    def test_env_supports_seeded_random_episode_windows(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.005, 0.012, -0.003, 0.011, 0.004, -0.002, 0.008, 0.006, -0.001],
            },
            index=dates,
        )

        if not rl_module.GYMNASIUM_AVAILABLE:
            with self.assertRaisesRegex(ImportError, "gymnasium"):
                rl_module.WealthFirstEnv(returns, lookback=2, episode_length=3, random_episode_start=True)
            return

        env = rl_module.WealthFirstEnv(
            returns,
            lookback=2,
            episode_length=3,
            random_episode_start=True,
            config=WealthFirstConfig(include_cash=True),
        )

        _, first_info = env.reset(seed=123)
        _, second_info = env.reset(seed=123)
        self.assertEqual(first_info["episode_start_step"], second_info["episode_start_step"])
        self.assertEqual(first_info["episode_end_step"] - first_info["episode_start_step"], 3)

        terminated = False
        steps_taken = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
            steps_taken += 1

        self.assertEqual(steps_taken, 3)


if __name__ == "__main__":
    unittest.main()