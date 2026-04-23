from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wealth_first.backtest import run_rolling_backtest
from wealth_first.data import load_returns_csv
from wealth_first.execution import PaperExchangeAdapter, build_execution_plan
from wealth_first.optimizer import WealthFirstConfig, optimize_weights, wealth_first_objective


class WealthFirstObjectiveTests(unittest.TestCase):
    def test_loss_penalty_makes_lossy_path_less_attractive(self) -> None:
        config = WealthFirstConfig(loss_penalty=12.0, include_cash=False)
        weights = np.array([1.0])

        steady_gains = pd.DataFrame({"SPY": [0.01, 0.02, 0.015, 0.01]})
        mixed_path = pd.DataFrame({"SPY": [0.01, -0.03, 0.015, -0.02]})

        gain_score = wealth_first_objective(weights, steady_gains, config)
        mixed_score = wealth_first_objective(weights, mixed_path, config)

        self.assertGreater(mixed_score, gain_score)

    def test_benchmark_aware_objective_requires_benchmark_returns(self) -> None:
        config = WealthFirstConfig(include_cash=False, benchmark_loss_penalty=4.0)
        returns = pd.DataFrame({"SPY": [0.01, -0.01, 0.015]})

        with self.assertRaisesRegex(ValueError, "requires benchmark returns"):
            wealth_first_objective(np.array([1.0]), returns, config)

    def test_single_asset_optimization_adds_cash_sleeve(self) -> None:
        returns = pd.DataFrame({"SPY": [-0.02, -0.01, -0.005, 0.001, -0.015]})
        result = optimize_weights(returns, WealthFirstConfig(loss_penalty=15.0, include_cash=True))

        self.assertIn("CASH", result.weights.index)
        self.assertGreater(result.weights["CASH"], 0.99)

    def test_per_sleeve_max_weight_caps_risky_asset(self) -> None:
        returns = pd.DataFrame({"SPY": [0.02, 0.015, 0.03, 0.01, 0.025]})

        result = optimize_weights(
            returns,
            WealthFirstConfig(
                loss_penalty=1.0,
                include_cash=True,
                max_weight_overrides={"SPY": 0.2},
            ),
        )

        self.assertAlmostEqual(result.weights["SPY"], 0.2, places=6)
        self.assertAlmostEqual(result.weights["CASH"], 0.8, places=6)

    def test_per_sleeve_min_weight_keeps_cash_reserve(self) -> None:
        returns = pd.DataFrame({"SPY": [0.02, 0.015, 0.03, 0.01, 0.025]})

        result = optimize_weights(
            returns,
            WealthFirstConfig(
                loss_penalty=1.0,
                include_cash=True,
                min_weight_overrides={"CASH": 0.35},
            ),
        )

        self.assertGreaterEqual(result.weights["CASH"], 0.35 - 1e-8)
        self.assertAlmostEqual(result.weights["SPY"], 0.65, places=6)

    def test_benchmark_aware_optimization_increases_benchmark_tracking_weight(self) -> None:
        returns = pd.DataFrame({"SPY": [0.03, -0.04, 0.03, -0.04, 0.03, -0.04]})
        base_config = WealthFirstConfig(loss_penalty=12.0, include_cash=True)
        benchmark_config = WealthFirstConfig(
            loss_penalty=12.0,
            include_cash=True,
            benchmark_gain_reward=0.5,
            benchmark_loss_penalty=20.0,
        )

        unconstrained = optimize_weights(returns, base_config)
        benchmark_aware = optimize_weights(returns, benchmark_config, benchmark_returns=returns["SPY"])

        self.assertGreater(benchmark_aware.weights["SPY"], unconstrained.weights["SPY"])

    def test_infeasible_per_sleeve_bounds_raise_error(self) -> None:
        returns = pd.DataFrame({"SPY": [0.01, 0.02, 0.03, 0.01]})

        with self.assertRaisesRegex(ValueError, "sum to more than 1"):
            optimize_weights(
                returns,
                WealthFirstConfig(
                    include_cash=True,
                    min_weight_overrides={"SPY": 0.8, "CASH": 0.3},
                ),
            )

    def test_turnover_penalty_uses_tradable_turnover_without_double_counting_cash(self) -> None:
        returns = pd.DataFrame({"SPY": [0.0, 0.0], "CASH": [0.0, 0.0]})
        weights = np.array([0.35, 0.65])
        previous_weights = np.array([0.60, 0.40])

        score = wealth_first_objective(
            weights,
            returns,
            WealthFirstConfig(loss_penalty=1.0, loss_power=1.0, turnover_penalty=1.0, include_cash=False),
            previous_weights=previous_weights,
        )

        self.assertAlmostEqual(score, 0.25)

    def test_transaction_and_slippage_costs_penalize_objective_on_tradable_turnover(self) -> None:
        returns = pd.DataFrame({"SPY": [0.0, 0.0], "CASH": [0.0, 0.0]})
        weights = np.array([0.35, 0.65])
        previous_weights = np.array([0.60, 0.40])

        score = wealth_first_objective(
            weights,
            returns,
            WealthFirstConfig(
                loss_penalty=1.0,
                loss_power=1.0,
                include_cash=False,
                transaction_cost_bps=10.0,
                slippage_bps=10.0,
            ),
            previous_weights=previous_weights,
        )

        self.assertAlmostEqual(score, 0.0005)

    def test_rolling_backtest_produces_positive_wealth_ratios(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015, -0.005, 0.007, 0.011, -0.004, 0.009],
            },
            index=dates,
        )

        result = run_rolling_backtest(
            returns,
            lookback=3,
            rebalance_frequency=1,
            config=WealthFirstConfig(include_cash=True, loss_penalty=8.0, turnover_penalty=0.01),
        )

        self.assertEqual(len(result.portfolio_returns), len(returns) - 3)
        self.assertListEqual(list(result.weights.columns), ["SPY", "CASH"])
        self.assertTrue((result.wealth_ratios > 0.0).all())

    def test_backtest_costs_reduce_net_returns(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.04, -0.05, 0.05, -0.04, 0.03, -0.03, 0.04, -0.04, 0.03, -0.02],
            },
            index=dates,
        )

        no_cost_result = run_rolling_backtest(
            returns,
            lookback=3,
            rebalance_frequency=1,
            config=WealthFirstConfig(include_cash=True, loss_penalty=10.0, turnover_penalty=0.0),
        )
        with_cost_result = run_rolling_backtest(
            returns,
            lookback=3,
            rebalance_frequency=1,
            config=WealthFirstConfig(
                include_cash=True,
                loss_penalty=10.0,
                turnover_penalty=0.0,
                transaction_cost_bps=10.0,
                slippage_bps=10.0,
            ),
        )

        self.assertGreater(with_cost_result.trading_costs.sum(), 0.0)
        self.assertLess(with_cost_result.wealth_index.iloc[-1], no_cost_result.wealth_index.iloc[-1])

    def test_backtest_target_weights_respect_per_sleeve_bounds_on_rebalance(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        returns = pd.DataFrame(
            {
                "SPY": [0.02, 0.01, 0.03, 0.01, 0.02, 0.015, 0.025, 0.01],
            },
            index=dates,
        )

        result = run_rolling_backtest(
            returns,
            lookback=3,
            rebalance_frequency=1,
            config=WealthFirstConfig(
                include_cash=True,
                loss_penalty=1.0,
                max_weight_overrides={"SPY": 0.25},
                min_weight_overrides={"CASH": 0.25},
            ),
        )

        self.assertTrue((result.weights["SPY"] <= 0.25 + 1e-8).all())
        self.assertTrue((result.weights["CASH"] >= 0.25 - 1e-8).all())

    def test_build_execution_plan_outputs_orders(self) -> None:
        target_weights = pd.Series({"SPY": 0.6, "CASH": 0.4})
        current_weights = pd.Series({"SPY": 0.2, "CASH": 0.8})

        plan = build_execution_plan(target_weights, current_weights=current_weights, equity=100_000.0, min_trade_weight=0.05)
        response = PaperExchangeAdapter().submit_allocation(plan)

        self.assertAlmostEqual(plan.turnover, 0.4)
        self.assertEqual(len(plan.orders), 1)
        self.assertEqual(plan.orders[0].symbol, "SPY")
        self.assertAlmostEqual(plan.target_cash_weight, 0.4)
        self.assertEqual(response["status"], "accepted")

    def test_load_returns_csv_parses_date_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sleeves.csv"
            csv_path.write_text(
                "date,TREND_FOLLOWING,MEAN_REVERSION\n"
                "2024-01-02,0.01,0.00\n"
                "2024-01-03,-0.02,0.01\n",
                encoding="utf-8",
            )

            loaded = load_returns_csv(csv_path)

        self.assertListEqual(list(loaded.columns), ["TREND_FOLLOWING", "MEAN_REVERSION"])
        self.assertEqual(str(loaded.index[0].date()), "2024-01-02")

    def test_backtest_accepts_external_benchmark(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.01, 0.0, 0.012, -0.002, 0.004, 0.006, 0.0, 0.008],
                "MEAN_REVERSION": [0.0, 0.005, -0.003, 0.007, 0.0, 0.004, 0.006, 0.0],
                "HEDGE_OVERLAY": [-0.002, 0.004, -0.003, 0.002, -0.001, 0.0, 0.001, -0.002],
            },
            index=dates,
        )
        benchmark = pd.Series([0.008, -0.01, 0.012, -0.004, 0.003, 0.007, -0.002, 0.009], index=dates, name="SPY_BENCHMARK")

        result = run_rolling_backtest(
            returns,
            lookback=3,
            rebalance_frequency=1,
            config=WealthFirstConfig(include_cash=True, loss_penalty=8.0),
            benchmark_returns=benchmark,
        )

        self.assertIsNotNone(result.benchmark_wealth_index)
        self.assertEqual(len(result.benchmark_wealth_index), len(result.portfolio_returns))

    def test_backtest_summary_reports_relative_metrics_when_benchmark_exists(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.011, 0.004, 0.012, -0.001, 0.005, 0.006, 0.004, 0.008],
                "MEAN_REVERSION": [0.0, 0.003, -0.002, 0.006, 0.001, 0.004, 0.005, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.008, -0.004, 0.007, -0.002, 0.003, 0.005, 0.002, 0.006], index=dates, name="SPY_BENCHMARK")

        result = run_rolling_backtest(
            returns,
            lookback=3,
            rebalance_frequency=1,
            config=WealthFirstConfig(include_cash=True, benchmark_loss_penalty=3.0),
            benchmark_returns=benchmark,
        )

        self.assertIn("benchmark_total_return", result.summary.index)
        self.assertIn("relative_total_return", result.summary.index)
        self.assertIn("tracking_error", result.summary.index)


if __name__ == "__main__":
    unittest.main()