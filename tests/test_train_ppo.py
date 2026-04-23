from __future__ import annotations

import unittest

import pandas as pd

from wealth_first.data_splits import chronological_train_validation_test_split, generate_walk_forward_splits, suggest_regime_balanced_split
from wealth_first.optimizer import WealthFirstConfig
from wealth_first.train_ppo import (
    _build_parser,
    _aggregate_comparison_maps,
    _aggregate_named_comparison_maps,
    _build_common_metric_comparison,
    _build_named_metric_comparison,
    _compute_rebalance_impact_records,
    _merge_weight_bound_overrides,
    _normalize_split_method_alias,
    _normalize_trade_budgets,
    _parse_weight_bound_overrides,
    _resolve_benchmark_regime_observation_flags,
    _simulate_static_hold_rollout_records,
    _simulate_trade_budget_rollout_records,
    _summarize_rebalance_impact_records,
    _summarize_policy_rollout_records,
)


class TrainPpoHelpersTests(unittest.TestCase):
    def test_chronological_split_preserves_chronology_and_window_coverage(self) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": range(20),
                "MEAN_REVERSION": range(100, 120),
            },
            index=dates,
        ).astype(float)

        split = chronological_train_validation_test_split(
            returns,
            lookback=3,
            validation_fraction=0.15,
            test_fraction=0.10,
        )

        self.assertEqual(split.train.rows, 10)
        self.assertEqual(split.validation.rows, 5)
        self.assertEqual(split.test.rows, 5)
        self.assertEqual(split.train.start_label, dates[0])
        self.assertEqual(split.train.end_label, dates[9])
        self.assertEqual(split.validation.start_label, dates[10])
        self.assertEqual(split.test.start_label, dates[15])
        self.assertEqual(split.test.end_label, dates[-1])

    def test_regime_balanced_split_returns_contiguous_windows_and_score(self) -> None:
        dates = pd.date_range("2020-01-01", periods=90, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": ([0.002] * 30) + ([0.015, -0.02] * 15) + ([0.004] * 30),
                "MEAN_REVERSION": ([0.001] * 30) + ([-0.01, 0.012] * 15) + ([0.003] * 30),
            },
            index=dates,
        )
        benchmark = pd.Series(([0.0015] * 30) + ([0.018, -0.022] * 15) + ([0.0035] * 30), index=dates, dtype=float)

        split = suggest_regime_balanced_split(
            returns,
            benchmark_returns=benchmark,
            lookback=5,
            validation_fraction=0.15,
            test_fraction=0.10,
            search_step=5,
        )

        self.assertEqual(split.method, "regime-balanced")
        self.assertEqual(split.train.start_index, 0)
        self.assertEqual(split.train.end_index, split.validation.start_index)
        self.assertEqual(split.validation.end_index, split.test.start_index)
        self.assertEqual(split.test.end_index, len(returns))
        self.assertTrue(split.score >= 0.0)
        self.assertGreater(split.regime_coverage["train"], 0.0)
        self.assertGreater(split.regime_coverage["validation"], 0.0)
        self.assertGreater(split.regime_coverage["test"], 0.0)
        self.assertLessEqual(abs(split.train.rows - 67), 10)
        self.assertLessEqual(abs(split.validation.rows - 14), 10)
        self.assertLessEqual(abs(split.test.rows - 9), 10)

    def test_parse_weight_bound_overrides_returns_min_and_max_maps(self) -> None:
        min_overrides, max_overrides = _parse_weight_bound_overrides([
            "TREND_FOLLOWING=0.00:0.50",
            "CASH=0.10:0.80",
        ])

        self.assertEqual(min_overrides, {"TREND_FOLLOWING": 0.0, "CASH": 0.1})
        self.assertEqual(max_overrides, {"TREND_FOLLOWING": 0.5, "CASH": 0.8})

    def test_merge_weight_bound_overrides_overlays_policy_specific_bounds(self) -> None:
        merged_min_overrides, merged_max_overrides = _merge_weight_bound_overrides(
            {"CASH": 0.1},
            {"HEDGE_OVERLAY": 0.25},
            {"TREND_FOLLOWING": 0.45},
            {"HEDGE_OVERLAY": 0.05, "CASH": 0.35},
        )

        self.assertEqual(merged_min_overrides, {"CASH": 0.1, "TREND_FOLLOWING": 0.45})
        self.assertEqual(merged_max_overrides, {"HEDGE_OVERLAY": 0.05, "CASH": 0.35})

    def test_resolve_benchmark_regime_observation_flags_supports_summary_only_and_relative_only(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(["--benchmark-regime-observations"])
        self.assertEqual(_resolve_benchmark_regime_observation_flags(args), (True, True))

        args = parser.parse_args(["--benchmark-regime-summary-only-observations"])
        self.assertEqual(_resolve_benchmark_regime_observation_flags(args), (True, False))

        args = parser.parse_args(["--benchmark-regime-relative-cumulative-only-observations"])
        self.assertEqual(_resolve_benchmark_regime_observation_flags(args), (False, True))

        args = parser.parse_args([])
        self.assertEqual(_resolve_benchmark_regime_observation_flags(args), (False, False))

    def test_build_parser_accepts_trend_early_risk_args(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-trend-turnover-cap",
                "0.01",
                "--early-rebalance-risk-trend-turnover-cap-after",
                "1",
                "--early-rebalance-risk-trend-turnover-cap-before",
                "3",
                "--early-rebalance-risk-trend-turnover-cap-benchmark-cumulative-return-max-threshold",
                "0.018193",
                "--early-rebalance-risk-trend-turnover-cap-target-trend-min-threshold",
                "0.562379",
                "--early-rebalance-risk-trend-turnover-cap-pre-trade-trend-min-threshold",
                "0.556989",
                "--early-rebalance-risk-trend-turnover-cap-delta-trend-min-threshold",
                "0.004167",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_trend_turnover_cap, 0.01)
        self.assertEqual(args.early_rebalance_risk_trend_turnover_cap_after, 1)
        self.assertEqual(args.early_rebalance_risk_trend_turnover_cap_before, 3)
        self.assertAlmostEqual(
            args.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold,
            0.018193,
        )
        self.assertAlmostEqual(args.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold, 0.562379)
        self.assertAlmostEqual(args.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold, 0.556989)
        self.assertAlmostEqual(args.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold, 0.004167)

    def test_build_parser_accepts_direct_cash_min_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-min-pre-trade-cash-weight",
                "0.188534",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight, 0.188534)

    def test_build_parser_accepts_direct_target_cash_min_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-target-cash-min-threshold",
                "0.186246",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_target_cash_min_threshold, 0.186246)

    def test_build_parser_accepts_direct_target_cash_max_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-target-cash-max-threshold",
                "0.215911",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_target_cash_max_threshold, 0.215911)

    def test_build_parser_accepts_direct_target_trend_max_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-target-trend-max-threshold",
                "0.589234",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_target_trend_max_threshold, 0.589234)

    def test_build_parser_accepts_direct_target_mean_reversion_max_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-target-mean-reversion-max-threshold",
                "0.215481",
            ]
        )

        self.assertAlmostEqual(
            args.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
            0.215481,
        )

    def test_build_parser_accepts_direct_delta_cash_min_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-delta-cash-min-threshold",
                "-0.01668",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_delta_cash_min_threshold, -0.01668)

    def test_build_parser_accepts_direct_delta_cash_max_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-delta-cash-max-threshold",
                "0.004022",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_delta_cash_max_threshold, 0.004022)

    def test_build_parser_accepts_direct_delta_trend_min_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-delta-trend-min-threshold",
                "0.004167",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_delta_trend_min_threshold, 0.004167)

    def test_build_parser_accepts_direct_allow_nonincreasing_risk_symbol_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-allow-nonincreasing-risk-symbol",
            ]
        )

        self.assertTrue(args.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol)
        self.assertFalse(args.early_rebalance_risk_turnover_cap_use_penalty_state_filters)

    def test_build_parser_accepts_primary_turnover_cap_max_applications_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-max-applications",
                "3",
            ]
        )

        self.assertEqual(args.early_rebalance_risk_turnover_cap_max_applications, 3)

    def test_build_parser_accepts_primary_turnover_cap_secondary_schedule_args(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-secondary-cap",
                "0.04",
                "--early-rebalance-risk-turnover-cap-secondary-after-applications",
                "2",
                "--early-rebalance-risk-turnover-cap-secondary-benchmark-cumulative-return-min-threshold",
                "0.02",
                "--early-rebalance-risk-turnover-cap-secondary-max-pre-trade-relative-wealth-ratio",
                "0.9995",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_secondary_cap, 0.04)
        self.assertEqual(args.early_rebalance_risk_turnover_cap_secondary_after_applications, 2)
        self.assertAlmostEqual(
            args.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold,
            0.02,
        )
        self.assertAlmostEqual(
            args.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio,
            0.9995,
        )

    def test_build_parser_accepts_direct_delta_mean_reversion_min_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-delta-mean-reversion-min-threshold",
                "-0.010352",
            ]
        )

        self.assertAlmostEqual(
            args.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
            -0.010352,
        )

    def test_build_parser_accepts_direct_proposed_turnover_min_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-proposed-turnover-min-threshold",
                "0.045717",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold, 0.045717)

    def test_build_parser_accepts_direct_proposed_turnover_max_early_risk_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-turnover-cap-proposed-turnover-max-threshold",
                "0.104431",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold, 0.104431)

    def test_build_parser_accepts_repeat_early_risk_args(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-repeat-turnover-cap",
                "0.05",
                "--early-rebalance-risk-repeat-action-smoothing",
                "0.5",
                "--early-rebalance-risk-repeat-turnover-cap-after",
                "1",
                "--early-rebalance-risk-repeat-turnover-cap-before",
                "3",
                "--early-rebalance-risk-repeat-symbol",
                "TREND_FOLLOWING",
                "--early-rebalance-risk-repeat-previous-cash-reduction-min",
                "0.01",
                "--early-rebalance-risk-repeat-previous-symbol-increase-min",
                "0.02",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_repeat_turnover_cap, 0.05)
        self.assertAlmostEqual(args.early_rebalance_risk_repeat_action_smoothing, 0.5)
        self.assertEqual(args.early_rebalance_risk_repeat_turnover_cap_after, 1)
        self.assertEqual(args.early_rebalance_risk_repeat_turnover_cap_before, 3)
        self.assertEqual(args.early_rebalance_risk_repeat_symbol, "TREND_FOLLOWING")
        self.assertAlmostEqual(args.early_rebalance_risk_repeat_previous_cash_reduction_min, 0.01)
        self.assertAlmostEqual(args.early_rebalance_risk_repeat_previous_symbol_increase_min, 0.02)

    def test_build_parser_accepts_mean_reversion_action_smoothing_arg(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-mean-reversion-action-smoothing",
                "0.25",
                "--early-rebalance-risk-mean-reversion-turnover-cap-before",
                "3",
                "--early-rebalance-risk-mean-reversion-turnover-cap-pre-trade-mean-reversion-min-threshold",
                "0.176047",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_mean_reversion_action_smoothing, 0.25)
        self.assertEqual(args.early_rebalance_risk_mean_reversion_turnover_cap_before, 3)
        self.assertAlmostEqual(
            args.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
            0.176047,
        )

    def test_build_parser_accepts_repeat_unrecovered_early_risk_args(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-repeat-unrecovered-turnover-cap",
                "0.04",
                "--early-rebalance-risk-repeat-unrecovered-turnover-cap-after",
                "1",
                "--early-rebalance-risk-repeat-unrecovered-turnover-cap-before",
                "3",
                "--early-rebalance-risk-repeat-unrecovered-symbol",
                "TREND_FOLLOWING",
                "--early-rebalance-risk-repeat-unrecovered-previous-cash-reduction-min",
                "0.01",
                "--early-rebalance-risk-repeat-unrecovered-previous-symbol-increase-min",
                "0.02",
                "--early-rebalance-risk-repeat-unrecovered-min-relative-wealth-recovery",
                "0.01",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_repeat_unrecovered_turnover_cap, 0.04)
        self.assertEqual(args.early_rebalance_risk_repeat_unrecovered_turnover_cap_after, 1)
        self.assertEqual(args.early_rebalance_risk_repeat_unrecovered_turnover_cap_before, 3)
        self.assertEqual(args.early_rebalance_risk_repeat_unrecovered_symbol, "TREND_FOLLOWING")
        self.assertAlmostEqual(args.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min, 0.01)
        self.assertAlmostEqual(args.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min, 0.02)
        self.assertAlmostEqual(args.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery, 0.01)

    def test_build_parser_accepts_cumulative_early_risk_args(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "--early-rebalance-risk-cumulative-turnover-cap",
                "0.05",
                "--early-rebalance-risk-cumulative-turnover-cap-after",
                "1",
                "--early-rebalance-risk-cumulative-turnover-cap-before",
                "3",
                "--early-rebalance-risk-cumulative-symbol",
                "TREND_FOLLOWING",
                "--early-rebalance-risk-cumulative-cash-reduction-budget",
                "0.03",
            ]
        )

        self.assertAlmostEqual(args.early_rebalance_risk_cumulative_turnover_cap, 0.05)
        self.assertEqual(args.early_rebalance_risk_cumulative_turnover_cap_after, 1)
        self.assertEqual(args.early_rebalance_risk_cumulative_turnover_cap_before, 3)
        self.assertEqual(args.early_rebalance_risk_cumulative_symbol, "TREND_FOLLOWING")
        self.assertAlmostEqual(args.early_rebalance_risk_cumulative_cash_reduction_budget, 0.03)

    def test_split_method_aliases_normalize_to_canonical_values(self) -> None:
        parser = _build_parser()

        chrono_args = parser.parse_args(["--split-method", "chrono"])
        self.assertEqual(_normalize_split_method_alias(chrono_args.split_method), "chronological")

        regime_args = parser.parse_args(["--split-method", "regime"])
        self.assertEqual(_normalize_split_method_alias(regime_args.split_method), "regime-balanced")

        canonical_args = parser.parse_args(["--split-method", "chronological"])
        self.assertEqual(_normalize_split_method_alias(canonical_args.split_method), "chronological")

    def test_normalize_trade_budgets_deduplicates_and_rejects_one_trade_case(self) -> None:
        self.assertEqual(_normalize_trade_budgets([1, 2, 3, 2]), [1, 2, 3])

        with self.assertRaisesRegex(ValueError, "positive integers"):
            _normalize_trade_budgets([0, 1])

    def test_summarize_policy_rollout_records_matches_expected_totals(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        records = pd.DataFrame(
            {
                "raw_reward": [0.2, -0.1, 0.25],
                "portfolio_return": [0.01, -0.02, 0.03],
                "gross_portfolio_return": [0.011, -0.018, 0.031],
                "gross_reward_component": [0.21, -0.08, 0.28],
                "reward_core_component": [0.2, -0.09, 0.26],
                "benchmark_reward_component": [0.01, -0.01, 0.02],
                "execution_cost_reward_drag": [-0.01, -0.01, -0.02],
                "turnover_penalty_component": [-0.002, -0.001, -0.0015],
                "weight_reg_penalty_component": [-0.001, -0.001, -0.001],
                "friction_reward_drag": [-0.012, -0.011, -0.0215],
                "proposed_turnover": [0.3, 0.2, 0.25],
                "turnover": [0.2, 0.1, 0.15],
                "trade_suppressed": [False, True, False],
                "execution_cost": [0.001, 0.0005, 0.0007],
                "rebalance_budget_exhausted": [False, False, True],
                "rebalance_cooldown_active": [False, True, False],
                "rebalance_cooldown_blocked": [False, True, False],
                "rebalance_cooldown_remaining": [1.0, 0.0, 1.0],
                "max_executed_rebalances": [4.0, 4.0, 4.0],
                "rebalance_cooldown_steps": [1.0, 1.0, 1.0],
                "early_rebalance_risk_window_active": [True, True, False],
                "early_rebalance_risk_turnover_cap_window_active": [True, True, False],
                "early_rebalance_risk_turnover_cap_condition_met": [True, False, False],
                "early_rebalance_risk_turnover_cap_applied": [True, False, False],
                "early_rebalance_risk_shallow_drawdown_turnover_cap_window_active": [False, True, False],
                "early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met": [False, True, False],
                "early_rebalance_risk_shallow_drawdown_turnover_cap_applied": [False, True, False],
                "early_rebalance_risk_mean_reversion_turnover_cap_window_active": [False, True, False],
                "early_rebalance_risk_mean_reversion_turnover_cap_condition_met": [False, True, False],
                "early_rebalance_risk_mean_reversion_action_smoothing_applied": [False, True, False],
                "early_rebalance_risk_mean_reversion_turnover_cap_applied": [False, True, False],
                "early_rebalance_risk_trend_turnover_cap_window_active": [True, False, False],
                "early_rebalance_risk_trend_turnover_cap_condition_met": [True, False, False],
                "early_rebalance_risk_trend_turnover_cap_applied": [True, False, False],
                "early_rebalance_risk_deep_drawdown_turnover_cap_window_active": [True, False, False],
                "early_rebalance_risk_deep_drawdown_turnover_cap_condition_met": [True, False, False],
                "early_rebalance_risk_deep_drawdown_turnover_cap_applied": [True, False, False],
                "early_rebalance_risk_repeat_turnover_cap_window_active": [False, True, True],
                "early_rebalance_risk_repeat_turnover_cap_condition_met": [False, True, False],
                "early_rebalance_risk_repeat_action_smoothing_applied": [False, True, False],
                "early_rebalance_risk_repeat_turnover_cap_applied": [False, True, False],
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active": [False, True, True],
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met": [False, True, False],
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_applied": [False, True, False],
                "early_rebalance_risk_cumulative_turnover_cap_window_active": [False, True, True],
                "early_rebalance_risk_cumulative_turnover_cap_condition_met": [False, True, False],
                "early_rebalance_risk_cumulative_turnover_cap_applied": [False, True, False],
                "early_rebalance_risk_penalty_applied": [True, False, False],
                "early_rebalance_risk_condition_met": [True, False, False],
                "early_rebalance_risk_penalty": [0.02, 0.02, 0.02],
                "early_rebalance_risk_turnover_cap": [0.15, 0.15, 0.15],
                "early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold": [-0.022896, -0.022896, -0.022896],
                "early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold": [0.0125, 0.0125, 0.0125],
                "early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight": [0.188534, 0.188534, 0.188534],
                "early_rebalance_risk_turnover_cap_target_cash_min_threshold": [0.186246, 0.186246, 0.186246],
                "early_rebalance_risk_turnover_cap_target_cash_max_threshold": [0.215911, 0.215911, 0.215911],
                "early_rebalance_risk_turnover_cap_target_trend_max_threshold": [0.589234, 0.589234, 0.589234],
                "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold": [0.215481, 0.215481, 0.215481],
                "early_rebalance_risk_turnover_cap_delta_cash_min_threshold": [-0.01668, -0.01668, -0.01668],
                "early_rebalance_risk_turnover_cap_delta_cash_max_threshold": [0.004022, 0.004022, 0.004022],
                "early_rebalance_risk_turnover_cap_delta_trend_min_threshold": [0.004167, 0.004167, 0.004167],
                "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold": [-0.010352, -0.010352, -0.010352],
                "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold": [0.045717, 0.045717, 0.045717],
                "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold": [0.104431, 0.104431, 0.104431],
                "early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio": [1.000688, 1.000688, 1.000688],
                "early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio": [1.00499, 1.00499, 1.00499],
                "early_rebalance_risk_turnover_cap_use_penalty_state_filters": [True, True, True],
                "early_rebalance_risk_turnover_cap_after": [2.0, 2.0, 2.0],
                "early_rebalance_risk_turnover_cap_before": [3.0, 3.0, 3.0],
                "early_rebalance_risk_turnover_cap_max_applications": [3.0, 3.0, 3.0],
                "early_rebalance_risk_turnover_cap_secondary_cap": [0.25, 0.25, 0.25],
                "early_rebalance_risk_turnover_cap_secondary_after_applications": [1.0, 1.0, 1.0],
                "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold": [0.02, 0.02, 0.02],
                "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio": [0.9995, 0.9995, 0.9995],
                "early_rebalance_risk_turnover_cap_applications": [1.0, 1.0, 1.0],
                "early_rebalance_risk_turnover_cap_max_applications_reached": [False, False, False],
                "early_rebalance_risk_turnover_cap_secondary_after_applications_reached": [False, True, True],
                "early_rebalance_risk_turnover_cap_secondary_state_condition_met": [False, True, True],
                "early_rebalance_risk_turnover_cap_secondary_active": [False, True, True],
                "early_rebalance_risk_turnover_cap_secondary_applied": [False, True, False],
                "early_rebalance_risk_turnover_cap_effective_cap": [0.15, 0.25, 0.25],
                "early_rebalance_risk_shallow_drawdown_turnover_cap": [0.0, 0.0, 0.0],
                "early_rebalance_risk_shallow_drawdown_turnover_cap_after": [1.0, 1.0, 1.0],
                "early_rebalance_risk_shallow_drawdown_turnover_cap_before": [3.0, 3.0, 3.0],
                "early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold": [0.212552, 0.212552, 0.212552],
                "early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold": [-0.031042, -0.031042, -0.031042],
                "early_rebalance_risk_mean_reversion_turnover_cap": [0.01, 0.01, 0.01],
                "early_rebalance_risk_mean_reversion_action_smoothing": [0.25, 0.25, 0.25],
                "early_rebalance_risk_mean_reversion_turnover_cap_after": [1.0, 1.0, 1.0],
                "early_rebalance_risk_mean_reversion_turnover_cap_before": [3.0, 3.0, 3.0],
                "early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold": [0.018193, 0.018193, 0.018193],
                "early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold": [0.175427, 0.175427, 0.175427],
                "early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold": [0.179722, 0.179722, 0.179722],
                "early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold": [-0.010352, -0.010352, -0.010352],
                "early_rebalance_risk_trend_turnover_cap": [0.01, 0.01, 0.01],
                "early_rebalance_risk_trend_turnover_cap_after": [1.0, 1.0, 1.0],
                "early_rebalance_risk_trend_turnover_cap_before": [3.0, 3.0, 3.0],
                "early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold": [0.018193, 0.018193, 0.018193],
                "early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold": [0.562379, 0.562379, 0.562379],
                "early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold": [0.556989, 0.556989, 0.556989],
                "early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold": [0.004167, 0.004167, 0.004167],
                "early_rebalance_risk_deep_drawdown_turnover_cap": [0.0, 0.0, 0.0],
                "early_rebalance_risk_deep_drawdown_turnover_cap_after": [0.0, 0.0, 0.0],
                "early_rebalance_risk_deep_drawdown_turnover_cap_before": [2.0, 2.0, 2.0],
                "early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold": [-0.03, -0.03, -0.03],
                "early_rebalance_risk_repeat_turnover_cap": [0.05, 0.05, 0.05],
                "early_rebalance_risk_repeat_action_smoothing": [0.5, 0.5, 0.5],
                "early_rebalance_risk_repeat_turnover_cap_after": [1.0, 1.0, 1.0],
                "early_rebalance_risk_repeat_turnover_cap_before": [3.0, 3.0, 3.0],
                "early_rebalance_risk_repeat_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "early_rebalance_risk_repeat_previous_cash_reduction_min": [0.01, 0.01, 0.01],
                "early_rebalance_risk_repeat_previous_symbol_increase_min": [0.02, 0.02, 0.02],
                "early_rebalance_risk_repeat_previous_cash_reduction": [0.0, 0.03, 0.04],
                "early_rebalance_risk_repeat_previous_symbol_increase": [0.0, 0.04, 0.05],
                "early_rebalance_risk_repeat_cash_reduction": [0.0, 0.02, 0.0],
                "early_rebalance_risk_repeat_symbol_increase": [0.0, 0.03, 0.0],
                "early_rebalance_risk_repeat_unrecovered_turnover_cap": [0.04, 0.04, 0.04],
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_after": [1.0, 1.0, 1.0],
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_before": [3.0, 3.0, 3.0],
                "early_rebalance_risk_repeat_unrecovered_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min": [0.01, 0.01, 0.01],
                "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min": [0.02, 0.02, 0.02],
                "early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery": [0.01, 0.01, 0.01],
                "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction": [0.0, 0.03, 0.04],
                "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase": [0.0, 0.04, 0.05],
                "early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio": [None, 1.0, 0.99],
                "early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery": [None, -0.01, 0.02],
                "early_rebalance_risk_repeat_unrecovered_cash_reduction": [0.0, 0.02, 0.0],
                "early_rebalance_risk_repeat_unrecovered_symbol_increase": [0.0, 0.03, 0.0],
                "early_rebalance_risk_cumulative_turnover_cap": [0.05, 0.05, 0.05],
                "early_rebalance_risk_cumulative_turnover_cap_after": [1.0, 1.0, 1.0],
                "early_rebalance_risk_cumulative_turnover_cap_before": [3.0, 3.0, 3.0],
                "early_rebalance_risk_cumulative_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "early_rebalance_risk_cumulative_cash_reduction_budget": [0.03, 0.03, 0.03],
                "early_rebalance_risk_cumulative_symbol_increase_budget": [0.05, 0.05, 0.05],
                "early_rebalance_risk_cumulative_prior_cash_reduction": [0.0, 0.02, 0.03],
                "early_rebalance_risk_cumulative_prior_symbol_increase": [0.0, 0.03, 0.04],
                "early_rebalance_risk_cumulative_cash_reduction": [0.02, 0.01, 0.0],
                "early_rebalance_risk_cumulative_symbol_increase": [0.03, 0.02, 0.0],
                "early_rebalance_risk_penalty_after": [1.0, 1.0, 1.0],
                "early_rebalance_risk_penalty_before": [4.0, 4.0, 4.0],
                "early_rebalance_risk_penalty_cash_max_threshold": [0.22, 0.22, 0.22],
                "early_rebalance_risk_penalty_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "early_rebalance_risk_penalty_symbol_min_weight": [0.55, 0.55, 0.55],
                "early_rebalance_risk_penalty_symbol_max_weight": [0.60, 0.60, 0.60],
                "early_rebalance_risk_penalty_benchmark_drawdown_min_threshold": [-0.02, -0.02, -0.02],
                "early_rebalance_risk_penalty_benchmark_drawdown_max_threshold": [-0.03, -0.03, -0.03],
                "early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio": [1.01, 1.01, 1.01],
                "early_benchmark_euphoria_window_active": [True, True, False],
                "early_benchmark_euphoria_turnover_cap_applied": [True, False, False],
                "early_benchmark_euphoria_penalty_applied": [True, False, False],
                "early_benchmark_euphoria_condition_met": [True, False, False],
                "early_benchmark_euphoria_penalty": [0.01, 0.01, 0.01],
                "early_benchmark_euphoria_turnover_cap": [0.10, 0.10, 0.10],
                "early_benchmark_euphoria_before": [2.0, 2.0, 2.0],
                "early_benchmark_euphoria_benchmark_drawdown_min_threshold": [-0.02, -0.02, -0.02],
                "early_benchmark_euphoria_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "late_rebalance_penalty": [0.01, 0.01, 0.01],
                "late_rebalance_penalty_after": [4.0, 4.0, 4.0],
                "late_rebalance_penalty_applied": [False, False, True],
                "late_rebalance_threshold_reached": [False, False, True],
                "late_rebalance_gate_active": [False, True, True],
                "late_rebalance_gate_blocked": [False, True, False],
                "late_rebalance_gate_condition_met": [False, False, True],
                "late_rebalance_gate_threshold_reached": [False, True, True],
                "late_rebalance_gate_after": [4.0, 4.0, 4.0],
                "late_rebalance_gate_cash_threshold": [0.17, 0.17, 0.17],
                "late_rebalance_gate_target_cash_min_threshold": [0.16, 0.16, 0.16],
                "late_rebalance_gate_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "late_rebalance_gate_symbol_max_weight": [0.66, 0.66, 0.66],
                "late_rebalance_gate_refinement_condition_met": [True, False, True],
                "late_rebalance_gate_cash_reduction_max": [0.0003, 0.0003, 0.0003],
                "late_rebalance_gate_symbol_increase_max": [0.003, 0.003, 0.003],
                "late_rebalance_gate_cash_reduction": [0.0, 0.01, 0.0],
                "late_rebalance_gate_symbol_increase": [0.0, 0.03, 0.005],
                "late_defensive_posture_window_active": [False, True, True],
                "late_defensive_posture_condition_met": [False, True, False],
                "late_defensive_posture_penalty_applied": [False, True, False],
                "state_trend_preservation_window_active": [True, False, True],
                "state_trend_preservation_condition_met": [True, False, False],
                "state_trend_preservation_guard_applied": [True, False, False],
                "late_defensive_posture_penalty": [0.015, 0.015, 0.015],
                "late_defensive_posture_penalty_after": [2.0, 2.0, 2.0],
                "late_defensive_posture_penalty_cash_min_threshold": [0.17, 0.17, 0.17],
                "late_defensive_posture_penalty_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "late_defensive_posture_penalty_symbol_max_weight": [0.66, 0.66, 0.66],
                "late_trend_mean_reversion_conflict_window_active": [False, True, True],
                "late_trend_mean_reversion_conflict_condition_met": [False, True, False],
                "late_trend_mean_reversion_conflict_penalty_applied": [False, True, False],
                "late_trend_mean_reversion_conflict_penalty": [0.02, 0.02, 0.02],
                "late_trend_mean_reversion_conflict_penalty_after": [1.0, 1.0, 1.0],
                "late_trend_mean_reversion_conflict_trend_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "late_trend_mean_reversion_conflict_trend_min_weight": [0.66, 0.66, 0.66],
                "late_trend_mean_reversion_conflict_mean_reversion_symbol": ["MEAN_REVERSION", "MEAN_REVERSION", "MEAN_REVERSION"],
                "late_trend_mean_reversion_conflict_mean_reversion_min_weight": [0.13, 0.13, 0.13],
                "state_trend_preservation_symbol": ["TREND_FOLLOWING", "TREND_FOLLOWING", "TREND_FOLLOWING"],
                "state_trend_preservation_cash_max_threshold": [0.22, 0.22, 0.22],
                "state_trend_preservation_symbol_min_weight": [0.64, 0.64, 0.64],
                "state_trend_preservation_max_symbol_reduction": [0.02, 0.02, 0.02],
                "early_rebalance_risk_penalty_component": [-0.02, 0.0, 0.0],
                "early_benchmark_euphoria_penalty_component": [-0.01, 0.0, 0.0],
                "late_rebalance_penalty_component": [0.0, 0.0, -0.01],
                "late_defensive_posture_penalty_component": [0.0, -0.015, 0.0],
                "late_trend_mean_reversion_conflict_penalty_component": [0.0, -0.02, 0.0],
                "executed_rebalances": [1.0, 1.0, 2.0],
                "excess_cash_weight": [0.2, 0.3, 0.1],
                "cash_weight_penalty_component": [-0.01, -0.015, -0.005],
                "benchmark_return": [0.005, -0.01, 0.02],
                "target_weight_TREND_FOLLOWING": [0.5, 0.4, 0.6],
                "gross_return_contribution_TREND_FOLLOWING": [0.004, -0.003, 0.005],
                "target_weight_CASH": [0.3, 0.4, 0.2],
            },
            index=dates,
        )

        summary = _summarize_policy_rollout_records(records)

        self.assertAlmostEqual(summary["total_return"], (1.01 * 0.98 * 1.03) - 1.0)
        self.assertAlmostEqual(summary["average_turnover"], 0.15)
        self.assertAlmostEqual(summary["total_turnover"], 0.45)
        self.assertAlmostEqual(summary["average_trading_cost"], (0.001 + 0.0005 + 0.0007) / 3)
        self.assertAlmostEqual(summary["average_cash_weight"], 0.3)
        self.assertAlmostEqual(summary["average_raw_reward"], (0.2 - 0.1 + 0.25) / 3)
        self.assertAlmostEqual(summary["average_execution_cost_reward_drag"], (-0.01 - 0.01 - 0.02) / 3)
        self.assertAlmostEqual(summary["average_proposed_turnover"], 0.25)
        self.assertAlmostEqual(summary["average_turnover_reduction"], 0.10)
        self.assertAlmostEqual(summary["rebalance_count"], 3.0)
        self.assertAlmostEqual(summary["rebalance_rate"], 1.0)
        self.assertAlmostEqual(summary["average_executed_turnover"], 0.15)
        self.assertAlmostEqual(summary["trade_suppression_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["rebalance_budget_exhaustion_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["rebalance_cooldown_active_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["rebalance_cooldown_block_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["average_early_rebalance_risk_penalty_component"], -0.02 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_max_applications_reached_rate"], 0.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_secondary_after_applications_reached_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_secondary_state_condition_met_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_secondary_active_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_secondary_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_shallow_drawdown_turnover_cap_window_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_shallow_drawdown_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_shallow_drawdown_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_turnover_cap_window_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_action_smoothing_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_trend_turnover_cap_window_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_trend_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_trend_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_deep_drawdown_turnover_cap_window_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_deep_drawdown_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_deep_drawdown_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_turnover_cap_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_action_smoothing_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_turnover_cap_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_turnover_cap_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_turnover_cap_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["average_early_benchmark_euphoria_penalty_component"], -0.01 / 3.0)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_turnover_cap_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_penalty_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["average_late_rebalance_penalty_component"], -0.01 / 3.0)
        self.assertAlmostEqual(summary["late_rebalance_penalty_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["late_rebalance_threshold_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["late_rebalance_gate_active_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["late_rebalance_gate_block_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["late_rebalance_gate_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["late_rebalance_gate_threshold_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["late_rebalance_gate_refinement_condition_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["average_late_defensive_posture_penalty_component"], -0.015 / 3.0)
        self.assertAlmostEqual(summary["late_defensive_posture_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["late_defensive_posture_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["late_defensive_posture_penalty_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["average_late_trend_mean_reversion_conflict_penalty_component"], -0.02 / 3.0)
        self.assertAlmostEqual(summary["late_trend_mean_reversion_conflict_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["late_trend_mean_reversion_conflict_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["late_trend_mean_reversion_conflict_penalty_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["state_trend_preservation_window_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["state_trend_preservation_condition_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["state_trend_preservation_guard_application_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["final_executed_rebalances"], 2.0)
        self.assertAlmostEqual(summary["final_early_rebalance_risk_turnover_cap_applications"], 1.0)
        self.assertAlmostEqual(summary["max_executed_rebalances"], 4.0)
        self.assertAlmostEqual(summary["final_rebalance_cooldown_remaining"], 1.0)
        self.assertAlmostEqual(summary["rebalance_cooldown_steps"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty"], 0.02)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap"], 0.15)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold"], -0.022896)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold"], 0.0125)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight"], 0.188534)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_target_cash_min_threshold"], 0.186246)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_target_cash_max_threshold"], 0.215911)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_target_trend_max_threshold"], 0.589234)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold"], 0.215481)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_delta_cash_min_threshold"], -0.01668)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_delta_cash_max_threshold"], 0.004022)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_delta_trend_min_threshold"], 0.004167)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold"], -0.010352)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold"], 0.045717)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold"], 0.104431)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio"], 1.000688)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio"], 1.00499)
        self.assertTrue(summary["early_rebalance_risk_turnover_cap_use_penalty_state_filters"])
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_after"], 2.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_before"], 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_max_applications"], 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_secondary_cap"], 0.25)
        self.assertAlmostEqual(summary["early_rebalance_risk_turnover_cap_secondary_after_applications"], 1.0)
        self.assertAlmostEqual(
            summary["early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold"],
            0.02,
        )
        self.assertAlmostEqual(
            summary["early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio"],
            0.9995,
        )
        self.assertAlmostEqual(summary["early_rebalance_risk_shallow_drawdown_turnover_cap"], 0.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_shallow_drawdown_turnover_cap_after"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_shallow_drawdown_turnover_cap_before"], 3.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold"], 0.212552)
        self.assertAlmostEqual(
            summary["early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold"],
            -0.031042,
        )
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_turnover_cap"], 0.01)
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_action_smoothing"], 0.25)
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_turnover_cap_after"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_mean_reversion_turnover_cap_before"], 3.0)
        self.assertAlmostEqual(
            summary["early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.018193,
        )
        self.assertAlmostEqual(
            summary["early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold"],
            0.175427,
        )
        self.assertAlmostEqual(
            summary["early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold"],
            0.179722,
        )
        self.assertAlmostEqual(
            summary["early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold"],
            -0.010352,
        )
        self.assertAlmostEqual(summary["early_rebalance_risk_trend_turnover_cap"], 0.01)
        self.assertAlmostEqual(summary["early_rebalance_risk_trend_turnover_cap_after"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_trend_turnover_cap_before"], 3.0)
        self.assertAlmostEqual(
            summary["early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold"],
            0.018193,
        )
        self.assertAlmostEqual(
            summary["early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold"],
            0.562379,
        )
        self.assertAlmostEqual(
            summary["early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold"],
            0.556989,
        )
        self.assertAlmostEqual(
            summary["early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold"],
            0.004167,
        )
        self.assertAlmostEqual(summary["early_rebalance_risk_deep_drawdown_turnover_cap"], 0.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_deep_drawdown_turnover_cap_after"], 0.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_deep_drawdown_turnover_cap_before"], 2.0)
        self.assertAlmostEqual(
            summary["early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold"],
            -0.03,
        )
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_turnover_cap"], 0.05)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_action_smoothing"], 0.5)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_turnover_cap_after"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_turnover_cap_before"], 3.0)
        self.assertEqual(summary["early_rebalance_risk_repeat_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_previous_cash_reduction_min"], 0.01)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_previous_symbol_increase_min"], 0.02)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_turnover_cap"], 0.04)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_turnover_cap_after"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_turnover_cap_before"], 3.0)
        self.assertEqual(summary["early_rebalance_risk_repeat_unrecovered_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min"], 0.01)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min"], 0.02)
        self.assertAlmostEqual(summary["early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery"], 0.01)
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_turnover_cap"], 0.05)
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_turnover_cap_after"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_turnover_cap_before"], 3.0)
        self.assertEqual(summary["early_rebalance_risk_cumulative_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_cash_reduction_budget"], 0.03)
        self.assertAlmostEqual(summary["early_rebalance_risk_cumulative_symbol_increase_budget"], 0.05)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_after"], 1.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_before"], 4.0)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_cash_max_threshold"], 0.22)
        self.assertEqual(summary["early_rebalance_risk_penalty_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_symbol_min_weight"], 0.55)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_symbol_max_weight"], 0.60)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_benchmark_drawdown_min_threshold"], -0.02)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_benchmark_drawdown_max_threshold"], -0.03)
        self.assertAlmostEqual(summary["early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio"], 1.01)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_penalty"], 0.01)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_turnover_cap"], 0.10)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_before"], 2.0)
        self.assertAlmostEqual(summary["early_benchmark_euphoria_benchmark_drawdown_min_threshold"], -0.02)
        self.assertEqual(summary["early_benchmark_euphoria_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["late_rebalance_penalty"], 0.01)
        self.assertAlmostEqual(summary["late_rebalance_penalty_after"], 4.0)
        self.assertAlmostEqual(summary["late_rebalance_gate_after"], 4.0)
        self.assertAlmostEqual(summary["late_rebalance_gate_cash_threshold"], 0.17)
        self.assertAlmostEqual(summary["late_rebalance_gate_target_cash_min_threshold"], 0.16)
        self.assertEqual(summary["late_rebalance_gate_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["late_rebalance_gate_symbol_max_weight"], 0.66)
        self.assertAlmostEqual(summary["late_rebalance_gate_cash_reduction_max"], 0.0003)
        self.assertAlmostEqual(summary["late_rebalance_gate_symbol_increase_max"], 0.003)
        self.assertAlmostEqual(summary["late_defensive_posture_penalty"], 0.015)
        self.assertAlmostEqual(summary["late_defensive_posture_penalty_after"], 2.0)
        self.assertAlmostEqual(summary["late_defensive_posture_penalty_cash_min_threshold"], 0.17)
        self.assertEqual(summary["late_defensive_posture_penalty_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["late_defensive_posture_penalty_symbol_max_weight"], 0.66)
        self.assertAlmostEqual(summary["late_trend_mean_reversion_conflict_penalty"], 0.02)
        self.assertAlmostEqual(summary["late_trend_mean_reversion_conflict_penalty_after"], 1.0)
        self.assertEqual(summary["late_trend_mean_reversion_conflict_trend_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["late_trend_mean_reversion_conflict_trend_min_weight"], 0.66)
        self.assertEqual(summary["late_trend_mean_reversion_conflict_mean_reversion_symbol"], "MEAN_REVERSION")
        self.assertAlmostEqual(summary["late_trend_mean_reversion_conflict_mean_reversion_min_weight"], 0.13)
        self.assertEqual(summary["state_trend_preservation_symbol"], "TREND_FOLLOWING")
        self.assertAlmostEqual(summary["state_trend_preservation_cash_max_threshold"], 0.22)
        self.assertAlmostEqual(summary["state_trend_preservation_symbol_min_weight"], 0.64)
        self.assertAlmostEqual(summary["state_trend_preservation_max_symbol_reduction"], 0.02)
        self.assertAlmostEqual(summary["average_excess_cash_weight"], 0.2)
        self.assertAlmostEqual(summary["average_cash_weight_penalty_component"], (-0.01 - 0.015 - 0.005) / 3)
        self.assertAlmostEqual(summary["average_target_weight_TREND_FOLLOWING"], 0.5)
        self.assertAlmostEqual(summary["average_gross_return_contribution_TREND_FOLLOWING"], (0.004 - 0.003 + 0.005) / 3)
        self.assertIn("relative_total_return", summary.index)

    def test_simulate_static_hold_rollout_records_trades_once_then_drifts(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.10, 0.00],
            },
            index=dates,
        )
        policy_rollout = pd.DataFrame(
            {
                "step_label": dates.astype(str),
                "pre_trade_weight_TREND_FOLLOWING": [0.50, 0.0],
                "pre_trade_weight_CASH": [0.50, 0.0],
                "target_weight_TREND_FOLLOWING": [0.60, 0.0],
                "target_weight_CASH": [0.40, 0.0],
            },
            index=dates,
        )
        config = WealthFirstConfig(include_cash=True, transaction_cost_bps=0.0, slippage_bps=0.0)

        static_hold_records = _simulate_static_hold_rollout_records(
            policy_rollout,
            returns,
            benchmark_returns=None,
            config=config,
        )

        self.assertAlmostEqual(static_hold_records.iloc[0]["turnover"], 0.10)
        self.assertAlmostEqual(static_hold_records.iloc[0]["gross_portfolio_return"], 0.06)
        self.assertAlmostEqual(static_hold_records.iloc[1]["turnover"], 0.0)
        self.assertAlmostEqual(static_hold_records.iloc[0]["target_weight_TREND_FOLLOWING"], 0.60)
        self.assertAlmostEqual(static_hold_records.iloc[0]["ending_weight_TREND_FOLLOWING"], 0.6226415094339622)
        self.assertAlmostEqual(static_hold_records.iloc[1]["pre_trade_weight_TREND_FOLLOWING"], 0.6226415094339622)
        self.assertAlmostEqual(static_hold_records.iloc[1]["target_weight_TREND_FOLLOWING"], 0.6226415094339622)

    def test_simulate_static_hold_rollout_records_include_pre_trade_relative_wealth_ratio(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.0, 0.0],
                "CASH": [0.0, 0.0],
            },
            index=dates,
        )
        benchmark = pd.Series([0.01, 0.0], index=dates, dtype=float)
        policy_rollout = pd.DataFrame(
            {
                "step_label": dates.astype(str),
                "pre_trade_weight_TREND_FOLLOWING": [0.50, 0.50],
                "pre_trade_weight_CASH": [0.50, 0.50],
                "target_weight_TREND_FOLLOWING": [0.50, 0.50],
                "target_weight_CASH": [0.50, 0.50],
            },
            index=dates,
        )
        config = WealthFirstConfig(include_cash=True, transaction_cost_bps=0.0, slippage_bps=0.0)

        static_hold_records = _simulate_static_hold_rollout_records(
            policy_rollout,
            returns,
            benchmark_returns=benchmark,
            config=config,
        )

        self.assertAlmostEqual(static_hold_records.iloc[0]["pre_trade_relative_wealth_ratio"], 1.0)
        self.assertAlmostEqual(static_hold_records.iloc[1]["pre_trade_relative_wealth_ratio"], 1.0 / 1.01)

    def test_simulate_static_hold_rollout_records_holds_first_step_target_even_without_initial_trade(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.0, 0.0, 0.0],
                "CASH": [0.0, 0.0, 0.0],
            },
            index=dates,
        )
        policy_rollout = pd.DataFrame(
            {
                "step_label": dates.astype(str),
                "pre_trade_weight_TREND_FOLLOWING": [0.50, 0.50, 0.60],
                "pre_trade_weight_CASH": [0.50, 0.50, 0.40],
                "target_weight_TREND_FOLLOWING": [0.50, 0.60, 0.70],
                "target_weight_CASH": [0.50, 0.40, 0.30],
            },
            index=dates,
        )
        config = WealthFirstConfig(include_cash=True, transaction_cost_bps=0.0, slippage_bps=0.0)

        static_hold_records = _simulate_static_hold_rollout_records(
            policy_rollout,
            returns,
            benchmark_returns=None,
            config=config,
        )

        self.assertAlmostEqual(static_hold_records.iloc[0]["turnover"], 0.0)
        self.assertAlmostEqual(static_hold_records.iloc[1]["turnover"], 0.0)
        self.assertAlmostEqual(static_hold_records.iloc[1]["target_weight_TREND_FOLLOWING"], 0.50)
        self.assertAlmostEqual(static_hold_records.iloc[2]["target_weight_TREND_FOLLOWING"], 0.50)

    def test_simulate_trade_budget_rollout_records_caps_executed_rebalances(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        policy_rollout = pd.DataFrame(
            {
                "step_label": dates.astype(str),
                "pre_trade_weight_TREND_FOLLOWING": [0.50, 0.50, 0.60, 0.70],
                "pre_trade_weight_CASH": [0.50, 0.50, 0.40, 0.30],
                "target_weight_TREND_FOLLOWING": [0.50, 0.60, 0.70, 0.80],
                "target_weight_CASH": [0.50, 0.40, 0.30, 0.20],
            },
            index=dates,
        )
        config = WealthFirstConfig(include_cash=True, transaction_cost_bps=0.0, slippage_bps=0.0)

        capped_records = _simulate_trade_budget_rollout_records(
            policy_rollout,
            returns,
            benchmark_returns=None,
            config=config,
            max_rebalances=2,
        )

        self.assertAlmostEqual(capped_records.iloc[0]["turnover"], 0.0)
        self.assertAlmostEqual(capped_records.iloc[1]["turnover"], 0.10)
        self.assertAlmostEqual(capped_records.iloc[2]["turnover"], 0.10)
        self.assertAlmostEqual(capped_records.iloc[3]["proposed_turnover"], 0.10)
        self.assertAlmostEqual(capped_records.iloc[3]["turnover"], 0.0)
        self.assertEqual(capped_records.iloc[3]["trade_suppressed"], True)
        self.assertAlmostEqual(capped_records.iloc[3]["target_weight_TREND_FOLLOWING"], 0.70)
        self.assertAlmostEqual(capped_records.iloc[3]["rebalances_executed"], 2.0)

    def test_simulate_trade_budget_rollout_records_counts_executed_rebalances_after_no_trade_start(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        policy_rollout = pd.DataFrame(
            {
                "step_label": dates.astype(str),
                "pre_trade_weight_TREND_FOLLOWING": [0.50, 0.50, 0.60, 0.70],
                "pre_trade_weight_CASH": [0.50, 0.50, 0.40, 0.30],
                "target_weight_TREND_FOLLOWING": [0.50, 0.60, 0.70, 0.80],
                "target_weight_CASH": [0.50, 0.40, 0.30, 0.20],
            },
            index=dates,
        )
        config = WealthFirstConfig(include_cash=True, transaction_cost_bps=0.0, slippage_bps=0.0)

        capped_records = _simulate_trade_budget_rollout_records(
            policy_rollout,
            returns,
            benchmark_returns=None,
            config=config,
            max_rebalances=1,
        )

        self.assertAlmostEqual(capped_records.iloc[0]["turnover"], 0.0)
        self.assertAlmostEqual(capped_records.iloc[1]["turnover"], 0.10)
        self.assertAlmostEqual(capped_records.iloc[2]["turnover"], 0.0)
        self.assertEqual(capped_records.iloc[2]["trade_suppressed"], True)
        self.assertAlmostEqual(capped_records.iloc[2]["target_weight_TREND_FOLLOWING"], 0.60)

    def test_compute_rebalance_impact_records_scores_each_executed_rebalance_interval(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.10, 0.0, 0.20],
            },
            index=dates,
        )
        policy_rollout = pd.DataFrame(
            {
                "step_label": dates.astype(str),
                "turnover": [0.20, 0.0, 0.50],
                "proposed_turnover": [0.20, 0.0, 0.50],
                "pre_trade_weight_TREND_FOLLOWING": [0.50, 0.70, 0.70],
                "pre_trade_weight_CASH": [0.50, 0.30, 0.30],
                "target_weight_TREND_FOLLOWING": [0.70, 0.70, 0.20],
                "target_weight_CASH": [0.30, 0.30, 0.80],
            },
            index=dates,
        )
        config = WealthFirstConfig(include_cash=True, transaction_cost_bps=0.0, slippage_bps=0.0)

        impact_records = _compute_rebalance_impact_records(
            policy_rollout,
            returns,
            benchmark_returns=None,
            config=config,
        )

        self.assertEqual(len(impact_records), 2)
        self.assertAlmostEqual(impact_records.iloc[0]["rebalance_number"], 1.0)
        self.assertAlmostEqual(impact_records.iloc[0]["first_step_return_delta"], 0.02)
        self.assertAlmostEqual(impact_records.iloc[0]["interval_total_return_delta"], 0.02)
        self.assertAlmostEqual(impact_records.iloc[1]["rebalance_number"], 2.0)
        self.assertAlmostEqual(impact_records.iloc[1]["first_step_return_delta"], -0.10)
        self.assertAlmostEqual(impact_records.iloc[1]["interval_total_return_delta"], -0.10)

    def test_simulate_trade_budget_rollout_records_matches_live_when_budget_allows_all_executed_trades(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        drifted_trend_weight = 0.77 / 1.07
        drifted_cash_weight = 0.30 / 1.07
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.10, 0.0, 0.20],
            },
            index=dates,
        )
        policy_rollout = pd.DataFrame(
            {
                "step_label": dates.astype(str),
                "turnover": [0.20, 0.0, 0.50],
                "proposed_turnover": [0.20, 0.0, 0.50],
                "pre_trade_weight_TREND_FOLLOWING": [0.50, drifted_trend_weight, drifted_trend_weight],
                "pre_trade_weight_CASH": [0.50, drifted_cash_weight, drifted_cash_weight],
                "target_weight_TREND_FOLLOWING": [0.70, drifted_trend_weight, 0.20],
                "target_weight_CASH": [0.30, drifted_cash_weight, 0.80],
            },
            index=dates,
        )
        config = WealthFirstConfig(include_cash=True, transaction_cost_bps=0.0, slippage_bps=0.0)

        all_budget_records = _simulate_trade_budget_rollout_records(
            policy_rollout,
            returns,
            benchmark_returns=None,
            config=config,
            max_rebalances=2,
        )

        self.assertAlmostEqual(all_budget_records.iloc[0]["gross_portfolio_return"], 0.07)
        self.assertAlmostEqual(all_budget_records.iloc[1]["gross_portfolio_return"], 0.0)
        self.assertAlmostEqual(all_budget_records.iloc[2]["gross_portfolio_return"], 0.04)

    def test_summarize_rebalance_impact_records_tracks_post_first_damage(self) -> None:
        records = pd.DataFrame(
            {
                "rebalance_number": [1.0, 2.0, 4.0],
                "turnover": [0.1, 0.2, 0.3],
                "proposed_turnover": [0.1, 0.2, 0.3],
                "interval_steps": [2.0, 1.0, 1.0],
                "first_step_return_delta": [0.02, -0.01, -0.03],
                "interval_total_return_delta": [0.02, -0.05, -0.02],
                "interval_gross_total_return_delta": [0.02, -0.05, -0.02],
            }
        )

        summary = _summarize_rebalance_impact_records(records)

        self.assertAlmostEqual(summary["rebalance_count"], 3.0)
        self.assertAlmostEqual(summary["average_interval_total_return_delta"], (-0.05) / 3.0)
        self.assertAlmostEqual(summary["first_rebalance_average_interval_total_return_delta"], 0.02)
        self.assertAlmostEqual(summary["post_first_rebalances_average_interval_total_return_delta"], -0.035)
        self.assertAlmostEqual(summary["post_third_rebalances_average_interval_total_return_delta"], -0.02)

    def test_build_named_metric_comparison_uses_requested_labels(self) -> None:
        left_summary = pd.Series({"total_return": 0.12})
        right_summary = pd.Series({"total_return": 0.10})

        comparison = _build_named_metric_comparison(left_summary, right_summary, left_label="policy", right_label="static_hold")

        self.assertAlmostEqual(comparison["total_return"]["policy"], 0.12)
        self.assertAlmostEqual(comparison["total_return"]["static_hold"], 0.10)
        self.assertAlmostEqual(comparison["total_return"]["difference"], 0.02)

    def test_build_common_metric_comparison_computes_difference(self) -> None:
        policy_summary = pd.Series({"total_return": 0.12, "max_drawdown": -0.08})
        optimizer_summary = pd.Series({"total_return": 0.10, "max_drawdown": -0.05})

        comparison = _build_common_metric_comparison(policy_summary, optimizer_summary)

        self.assertAlmostEqual(comparison["total_return"]["difference"], 0.02)
        self.assertAlmostEqual(comparison["max_drawdown"]["difference"], -0.03)

    def test_aggregate_comparison_maps_means_numeric_fields(self) -> None:
        comparisons = [
            {
                "total_return": {"policy": 0.10, "optimizer": 0.20, "difference": -0.10},
                "average_turnover": {"policy": 0.18, "optimizer": 0.05, "difference": 0.13},
            },
            {
                "total_return": {"policy": 0.00, "optimizer": 0.10, "difference": -0.10},
                "average_turnover": {"policy": 0.15, "optimizer": 0.04, "difference": 0.11},
            },
        ]

        aggregated = _aggregate_comparison_maps(comparisons)

        self.assertAlmostEqual(aggregated["total_return"]["policy"], 0.05)
        self.assertAlmostEqual(aggregated["total_return"]["optimizer"], 0.15)
        self.assertAlmostEqual(aggregated["average_turnover"]["difference"], 0.12)

    def test_aggregate_named_comparison_maps_supports_static_hold_fields(self) -> None:
        comparisons = [
            {
                "total_return": {"policy": 0.12, "static_hold": 0.10, "difference": 0.02},
            },
            {
                "total_return": {"policy": 0.08, "static_hold": 0.07, "difference": 0.01},
            },
        ]

        aggregated = _aggregate_named_comparison_maps(comparisons, left_field="policy", right_field="static_hold")

        self.assertAlmostEqual(aggregated["total_return"]["policy"], 0.10)
        self.assertAlmostEqual(aggregated["total_return"]["static_hold"], 0.085)
        self.assertAlmostEqual(aggregated["total_return"]["difference"], 0.015)

    def test_generate_walk_forward_splits_returns_ordered_recent_folds(self) -> None:
        dates = pd.date_range("2021-01-01", periods=120, freq="B")
        returns = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.002] * 40 + [0.01, -0.012] * 20 + [0.003] * 40,
                "MEAN_REVERSION": [0.001] * 40 + [-0.008, 0.01] * 20 + [0.002] * 40,
            },
            index=dates,
        )
        benchmark = pd.Series([0.0015] * 40 + [0.012, -0.014] * 20 + [0.0025] * 40, index=dates, dtype=float)

        splits = generate_walk_forward_splits(
            returns,
            benchmark_returns=benchmark,
            lookback=5,
            validation_rows=18,
            test_rows=12,
            step_rows=12,
            max_splits=3,
        )

        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[-1].test.end_index, len(returns))
        self.assertLess(splits[0].test.end_index, splits[1].test.end_index)
        self.assertLess(splits[1].test.end_index, splits[2].test.end_index)
        for split in splits:
            self.assertEqual(split.train.start_index, 0)
            self.assertEqual(split.train.end_index, split.validation.start_index)
            self.assertEqual(split.validation.end_index, split.test.start_index)


if __name__ == "__main__":
    unittest.main()