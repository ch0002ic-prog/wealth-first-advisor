from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from wealth_first.main2 import Main2Config, Main2SPYEnv, _append_main2_progress_event, build_main2_feature_frame, build_policy_total_return_comparison_summary, compute_main2_dynamic_floor_components


class Main2Tests(unittest.TestCase):
    def test_append_main2_progress_event_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_path = Path(temp_dir) / "main2.progress.jsonl"

            _append_main2_progress_event(
                progress_path,
                "training_checkpoint",
                split="chrono",
                seed=7,
                fold="fold_01",
                step=32,
                progress_fraction=0.5,
                validation_slice_scores=[0.12, 0.15],
            )

            lines = progress_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["event_type"], "training_checkpoint")
            self.assertEqual(payload["split"], "chrono")
            self.assertEqual(payload["seed"], 7)
            self.assertEqual(payload["fold"], "fold_01")
            self.assertEqual(payload["step"], 32)
            self.assertEqual(payload["validation_slice_scores"], [0.12, 0.15])
            self.assertIn("timestamp", payload)

    def test_build_main2_feature_frame_uses_lagged_returns(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        spy_returns = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.0, 0.01], index=dates, dtype=float)

        features = build_main2_feature_frame(spy_returns)

        self.assertFalse(features.isna().any().any())
        self.assertAlmostEqual(features.loc[dates[3], "ret_1"], spy_returns.iloc[2])
        self.assertNotAlmostEqual(features.loc[dates[3], "ret_1"], spy_returns.iloc[3])

    def test_build_main2_feature_frame_includes_long_window_features(self) -> None:
        import numpy as np
        # Build a series with a clear prolonged bear (252 days down) followed by recovery.
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = [-0.001] * 252 + [0.005] * 48
        spy_returns = pd.Series(returns, index=dates, dtype=float)

        features = build_main2_feature_frame(spy_returns)

        self.assertIn("ret_252", features.columns, "ret_252 feature must be present")
        self.assertIn("drawdown_252", features.columns, "drawdown_252 feature must be present")
        self.assertFalse(features.isna().any().any())

        # ret_252 at day 253 should be the lagged cumulative product of the first 252 returns (all -0.001)
        expected_ret252 = (0.999 ** 252) - 1.0
        self.assertAlmostEqual(features.loc[dates[252], "ret_252"], expected_ret252, places=4)

        # drawdown_252 at the recovery start should be very negative (price far below 252-day rolling peak)
        self.assertLess(features.loc[dates[252], "drawdown_252"], -0.10,
                        "drawdown_252 should be deeply negative after prolonged decline")

    def test_main2_env_respects_no_trade_band(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        features = pd.DataFrame({"ret_1": [0.0, 0.0, 0.0], "vol_5": [0.0, 0.0, 0.0]}, index=dates)
        spy_returns = pd.Series([0.0, 0.0, 0.0], index=dates, dtype=float)
        config = Main2Config(
            lookback=2,
            action_bins=2,
            action_smoothing=0.5,
            no_trade_band=0.31,
            initial_spy_weight=0.4,
            min_spy_weight=0.0,
            max_spy_weight=1.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
        env = Main2SPYEnv(features, spy_returns, start_index=0, end_index=1, config=config, random_episode_start=False)

        env.reset()
        _, _, _, info = env.step(1)

        self.assertAlmostEqual(float(info["target_spy_weight"]), 0.4)
        self.assertAlmostEqual(float(info["turnover"]), 0.0)

    def test_main2_env_applies_smoothing_when_trade_executes(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        features = pd.DataFrame({"ret_1": [0.0, 0.0, 0.0], "vol_5": [0.0, 0.0, 0.0]}, index=dates)
        spy_returns = pd.Series([0.0, 0.0, 0.0], index=dates, dtype=float)
        config = Main2Config(
            lookback=2,
            action_bins=2,
            action_smoothing=0.5,
            no_trade_band=0.29,
            initial_spy_weight=0.4,
            min_spy_weight=0.0,
            max_spy_weight=1.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
        env = Main2SPYEnv(features, spy_returns, start_index=0, end_index=1, config=config, random_episode_start=False)

        env.reset()
        _, _, _, info = env.step(1)

        self.assertAlmostEqual(float(info["target_spy_weight"]), 0.7)
        self.assertAlmostEqual(float(info["turnover"]), 0.3)

    def test_main2_env_action_grid_respects_spy_weight_bounds(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        features = pd.DataFrame({"ret_1": [0.0, 0.0, 0.0], "vol_5": [0.0, 0.0, 0.0]}, index=dates)
        spy_returns = pd.Series([0.0, 0.0, 0.0], index=dates, dtype=float)
        config = Main2Config(
            lookback=2,
            action_bins=5,
            initial_spy_weight=1.0,
            min_spy_weight=0.35,
            max_spy_weight=0.85,
        )
        env = Main2SPYEnv(features, spy_returns, start_index=0, end_index=1, config=config, random_episode_start=False)

        self.assertAlmostEqual(float(env.action_grid[0]), 0.35)
        self.assertAlmostEqual(float(env.action_grid[-1]), 0.85)

        env.reset()
        self.assertAlmostEqual(env.current_spy_weight, 0.85)

    def test_main2_env_applies_trend_floor_when_recent_path_is_strong(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        features = pd.DataFrame({"ret_1": [0.0] * 6, "vol_5": [0.0] * 6}, index=dates)
        spy_returns = pd.Series([0.02, 0.01, 0.015, 0.01, 0.0, 0.0], index=dates, dtype=float)
        config = Main2Config(
            lookback=2,
            action_bins=3,
            initial_spy_weight=0.8,
            min_spy_weight=0.65,
            max_spy_weight=1.0,
            trend_floor_min_spy_weight=0.9,
            trend_floor_lookback=4,
            trend_floor_return_threshold=0.03,
            trend_floor_drawdown_threshold=-0.02,
            action_smoothing=0.0,
            no_trade_band=0.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
        env = Main2SPYEnv(features, spy_returns, start_index=4, end_index=4, config=config, random_episode_start=False)

        env.reset()
        _, _, _, info = env.step(0)

        self.assertAlmostEqual(float(info["dynamic_min_spy_weight"]), 0.9)
        self.assertAlmostEqual(float(info["target_spy_weight"]), 0.9)

    def test_main2_env_bear_oversample_weight_biases_starts_toward_drawdown(self) -> None:
        # Build a return series: 50 up days then 50 down days.
        # With bear_oversample_weight>0, resets should favour starts in the down half.
        import numpy as np
        n = 100
        dates = pd.date_range("2024-01-01", periods=n + 2, freq="B")
        returns = [0.01] * 50 + [-0.01] * 50
        spy_returns = pd.Series(returns + [0.0, 0.0], index=dates, dtype=float)
        features = pd.DataFrame({"ret_1": [0.0] * (n + 2), "vol_5": [0.0] * (n + 2)}, index=dates)
        config = Main2Config(
            lookback=2, action_bins=2, episode_length=10,
            bear_oversample_weight=20.0,
            trend_floor_min_spy_weight=None, participation_floor_min_spy_weight=None,
            recovery_floor_min_spy_weight=None,
        )
        env = Main2SPYEnv(features, spy_returns, start_index=0, end_index=n - 1, config=config, random_episode_start=True)

        starts = []
        for i in range(500):
            env.reset()
            starts.append(env.current_index)

        # Majority of episode starts should be in the down half (index >= 50)
        bear_starts = sum(1 for s in starts if s >= 50)
        self.assertGreater(bear_starts / len(starts), 0.55, "Bear oversample should bias starts toward drawdown days")

    def test_main2_env_drawdown_penalty_penalises_reward_in_drawdown(self) -> None:
        # Two steps: first step produces a 2% loss (establishes drawdown), second step also loses 2%.
        # With drawdown_penalty>0 the second step should have a more-negative reward than without it.
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        features = pd.DataFrame({"ret_1": [0.0] * 4, "vol_5": [0.0] * 4}, index=dates)
        spy_returns = pd.Series([-0.02, -0.02, -0.02, -0.02], index=dates, dtype=float)
        base_kwargs = dict(
            lookback=2, action_bins=2, action_smoothing=0.0, no_trade_band=0.0,
            initial_spy_weight=1.0, min_spy_weight=0.0, max_spy_weight=1.0,
            transaction_cost_bps=0.0, slippage_bps=0.0,
            trend_floor_min_spy_weight=None, participation_floor_min_spy_weight=None,
            recovery_floor_min_spy_weight=None,
        )

        env_no_penalty = Main2SPYEnv(
            features, spy_returns, start_index=0, end_index=3,
            config=Main2Config(**base_kwargs, drawdown_penalty=0.0), random_episode_start=False,
        )
        env_penalty = Main2SPYEnv(
            features, spy_returns, start_index=0, end_index=3,
            config=Main2Config(**base_kwargs, drawdown_penalty=2.0), random_episode_start=False,
        )

        env_no_penalty.reset()
        env_penalty.reset()
        # Step 1: both envs take same action (full SPY, grid index 1)
        env_no_penalty.step(1)
        env_penalty.step(1)
        # Step 2: now in a drawdown — reward should be lower with penalty
        _, r_no_penalty, _, _ = env_no_penalty.step(1)
        _, r_penalty, _, _ = env_penalty.step(1)

        self.assertLess(r_penalty, r_no_penalty)

    def test_main2_dynamic_floor_components_apply_participation_floor_on_long_trend_pullback(self) -> None:
        dates = pd.date_range("2024-01-01", periods=140, freq="B")
        spy_returns = pd.Series(([0.002] * 110) + ([-0.001] * 15) + ([0.001] * 15), index=dates, dtype=float)
        config = Main2Config(
            participation_floor_min_spy_weight=0.85,
            participation_floor_lookback=126,
            participation_floor_return_threshold=0.02,
            participation_floor_ma_gap_threshold=0.02,
            participation_floor_drawdown_threshold=-0.05,
            trend_floor_min_spy_weight=None,
        )

        components = compute_main2_dynamic_floor_components(spy_returns, row_index=len(spy_returns), config=config)

        self.assertTrue(bool(components["participation_floor_active"]))
        self.assertAlmostEqual(float(components["dynamic_min_spy_weight"]), 0.85)

    def test_main2_dynamic_floor_components_apply_recovery_floor_after_deep_drawdown(self) -> None:
        dates = pd.date_range("2024-01-01", periods=160, freq="B")
        spy_returns = pd.Series(([0.001] * 50) + ([-0.006] * 60) + ([0.005] * 30) + ([0.003] * 20), index=dates, dtype=float)
        config = Main2Config(
            recovery_floor_min_spy_weight=0.8,
            recovery_floor_long_lookback=126,
            recovery_floor_drawdown_threshold=-0.10,
            recovery_floor_short_lookback=21,
            recovery_floor_return_threshold=0.01,
            recovery_floor_ma_gap_threshold=0.0,
            trend_floor_min_spy_weight=None,
        )

        components = compute_main2_dynamic_floor_components(spy_returns, row_index=len(spy_returns), config=config)

        self.assertTrue(bool(components["recovery_floor_active"]))
        self.assertAlmostEqual(float(components["dynamic_min_spy_weight"]), 0.8)

    def test_main2_dynamic_floor_components_relax_trend_floor_on_early_crack(self) -> None:
        dates = pd.date_range("2024-01-01", periods=147, freq="B")
        spy_returns = pd.Series(([0.0035] * 126) + ([-0.0015] * 21), index=dates, dtype=float)
        config = Main2Config(
            min_spy_weight=0.65,
            trend_floor_min_spy_weight=0.9,
            trend_floor_lookback=63,
            trend_floor_return_threshold=0.08,
            trend_floor_drawdown_threshold=-0.10,
            early_crack_floor_min_spy_weight=0.75,
            early_crack_long_lookback=126,
            early_crack_long_return_threshold=0.10,
            early_crack_long_ma_gap_threshold=0.04,
            early_crack_long_drawdown_threshold=-0.06,
            early_crack_trend_ma_gap_threshold=0.02,
            early_crack_short_lookback=21,
            early_crack_short_return_threshold=0.0,
            early_crack_short_ma_gap_threshold=0.0,
            early_crack_short_drawdown_threshold=-0.02,
            participation_floor_min_spy_weight=None,
            recovery_floor_min_spy_weight=None,
        )

        components = compute_main2_dynamic_floor_components(spy_returns, row_index=len(spy_returns), config=config)

        self.assertTrue(bool(components["trend_floor_active"]))
        self.assertTrue(bool(components["early_crack_active"]))
        self.assertAlmostEqual(float(components["effective_trend_floor_min_spy_weight"]), 0.75)
        self.assertAlmostEqual(float(components["dynamic_min_spy_weight"]), 0.75)

    def test_main2_dynamic_floor_components_early_crack_does_not_override_participation_floor(self) -> None:
        dates = pd.date_range("2024-01-01", periods=147, freq="B")
        spy_returns = pd.Series(([0.0035] * 126) + ([-0.0015] * 21), index=dates, dtype=float)
        config = Main2Config(
            min_spy_weight=0.65,
            trend_floor_min_spy_weight=0.9,
            trend_floor_lookback=63,
            trend_floor_return_threshold=0.08,
            trend_floor_drawdown_threshold=-0.10,
            participation_floor_min_spy_weight=0.85,
            participation_floor_lookback=126,
            participation_floor_return_threshold=0.10,
            participation_floor_ma_gap_threshold=0.04,
            participation_floor_drawdown_threshold=-0.10,
            early_crack_floor_min_spy_weight=0.75,
            early_crack_long_lookback=126,
            early_crack_long_return_threshold=0.10,
            early_crack_long_ma_gap_threshold=0.04,
            early_crack_long_drawdown_threshold=-0.06,
            early_crack_trend_ma_gap_threshold=0.02,
            early_crack_short_lookback=21,
            early_crack_short_return_threshold=0.0,
            early_crack_short_ma_gap_threshold=0.0,
            early_crack_short_drawdown_threshold=-0.02,
            recovery_floor_min_spy_weight=None,
        )

        components = compute_main2_dynamic_floor_components(spy_returns, row_index=len(spy_returns), config=config)

        self.assertTrue(bool(components["early_crack_active"]))
        self.assertTrue(bool(components["participation_floor_active"]))
        self.assertAlmostEqual(float(components["effective_trend_floor_min_spy_weight"]), 0.75)
        self.assertAlmostEqual(float(components["dynamic_min_spy_weight"]), 0.85)

    def test_main2_dynamic_floor_components_apply_constructive_crack_cap_after_recent_constructive_regime(self) -> None:
        dates = pd.date_range("2024-01-01", periods=137, freq="B")
        spy_returns = pd.Series(([0.0013] * 126) + ([-0.0015] * 8) + ([-0.003] * 3), index=dates, dtype=float)
        config = Main2Config(
            min_spy_weight=0.65,
            trend_floor_min_spy_weight=None,
            participation_floor_min_spy_weight=0.85,
            participation_floor_lookback=126,
            participation_floor_return_threshold=0.02,
            participation_floor_ma_gap_threshold=0.02,
            participation_floor_drawdown_threshold=-0.05,
            recovery_floor_min_spy_weight=None,
            constructive_crack_cap_min_spy_weight=0.75,
            constructive_crack_cap_recent_constructive_lookback=15,
            constructive_crack_cap_long_lookback=126,
            constructive_crack_cap_long_return_threshold=0.10,
            constructive_crack_cap_long_ma_gap_threshold=0.03,
            constructive_crack_cap_long_drawdown_threshold=-0.04,
            constructive_crack_cap_short_lookback=21,
            constructive_crack_cap_short_return_threshold=0.0,
            constructive_crack_cap_short_ma_gap_threshold=0.0,
            constructive_crack_cap_short_drawdown_threshold=-0.015,
            constructive_crack_cap_current_trend_return_cap=0.05,
            constructive_crack_cap_current_trend_ma_gap_cap=0.015,
            constructive_crack_cap_current_long_return_min=0.09,
            constructive_crack_cap_current_long_return_max=0.15,
        )

        components = compute_main2_dynamic_floor_components(spy_returns, row_index=len(spy_returns), config=config)

        self.assertTrue(bool(components["participation_floor_active"]))
        self.assertTrue(bool(components["constructive_crack_recent_constructive_active"]))
        self.assertTrue(bool(components["constructive_crack_cap_active"]))
        self.assertAlmostEqual(float(components["dynamic_min_spy_weight"]), 0.75)

    def test_main2_dynamic_floor_components_constructive_crack_cap_requires_recent_constructive_memory(self) -> None:
        dates = pd.date_range("2024-01-01", periods=137, freq="B")
        spy_returns = pd.Series(([0.0013] * 126) + ([-0.0015] * 8) + ([-0.003] * 3), index=dates, dtype=float)
        config = Main2Config(
            min_spy_weight=0.65,
            trend_floor_min_spy_weight=None,
            participation_floor_min_spy_weight=0.85,
            participation_floor_lookback=126,
            participation_floor_return_threshold=0.02,
            participation_floor_ma_gap_threshold=0.02,
            participation_floor_drawdown_threshold=-0.05,
            recovery_floor_min_spy_weight=None,
            constructive_crack_cap_min_spy_weight=0.75,
            constructive_crack_cap_recent_constructive_lookback=2,
            constructive_crack_cap_long_lookback=126,
            constructive_crack_cap_long_return_threshold=0.10,
            constructive_crack_cap_long_ma_gap_threshold=0.03,
            constructive_crack_cap_long_drawdown_threshold=-0.04,
            constructive_crack_cap_short_lookback=21,
            constructive_crack_cap_short_return_threshold=0.0,
            constructive_crack_cap_short_ma_gap_threshold=0.0,
            constructive_crack_cap_short_drawdown_threshold=-0.015,
            constructive_crack_cap_current_trend_return_cap=0.05,
            constructive_crack_cap_current_trend_ma_gap_cap=0.015,
            constructive_crack_cap_current_long_return_min=0.09,
            constructive_crack_cap_current_long_return_max=0.15,
        )

        components = compute_main2_dynamic_floor_components(spy_returns, row_index=len(spy_returns), config=config)

        self.assertTrue(bool(components["participation_floor_active"]))
        self.assertFalse(bool(components["constructive_crack_recent_constructive_active"]))
        self.assertFalse(bool(components["constructive_crack_cap_active"]))
        self.assertAlmostEqual(float(components["dynamic_min_spy_weight"]), 0.85)

    def test_main2_dynamic_floor_components_constructive_crack_cap_sticky_days_keeps_cap_active(self) -> None:
        # 126 constructive days + 8 mild crack + 3 deep crack: cap fires at row 137.
        # Extend with 1 large recovery day (+5%): short conditions recover, cap would NOT fire at row 138.
        # With sticky_days=1 the cap should remain active; with sticky_days=0 it should not.
        dates = pd.date_range("2024-01-01", periods=138, freq="B")
        spy_returns = pd.Series(
            ([0.0013] * 126) + ([-0.0015] * 8) + ([-0.003] * 3) + [0.05],
            index=dates,
            dtype=float,
        )
        base_kwargs = dict(
            min_spy_weight=0.65,
            trend_floor_min_spy_weight=None,
            participation_floor_min_spy_weight=0.85,
            participation_floor_lookback=126,
            participation_floor_return_threshold=0.02,
            participation_floor_ma_gap_threshold=0.02,
            participation_floor_drawdown_threshold=-0.05,
            recovery_floor_min_spy_weight=None,
            constructive_crack_cap_min_spy_weight=0.75,
            constructive_crack_cap_recent_constructive_lookback=15,
            constructive_crack_cap_long_lookback=126,
            constructive_crack_cap_long_return_threshold=0.10,
            constructive_crack_cap_long_ma_gap_threshold=0.03,
            constructive_crack_cap_long_drawdown_threshold=-0.04,
            constructive_crack_cap_short_lookback=21,
            constructive_crack_cap_short_return_threshold=0.0,
            constructive_crack_cap_short_ma_gap_threshold=0.0,
            constructive_crack_cap_short_drawdown_threshold=-0.015,
            constructive_crack_cap_current_trend_return_cap=0.05,
            constructive_crack_cap_current_trend_ma_gap_cap=0.015,
            constructive_crack_cap_current_long_return_min=0.09,
            constructive_crack_cap_current_long_return_max=0.15,
        )
        # Confirm cap fires at row 137 (the last -0.003 day).
        config_no_sticky = Main2Config(**base_kwargs, constructive_crack_cap_sticky_days=0)
        at_crack = compute_main2_dynamic_floor_components(spy_returns, row_index=137, config=config_no_sticky)
        self.assertTrue(bool(at_crack["constructive_crack_cap_active"]))
        self.assertAlmostEqual(float(at_crack["dynamic_min_spy_weight"]), 0.75)

        # Without sticky, cap is NOT active 1 day later (big +5% recovery day).
        after_recovery_no_sticky = compute_main2_dynamic_floor_components(
            spy_returns, row_index=138, config=config_no_sticky
        )
        self.assertFalse(bool(after_recovery_no_sticky["constructive_crack_cap_active"]))

        # With sticky_days=1, cap IS still active 1 day later.
        config_sticky_1 = Main2Config(**base_kwargs, constructive_crack_cap_sticky_days=1)
        after_recovery_sticky = compute_main2_dynamic_floor_components(
            spy_returns, row_index=138, config=config_sticky_1
        )
        self.assertTrue(bool(after_recovery_sticky["constructive_crack_cap_active"]))
        self.assertTrue(bool(after_recovery_sticky["constructive_crack_cap_sticky_active"]))
        self.assertAlmostEqual(float(after_recovery_sticky["dynamic_min_spy_weight"]), 0.75)

    def test_build_policy_total_return_comparison_summary_counts_row_wins(self) -> None:
        current = pd.DataFrame(
            {
                "split": ["chrono", "chrono"],
                "seed": [7, 7],
                "fold": ["fold_01", "fold_01"],
                "phase": ["validation", "test"],
                "phase_start": ["2024-01-01", "2024-02-01"],
                "phase_end": ["2024-01-31", "2024-02-29"],
                "policy_total_return": [0.05, 0.01],
                "delta_total_return_vs_static_hold": [0.01, -0.01],
            }
        )
        candidate = pd.DataFrame(
            {
                "split": ["chrono", "chrono"],
                "seed": [7, 7],
                "fold": ["fold_01", "fold_01"],
                "phase": ["validation", "test"],
                "phase_start": ["2024-01-01", "2024-02-01"],
                "phase_end": ["2024-01-31", "2024-02-29"],
                "policy_total_return": [0.07, -0.02],
                "delta_total_return_vs_static_hold": [0.03, -0.04],
            }
        )

        summary = build_policy_total_return_comparison_summary(candidate, current)

        self.assertEqual(summary["shared_rows"], 2)
        self.assertEqual(summary["main2_win_rows"], 1)
        self.assertEqual(summary["current_win_rows"], 1)


if __name__ == "__main__":
    unittest.main()