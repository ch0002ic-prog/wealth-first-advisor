from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} for tests.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


compare_ppo_eval_details = _load_module("compare_ppo_eval_details", "scripts/compare_ppo_eval_details.py")
analyze_ppo_weak_windows = _load_module("analyze_ppo_weak_windows", "scripts/analyze_ppo_weak_windows.py")
analyze_regimeobs_policy_drift = _load_module("analyze_regimeobs_policy_drift", "scripts/analyze_regimeobs_policy_drift.py")
analyze_regimeobs_observation_ablation = _load_module(
    "analyze_regimeobs_observation_ablation",
    "scripts/analyze_regimeobs_observation_ablation.py",
)
analyze_regimeobs_selective_early_counterfactual = _load_module(
    "analyze_regimeobs_selective_early_counterfactual",
    "scripts/analyze_regimeobs_selective_early_counterfactual.py",
)
analyze_regimeobs_repair_surface_compatibility = _load_module(
    "analyze_regimeobs_repair_surface_compatibility",
    "scripts/analyze_regimeobs_repair_surface_compatibility.py",
)
postmortem_frozen_ppo_window = _load_module(
    "postmortem_frozen_ppo_window",
    "scripts/postmortem_frozen_ppo_window.py",
)
evaluate_frozen_ppo_baseline = _load_module(
    "evaluate_frozen_ppo_baseline",
    "scripts/evaluate_frozen_ppo_baseline.py",
)


class PpoAnalysisScriptTests(unittest.TestCase):
    def _write_postmortem_seed_run(self, artifacts_dir: Path, prefix: str) -> None:
        seed_dir = artifacts_dir / f"{prefix}7"
        fold_dir = seed_dir / "fold_01"
        fold_dir.mkdir(parents=True)

        (seed_dir / "split_windows.json").write_text(
            json.dumps(
                {
                    "walk_forward_folds": [
                        {
                            "validation": {"start": "2024-01-01", "end": "2024-01-31"},
                            "test": {"start": "2024-02-01", "end": "2024-02-29"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        policy_summary = {
            "total_return": -0.075,
            "relative_total_return": -0.056122448979591844,
            "average_cash_weight": 0.10,
            "average_target_weight_TREND_FOLLOWING": 0.80,
            "total_turnover": 0.20,
            "rebalance_count": 1.0,
            "late_defensive_posture_penalty_application_rate": 0.0,
        }
        static_hold_summary = {
            "total_return": -0.05,
            "relative_total_return": -0.030612244897959218,
            "average_cash_weight": 0.20,
            "average_target_weight_TREND_FOLLOWING": 0.60,
            "total_turnover": 0.0,
            "rebalance_count": 0.0,
        }
        optimizer_summary = {
            "total_return": -0.06,
            "relative_total_return": -0.04081632653061229,
            "average_cash_weight": 0.15,
            "average_target_weight_TREND_FOLLOWING": 0.70,
            "total_turnover": 0.05,
            "rebalance_count": 1.0,
        }
        (fold_dir / "test_policy_summary.json").write_text(json.dumps(policy_summary), encoding="utf-8")
        (fold_dir / "test_static_hold_summary.json").write_text(json.dumps(static_hold_summary), encoding="utf-8")
        (fold_dir / "test_optimizer_summary.json").write_text(json.dumps(optimizer_summary), encoding="utf-8")

        for label, difference in [("trade_budget_1", 0.01), ("trade_budget_2", 0.015), ("trade_budget_3", 0.02)]:
            (fold_dir / f"policy_vs_{label}_comparison.json").write_text(
                json.dumps({"test": {"total_return": {"difference": difference}}}),
                encoding="utf-8",
            )

        pd.DataFrame(
            [
                {
                    "step_label": "2024-02-01",
                    "turnover": 0.20,
                    "execution_cost": 0.0,
                    "benchmark_return": -0.02,
                    "benchmark_regime_cumulative_return": 0.01,
                    "benchmark_regime_drawdown": -0.12,
                    "pre_trade_weight_CASH": 0.20,
                    "pre_trade_weight_TREND_FOLLOWING": 0.60,
                    "pre_trade_weight_MEAN_REVERSION": 0.20,
                    "target_weight_CASH": 0.10,
                    "target_weight_TREND_FOLLOWING": 0.80,
                    "target_weight_MEAN_REVERSION": 0.10,
                    "asset_return_CASH": 0.0,
                    "asset_return_TREND_FOLLOWING": -0.10,
                    "asset_return_MEAN_REVERSION": 0.05,
                }
            ]
        ).to_csv(fold_dir / "test_policy_rollout.csv", index=False)

        pd.DataFrame(
            [
                {
                    "rebalance_event": 1,
                    "rebalance_number": 1.0,
                    "step_label": "2024-02-01",
                    "interval_end_label": "2024-02-01",
                    "interval_steps": 1.0,
                    "turnover": 0.20,
                    "proposed_turnover": 0.20,
                    "policy_first_step_return": -0.075,
                    "hold_first_step_return": -0.05,
                    "first_step_return_delta": -0.025,
                    "policy_interval_total_return": -0.075,
                    "hold_interval_total_return": -0.05,
                    "interval_total_return_delta": -0.025,
                    "policy_interval_gross_total_return": -0.075,
                    "hold_interval_gross_total_return": -0.05,
                    "interval_gross_total_return_delta": -0.025,
                    "policy_interval_relative_total_return": -0.056122448979591844,
                    "hold_interval_relative_total_return": -0.030612244897959218,
                    "interval_relative_total_return_delta": -0.025510204081632626,
                    "pre_trade_weight_TREND_FOLLOWING": 0.60,
                    "pre_trade_weight_MEAN_REVERSION": 0.20,
                    "pre_trade_weight_CASH": 0.20,
                    "target_weight_TREND_FOLLOWING": 0.80,
                    "target_weight_MEAN_REVERSION": 0.10,
                    "target_weight_CASH": 0.10,
                }
            ]
        ).to_csv(fold_dir / "test_rebalance_impacts.csv", index=False)

    def test_compare_ppo_eval_details_handles_chrono_only_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_path = temp_path / "base.csv"
            candidate_path = temp_path / "candidate.csv"
            output_prefix = temp_path / "comparison"

            base_path.write_text(
                "split,seed,fold,phase,phase_start,phase_end,delta_total_return_vs_static_hold,delta_total_return_vs_optimizer\n"
                "chrono,7,fold_01,test,2024-01-01,2024-03-31,-0.02,-0.01\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                "split,seed,fold,phase,phase_start,phase_end,delta_total_return_vs_static_hold,delta_total_return_vs_optimizer\n"
                "chrono,7,fold_01,test,2024-01-01,2024-03-31,0.01,0.02\n",
                encoding="utf-8",
            )

            result = compare_ppo_eval_details.main(
                [
                    "--base-detail-csv",
                    str(base_path),
                    "--candidate-detail-csv",
                    str(candidate_path),
                    "--output-prefix",
                    str(output_prefix),
                ]
            )

            self.assertEqual(result, 0)
            summary = json.loads(output_prefix.with_name(f"{output_prefix.name}_summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["chrono_mean_improve_vs_static_hold"], 0.03)
            self.assertIsNone(summary["regime_mean_improve_vs_static_hold"])
            self.assertEqual(summary["mean_improve_vs_static_hold_by_split"], {"chrono": 0.03})

    def test_seed_discovery_ignores_suffixed_experiment_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            (artifacts_dir / "pilot_seed_7").mkdir()
            (artifacts_dir / "pilot_seed_17_secondarycap004_catchup").mkdir()

            postmortem_dirs = postmortem_frozen_ppo_window._discover_seed_dirs(artifacts_dir, "pilot_seed_")
            evaluate_dirs = evaluate_frozen_ppo_baseline._discover_seed_dirs(artifacts_dir, "pilot_seed_")

            self.assertEqual([path.name for path in postmortem_dirs], ["pilot_seed_7"])
            self.assertEqual([path.name for path in evaluate_dirs], ["pilot_seed_7"])

    def test_analyze_ppo_weak_windows_requires_at_least_one_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            analyze_ppo_weak_windows._requested_splits(None, None)

        self.assertEqual(
            analyze_ppo_weak_windows._requested_splits(None, "candidate_chrono_seed_"),
            [("chrono", "candidate_chrono_seed_")],
        )

    def test_regimeobs_scripts_fall_back_to_archived_artifacts(self) -> None:
        modules = [
            analyze_regimeobs_policy_drift,
            analyze_regimeobs_observation_ablation,
            analyze_regimeobs_selective_early_counterfactual,
            analyze_regimeobs_repair_surface_compatibility,
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifacts_dir = temp_path / "artifacts"
            archive_dir = artifacts_dir / "archive" / "ruled_out_2026-04-16"
            archived_compare_detail = archive_dir / "ppo_frozen_live_baseline_regimeobs_vs_current_detail.csv"
            archived_compare_detail.parent.mkdir(parents=True)
            archived_compare_detail.write_text("split,seed,fold,phase\n", encoding="utf-8")

            archived_run_dir = archive_dir / "ppo_frozen_live_baseline_regimeobs_chrono_seed_7" / "fold_1"
            archived_run_dir.mkdir(parents=True)
            archived_rollout = archived_run_dir / "test_policy_rollout.csv"
            archived_rollout.write_text("step_label,turnover,executed_rebalances\n", encoding="utf-8")
            archived_summary = archived_run_dir / "test_policy_summary.json"
            archived_summary.write_text("{}", encoding="utf-8")

            original_values: list[tuple[object, Path, Path]] = []
            for module in modules:
                original_values.append((module, module.ARTIFACTS, module.RULED_OUT_ARCHIVE))
                module.ARTIFACTS = artifacts_dir
                module.RULED_OUT_ARCHIVE = archive_dir

            try:
                row = {
                    "split": "chrono",
                    "seed": 7,
                    "fold": "fold_1",
                    "phase": "test",
                }

                self.assertEqual(
                    analyze_regimeobs_policy_drift._preferred_artifact_path(
                        "ppo_frozen_live_baseline_regimeobs_vs_current_detail.csv"
                    ),
                    archived_compare_detail,
                )
                self.assertEqual(
                    analyze_regimeobs_policy_drift._rollout_path(
                        analyze_regimeobs_policy_drift.CANDIDATE_PREFIX_BY_SPLIT,
                        row,
                    ),
                    archived_rollout,
                )
                self.assertEqual(
                    analyze_regimeobs_observation_ablation._artifact_dir(
                        analyze_regimeobs_observation_ablation.CANDIDATE_PREFIX_BY_SPLIT,
                        row,
                    ),
                    archived_run_dir,
                )
                self.assertEqual(
                    analyze_regimeobs_selective_early_counterfactual._candidate_rollout_path(row),
                    archived_rollout,
                )
                self.assertEqual(
                    analyze_regimeobs_selective_early_counterfactual._candidate_summary_path(row),
                    archived_summary,
                )
                self.assertEqual(
                    analyze_regimeobs_repair_surface_compatibility._candidate_rollout_path(row),
                    archived_rollout,
                )
            finally:
                for module, original_artifacts, original_archive in original_values:
                    module.ARTIFACTS = original_artifacts
                    module.RULED_OUT_ARCHIVE = original_archive

    def test_postmortem_window_runtime_overlay_respects_split_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifacts_dir = temp_path / "artifacts"
            artifacts_dir.mkdir()
            self._write_postmortem_seed_run(artifacts_dir, "pilot_seed_")

            regime_output = temp_path / "regime_postmortem"
            regime_result = postmortem_frozen_ppo_window.main(
                [
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--prefix",
                    "pilot_seed_",
                    "--split-label",
                    "regime",
                    "--split-name",
                    "regime",
                    "--runtime-overlay",
                    "robust-regime-only",
                    "--output-prefix",
                    str(regime_output),
                ]
            )

            self.assertEqual(regime_result, 0)
            regime_detail = pd.read_csv(regime_output.with_name(f"{regime_output.name}_detail.csv"))
            regime_summary = json.loads(
                regime_output.with_name(f"{regime_output.name}_summary.json").read_text(encoding="utf-8")
            )

            self.assertAlmostEqual(float(regime_detail.loc[0, "policy_total_return"]), -0.05, places=6)
            self.assertGreater(float(regime_detail.loc[0, "delta_total_return_vs_saved_policy"]), 0.0)
            self.assertEqual(int(regime_detail.loc[0, "overlay_applied_steps"]), 1)
            self.assertEqual(int(regime_detail.loc[0, "overlay_suppressed_steps"]), 1)
            self.assertTrue(regime_summary["runtime_overlay"]["active"])
            self.assertEqual(regime_summary["runtime_overlay"]["mode"], "robust-regime-only")
            self.assertEqual(regime_summary["event_summary"]["rows"], 0)

            chrono_output = temp_path / "chrono_postmortem"
            chrono_result = postmortem_frozen_ppo_window.main(
                [
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--prefix",
                    "pilot_seed_",
                    "--split-label",
                    "chrono",
                    "--split-name",
                    "chrono",
                    "--runtime-overlay",
                    "robust-regime-only",
                    "--output-prefix",
                    str(chrono_output),
                ]
            )

            self.assertEqual(chrono_result, 0)
            chrono_detail = pd.read_csv(chrono_output.with_name(f"{chrono_output.name}_detail.csv"))
            chrono_summary = json.loads(
                chrono_output.with_name(f"{chrono_output.name}_summary.json").read_text(encoding="utf-8")
            )

            self.assertAlmostEqual(float(chrono_detail.loc[0, "delta_total_return_vs_saved_policy"]), 0.0, places=9)
            self.assertEqual(int(chrono_detail.loc[0, "overlay_applied_steps"]), 0)
            self.assertFalse(chrono_summary["runtime_overlay"]["active"])
            self.assertEqual(chrono_summary["event_summary"]["rows"], 1)
            self.assertGreater(
                float(regime_detail.loc[0, "delta_total_return_vs_trade_budget_1"]),
                float(chrono_detail.loc[0, "delta_total_return_vs_trade_budget_1"]),
            )

    def test_postmortem_runtime_overlay_soft_premr_replays_smoothed_weights(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "step_label": "2024-02-01",
                    "benchmark_return": 0.0,
                    "benchmark_regime_cumulative_return": 0.03,
                    "benchmark_regime_drawdown": -0.02,
                    "pre_trade_weight_CASH": 0.20,
                    "pre_trade_weight_TREND_FOLLOWING": 0.60,
                    "pre_trade_weight_MEAN_REVERSION": 0.20,
                    "target_weight_CASH": 0.15,
                    "target_weight_TREND_FOLLOWING": 0.65,
                    "target_weight_MEAN_REVERSION": 0.20,
                    "asset_return_CASH": 0.0,
                    "asset_return_TREND_FOLLOWING": 0.0,
                    "asset_return_MEAN_REVERSION": 0.0,
                },
                {
                    "step_label": "2024-02-02",
                    "benchmark_return": 0.0,
                    "benchmark_regime_cumulative_return": 0.01,
                    "benchmark_regime_drawdown": -0.02,
                    "pre_trade_weight_CASH": 0.15,
                    "pre_trade_weight_TREND_FOLLOWING": 0.65,
                    "pre_trade_weight_MEAN_REVERSION": 0.20,
                    "target_weight_CASH": 0.05,
                    "target_weight_TREND_FOLLOWING": 0.80,
                    "target_weight_MEAN_REVERSION": 0.15,
                    "asset_return_CASH": 0.0,
                    "asset_return_TREND_FOLLOWING": -0.10,
                    "asset_return_MEAN_REVERSION": 0.05,
                },
            ]
        )

        replayed_frame, metrics = postmortem_frozen_ppo_window._replay_runtime_overlay_frame(
            frame,
            runtime_overlay="soft-premr",
            trend_symbol="TREND_FOLLOWING",
        )

        self.assertEqual(int(metrics["overlay_applied_steps"]), 1)
        self.assertEqual(int(metrics["overlay_suppressed_steps"]), 0)
        self.assertAlmostEqual(float(metrics["policy_turnover"]), 0.20, places=6)
        self.assertAlmostEqual(float(replayed_frame.loc[1, "target_weight_TREND_FOLLOWING"]), 0.7625, places=6)
        self.assertAlmostEqual(float(replayed_frame.loc[1, "target_weight_MEAN_REVERSION"]), 0.1625, places=6)
        self.assertAlmostEqual(float(replayed_frame.loc[1, "target_weight_CASH"]), 0.075, places=6)

    def test_evaluate_runtime_overlay_soft_premr_replays_smoothed_weights(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "step_label": "2024-02-01",
                    "benchmark_return": 0.0,
                    "benchmark_regime_cumulative_return": 0.03,
                    "benchmark_regime_drawdown": -0.02,
                    "pre_trade_weight_CASH": 0.20,
                    "pre_trade_weight_TREND_FOLLOWING": 0.60,
                    "pre_trade_weight_MEAN_REVERSION": 0.20,
                    "target_weight_CASH": 0.15,
                    "target_weight_TREND_FOLLOWING": 0.65,
                    "target_weight_MEAN_REVERSION": 0.20,
                    "asset_return_CASH": 0.0,
                    "asset_return_TREND_FOLLOWING": 0.0,
                    "asset_return_MEAN_REVERSION": 0.0,
                },
                {
                    "step_label": "2024-02-02",
                    "benchmark_return": 0.0,
                    "benchmark_regime_cumulative_return": 0.01,
                    "benchmark_regime_drawdown": -0.02,
                    "pre_trade_weight_CASH": 0.15,
                    "pre_trade_weight_TREND_FOLLOWING": 0.65,
                    "pre_trade_weight_MEAN_REVERSION": 0.20,
                    "target_weight_CASH": 0.05,
                    "target_weight_TREND_FOLLOWING": 0.80,
                    "target_weight_MEAN_REVERSION": 0.15,
                    "asset_return_CASH": 0.0,
                    "asset_return_TREND_FOLLOWING": -0.10,
                    "asset_return_MEAN_REVERSION": 0.05,
                },
            ]
        )

        metrics = evaluate_frozen_ppo_baseline._replay_runtime_overlay(
            frame,
            runtime_overlay="soft-premr",
            trend_symbol="TREND_FOLLOWING",
        )

        self.assertEqual(int(metrics["overlay_applied_steps"]), 1)
        self.assertEqual(int(metrics["overlay_suppressed_steps"]), 0)
        self.assertAlmostEqual(float(metrics["policy_turnover"]), 0.20, places=6)
        self.assertAlmostEqual(float(metrics["policy_cash_weight"]), 0.1125, places=6)
        self.assertAlmostEqual(float(metrics["policy_trend_weight"]), 0.70625, places=6)


if __name__ == "__main__":
    unittest.main()