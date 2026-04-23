from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_frozen_ppo_baseline.py"
MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_frozen_ppo_baseline", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("Could not load evaluate_frozen_ppo_baseline module for tests.")
evaluate_frozen_ppo_baseline = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(evaluate_frozen_ppo_baseline)


class EvaluateFrozenPpoBaselineTests(unittest.TestCase):
    def _write_summary(self, path: Path, *, total_return: float, relative_total_return: float, cash_weight: float, trend_weight: float, turnover: float, rebalances: float) -> None:
        payload = {
            "total_return": total_return,
            "relative_total_return": relative_total_return,
            "average_cash_weight": cash_weight,
            "average_target_weight_TREND_FOLLOWING": trend_weight,
            "total_turnover": turnover,
            "rebalance_count": rebalances,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_seed_run(self, artifacts_dir: Path, prefix: str) -> None:
        seed_dir = artifacts_dir / f"{prefix}7"
        fold_dir = seed_dir / "fold_1"
        fold_dir.mkdir(parents=True)
        (seed_dir / "split_windows.json").write_text(
            json.dumps(
                {
                    "walk_forward_folds": [
                        {
                            "validation": {"start": "2024-01-01", "end": "2024-03-31"},
                            "test": {"start": "2024-04-01", "end": "2024-06-30"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self._write_summary(
            fold_dir / "validation_policy_summary.json",
            total_return=0.08,
            relative_total_return=0.02,
            cash_weight=0.20,
            trend_weight=0.65,
            turnover=0.40,
            rebalances=4.0,
        )
        self._write_summary(
            fold_dir / "validation_static_hold_summary.json",
            total_return=0.06,
            relative_total_return=0.01,
            cash_weight=0.30,
            trend_weight=0.55,
            turnover=0.00,
            rebalances=0.0,
        )
        self._write_summary(
            fold_dir / "validation_optimizer_summary.json",
            total_return=0.07,
            relative_total_return=0.015,
            cash_weight=0.25,
            trend_weight=0.60,
            turnover=0.10,
            rebalances=1.0,
        )
        self._write_summary(
            fold_dir / "test_policy_summary.json",
            total_return=0.04,
            relative_total_return=0.01,
            cash_weight=0.18,
            trend_weight=0.68,
            turnover=0.30,
            rebalances=3.0,
        )
        self._write_summary(
            fold_dir / "test_static_hold_summary.json",
            total_return=0.03,
            relative_total_return=0.005,
            cash_weight=0.28,
            trend_weight=0.58,
            turnover=0.00,
            rebalances=0.0,
        )
        self._write_summary(
            fold_dir / "test_optimizer_summary.json",
            total_return=0.035,
            relative_total_return=0.006,
            cash_weight=0.22,
            trend_weight=0.62,
            turnover=0.08,
            rebalances=1.0,
        )

    def _write_overlay_rollout(self, path: Path) -> None:
        pd.DataFrame(
            [
                {
                    "step_label": "2024-01-01",
                    "turnover": 0.20,
                    "execution_cost": 0.0,
                    "benchmark_return": -0.02,
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
        ).to_csv(path, index=False)

    def _write_overlay_seed_run(self, artifacts_dir: Path, prefix: str) -> None:
        seed_dir = artifacts_dir / f"{prefix}7"
        fold_dir = seed_dir / "fold_1"
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
        for phase in ["validation", "test"]:
            self._write_summary(
                fold_dir / f"{phase}_policy_summary.json",
                total_return=-0.075,
                relative_total_return=-0.056122448979591844,
                cash_weight=0.10,
                trend_weight=0.80,
                turnover=0.20,
                rebalances=1.0,
            )
            self._write_summary(
                fold_dir / f"{phase}_static_hold_summary.json",
                total_return=-0.05,
                relative_total_return=-0.030612244897959218,
                cash_weight=0.20,
                trend_weight=0.60,
                turnover=0.0,
                rebalances=0.0,
            )
            self._write_summary(
                fold_dir / f"{phase}_optimizer_summary.json",
                total_return=-0.06,
                relative_total_return=-0.04081632653061229,
                cash_weight=0.15,
                trend_weight=0.70,
                turnover=0.05,
                rebalances=1.0,
            )
            self._write_overlay_rollout(fold_dir / f"{phase}_policy_rollout.csv")

    def test_requested_splits_requires_at_least_one_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            evaluate_frozen_ppo_baseline._requested_splits(None, None)

    def test_main_supports_chrono_only_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifacts_dir = temp_path / "artifacts"
            artifacts_dir.mkdir()
            self._write_seed_run(artifacts_dir, "pilot_chrono_seed_")

            output_prefix = temp_path / "chrono_only_eval"
            result = evaluate_frozen_ppo_baseline.main(
                [
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--chrono-prefix",
                    "pilot_chrono_seed_",
                    "--output-prefix",
                    str(output_prefix),
                ]
            )

            self.assertEqual(result, 0)
            detail = pd.read_csv(output_prefix.with_name(f"{output_prefix.name}_detail.csv"))
            summary = json.loads(output_prefix.with_name(f"{output_prefix.name}_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(set(detail["split"]), {"chrono"})
            self.assertEqual(len(detail), 2)
            self.assertIn("chrono", summary["by_split"])
            self.assertNotIn("regime", summary["by_split"])

    def test_main_applies_robust_regime_only_runtime_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifacts_dir = temp_path / "artifacts"
            artifacts_dir.mkdir()
            self._write_overlay_seed_run(artifacts_dir, "pilot_regime_seed_")
            self._write_overlay_seed_run(artifacts_dir, "pilot_chrono_seed_")

            output_prefix = temp_path / "overlay_eval"
            result = evaluate_frozen_ppo_baseline.main(
                [
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--regime-prefix",
                    "pilot_regime_seed_",
                    "--chrono-prefix",
                    "pilot_chrono_seed_",
                    "--runtime-overlay",
                    "robust-regime-only",
                    "--output-prefix",
                    str(output_prefix),
                ]
            )

            self.assertEqual(result, 0)
            detail = pd.read_csv(output_prefix.with_name(f"{output_prefix.name}_detail.csv"))
            summary = json.loads(output_prefix.with_name(f"{output_prefix.name}_summary.json").read_text(encoding="utf-8"))

            chrono_rows = detail[detail["split"] == "chrono"]
            regime_rows = detail[detail["split"] == "regime"]

            self.assertTrue((chrono_rows["delta_total_return_vs_saved_policy"] == 0.0).all())
            self.assertTrue((chrono_rows["overlay_applied_steps"] == 0).all())
            self.assertTrue((regime_rows["delta_total_return_vs_saved_policy"] > 0.0).all())
            self.assertTrue((regime_rows["overlay_applied_steps"] == 1).all())
            self.assertEqual(summary["runtime_overlay"]["mode"], "robust-regime-only")
            self.assertEqual(summary["by_split"]["chrono"]["mean_delta_total_return_vs_saved_policy"], 0.0)
            self.assertGreater(summary["by_split"]["regime"]["mean_delta_total_return_vs_saved_policy"], 0.0)


if __name__ == "__main__":
    unittest.main()