from __future__ import annotations

import unittest

import pandas as pd

import wealth_first
from wealth_first.compare import compare_return_streams


class SleeveComparisonTests(unittest.TestCase):
    def test_package_root_exposes_compare_lazily(self) -> None:
        self.assertTrue(callable(wealth_first.compare_return_streams))

    def test_compare_identical_returns_is_exact(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        reference = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.01, 0.0, -0.005, 0.003],
                "MEAN_REVERSION": [0.0, 0.002, 0.001, -0.001],
            },
            index=dates,
        )

        report = compare_return_streams(reference, reference.copy())
        frame = report.to_frame().set_index("reference_column")

        self.assertEqual(report.overlapping_dates, 4)
        self.assertAlmostEqual(frame.loc["TREND_FOLLOWING", "mean_abs_error"], 0.0)
        self.assertAlmostEqual(frame.loc["TREND_FOLLOWING", "mismatch_rate"], 0.0)
        self.assertAlmostEqual(frame.loc["MEAN_REVERSION", "correlation"], 1.0)

    def test_compare_supports_column_mapping_and_missing_dates(self) -> None:
        reference_dates = pd.date_range("2024-01-01", periods=4, freq="B")
        candidate_dates = reference_dates[1:]
        reference = pd.DataFrame(
            {
                "TREND_FOLLOWING": [0.01, 0.0, -0.005, 0.003],
            },
            index=reference_dates,
        )
        candidate = pd.DataFrame(
            {
                "tv_trend": [0.0, -0.006, 0.003],
                "unused_column": [1.0, 1.0, 1.0],
            },
            index=candidate_dates,
        )

        report = compare_return_streams(reference, candidate, candidate_to_reference={"tv_trend": "TREND_FOLLOWING"}, tolerance=5e-4)
        frame = report.to_frame().set_index("reference_column")

        self.assertEqual(report.reference_only_dates, 1)
        self.assertEqual(report.candidate_only_dates, 0)
        self.assertListEqual(report.candidate_only_columns, ["unused_column"])
        self.assertEqual(int(frame.loc["TREND_FOLLOWING", "overlapping_rows"]), 3)
        self.assertGreater(frame.loc["TREND_FOLLOWING", "mean_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()