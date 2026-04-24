#!/usr/bin/env python3
"""Compare main4 vs main3 metrics."""
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _load_main4_results() -> tuple[pd.DataFrame, str]:
    repro_path = ARTIFACTS_DIR / "main4_repro_main4_gate009_f3_s10_detail.csv"
    if repro_path.exists():
        return pd.read_csv(repro_path), repro_path.name

    candidates = sorted(ARTIFACTS_DIR.glob("main4_gate009_f3_s*_summary.csv"))
    if not candidates:
        raise FileNotFoundError("No main4 artifacts found. Run scripts/run_main4_repro_suite.py first.")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return pd.read_csv(latest), latest.name

# Load main3 baseline
with open(ARTIFACTS_DIR / "main3_repro_baseline_locked.json") as f:
    main3_baseline = json.load(f)

# Load main4 results
main4_df, main4_source = _load_main4_results()

if "phase" in main4_df.columns:
    test_df = main4_df.loc[main4_df["phase"] == "test"].copy()
    if test_df.empty:
        test_df = main4_df.copy()
else:
    test_df = main4_df.copy()

relative_column = "policy_relative_total_return"
absolute_column = "policy_total_return"
if relative_column not in test_df.columns:
    raise KeyError(
        f"Expected column '{relative_column}' in {main4_source}. Available columns: {list(test_df.columns)}"
    )
if absolute_column not in test_df.columns:
    raise KeyError(
        f"Expected column '{absolute_column}' in {main4_source}. Available columns: {list(test_df.columns)}"
    )

# Compute main4 aggregate metrics
main4_mean_relative = test_df[relative_column].mean()
main4_beat_hold = (test_df[relative_column] > 0).sum() / len(test_df)

# Get main3 baseline for comparison
main3_gate009 = main3_baseline["cases"]["G_gate009_f3_s10"]

print("=" * 70)
print("MAIN3 vs MAIN4 Comparison (Gate 009)")
print("=" * 70)
print(f"\nMain3 Gate 009 Baseline (10 seeds, 3 folds each):")
print(f"  Mean test relative return:  {main3_gate009['mean_test_relative']:10.6f}")
print(f"  Beat hold rate:             {main3_gate009['beat_rate']:10.1%}")
print(f"  Active rate:                {main3_gate009['active_rate']:10.1%}")
print(f"  Mean turnover:              {main3_gate009['mean_turnover']:10.6f}")

print(f"\nMain4 ({main4_source}, {len(test_df)} rows):")
print(f"  Mean test relative return:  {main4_mean_relative:10.6f}")
print(f"  Beat hold rate:             {main4_beat_hold:10.1%}")
print(f"  Active rate:                {test_df['active'].mean():10.1%}")
print(f"  Mean turnover:              {test_df['mean_turnover'].mean():10.6f}")

print(f"\nRelative Outperformance (Main4 vs Main3):")
print(f"  Delta in mean relative return: {main4_mean_relative - main3_gate009['mean_test_relative']:+10.6f}")
print(f"  Delta in beat hold rate:      {main4_beat_hold - main3_gate009['beat_rate']:+10.1%}")

print("\n" + "=" * 70)
print(f"\nMain4 Detail (fold aggregates):")
fold_summary = (
    test_df.groupby(["fold", "phase"], as_index=False)
    .agg(
        n_rows=(relative_column, "size"),
        mean_relative=(relative_column, "mean"),
        mean_absolute=(absolute_column, "mean"),
        mean_turnover=("mean_turnover", "mean"),
    )
)
for _, row in fold_summary.iterrows():
    print(
        f"  Fold {row['fold']} ({row['phase']:6s}): "
        f"n={int(row['n_rows'])}, "
        f"rel_mean={row['mean_relative']:+8.6f}, "
        f"abs_mean={row['mean_absolute']:8.6f}, "
        f"turnover_mean={row['mean_turnover']:.6f}"
    )

if "seed" in test_df.columns:
    seed_stats = test_df.groupby("seed", as_index=False)[relative_column].mean()
    print("\nSeed Variability Check:")
    print(f"  Seeds: {len(seed_stats)}")
    print(f"  Mean of seed means: {seed_stats[relative_column].mean():+.6f}")
    print(f"  Std of seed means:  {seed_stats[relative_column].std(ddof=0):.6f}")
    print(f"  Min/Max seed means: {seed_stats[relative_column].min():+.6f} / {seed_stats[relative_column].max():+.6f}")
