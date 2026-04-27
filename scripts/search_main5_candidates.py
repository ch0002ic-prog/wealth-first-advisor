#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCENARIOS = [
    ("baseline_b20_s4242", 20, 4242),
    ("altblock_b16_s4242", 16, 4242),
    ("altseed_b20_s5252", 20, 5252),
    ("deepcheck_b12_s8080", 12, 8080),
]
SEEDS = [5, 17]
COSTS = [22, 25]


def eval_candidate(root: Path, name: str, horizon: int, no_trade_band: float, scale_abs: float, reps: int) -> tuple[int, float, float]:
    out_dir = root / "artifacts" / "main5_candidate_search" / f"{name}_r{reps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[float, float]] = []

    for scenario_name, block_size, path_seed in SCENARIOS:
        for seed in SEEDS:
            for cost in COSTS:
                run_dir = out_dir / f"{scenario_name}_s{seed}_c{cost}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    ".venv/bin/python",
                    "src/wealth_first/main5.py",
                    "--gate", "055",
                    "--gate-scale", "bps",
                    "--n-folds", "5",
                    "--seed", str(seed),
                    "--output-dir", str(run_dir),
                    "--transaction-cost-bps", str(cost),
                    "--slippage-bps", str(cost),
                    "--ridge-l2", "0.020815",
                    "--min-signal-scale", str(-scale_abs),
                    "--max-signal-scale", str(scale_abs),
                    "--min-spy-weight", "0.809134",
                    "--max-spy-weight", "1.058647",
                    "--initial-spy-weight", "1.0",
                    "--action-smoothing", "1.1425",
                    "--no-trade-band", str(no_trade_band),
                    "--scale-turnover-penalty", "2.9952",
                    "--validation-relative-floor-target", "0.0",
                    "--validation-relative-floor-penalty", "1.25",
                    "--validation-hard-min-relative-return", "-0.001",
                    "--validation-tail-bootstrap-reps", str(reps),
                    "--validation-tail-bootstrap-block-size", str(block_size),
                    "--validation-tail-bootstrap-quantile", "0.05",
                    "--validation-tail-bootstrap-floor-target", "-0.003",
                    "--validation-tail-bootstrap-penalty", "2.0",
                    "--validation-tail-bootstrap-hard-min", "-0.007",
                    "--validation-tail-bootstrap-objective-weight", "4.0",
                    "--validation-tail-bootstrap-seed", str(path_seed + seed),
                    "--min-robust-min-relative", "0.0",
                    "--min-active-fraction", "0.01",
                    "--forward-horizon", str(horizon),
                    "--path-bootstrap-reps", str(reps),
                    "--path-bootstrap-block-size", str(block_size),
                    "--path-bootstrap-seed", str(path_seed + seed),
                    "--no-fail-on-gate",
                ]
                subprocess.run(
                    cmd,
                    cwd=root,
                    env={**os.environ, "PYTHONPATH": str(root / "src")},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                detail_path = run_dir / f"main5_gate055_f5_s{seed}_detailed.json"
                if not detail_path.exists():
                    rows.append((-1.0, -1.0))
                    continue
                detail = json.loads(detail_path.read_text(encoding="utf-8"))
                summary = detail.get("summary_metrics", {})
                p05 = float(summary.get("path_bootstrap_robust_min_test_relative_p05", -1.0))
                mean_test = float(summary.get("mean_test_relative_total_return", 0.0))
                rows.append((p05, mean_test))

    breaches = sum(1 for p05, _ in rows if p05 < -0.007)
    min_p05 = min(p05 for p05, _ in rows)
    mean_test = sum(mt for _, mt in rows) / len(rows)
    return breaches, min_p05, mean_test


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        ("h10_ntb035_s025", 10, 0.035, 0.25),
        ("h10_ntb040_s020", 10, 0.040, 0.20),
        ("h10_ntb045_s020", 10, 0.045, 0.20),
        ("h10_ntb050_s020", 10, 0.050, 0.20),
        ("h10_ntb050_s015", 10, 0.050, 0.15),
        ("h14_ntb040_s020", 14, 0.040, 0.20),
        ("h14_ntb050_s015", 14, 0.050, 0.15),
        ("h21_ntb040_s020", 21, 0.040, 0.20),
    ]

    reps = 40
    results = []
    for name, horizon, ntb, scale_abs in candidates:
        breaches, min_p05, mean_test = eval_candidate(
            root=root,
            name=name,
            horizon=horizon,
            no_trade_band=ntb,
            scale_abs=scale_abs,
            reps=reps,
        )
        results.append((name, breaches, min_p05, mean_test))
        print(f"{name:20s} breaches={breaches:2d} min_p05={min_p05: .6f} mean_test={mean_test: .6f}")

    print("\nTop candidates by breaches/min_p05:")
    for name, breaches, min_p05, mean_test in sorted(results, key=lambda r: (r[1], -r[2]), reverse=False):
        print(f"{name:20s} breaches={breaches:2d} min_p05={min_p05: .6f} mean_test={mean_test: .6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
