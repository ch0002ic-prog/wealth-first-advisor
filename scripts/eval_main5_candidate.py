#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main() -> int:
	root = Path(__file__).resolve().parents[1]
	scenarios = [
		("baseline_b20_s4242", 20, 4242),
		("altblock_b16_s4242", 16, 4242),
		("altseed_b20_s5252", 20, 5252),
		("deepcheck_b12_s8080", 12, 8080),
	]
	seeds = [5, 17]
	costs = [22, 25]

	out_dir = root / "artifacts" / "main5_candidate_h10_ntb050_s020"
	out_dir.mkdir(parents=True, exist_ok=True)

	rows: list[tuple[str, int, int, float, float, bool]] = []

	for scenario_name, block_size, path_seed in scenarios:
		for seed in seeds:
			for cost in costs:
				run_dir = out_dir / f"{scenario_name}_s{seed}_c{cost}"
				run_dir.mkdir(parents=True, exist_ok=True)
				cmd = [
					".venv/bin/python",
					"src/wealth_first/main5.py",
					"--gate",
					"055",
					"--gate-scale",
					"bps",
					"--n-folds",
					"5",
					"--seed",
					str(seed),
					"--output-dir",
					str(run_dir),
					"--transaction-cost-bps",
					str(cost),
					"--slippage-bps",
					str(cost),
					"--ridge-l2",
					"0.020815",
					"--min-signal-scale",
					"-0.20",
					"--max-signal-scale",
					"0.20",
					"--min-spy-weight",
					"0.809134",
					"--max-spy-weight",
					"1.058647",
					"--initial-spy-weight",
					"1.0",
					"--action-smoothing",
					"1.1425",
					"--no-trade-band",
					"0.05",
					"--scale-turnover-penalty",
					"2.9952",
					"--validation-relative-floor-target",
					"0.0",
					"--validation-relative-floor-penalty",
					"1.25",
					"--validation-hard-min-relative-return",
					"-0.001",
					"--validation-tail-bootstrap-reps",
					"80",
					"--validation-tail-bootstrap-block-size",
					str(block_size),
					"--validation-tail-bootstrap-quantile",
					"0.05",
					"--validation-tail-bootstrap-floor-target",
					"-0.003",
					"--validation-tail-bootstrap-penalty",
					"2.0",
					"--validation-tail-bootstrap-hard-min",
					"-0.007",
					"--validation-tail-bootstrap-objective-weight",
					"4.0",
					"--validation-tail-bootstrap-seed",
					str(path_seed + seed),
					"--min-robust-min-relative",
					"0.0",
					"--min-active-fraction",
					"0.01",
					"--forward-horizon",
					"10",
					"--path-bootstrap-reps",
					"80",
					"--path-bootstrap-block-size",
					str(block_size),
					"--path-bootstrap-seed",
					str(path_seed + seed),
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

				detailed_path = run_dir / f"main5_gate055_f5_s{seed}_detailed.json"
				if not detailed_path.exists():
					continue
				detail = json.loads(detailed_path.read_text(encoding="utf-8"))
				summary = detail.get("summary_metrics", {})
				gate_passed = bool(detail.get("gate_checks", {}).get("overall_passed", False))
				p05 = float(summary.get("path_bootstrap_robust_min_test_relative_p05", -1.0))
				mean_test = float(summary.get("mean_test_relative_total_return", 0.0))
				rows.append((scenario_name, seed, cost, p05, mean_test, gate_passed))

	rows.sort(key=lambda x: x[3])
	for scenario_name, seed, cost, p05, mean_test, gate_passed in rows:
		print(
			f"{scenario_name:20s} s={seed} c={cost} "
			f"p05={p05: .6f} mean={mean_test: .6f} gate={gate_passed}"
		)

	breaches = sum(1 for _, _, _, p05, _, _ in rows if p05 < -0.007)
	print(f"breaches {breaches} of {len(rows)}")
	if rows:
		print(f"min_p05 {min(r[3] for r in rows):.6f}")
		print(f"mean_mean_test {sum(r[4] for r in rows) / len(rows):.6f}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
