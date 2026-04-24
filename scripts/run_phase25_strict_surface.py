#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
ART = REPO / "artifacts"

BASE_CMD = [
    str(PY),
    "scripts/run_main4_wealthfirst_phase2.py",
    "--mode", "full",
    "--tail-worst-decile-threshold", "0.0",
    "--robust-min-threshold", "0.0",
    "--max-mean-turnover", "0.0015",
    "--min-worst-daily-relative-return", "-0.010",
    "--max-worst-relative-drawdown", "0.030",
    "--min-path-bootstrap-robust-min-p05", "-0.006",
    "--strict-path-bootstrap-gate-mode", "quantile",
    "--bootstrap-reps", "400",
    "--path-bootstrap-reps", "300",
    "--path-bootstrap-block-size", "20",
    "--path-bootstrap-seed", "4242",
    "--auto-calibrate-gates",
    "--auto-calibrate-apply",
    "--auto-calibrate-min-feasible-profiles", "2",
    "--selection-mode", "strict_only",
]

surface_rows = []

# Sweep 1: activity x cap at quantile=0.5
activity_values = [0.02, 0.019, 0.018, 0.0]
cap_values = [0.0, 0.0005, 0.0010, 0.0020, 0.0030, 0.0040, -1.0]
for min_step in activity_values:
    for cap in cap_values:
        label = f"ms{str(min_step).replace('.', '')}_cap{str(cap).replace('.', '').replace('-', 'neg')}"
        prefix = f"main4_phase25a_surface_{label}"
        cmd = BASE_CMD + [
            "--output-prefix", prefix,
            "--min-mean-executed-step-rate", str(min_step),
            "--strict-path-bootstrap-gate-quantile", "0.5",
            "--strict-path-bootstrap-gate-max-relaxation", str(cap),
        ]
        subprocess.run(cmd, cwd=REPO, check=True)
        summary = json.loads((ART / f"{prefix}_summary.json").read_text())
        pr = summary["promotion_report"]
        audit = pr.get("strict_path_bootstrap_gate_audit", {})
        cap_audit = audit.get("relaxation_cap", {})
        surface_rows.append(
            {
                "experiment": "activity_x_cap",
                "min_mean_executed_step_rate": min_step,
                "strict_path_bootstrap_gate_quantile": 0.5,
                "strict_path_bootstrap_gate_max_relaxation": cap,
                "strict_feasible": bool(pr.get("strict_feasible")),
                "strict_feasible_profile_count": int(pr.get("strict_feasible_profile_count", 0)),
                "strict_best_eligible_profile": pr.get("strict_best_eligible_profile"),
                "selected_best_profile": summary.get("best_profile"),
                "path_bootstrap_fail_count": int(pr.get("strict_failed_gate_counts", {}).get("path_bootstrap_robust_min_p05", 0)),
                "mean_executed_step_fail_count": int(pr.get("strict_failed_gate_counts", {}).get("mean_executed_step_rate", 0)),
                "resolved_threshold_before_cap": audit.get("applied_threshold"),
                "resolved_threshold_after_cap": cap_audit.get("resolved_threshold_after_cap", audit.get("applied_threshold")),
                "cap_applied": bool(cap_audit.get("cap_applied", False)),
                "governance_strict_blocked": bool(pr.get("governance_status", {}).get("strict_blocked", False)),
                "source_summary": f"artifacts/{prefix}_summary.json",
            }
        )

# Sweep 2: quantile sensitivity with fixed cap and activity gate
quantile_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for q in quantile_values:
    prefix = f"main4_phase25a_quantile_q{str(q).replace('.', '')}"
    cmd = BASE_CMD + [
        "--output-prefix", prefix,
        "--min-mean-executed-step-rate", "0.02",
        "--strict-path-bootstrap-gate-quantile", str(q),
        "--strict-path-bootstrap-gate-max-relaxation", "0.001",
    ]
    subprocess.run(cmd, cwd=REPO, check=True)
    summary = json.loads((ART / f"{prefix}_summary.json").read_text())
    pr = summary["promotion_report"]
    audit = pr.get("strict_path_bootstrap_gate_audit", {})
    cap_audit = audit.get("relaxation_cap", {})
    surface_rows.append(
        {
            "experiment": "quantile_sensitivity",
            "min_mean_executed_step_rate": 0.02,
            "strict_path_bootstrap_gate_quantile": q,
            "strict_path_bootstrap_gate_max_relaxation": 0.001,
            "strict_feasible": bool(pr.get("strict_feasible")),
            "strict_feasible_profile_count": int(pr.get("strict_feasible_profile_count", 0)),
            "strict_best_eligible_profile": pr.get("strict_best_eligible_profile"),
            "selected_best_profile": summary.get("best_profile"),
            "path_bootstrap_fail_count": int(pr.get("strict_failed_gate_counts", {}).get("path_bootstrap_robust_min_p05", 0)),
            "mean_executed_step_fail_count": int(pr.get("strict_failed_gate_counts", {}).get("mean_executed_step_rate", 0)),
            "resolved_threshold_before_cap": audit.get("applied_threshold"),
            "resolved_threshold_after_cap": cap_audit.get("resolved_threshold_after_cap", audit.get("applied_threshold")),
            "cap_applied": bool(cap_audit.get("cap_applied", False)),
            "governance_strict_blocked": bool(pr.get("governance_status", {}).get("strict_blocked", False)),
            "source_summary": f"artifacts/{prefix}_summary.json",
        }
    )

# Rollup
rollup = {
    "rows": surface_rows,
    "summary": {
        "activity_x_cap": {
            "total": sum(1 for r in surface_rows if r["experiment"] == "activity_x_cap"),
            "strict_feasible": sum(1 for r in surface_rows if r["experiment"] == "activity_x_cap" and r["strict_feasible"]),
        },
        "quantile_sensitivity": {
            "total": sum(1 for r in surface_rows if r["experiment"] == "quantile_sensitivity"),
            "strict_feasible": sum(1 for r in surface_rows if r["experiment"] == "quantile_sensitivity" and r["strict_feasible"]),
        },
    },
}

json_path = ART / "main4_phase25a_strict_surface_summary.json"
csv_path = ART / "main4_phase25a_strict_surface_summary.csv"
json_path.write_text(json.dumps(rollup, indent=2))
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(surface_rows[0].keys()))
    writer.writeheader()
    writer.writerows(surface_rows)

print("WROTE", json_path)
print("WROTE", csv_path)
print(json.dumps(rollup["summary"], indent=2))
