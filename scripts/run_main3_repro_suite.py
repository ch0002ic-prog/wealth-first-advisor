from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class Case:
    name: str
    output_prefix: str
    args: list[str]


CASES = [
    Case(
        name="A_locked_f3_s10",
        output_prefix="artifacts/main3_repro_A_f3_s10",
        args=[
            "--seeds", "7", "17", "27", "37", "47", "57", "67", "77", "87", "97",
            "--walk-forward-folds", "3",
            "--coarse-candidates", "260",
            "--refine-candidates", "220",
            "--validation-fraction", "0.15",
            "--test-fraction", "0.10",
            "--min-spy-weight", "0.80",
            "--max-spy-weight", "1.05",
            "--regime-shift-penalty", "0.25",
            "--max-regime-shift", "0.50",
            "--tactical-scale-max", "0.22",
            "--min-validation-relative-return-for-activation", "0.001",
        ],
    ),
    Case(
        name="G_gate009_f3_s10",
        output_prefix="artifacts/main3_repro_G_gate009_f3_s10",
        args=[
            "--seeds", "7", "17", "27", "37", "47", "57", "67", "77", "87", "97",
            "--walk-forward-folds", "3",
            "--coarse-candidates", "260",
            "--refine-candidates", "220",
            "--validation-fraction", "0.15",
            "--test-fraction", "0.10",
            "--min-spy-weight", "0.80",
            "--max-spy-weight", "1.05",
            "--regime-shift-penalty", "0.25",
            "--max-regime-shift", "0.50",
            "--tactical-scale-max", "0.22",
            "--min-validation-relative-return-for-activation", "0.009",
        ],
    ),
    Case(
        name="I_gate009_f3_s20",
        output_prefix="artifacts/main3_repro_I_gate009_f3_s20",
        args=[
            "--seeds",
            "7", "12", "17", "22", "27", "32", "37", "42", "47", "52",
            "57", "62", "67", "72", "77", "82", "87", "92", "97", "102",
            "--walk-forward-folds", "3",
            "--coarse-candidates", "260",
            "--refine-candidates", "220",
            "--validation-fraction", "0.15",
            "--test-fraction", "0.10",
            "--min-spy-weight", "0.80",
            "--max-spy-weight", "1.05",
            "--regime-shift-penalty", "0.25",
            "--max-regime-shift", "0.50",
            "--tactical-scale-max", "0.22",
            "--min-validation-relative-return-for-activation", "0.009",
        ],
    ),
]


def run_case(case: Case) -> Path:
    cmd = [
        str(PYTHON_BIN),
        "-m",
        "wealth_first.main3",
        "--output-prefix",
        case.output_prefix,
        *case.args,
    ]
    print("\\nRunning", case.name)
    print("Command:", shlex.join(["PYTHONPATH=src", *cmd]))
    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )
    return PROJECT_ROOT / f"{case.output_prefix}_detail.csv"


def summarize(detail_path: Path, case_name: str) -> dict[str, object]:
    detail = pd.read_csv(detail_path)
    test = detail[detail["phase"] == "test"].copy()
    x = test["policy_relative_total_return"].to_numpy(dtype=float)
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else 0.0
    mean = float(np.mean(x)) if n > 0 else 0.0
    return {
        "case": case_name,
        "rows": n,
        "mean_test_relative": mean,
        "ci95_lo": mean - 1.96 * se,
        "ci95_hi": mean + 1.96 * se,
        "beat_rate": float(np.mean(x > 0.0)) if n > 0 else 0.0,
        "active_rate": float(test["train_diag_active_selected"].mean()) if n > 0 else 0.0,
        "mean_turnover": float(test["policy_turnover"].mean()) if n > 0 else 0.0,
    }


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing venv Python: {PYTHON_BIN}")

    rows: list[dict[str, object]] = []
    for case in CASES:
        detail_path = run_case(case)
        rows.append(summarize(detail_path, case.name))

    out_csv = PROJECT_ROOT / "artifacts" / "main3_repro_suite_summary.csv"
    out_json = PROJECT_ROOT / "artifacts" / "main3_repro_suite_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(rows)
    table.round(6).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\\nSaved:")
    print(out_csv)
    print(out_json)
    print("\\n", table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
