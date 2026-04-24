from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from wealth_first.main3_structural_diagnosis import run_diagnosis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _run_main4_stress_suite() -> Path:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing venv Python: {PYTHON_BIN}")

    cmd = [
        str(PYTHON_BIN),
        "scripts/run_main4_stress_suite.py",
    ]
    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**dict(), **{"PYTHONPATH": "src"}, **dict(**{})},
        check=True,
    )
    return PROJECT_ROOT / "artifacts" / "main4_stress_suite_summary.json"


def _build_recommendations(structural_summary: dict, main4_summary: dict) -> list[str]:
    rich_test = structural_summary["by_model_phase"]["rich15:test"]["mean_relative_total_return"]
    base_test = structural_summary["by_model_phase"]["base6:test"]["mean_relative_total_return"]
    feature_uplift = structural_summary["uplift"].get("test_mean_uplift", 0.0)
    best_case = str(main4_summary.get("best_case", "unknown"))
    best_relative = float(main4_summary.get("best_mean_test_relative", 0.0))
    improvement_vs_baseline = float(main4_summary.get("improvement_vs_baseline", 0.0))

    recommendations: list[str] = []
    if feature_uplift < 0.005 and rich_test < 0.0 and base_test < 0.0:
        recommendations.append(
            "Do not spend the next cycle on more linear feature expansion alone; rich15 only marginally improves main3 and remains negative on test."
        )
    recommendations.append(
        "Prioritize the medium-capacity path around main4: extend regime features, tighten action gating, and keep the evaluation harness friction-aware."
    )
    if improvement_vs_baseline <= 0.0:
        recommendations.append(
            "Do not start a new DRL branch yet; the current main4 stress suite is not showing a meaningful uplift over the promoted baseline."
        )
    else:
        recommendations.append(
            f"Use '{best_case}' as the current promoted main4 stress configuration and rerun deeper validation before adding model complexity."
        )
    if best_relative > 0.0:
        recommendations.append(
            "If you revisit DRL later, constrain it to a bounded-delta overlay that must beat main4 under the same walk-forward and friction tests."
        )
    recommendations.append(
        "Add an experiment harness that ranks candidates by full-sample metrics, decisive-only metrics, decision rate, and turnover so improvements are not hidden by abstention or inactivity."
    )
    return recommendations


def _render_markdown(report: dict) -> str:
    structural = report["structural_diagnosis"]
    main4 = report["main4_stress_suite"]
    return "\n".join(
        [
            "# Optimization Research Audit",
            "",
            "## Key Findings",
            f"- main3 base6 test mean relative return: {structural['by_model_phase']['base6:test']['mean_relative_total_return']:.6f}",
            f"- main3 rich15 test mean relative return: {structural['by_model_phase']['rich15:test']['mean_relative_total_return']:.6f}",
            f"- main3 rich15 test uplift vs base6: {structural['uplift'].get('test_mean_uplift', 0.0):.6f}",
            f"- main4 best stress case: {main4['best_case']}",
            f"- main4 best mean relative return: {main4['best_mean_test_relative']:.6f}",
            f"- main4 uplift vs baseline: {main4['improvement_vs_baseline']:.6f}",
            "",
            "## Recommended Priorities",
            *[f"- {item}" for item in report["recommendations"]],
            "",
            "## Generated Artifacts",
            f"- structural summary: {report['artifacts']['structural_summary_json']}",
            f"- main4 stress summary: {report['artifacts']['main4_stress_suite_summary_json']}",
            f"- audit summary: {report['artifacts']['audit_summary_json']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight optimization research audit for main3 and main4 paths.")
    parser.add_argument("--structural-output-prefix", default="artifacts/optimization_research_structural")
    parser.add_argument("--audit-output-prefix", default="artifacts/optimization_research_audit")
    args = parser.parse_args(argv)

    structural_args = SimpleNamespace(
        returns_csv="data/demo_sleeves.csv",
        benchmark_column="SPY_BENCHMARK",
        date_column="date",
        start=None,
        end=None,
        walk_forward_folds=3,
        validation_fraction=0.15,
        test_fraction=0.10,
        ridge_l2=20.0,
        min_spy_weight=0.65,
        max_spy_weight=1.0,
        smoothing_alpha=0.15,
        no_trade_band=0.01,
        turnover_cost=0.001,
        output_prefix=args.structural_output_prefix,
    )
    structural_detail, structural_summary = run_diagnosis(structural_args)

    structural_prefix = PROJECT_ROOT / args.structural_output_prefix
    structural_detail_path = structural_prefix.with_name(f"{structural_prefix.name}_detail.csv")
    structural_summary_path = structural_prefix.with_name(f"{structural_prefix.name}_summary.json")
    structural_detail.round(6).to_csv(structural_detail_path, index=False)
    structural_summary_path.write_text(json.dumps(structural_summary, indent=2), encoding="utf-8")

    main4_summary_path = _run_main4_stress_suite()
    main4_summary = json.loads(main4_summary_path.read_text(encoding="utf-8"))

    recommendations = _build_recommendations(structural_summary, main4_summary)

    audit_prefix = PROJECT_ROOT / args.audit_output_prefix
    audit_summary_path = audit_prefix.with_name(f"{audit_prefix.name}_summary.json")
    audit_report_path = audit_prefix.with_name(f"{audit_prefix.name}.md")
    report = {
        "structural_diagnosis": structural_summary,
        "main4_stress_suite": main4_summary,
        "recommendations": recommendations,
        "artifacts": {
            "structural_detail_csv": structural_detail_path,
            "structural_summary_json": structural_summary_path,
            "main4_stress_suite_summary_json": main4_summary_path,
            "audit_summary_json": audit_summary_path,
            "audit_markdown": audit_report_path,
        },
    }
    audit_summary_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    audit_report_path.write_text(_render_markdown(_json_ready(report)), encoding="utf-8")

    print(audit_summary_path)
    print(audit_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())