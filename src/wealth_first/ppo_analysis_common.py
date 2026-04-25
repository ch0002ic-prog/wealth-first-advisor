from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
RULED_OUT_ARCHIVE = ARTIFACTS / "archive" / "ruled_out_2026-04-16"


def discover_seed_dirs(artifacts_dir: Path, prefix: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matched: list[tuple[int, Path]] = []
    for path in artifacts_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        matched.append((int(match.group(1)), path))
    return [path for _, path in sorted(matched, key=lambda item: item[0])]


def requested_splits(regime_prefix: str | None, chrono_prefix: str | None) -> list[tuple[str, str]]:
    requested: list[tuple[str, str]] = []
    if regime_prefix:
        requested.append(("regime", regime_prefix))
    if chrono_prefix:
        requested.append(("chrono", chrono_prefix))
    if not requested:
        raise ValueError("At least one split prefix must be provided.")
    return requested


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fold_dirs(seed_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)$", path.name)
        return (int(match.group(1)) if match else math.inf, path.name)

    return sorted([path for path in seed_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")], key=sort_key)


def phase_window(seed_dir: Path, fold_name: str, phase: str) -> tuple[str | None, str | None]:
    split_windows = seed_dir / "split_windows.json"
    if not split_windows.exists():
        return (None, None)
    payload = read_json(split_windows)
    folds = payload.get("walk_forward_folds", [])
    match = re.search(r"(\d+)$", fold_name)
    if match is None:
        return (None, None)
    fold_index = int(match.group(1)) - 1
    if fold_index < 0 or fold_index >= len(folds):
        return (None, None)
    window = folds[fold_index].get(phase, {})
    return (window.get("start"), window.get("end"))


def preferred_artifact_path(filename: str, *, artifacts: Path = ARTIFACTS, archive: Path = RULED_OUT_ARCHIVE) -> Path:
    primary = artifacts / filename
    if primary.exists():
        return primary
    return archive / filename


def artifact_run_dir(
    prefix_by_split: dict[str, str],
    row: dict[str, Any],
    *,
    artifacts: Path = ARTIFACTS,
    archive: Path = RULED_OUT_ARCHIVE,
) -> Path:
    split = str(row["split"])
    seed = int(row["seed"])
    fold = str(row["fold"])
    prefix = prefix_by_split[split]
    primary = artifacts / f"{prefix}{seed}" / fold
    if primary.exists():
        return primary
    return archive / f"{prefix}{seed}" / fold


def artifact_file(
    prefix_by_split: dict[str, str],
    row: dict[str, Any],
    filename: str,
    *,
    artifacts: Path = ARTIFACTS,
    archive: Path = RULED_OUT_ARCHIVE,
) -> Path:
    return artifact_run_dir(prefix_by_split, row, artifacts=artifacts, archive=archive) / filename


def _weight_symbols(frame: pd.DataFrame) -> list[str]:
    symbols: list[str] = []
    for column in frame.columns:
        if column.startswith("target_weight_"):
            symbols.append(column.removeprefix("target_weight_"))
    return symbols


def _original_turnover(frame: pd.DataFrame, symbols: list[str]) -> float:
    if "turnover" in frame.columns:
        return float(frame["turnover"].fillna(0.0).sum())
    turnover = 0.0
    for _, row in frame.iterrows():
        turnover += 0.5 * sum(
            abs(float(row.get(f"target_weight_{symbol}", 0.0)) - float(row.get(f"pre_trade_weight_{symbol}", 0.0)))
            for symbol in symbols
        )
    return float(turnover)


def _portfolio_total_return(frame: pd.DataFrame, symbols: list[str]) -> tuple[float, float | None]:
    policy_wealth = 1.0
    benchmark_wealth = 1.0
    benchmark_seen = "benchmark_return" in frame.columns
    for _, row in frame.iterrows():
        period_return = 0.0
        for symbol in symbols:
            period_return += float(row.get(f"target_weight_{symbol}", 0.0)) * float(row.get(f"asset_return_{symbol}", 0.0))
        period_return -= float(row.get("execution_cost", 0.0))
        policy_wealth *= 1.0 + period_return
        if benchmark_seen:
            benchmark_wealth *= 1.0 + float(row.get("benchmark_return", 0.0))
    total_return = policy_wealth - 1.0
    if not benchmark_seen:
        return (float(total_return), None)
    relative_total_return = (policy_wealth / benchmark_wealth) - 1.0
    return (float(total_return), float(relative_total_return))


def replay_runtime_overlay_frame(
    frame: pd.DataFrame,
    *,
    runtime_overlay: str | None,
    trend_symbol: str = "TREND_FOLLOWING",
    allow_overlay: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    replayed = frame.copy(deep=True)
    symbols = _weight_symbols(replayed)
    policy_turnover = _original_turnover(replayed, symbols)
    overlay_applied_steps = 0
    overlay_suppressed_steps = 0
    previous_targets: dict[str, float] | None = None

    for index, row in replayed.iterrows():
        current_targets = {symbol: float(row.get(f"target_weight_{symbol}", 0.0)) for symbol in symbols}
        if allow_overlay and runtime_overlay == "robust-regime-only":
            mean_reversion_weight = float(row.get("pre_trade_weight_MEAN_REVERSION", 0.0))
            drawdown = float(row.get("benchmark_regime_drawdown", 0.0))
            if drawdown <= -0.10 or mean_reversion_weight >= 0.186:
                for symbol in symbols:
                    replayed.at[index, f"target_weight_{symbol}"] = float(row.get(f"pre_trade_weight_{symbol}", 0.0))
                overlay_applied_steps += 1
                overlay_suppressed_steps += 1
                current_targets = {
                    symbol: float(replayed.at[index, f"target_weight_{symbol}"])
                    for symbol in symbols
                }
        elif allow_overlay and runtime_overlay == "soft-premr" and previous_targets is not None:
            regime_cumulative_return = float(row.get("benchmark_regime_cumulative_return", 0.0))
            if regime_cumulative_return <= 0.015:
                for symbol in symbols:
                    smoothed = previous_targets[symbol] + 0.75 * (current_targets[symbol] - previous_targets[symbol])
                    replayed.at[index, f"target_weight_{symbol}"] = smoothed
                overlay_applied_steps += 1
                current_targets = {
                    symbol: float(replayed.at[index, f"target_weight_{symbol}"])
                    for symbol in symbols
                }
        previous_targets = current_targets

    total_return, relative_total_return = _portfolio_total_return(replayed, symbols)
    metrics: dict[str, float | int] = {
        "overlay_applied_steps": int(overlay_applied_steps),
        "overlay_suppressed_steps": int(overlay_suppressed_steps),
        "policy_total_return": float(total_return),
        "policy_turnover": float(policy_turnover),
        "policy_cash_weight": float(replayed.get("target_weight_CASH", pd.Series([0.0])).mean()),
        "policy_trend_weight": float(replayed.get(f"target_weight_{trend_symbol}", pd.Series([0.0])).mean()),
    }
    if relative_total_return is not None:
        metrics["policy_relative_total_return"] = float(relative_total_return)
    return (replayed, metrics)
