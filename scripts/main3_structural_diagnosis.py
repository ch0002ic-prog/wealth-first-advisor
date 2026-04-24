from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wealth_first import main3
from wealth_first.data_splits import generate_walk_forward_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FoldSlice:
    fold: int
    phase: str
    start: int
    end: int


def _build_rich_features(spy: pd.Series) -> pd.DataFrame:
    base = main3._build_features(spy).astype(float)
    out = base.copy()
    out["ret_5"] = (
        (1.0 + spy.shift(1)).rolling(5, min_periods=1).apply(np.prod, raw=True) - 1.0
    ).fillna(0.0)
    out["vol_5"] = spy.rolling(5, min_periods=2).std(ddof=0).shift(1).fillna(0.0)
    out["vol_ratio_5_21"] = (
        out["vol_5"] / out["vol_21"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["trend_x_dd"] = out["ret_21"] * out["dd_63"]
    out["trend_x_vol"] = out["ret_21"] * out["vol_21"]
    out["ma_x_dd"] = out["ma_gap_63"] * out["dd_63"]
    out["ret_accel"] = out["ret_21"] - out["ret_63"]
    out["vol_chg"] = out["vol_21"].diff().fillna(0.0)
    out["dd_chg"] = out["dd_63"].diff().fillna(0.0)
    return out


def _fold_slices(spy: pd.Series, folds: int, validation_fraction: float, test_fraction: float) -> list[FoldSlice]:
    wf = generate_walk_forward_splits(
        spy.to_frame(name="SPY"),
        benchmark_returns=spy,
        lookback=20,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        max_splits=folds,
    )
    out: list[FoldSlice] = []
    for i, split in enumerate(wf, start=1):
        out.append(FoldSlice(i, "validation", split.validation.start_index, split.validation.end_index - 1))
        out.append(FoldSlice(i, "test", split.test.start_index, split.test.end_index - 1))
    return out


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xt = np.c_[np.ones(len(x)), x]
    a = xt.T @ xt
    reg = np.eye(a.shape[0])
    reg[0, 0] = 0.0
    b = xt.T @ y
    return np.linalg.solve(a + l2 * reg, b)


def _predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)), x] @ w


def _strategy_relative_return(
    pred: np.ndarray,
    ret: np.ndarray,
    min_weight: float = 0.65,
    max_weight: float = 1.0,
    smoothing_alpha: float = 0.15,
    no_trade_band: float = 0.01,
    turnover_cost: float = 0.001,
) -> tuple[float, float, float]:
    # Smooth logistic mapping keeps this close to the main3 policy shape.
    z = (pred - np.median(pred)) / (np.std(pred) + 1e-12)
    target = 1.0 / (1.0 + np.exp(-z))
    target = min_weight + (max_weight - min_weight) * target

    weights = np.empty_like(target)
    weights[0] = 1.0
    for t in range(1, len(target)):
        cand = weights[t - 1] + smoothing_alpha * (target[t] - weights[t - 1])
        if abs(cand - weights[t - 1]) < no_trade_band:
            cand = weights[t - 1]
        weights[t] = float(np.clip(cand, min_weight, max_weight))

    turns = np.abs(np.diff(weights, prepend=weights[0]))
    strat = weights * ret - turns * turnover_cost
    bench = ret
    rel = (1.0 + strat).prod() / (1.0 + bench).prod() - 1.0
    return float(rel), float(turns.mean()), float((1.0 - weights).mean())


def run_diagnosis(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    spy = main3._load_spy_returns(args.returns_csv, args.benchmark_column, args.date_column, args.start, args.end)
    feat_base = main3._build_features(spy).astype(float)
    feat_rich = _build_rich_features(spy)
    slices = _fold_slices(spy, args.walk_forward_folds, args.validation_fraction, args.test_fraction)

    base_cols = ["ret_1", "ret_21", "ret_63", "ma_gap_63", "dd_63", "vol_21"]
    rich_cols = base_cols + [
        "ret_5",
        "vol_5",
        "vol_ratio_5_21",
        "trend_x_dd",
        "trend_x_vol",
        "ma_x_dd",
        "ret_accel",
        "vol_chg",
        "dd_chg",
    ]

    rows: list[dict[str, float | int | str]] = []

    for model_name, feat, cols in [
        ("base6", feat_base, base_cols),
        ("rich15", feat_rich, rich_cols),
    ]:
        # Use each fold's train segment for normalization and fit.
        wf = generate_walk_forward_splits(
            spy.to_frame(name="SPY"),
            benchmark_returns=spy,
            lookback=20,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            max_splits=args.walk_forward_folds,
        )

        for fold_idx, split in enumerate(wf, start=1):
            tr_start = split.train.start_index
            tr_end = split.train.end_index - 1
            x_train = feat.iloc[tr_start : tr_end + 1][cols]
            mu = x_train.mean(axis=0)
            sd = x_train.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
            xn = ((feat[cols] - mu) / sd).fillna(0.0)

            w = _fit_ridge(
                xn.iloc[tr_start : tr_end + 1].to_numpy(),
                spy.iloc[tr_start : tr_end + 1].to_numpy(),
                l2=args.ridge_l2,
            )

            for phase_name, start_idx, end_idx in [
                ("validation", split.validation.start_index, split.validation.end_index - 1),
                ("test", split.test.start_index, split.test.end_index - 1),
            ]:
                x_phase = xn.iloc[start_idx : end_idx + 1].to_numpy()
                y_phase = spy.iloc[start_idx : end_idx + 1].to_numpy()
                pred = _predict(w, x_phase)
                ic = float(np.corrcoef(pred, y_phase)[0, 1]) if len(y_phase) > 2 else float("nan")
                sign_acc = float((np.sign(pred) == np.sign(y_phase)).mean())

                rel, turnover, cash = _strategy_relative_return(
                    pred,
                    y_phase,
                    min_weight=args.min_spy_weight,
                    max_weight=args.max_spy_weight,
                    smoothing_alpha=args.smoothing_alpha,
                    no_trade_band=args.no_trade_band,
                    turnover_cost=args.turnover_cost,
                )

                rows.append(
                    {
                        "model": model_name,
                        "fold": fold_idx,
                        "phase": phase_name,
                        "pred_ic": ic,
                        "sign_accuracy": sign_acc,
                        "relative_total_return": rel,
                        "avg_turnover": turnover,
                        "avg_cash_weight": cash,
                    }
                )

    detail = pd.DataFrame(rows)

    summary = {
        "overall": {
            "rows": int(len(detail)),
        },
        "by_model_phase": {},
        "uplift": {},
    }

    g = detail.groupby(["model", "phase"])
    for (model, phase), grp in g:
        summary["by_model_phase"][f"{model}:{phase}"] = {
            "rows": int(len(grp)),
            "mean_pred_ic": float(grp["pred_ic"].mean()),
            "mean_sign_accuracy": float(grp["sign_accuracy"].mean()),
            "mean_relative_total_return": float(grp["relative_total_return"].mean()),
            "mean_avg_turnover": float(grp["avg_turnover"].mean()),
            "mean_avg_cash_weight": float(grp["avg_cash_weight"].mean()),
        }

    piv = detail.pivot_table(
        index=["fold", "phase"],
        columns="model",
        values="relative_total_return",
    )
    if "base6" in piv.columns and "rich15" in piv.columns:
        uplift = (piv["rich15"] - piv["base6"]).dropna()
        summary["uplift"] = {
            "mean_uplift": float(uplift.mean()) if len(uplift) else 0.0,
            "test_mean_uplift": float(uplift.xs("test", level="phase").mean()) if len(uplift) else 0.0,
            "validation_mean_uplift": float(uplift.xs("validation", level="phase").mean()) if len(uplift) else 0.0,
        }

    return detail, summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Structural diagnosis for main3 signal quality and tradability.")
    p.add_argument("--returns-csv", default="data/demo_sleeves.csv")
    p.add_argument("--benchmark-column", default="SPY_BENCHMARK")
    p.add_argument("--date-column", default="date")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--walk-forward-folds", type=int, default=3)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--ridge-l2", type=float, default=20.0)
    p.add_argument("--min-spy-weight", type=float, default=0.65)
    p.add_argument("--max-spy-weight", type=float, default=1.0)
    p.add_argument("--smoothing-alpha", type=float, default=0.15)
    p.add_argument("--no-trade-band", type=float, default=0.01)
    p.add_argument("--turnover-cost", type=float, default=0.001)
    p.add_argument("--output-prefix", default="artifacts/main3_structural_diag")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    detail, summary = run_diagnosis(args)

    out_prefix = Path(args.output_prefix)
    if not out_prefix.is_absolute():
        out_prefix = PROJECT_ROOT / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    detail_path = out_prefix.with_name(f"{out_prefix.name}_detail.csv")
    summary_path = out_prefix.with_name(f"{out_prefix.name}_summary.json")

    detail.round(6).to_csv(detail_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(detail_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
