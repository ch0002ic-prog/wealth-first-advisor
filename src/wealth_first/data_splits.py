from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    start_index: int
    end_index: int
    start_label: object
    end_label: object
    rows: int


@dataclass(frozen=True)
class SuggestedTimeSeriesSplit:
    method: str
    train: SplitWindow
    validation: SplitWindow
    test: SplitWindow
    score: float
    search_step: int
    target_rows: dict[str, int]
    regime_coverage: dict[str, float]
    regime_distance: dict[str, float]


def _coerce_returns_frame(returns: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    elif isinstance(returns, pd.Series):
        column_name = returns.name if returns.name is not None else "asset_1"
        frame = returns.to_frame(name=str(column_name))
    else:
        raise TypeError("Returns must be provided as a pandas DataFrame or Series.")

    frame = frame.astype(float).dropna(how="any")
    if frame.empty:
        raise ValueError("Returns data is empty after dropping missing values.")
    return frame.sort_index()


def _coerce_reference_series(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
) -> pd.Series:
    if benchmark_returns is None:
        return returns.mean(axis=1).rename("reference_return")

    aligned = pd.Series(benchmark_returns, copy=True).astype(float).sort_index().reindex(returns.index)
    if aligned.isna().any():
        raise ValueError("Benchmark returns could not be aligned to the returns index.")
    return aligned.rename("reference_return")


def _quantile_bucket(values: pd.Series, bucket_count: int, labels: list[str]) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=bucket_count, labels=labels)


def build_regime_labels(
    returns: pd.DataFrame | pd.Series,
    benchmark_returns: pd.Series | None = None,
    lookback: int = 20,
) -> pd.Series:
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")

    returns_frame = _coerce_returns_frame(returns)
    reference_returns = _coerce_reference_series(returns_frame, benchmark_returns)
    rolling_volatility = reference_returns.rolling(lookback, min_periods=lookback).std(ddof=0)
    rolling_trend = reference_returns.rolling(lookback, min_periods=lookback).sum()
    reference_wealth = (1.0 + reference_returns).cumprod()
    drawdown = reference_wealth / reference_wealth.cummax() - 1.0

    regime_frame = pd.DataFrame(
        {
            "volatility": rolling_volatility,
            "trend": rolling_trend,
            "drawdown": drawdown,
        },
        index=returns_frame.index,
    ).dropna()
    if regime_frame.empty:
        raise ValueError("Not enough observations to compute regime labels for the requested lookback.")

    volatility_bucket = _quantile_bucket(
        regime_frame["volatility"],
        bucket_count=3,
        labels=["vol_low", "vol_mid", "vol_high"],
    ).astype(str)
    drawdown_bucket = _quantile_bucket(
        regime_frame["drawdown"],
        bucket_count=2,
        labels=["drawdown_stress", "drawdown_calm"],
    ).astype(str)
    trend_bucket = np.where(regime_frame["trend"] >= 0.0, "trend_up", "trend_down")

    regime_labels = volatility_bucket + "|" + drawdown_bucket + "|" + trend_bucket
    return regime_labels.rename("regime_label").reindex(returns_frame.index)


def _window_from_bounds(index: pd.Index, start_index: int, end_index: int) -> SplitWindow:
    if start_index < 0 or end_index > len(index) or start_index >= end_index:
        raise ValueError("Invalid split window bounds.")

    return SplitWindow(
        start_index=start_index,
        end_index=end_index,
        start_label=index[start_index],
        end_label=index[end_index - 1],
        rows=end_index - start_index,
    )


def _histogram_distance(labels: pd.Series, reference_labels: pd.Index, reference_histogram: pd.Series) -> tuple[float, float]:
    window_counts = labels.value_counts().reindex(reference_labels, fill_value=0).astype(float)
    total = float(window_counts.sum())
    if total <= 0:
        return float("inf"), 0.0

    window_histogram = window_counts / total
    coverage = float((window_counts > 0).sum() / len(reference_labels)) if len(reference_labels) > 0 else 1.0
    distance = float(np.abs(window_histogram - reference_histogram).sum())
    return distance, coverage


def _describe_split_windows(
    method: str,
    index: pd.Index,
    train_window: SplitWindow,
    validation_window: SplitWindow,
    test_window: SplitWindow,
    regime_labels: pd.Series | None,
    search_step: int,
    target_rows: dict[str, int],
) -> SuggestedTimeSeriesSplit:
    regime_coverage = {"train": 1.0, "validation": 1.0, "test": 1.0}
    regime_distance = {"train": 0.0, "validation": 0.0, "test": 0.0}
    score = 0.0

    if regime_labels is not None:
        full_labels = regime_labels.dropna()
        reference_labels = full_labels.value_counts().sort_index().index
        reference_histogram = full_labels.value_counts(normalize=True).reindex(reference_labels, fill_value=0.0)
        window_map = {
            "train": train_window,
            "validation": validation_window,
            "test": test_window,
        }
        for name, window in window_map.items():
            window_labels = regime_labels.iloc[window.start_index : window.end_index].dropna()
            distance, coverage = _histogram_distance(window_labels, reference_labels, reference_histogram)
            regime_distance[name] = distance
            regime_coverage[name] = coverage
        size_penalty = (
            abs(train_window.rows - target_rows["train"])
            + abs(validation_window.rows - target_rows["validation"])
            + abs(test_window.rows - target_rows["test"])
        ) / len(index)
        score = (
            1.5 * regime_distance["train"]
            + regime_distance["validation"]
            + regime_distance["test"]
            + 2.0 * (1.0 - regime_coverage["train"])
            + 1.0 * (1.0 - regime_coverage["validation"])
            + 1.0 * (1.0 - regime_coverage["test"])
            + 2.0 * size_penalty
        )

    return SuggestedTimeSeriesSplit(
        method=method,
        train=train_window,
        validation=validation_window,
        test=test_window,
        score=float(score),
        search_step=search_step,
        target_rows=target_rows,
        regime_coverage=regime_coverage,
        regime_distance=regime_distance,
    )


def chronological_train_validation_test_split(
    returns: pd.DataFrame | pd.Series,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.10,
    lookback: int = 20,
) -> SuggestedTimeSeriesSplit:
    returns_frame = _coerce_returns_frame(returns)
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation_fraction + test_fraction must leave a meaningful training window.")

    minimum_window = lookback + 2
    total_rows = len(returns_frame)
    target_validation_rows = max(minimum_window, int(round(total_rows * validation_fraction)))
    target_test_rows = max(minimum_window, int(round(total_rows * test_fraction)))
    train_rows = total_rows - target_validation_rows - target_test_rows
    if train_rows < minimum_window:
        raise ValueError("Not enough return observations to build train, validation, and test windows.")

    train = _window_from_bounds(returns_frame.index, 0, train_rows)
    validation = _window_from_bounds(returns_frame.index, train.end_index, train.end_index + target_validation_rows)
    test = _window_from_bounds(returns_frame.index, validation.end_index, total_rows)
    return _describe_split_windows(
        method="chronological",
        index=returns_frame.index,
        train_window=train,
        validation_window=validation,
        test_window=test,
        regime_labels=None,
        search_step=1,
        target_rows={
            "train": train.rows,
            "validation": target_validation_rows,
            "test": target_test_rows,
        },
    )


def suggest_regime_balanced_split(
    returns: pd.DataFrame | pd.Series,
    benchmark_returns: pd.Series | None = None,
    lookback: int = 20,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.10,
    min_window_size: int | None = None,
    search_step: int | None = None,
) -> SuggestedTimeSeriesSplit:
    returns_frame = _coerce_returns_frame(returns)
    regime_labels = build_regime_labels(returns_frame, benchmark_returns=benchmark_returns, lookback=lookback)
    full_labels = regime_labels.dropna()
    if full_labels.empty:
        raise ValueError("Not enough observations to compute regime labels for splitting.")

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation_fraction + test_fraction must leave a meaningful training window.")

    total_rows = len(returns_frame)
    minimum_window = max(lookback + 2, min_window_size or 0)
    target_validation_rows = max(minimum_window, int(round(total_rows * validation_fraction)))
    target_test_rows = max(minimum_window, int(round(total_rows * test_fraction)))
    target_train_rows = total_rows - target_validation_rows - target_test_rows
    if target_train_rows < minimum_window:
        raise ValueError("Not enough return observations to build train, validation, and test windows.")

    if search_step is None:
        search_step = max(5, total_rows // 200)
    size_tolerance = max(search_step, int(round(total_rows * 0.05)))

    reference_labels = full_labels.value_counts().sort_index().index
    reference_histogram = full_labels.value_counts(normalize=True).reindex(reference_labels, fill_value=0.0)

    def candidate_positions(start: int, stop: int, target: int) -> list[int]:
        positions = list(range(start, stop + 1, search_step))
        positions.extend([start, stop, target])
        return sorted({position for position in positions if start <= position <= stop})

    train_end_start = max(minimum_window, target_train_rows - size_tolerance)
    train_end_stop = min(total_rows - 2 * minimum_window, target_train_rows + size_tolerance)
    train_end_candidates = candidate_positions(train_end_start, train_end_stop, target_train_rows)
    best_split: SuggestedTimeSeriesSplit | None = None
    best_score = float("inf")

    for train_end in train_end_candidates:
        validation_target_end = train_end + target_validation_rows
        validation_end_candidates = candidate_positions(
            max(train_end + minimum_window, validation_target_end - size_tolerance),
            min(total_rows - minimum_window, validation_target_end + size_tolerance),
            validation_target_end,
        )
        for validation_end in validation_end_candidates:
            train_window = _window_from_bounds(returns_frame.index, 0, train_end)
            validation_window = _window_from_bounds(returns_frame.index, train_end, validation_end)
            test_window = _window_from_bounds(returns_frame.index, validation_end, total_rows)

            train_regimes = regime_labels.iloc[train_window.start_index : train_window.end_index].dropna()
            validation_regimes = regime_labels.iloc[validation_window.start_index : validation_window.end_index].dropna()
            test_regimes = regime_labels.iloc[test_window.start_index : test_window.end_index].dropna()
            if train_regimes.empty or validation_regimes.empty or test_regimes.empty:
                continue

            train_distance, train_coverage = _histogram_distance(train_regimes, reference_labels, reference_histogram)
            validation_distance, validation_coverage = _histogram_distance(validation_regimes, reference_labels, reference_histogram)
            test_distance, test_coverage = _histogram_distance(test_regimes, reference_labels, reference_histogram)
            size_penalty = (
                abs(train_window.rows - target_train_rows)
                + abs(validation_window.rows - target_validation_rows)
                + abs(test_window.rows - target_test_rows)
            ) / total_rows
            score = (
                1.5 * train_distance
                + validation_distance
                + test_distance
                + 2.0 * (1.0 - train_coverage)
                + 1.0 * (1.0 - validation_coverage)
                + 1.0 * (1.0 - test_coverage)
                + 2.0 * size_penalty
            )

            if score < best_score:
                best_score = score
                best_split = _describe_split_windows(
                    method="regime-balanced",
                    index=returns_frame.index,
                    train_window=train_window,
                    validation_window=validation_window,
                    test_window=test_window,
                    regime_labels=regime_labels,
                    search_step=search_step,
                    target_rows={
                        "train": target_train_rows,
                        "validation": target_validation_rows,
                        "test": target_test_rows,
                    },
                )

    if best_split is None:
        raise ValueError("Could not identify a feasible regime-balanced split.")
    return best_split


def generate_walk_forward_splits(
    returns: pd.DataFrame | pd.Series,
    benchmark_returns: pd.Series | None = None,
    lookback: int = 20,
    validation_rows: int | None = None,
    test_rows: int | None = None,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.10,
    step_rows: int | None = None,
    min_train_rows: int | None = None,
    max_splits: int | None = None,
) -> list[SuggestedTimeSeriesSplit]:
    returns_frame = _coerce_returns_frame(returns)
    total_rows = len(returns_frame)
    minimum_window = max(lookback + 2, min_train_rows or 0)
    if validation_rows is None:
        validation_rows = max(minimum_window, int(round(total_rows * validation_fraction)))
    if test_rows is None:
        test_rows = max(minimum_window, int(round(total_rows * test_fraction)))
    if validation_rows < minimum_window or test_rows < minimum_window:
        raise ValueError("validation_rows and test_rows must each be at least lookback + 2.")
    if step_rows is None:
        step_rows = test_rows
    if step_rows <= 0:
        raise ValueError("step_rows must be positive.")

    regime_labels = build_regime_labels(returns_frame, benchmark_returns=benchmark_returns, lookback=lookback)
    target_train_rows = total_rows - validation_rows - test_rows
    if target_train_rows < minimum_window:
        target_train_rows = minimum_window

    last_test_end = total_rows
    candidate_splits: list[SuggestedTimeSeriesSplit] = []
    while True:
        validation_end = last_test_end - test_rows
        train_end = validation_end - validation_rows
        if train_end < minimum_window:
            break

        train_window = _window_from_bounds(returns_frame.index, 0, train_end)
        validation_window = _window_from_bounds(returns_frame.index, train_end, validation_end)
        test_window = _window_from_bounds(returns_frame.index, validation_end, last_test_end)
        candidate_splits.append(
            _describe_split_windows(
                method="walk-forward-anchored",
                index=returns_frame.index,
                train_window=train_window,
                validation_window=validation_window,
                test_window=test_window,
                regime_labels=regime_labels,
                search_step=step_rows,
                target_rows={
                    "train": target_train_rows,
                    "validation": validation_rows,
                    "test": test_rows,
                },
            )
        )
        last_test_end -= step_rows

    candidate_splits = list(reversed(candidate_splits))
    if max_splits is not None and max_splits > 0:
        candidate_splits = candidate_splits[-max_splits:]
    if not candidate_splits:
        raise ValueError("Could not build any feasible walk-forward splits.")
    return candidate_splits
