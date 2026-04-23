from __future__ import annotations

import pandas as pd


def compute_tradable_turnover(
    target_weights: pd.Series,
    current_weights: pd.Series | None,
    cash_symbol: str,
) -> float:
    if current_weights is None:
        return 0.0

    weight_deltas = (target_weights - current_weights).drop(labels=[cash_symbol], errors="ignore")
    return float(weight_deltas.abs().sum())


def compute_execution_cost(
    turnover: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> float:
    cost_bps = max(transaction_cost_bps, 0.0) + max(slippage_bps, 0.0)
    return max(turnover, 0.0) * (cost_bps / 10_000.0)