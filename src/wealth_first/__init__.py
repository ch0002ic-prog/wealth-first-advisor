from __future__ import annotations

from typing import TYPE_CHECKING

from wealth_first.backtest import BacktestResult, run_rolling_backtest, summarize_performance
from wealth_first.data import add_cash_sleeve, download_price_history, download_returns, load_returns_csv, prices_to_returns
from wealth_first.execution import ExecutionPlan, ExchangeAdapter, PaperExchangeAdapter, TargetOrder, WebhookExchangeAdapter, build_execution_plan
from wealth_first.optimizer import OptimizationResult, WealthFirstConfig, optimize_weights, wealth_first_objective
from wealth_first.sleeves import build_demo_strategy_sleeves

if TYPE_CHECKING:
    from wealth_first.compare import ColumnComparison, ComparisonReport

__all__ = [
    "BacktestResult",
    "ColumnComparison",
    "ComparisonReport",
    "ExecutionPlan",
    "ExchangeAdapter",
    "OptimizationResult",
    "PaperExchangeAdapter",
    "TargetOrder",
    "WealthFirstConfig",
    "WebhookExchangeAdapter",
    "add_cash_sleeve",
    "build_execution_plan",
    "build_demo_strategy_sleeves",
    "compare_return_streams",
    "download_price_history",
    "download_returns",
    "load_returns_csv",
    "optimize_weights",
    "prices_to_returns",
    "run_rolling_backtest",
    "summarize_performance",
    "wealth_first_objective",
]


def __getattr__(name: str):
    if name in {"ColumnComparison", "ComparisonReport", "compare_return_streams"}:
        from wealth_first.compare import ColumnComparison, ComparisonReport, compare_return_streams

        lazy_exports = {
            "ColumnComparison": ColumnComparison,
            "ComparisonReport": ComparisonReport,
            "compare_return_streams": compare_return_streams,
        }
        value = lazy_exports[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module 'wealth_first' has no attribute '{name}'")