from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wealth_first.data import add_cash_sleeve
from wealth_first.optimizer import WealthFirstConfig, _coerce_returns_frame, optimize_weights
from wealth_first.rebalance import compute_execution_cost, compute_tradable_turnover


@dataclass(frozen=True)
class BacktestResult:
    initial_wealth: float
    weights: pd.DataFrame
    ending_weights: pd.DataFrame
    gross_portfolio_returns: pd.Series
    portfolio_returns: pd.Series
    turnover: pd.Series
    trading_costs: pd.Series
    wealth_ratios: pd.Series
    wealth_index: pd.Series
    benchmark_wealth_index: pd.Series | None
    summary: pd.Series


def _update_post_return_weights(weights: pd.Series, realized_returns: pd.Series, epsilon: float) -> pd.Series:
    period_values = weights * (1.0 + realized_returns)
    total_value = float(period_values.sum())
    if total_value <= epsilon:
        return weights.copy()
    return period_values / total_value


def summarize_performance(
    portfolio_returns: pd.Series,
    initial_wealth: float = 1.0,
    periods_per_year: int = 252,
) -> pd.Series:
    returns = portfolio_returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("Portfolio returns are empty.")

    wealth_ratios = 1.0 + returns
    wealth_index = initial_wealth * wealth_ratios.cumprod()
    terminal_multiple = wealth_index.iloc[-1] / initial_wealth
    annualized_return = terminal_multiple ** (periods_per_year / len(returns)) - 1.0
    annualized_volatility = returns.std(ddof=0) * np.sqrt(periods_per_year)
    downside_returns = returns.clip(upper=0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(periods_per_year)
    max_drawdown = float((wealth_index / wealth_index.cummax() - 1.0).min())

    return pd.Series(
        {
            "periods": float(len(returns)),
            "total_return": float(wealth_index.iloc[-1] / initial_wealth - 1.0),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility),
            "max_drawdown": max_drawdown,
            "hit_rate": float((returns > 0.0).mean()),
            "loss_rate": float((returns < 0.0).mean()),
            "average_step_return": float(returns.mean()),
            "worst_step_return": float(returns.min()),
            "best_step_return": float(returns.max()),
            "sharpe_like": float(annualized_return / annualized_volatility) if annualized_volatility > 0 else np.nan,
            "sortino_like": float(annualized_return / downside_deviation) if downside_deviation > 0 else np.nan,
        },
        name="summary",
    )


def run_rolling_backtest(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    lookback: int = 63,
    rebalance_frequency: int = 1,
    config: WealthFirstConfig | None = None,
    initial_wealth: float = 1.0,
    periods_per_year: int = 252,
    benchmark_symbol: str | None = None,
    benchmark_returns: pd.Series | None = None,
) -> BacktestResult:
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")
    if rebalance_frequency <= 0:
        raise ValueError("rebalance_frequency must be positive.")

    effective_config = config or WealthFirstConfig()
    base_returns = _coerce_returns_frame(returns).sort_index()
    working_returns = add_cash_sleeve(base_returns, effective_config.cash_symbol) if effective_config.include_cash else base_returns.copy()
    if len(working_returns) <= lookback:
        raise ValueError("Not enough return observations for the requested lookback.")

    benchmark_series = None
    if benchmark_returns is not None:
        benchmark_series = pd.Series(benchmark_returns, copy=True).astype(float).sort_index().reindex(base_returns.index)
        if benchmark_series.isna().any():
            raise ValueError("Benchmark returns could not be aligned to the backtest index.")
    else:
        benchmark_column = benchmark_symbol if benchmark_symbol in base_returns.columns else None
        if benchmark_column is None and base_returns.shape[1] == 1:
            benchmark_column = str(base_returns.columns[0])
        if benchmark_column is not None:
            benchmark_series = base_returns.loc[:, benchmark_column].astype(float)

    if (effective_config.benchmark_gain_reward > 0 or effective_config.benchmark_loss_penalty > 0) and benchmark_series is None:
        raise ValueError("Benchmark-aware optimization requires benchmark returns or a benchmark symbol.")

    backtest_index = working_returns.index[lookback:]
    current_weights: pd.Series | None = None
    weight_history: list[np.ndarray] = []
    ending_weight_history: list[np.ndarray] = []
    gross_return_history: list[float] = []
    portfolio_return_history: list[float] = []
    turnover_history: list[float] = []
    trading_cost_history: list[float] = []

    for offset, row_number in enumerate(range(lookback, len(working_returns))):
        turnover = 0.0
        if current_weights is None or offset % rebalance_frequency == 0:
            estimation_window = working_returns.iloc[row_number - lookback : row_number]
            optimization_result = optimize_weights(
                estimation_window,
                config=effective_config,
                previous_weights=current_weights,
                initial_weights=current_weights,
                benchmark_returns=None if benchmark_series is None else benchmark_series.iloc[row_number - lookback : row_number],
            )
            target_weights = optimization_result.weights
            turnover = compute_tradable_turnover(target_weights, current_weights, effective_config.cash_symbol)
            current_weights = target_weights

        realized_returns = working_returns.iloc[row_number]
        gross_portfolio_return = float(realized_returns @ current_weights)
        execution_cost = compute_execution_cost(
            turnover,
            effective_config.transaction_cost_bps,
            effective_config.slippage_bps,
        )
        gross_wealth_ratio = max(1.0 + gross_portfolio_return, effective_config.epsilon)
        net_wealth_ratio = max(1.0 - execution_cost, 0.0) * gross_wealth_ratio
        gross_return_history.append(gross_portfolio_return)
        portfolio_return_history.append(net_wealth_ratio - 1.0)
        turnover_history.append(turnover)
        trading_cost_history.append(execution_cost)
        weight_history.append(current_weights.to_numpy(copy=True))
        current_weights = _update_post_return_weights(current_weights, realized_returns, effective_config.epsilon)
        ending_weight_history.append(current_weights.to_numpy(copy=True))

    weights = pd.DataFrame(weight_history, index=backtest_index, columns=working_returns.columns)
    ending_weights = pd.DataFrame(ending_weight_history, index=backtest_index, columns=working_returns.columns)
    gross_portfolio_returns = pd.Series(gross_return_history, index=backtest_index, name="gross_portfolio_return")
    portfolio_returns = pd.Series(portfolio_return_history, index=backtest_index, name="portfolio_return")
    turnover_series = pd.Series(turnover_history, index=backtest_index, name="turnover")
    trading_costs = pd.Series(trading_cost_history, index=backtest_index, name="trading_cost")
    wealth_ratios = 1.0 + portfolio_returns
    wealth_index = initial_wealth * wealth_ratios.cumprod()
    gross_wealth_index = initial_wealth * (1.0 + gross_portfolio_returns).cumprod()
    summary = summarize_performance(
        portfolio_returns=portfolio_returns,
        initial_wealth=initial_wealth,
        periods_per_year=periods_per_year,
    )
    summary.loc["gross_total_return"] = float(gross_wealth_index.iloc[-1] / initial_wealth - 1.0)
    summary.loc["average_turnover"] = float(turnover_series.mean())
    summary.loc["total_turnover"] = float(turnover_series.sum())
    summary.loc["average_trading_cost"] = float(trading_costs.mean())
    summary.loc["cost_drag"] = float(gross_wealth_index.iloc[-1] / initial_wealth - wealth_index.iloc[-1] / initial_wealth)

    if effective_config.cash_symbol in weights.columns:
        summary.loc["average_cash_weight"] = float(weights[effective_config.cash_symbol].mean())
        summary.loc["average_risky_weight"] = float(1.0 - weights[effective_config.cash_symbol].mean())

    benchmark_wealth_index = None
    if benchmark_series is not None:
        benchmark_series = benchmark_series.reindex(backtest_index)
        if benchmark_series.isna().any():
            raise ValueError("Benchmark returns could not be aligned to the backtest index.")
        benchmark_wealth_index = initial_wealth * (1.0 + benchmark_series).cumprod()
        active_returns = portfolio_returns - benchmark_series
        tracking_error = float(active_returns.std(ddof=0) * np.sqrt(periods_per_year))
        annualized_active_return = float(active_returns.mean() * periods_per_year)
        summary.loc["benchmark_total_return"] = float(benchmark_wealth_index.iloc[-1] / initial_wealth - 1.0)
        summary.loc["relative_total_return"] = float(wealth_index.iloc[-1] / benchmark_wealth_index.iloc[-1] - 1.0)
        summary.loc["average_active_return"] = float(active_returns.mean())
        summary.loc["tracking_error"] = tracking_error
        summary.loc["information_ratio"] = annualized_active_return / tracking_error if tracking_error > 0 else np.nan

    return BacktestResult(
        initial_wealth=initial_wealth,
        weights=weights,
        ending_weights=ending_weights,
        gross_portfolio_returns=gross_portfolio_returns,
        portfolio_returns=portfolio_returns,
        turnover=turnover_series,
        trading_costs=trading_costs,
        wealth_ratios=wealth_ratios,
        wealth_index=wealth_index,
        benchmark_wealth_index=benchmark_wealth_index,
        summary=summary,
    )