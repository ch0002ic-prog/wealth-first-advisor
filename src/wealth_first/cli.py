from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wealth_first.backtest import run_rolling_backtest
from wealth_first.data import download_returns, load_returns_csv
from wealth_first.optimizer import WealthFirstConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a wealth-first optimization backtest.")
    parser.add_argument("--symbols", nargs="+", default=["SPY"], help="Tickers to include in the optimization universe.")
    parser.add_argument("--returns-csv", default=None, help="Optional CSV file containing precomputed return streams for multiple sleeves.")
    parser.add_argument("--date-column", default=None, help="Optional date column name when loading returns from CSV.")
    parser.add_argument("--benchmark-column", default=None, help="Optional benchmark return column, especially for returns CSV workflows.")
    parser.add_argument(
        "--exclude-benchmark-from-universe",
        action="store_true",
        help="When using a returns CSV, remove the benchmark column from the optimization universe while still reporting it as a benchmark.",
    )
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date.")
    parser.add_argument("--end", default=None, help="Optional backtest end date.")
    parser.add_argument("--lookback", type=int, default=63, help="Number of observations used for each optimization window.")
    parser.add_argument("--rebalance-frequency", type=int, default=5, help="How often to re-optimize, in periods.")
    parser.add_argument("--objective-mode", choices=["piecewise", "log_wealth"], default="piecewise")
    parser.add_argument("--loss-penalty", type=float, default=8.0)
    parser.add_argument("--gain-reward", type=float, default=1.0)
    parser.add_argument("--gain-power", type=float, default=1.0)
    parser.add_argument("--loss-power", type=float, default=1.5)
    parser.add_argument("--benchmark-gain-reward", type=float, default=0.0, help="Reward applied to benchmark-relative upside when benchmark returns are supplied.")
    parser.add_argument("--benchmark-loss-penalty", type=float, default=0.0, help="Penalty applied to benchmark-relative downside when benchmark returns are supplied.")
    parser.add_argument("--turnover-penalty", type=float, default=0.001)
    parser.add_argument("--weight-reg", type=float, default=0.0001)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0, help="One-way transaction cost in basis points per unit traded.")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="One-way slippage in basis points per unit traded.")
    parser.add_argument(
        "--weight-bound",
        action="append",
        default=[],
        metavar="SYMBOL=MIN:MAX",
        help="Optional per-sleeve min/max allocation override. May be repeated and can include CASH.",
    )
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--no-cash", action="store_true", help="Disable the synthetic cash sleeve.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for CSV outputs.")
    return parser


def _format_metric(name: str, value: float) -> str:
    percent_metrics = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "hit_rate",
        "loss_rate",
        "average_step_return",
        "worst_step_return",
        "best_step_return",
        "benchmark_total_return",
        "relative_total_return",
        "average_active_return",
        "tracking_error",
        "gross_total_return",
        "average_turnover",
        "total_turnover",
        "average_trading_cost",
        "cost_drag",
        "average_cash_weight",
        "average_risky_weight",
    }
    if name in percent_metrics:
        return f"{value:.2%}"
    return f"{value:.4f}"


def _print_summary(summary: pd.Series) -> None:
    print("Summary")
    for name, value in summary.items():
        print(f"  {name}: {_format_metric(name, float(value))}")


def _filter_returns_by_date(
    returns: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex):
        return returns

    filtered = returns.sort_index()
    if start:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start)]
    if end:
        filtered = filtered.loc[filtered.index <= pd.Timestamp(end)]
    if filtered.empty:
        raise ValueError("No return observations remain after applying the requested date filter.")
    return filtered


def _parse_weight_bound_overrides(raw_bounds: list[str]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    min_overrides: dict[str, float] = {}
    max_overrides: dict[str, float] = {}

    for raw_bound in raw_bounds:
        symbol_text, separator, bounds_text = raw_bound.partition("=")
        if not separator or ":" not in bounds_text:
            raise ValueError(f"Invalid weight bound '{raw_bound}'. Expected SYMBOL=MIN:MAX.")

        minimum_text, maximum_text = bounds_text.split(":", 1)
        symbol = symbol_text.strip()
        if not symbol:
            raise ValueError(f"Invalid weight bound '{raw_bound}'. Sleeve name cannot be empty.")
        if symbol in min_overrides:
            raise ValueError(f"Duplicate weight bound provided for '{symbol}'.")

        min_overrides[symbol] = float(minimum_text.strip())
        max_overrides[symbol] = float(maximum_text.strip())

    return (min_overrides or None, max_overrides or None)


def _save_outputs(output_dir: Path, backtest_result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_result.weights.to_csv(output_dir / "weights.csv")
    backtest_result.ending_weights.to_csv(output_dir / "ending_weights.csv")
    backtest_result.gross_portfolio_returns.to_csv(output_dir / "gross_portfolio_returns.csv", header=True)
    backtest_result.portfolio_returns.to_csv(output_dir / "portfolio_returns.csv", header=True)
    backtest_result.turnover.to_csv(output_dir / "turnover.csv", header=True)
    backtest_result.trading_costs.to_csv(output_dir / "trading_costs.csv", header=True)
    backtest_result.wealth_index.to_csv(output_dir / "wealth_index.csv", header=True)
    if backtest_result.benchmark_wealth_index is not None:
        backtest_result.benchmark_wealth_index.to_csv(output_dir / "benchmark_wealth_index.csv", header=True)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    min_weight_overrides, max_weight_overrides = _parse_weight_bound_overrides(args.weight_bound)

    benchmark_returns = None
    benchmark_symbol = args.benchmark_column
    benchmark_aware_requested = args.benchmark_gain_reward > 0 or args.benchmark_loss_penalty > 0
    if args.returns_csv:
        if benchmark_aware_requested and benchmark_symbol is None:
            raise ValueError("Benchmark-aware optimization with returns CSV requires --benchmark-column.")
        returns = load_returns_csv(args.returns_csv, date_column=args.date_column)
        returns = _filter_returns_by_date(returns, start=args.start, end=args.end)
        if benchmark_symbol and args.exclude_benchmark_from_universe:
            if benchmark_symbol not in returns.columns:
                raise ValueError(f"Benchmark column '{benchmark_symbol}' was not found in the returns CSV.")
            benchmark_returns = returns[benchmark_symbol].copy()
            returns = returns.drop(columns=[benchmark_symbol])
    else:
        returns = download_returns(symbols=args.symbols, start=args.start, end=args.end)
        if benchmark_symbol is None:
            benchmark_symbol = args.symbols[0].upper()

    config = WealthFirstConfig(
        loss_penalty=args.loss_penalty,
        gain_reward=args.gain_reward,
        gain_power=args.gain_power,
        loss_power=args.loss_power,
        benchmark_gain_reward=args.benchmark_gain_reward,
        benchmark_loss_penalty=args.benchmark_loss_penalty,
        turnover_penalty=args.turnover_penalty,
        weight_reg=args.weight_reg,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        objective_mode=args.objective_mode,
        min_weight_overrides=min_weight_overrides,
        max_weight_overrides=max_weight_overrides,
        include_cash=not args.no_cash,
    )
    backtest_result = run_rolling_backtest(
        returns=returns,
        lookback=args.lookback,
        rebalance_frequency=args.rebalance_frequency,
        config=config,
        periods_per_year=args.periods_per_year,
        benchmark_symbol=benchmark_symbol,
        benchmark_returns=benchmark_returns,
    )

    _print_summary(backtest_result.summary)
    print("\nLatest weights")
    latest_weights = backtest_result.weights.iloc[-1].sort_values(ascending=False)
    for symbol, weight in latest_weights.items():
        print(f"  {symbol}: {weight:.2%}")

    if backtest_result.benchmark_wealth_index is not None:
        benchmark_total_return = backtest_result.benchmark_wealth_index.iloc[-1] / backtest_result.initial_wealth - 1.0
        print(f"\nBenchmark total return: {benchmark_total_return:.2%}")

    if args.output_dir:
        _save_outputs(Path(args.output_dir), backtest_result)
        print(f"Saved CSV outputs to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())