from __future__ import annotations

import pandas as pd


def build_demo_strategy_sleeves(
    prices: pd.DataFrame,
    base_symbol: str = "SPY",
    warmup_period: int = 200,
) -> pd.DataFrame:
    if base_symbol not in prices.columns:
        raise ValueError(f"Base symbol '{base_symbol}' was not found in the price history.")

    base_prices = prices[base_symbol].dropna().astype(float)
    if len(base_prices) <= warmup_period:
        raise ValueError("Not enough price history to build the demo strategy sleeves.")

    base_returns = base_prices.pct_change().dropna()

    fast_trend = base_prices.rolling(50).mean()
    slow_trend = base_prices.rolling(200).mean()
    trend_signal = (fast_trend > slow_trend).shift(1).reindex(base_returns.index).fillna(False).astype(float)
    trend_following_returns = trend_signal * base_returns

    pullback_signal = (base_prices.pct_change(5) < -0.03).shift(1).reindex(base_returns.index).fillna(False).astype(float)
    mean_reversion_returns = pullback_signal * base_returns

    realized_volatility = base_returns.rolling(20).std()
    volatility_threshold = realized_volatility.rolling(60).median()
    hedge_signal = (realized_volatility > volatility_threshold).shift(1).reindex(base_returns.index).fillna(False).astype(float)
    hedge_overlay_returns = hedge_signal * (-0.35 * base_returns)

    sleeves = pd.DataFrame(
        {
            "TREND_FOLLOWING": trend_following_returns,
            "MEAN_REVERSION": mean_reversion_returns,
            "HEDGE_OVERLAY": hedge_overlay_returns,
            f"{base_symbol}_BENCHMARK": base_returns,
        },
        index=base_returns.index,
    ).fillna(0.0)

    if len(sleeves) <= warmup_period:
        raise ValueError("Not enough post-warmup observations to build the demo strategy sleeves.")
    return sleeves.iloc[warmup_period:].copy()