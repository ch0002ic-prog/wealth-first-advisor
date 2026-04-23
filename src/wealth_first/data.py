from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    if isinstance(symbols, str):
        return [symbols]
    normalized = [str(symbol).upper() for symbol in symbols]
    if not normalized:
        raise ValueError("At least one symbol is required.")
    return normalized


def download_price_history(
    symbols: str | Sequence[str],
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to download market data.") from exc

    normalized_symbols = _normalize_symbols(symbols)
    raw_data = yf.download(
        normalized_symbols,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if raw_data.empty:
        raise ValueError("No market data was returned for the requested symbols and date range.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        for price_field in ("Close", "Adj Close"):
            if price_field in raw_data.columns.get_level_values(0):
                prices = raw_data[price_field]
                break
        else:
            raise ValueError("Could not locate a close price column in the downloaded data.")
    else:
        if "Close" in raw_data.columns:
            prices = raw_data[["Close"]].rename(columns={"Close": normalized_symbols[0]})
        elif "Adj Close" in raw_data.columns:
            prices = raw_data[["Adj Close"]].rename(columns={"Adj Close": normalized_symbols[0]})
        else:
            raise ValueError("Could not locate a close price column in the downloaded data.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=normalized_symbols[0])

    prices = prices.dropna(how="all")
    prices.columns = [str(column).upper() for column in prices.columns]
    return prices


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Price history is empty.")
    returns = prices.sort_index().pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough price history to compute returns.")
    return returns


def download_returns(
    symbols: str | Sequence[str],
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    return prices_to_returns(download_price_history(symbols=symbols, start=start, end=end, auto_adjust=auto_adjust))


def load_returns_csv(
    file_path: str | Path,
    date_column: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    if frame.empty:
        raise ValueError("The returns CSV is empty.")

    frame.columns = [str(column) for column in frame.columns]
    datetime_index = None
    if date_column is not None:
        if date_column not in frame.columns:
            raise ValueError(f"Date column '{date_column}' was not found in the returns CSV.")
        datetime_index = pd.to_datetime(frame[date_column], errors="coerce")
        if datetime_index.isna().any():
            raise ValueError(f"Date column '{date_column}' could not be parsed as datetimes.")
        frame = frame.drop(columns=[date_column])
    else:
        first_column = frame.columns[0]
        parsed_first_column = pd.to_datetime(frame[first_column], errors="coerce")
        if parsed_first_column.notna().all() and not pd.api.types.is_numeric_dtype(frame[first_column]):
            datetime_index = parsed_first_column
            frame = frame.drop(columns=[first_column])

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    if numeric_frame.empty or numeric_frame.shape[1] == 0:
        raise ValueError("The returns CSV does not contain any numeric strategy columns.")

    if datetime_index is not None:
        numeric_frame.index = pd.DatetimeIndex(datetime_index, name="date")

    numeric_frame = numeric_frame.dropna(how="any")
    if numeric_frame.empty:
        raise ValueError("The returns CSV contains no complete numeric rows after parsing.")

    return numeric_frame.sort_index()


def add_cash_sleeve(
    returns: pd.DataFrame,
    cash_symbol: str = "CASH",
    cash_return: float = 0.0,
) -> pd.DataFrame:
    if cash_symbol in returns.columns:
        return returns.copy()

    augmented = returns.copy()
    augmented[cash_symbol] = float(cash_return)
    ordered_columns = [column for column in returns.columns] + [cash_symbol]
    return augmented[ordered_columns]