from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wealth_first.data import download_price_history
from wealth_first.sleeves import build_demo_strategy_sleeves


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build demo multi-sleeve return streams from SPY price history.")
    parser.add_argument("--symbol", default="SPY", help="Base symbol used to derive the demo sleeves.")
    parser.add_argument("--start", default="2010-01-01", help="Price history start date.")
    parser.add_argument("--end", default=None, help="Optional price history end date.")
    parser.add_argument("--output", default="data/demo_sleeves.csv", help="CSV file path for the generated sleeve returns.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    prices = download_price_history(symbols=[args.symbol], start=args.start, end=args.end)
    sleeves = build_demo_strategy_sleeves(prices, base_symbol=args.symbol.upper())

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sleeves.to_csv(output_path, index_label="date")
    print(f"Saved {len(sleeves)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())