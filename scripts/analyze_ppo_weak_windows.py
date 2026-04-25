from __future__ import annotations

import argparse
from pathlib import Path

from wealth_first.ppo_analysis_common import requested_splits, write_json


def _requested_splits(regime_prefix: str | None, chrono_prefix: str | None) -> list[tuple[str, str]]:
    return requested_splits(regime_prefix, chrono_prefix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize which PPO split prefixes are requested.")
    parser.add_argument("--regime-prefix")
    parser.add_argument("--chrono-prefix")
    parser.add_argument("--output-prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits = _requested_splits(args.regime_prefix, args.chrono_prefix)
    if args.output_prefix:
        output_prefix = Path(args.output_prefix)
        write_json(
            output_prefix.with_name(f"{output_prefix.name}_summary.json"),
            {"requested_splits": [{"split": split, "prefix": prefix} for split, prefix in splits]},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
