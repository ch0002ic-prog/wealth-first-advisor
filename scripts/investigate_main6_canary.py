#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main6_canary_runs"

DEFAULT_RETURNS_CSV = "data/demo_sleeves.csv"
DEFAULT_BENCHMARK_COLUMN = "SPY_BENCHMARK"
DEFAULT_DATE_COLUMN = "date"
DEFAULT_SCENARIOS = [
    ("baseline_b20_s4242", 20, 4242),
    ("altblock_b16_s4242", 16, 4242),
    ("altseed_b20_s5252", 20, 5252),
    ("deepcheck_b12_s8080", 12, 8080),
]


@dataclass(frozen=True)
class Candidate:
    """Candidate configuration. All fields must be set explicitly in candidate definitions.
    
    Note: path bootstrap block_size and seed are scenario-driven. Validation tail bootstrap
    block_size can optionally be overridden per candidate for sensitivity sweeps.
    All other optimization knobs must have explicit values to avoid silent inheritance
    of promoted main6 defaults.
    """
    name: str
    # Core knobs: no-trade band and turnover penalty
    no_trade_band: float = 0.005  # main6 promoted default
    scale_turnover_penalty: float = 2.9952  # main6 promoted default
    # Validation floor constraints
    validation_relative_floor_target: float | None = 0.00015  # main6 promoted default
    validation_relative_floor_penalty: float = 3.0  # main6 promoted default
    validation_hard_min_relative_return: float | None = None
    # Tail-risk bootstrap settings
    validation_tail_bootstrap_reps: int = 320  # main6 promoted default
    validation_tail_bootstrap_block_size: int | None = None  # None => scenario block size
    validation_tail_bootstrap_quantile: float = 0.05  # main6 promoted default
    validation_tail_bootstrap_floor_target: float | None = -0.003  # main6 promoted default
    validation_tail_bootstrap_penalty: float = 2.0  # main6 promoted default
    validation_tail_bootstrap_hard_min: float | None = -0.007  # main6 promoted default
    validation_tail_bootstrap_objective_weight: float = 0.5  # main6 promoted default
    # Path bootstrap settings
    path_bootstrap_reps: int = 80  # main6 promoted default


CANDIDATES = [
    # baseline: old defaults for comparison (high ntb, zero-weight tail risk optimization)
    Candidate(
        name="baseline",
        no_trade_band=0.02,
        scale_turnover_penalty=0.0,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=0,
        validation_tail_bootstrap_floor_target=None,
        validation_tail_bootstrap_penalty=0.0,
        validation_tail_bootstrap_hard_min=None,
        validation_tail_bootstrap_objective_weight=0.0,
        path_bootstrap_reps=0,
    ),
    # baseline_ntb010: tighter ntb, otherwise old defaults
    Candidate(
        name="baseline_ntb010",
        no_trade_band=0.01,
        scale_turnover_penalty=0.0,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=0,
        validation_tail_bootstrap_floor_target=None,
        validation_tail_bootstrap_penalty=0.0,
        validation_tail_bootstrap_hard_min=None,
        validation_tail_bootstrap_objective_weight=0.0,
        path_bootstrap_reps=0,
    ),
    # baseline_ntb005: promoted default ntb, otherwise old defaults
    Candidate(
        name="baseline_ntb005",
        no_trade_band=0.005,
        scale_turnover_penalty=0.0,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=0,
        validation_tail_bootstrap_floor_target=None,
        validation_tail_bootstrap_penalty=0.0,
        validation_tail_bootstrap_hard_min=None,
        validation_tail_bootstrap_objective_weight=0.0,
        path_bootstrap_reps=0,
    ),
    # objw05: turnover + tail-risk optimization, ntb=0.02
    Candidate(
        name="objw05",
        no_trade_band=0.02,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw10: higher tail-risk weight, ntb=0.02
    Candidate(
        name="objw10",
        no_trade_band=0.02,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=1.0,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010: turnover + tail-risk optimization, ntb=0.01
    Candidate(
        name="objw05_ntb010",
        no_trade_band=0.01,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005: turnover + tail-risk optimization, ntb=0.005 (promoted default)
    Candidate(
        name="objw05_ntb005",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010_pen150: lower turnover penalty, ntb=0.01
    Candidate(
        name="objw05_ntb010_pen150",
        no_trade_band=0.01,
        scale_turnover_penalty=1.5,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010_pen100: even lower turnover penalty, ntb=0.01
    Candidate(
        name="objw05_ntb010_pen100",
        no_trade_band=0.01,
        scale_turnover_penalty=1.0,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010_floor: + relative floor constraint, ntb=0.01
    Candidate(
        name="objw05_ntb010_floor",
        no_trade_band=0.01,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0002,
        validation_relative_floor_penalty=4.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010_hardmin: + hard min return constraint, ntb=0.01
    Candidate(
        name="objw05_ntb010_hardmin",
        no_trade_band=0.01,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_hard_min_relative_return=0.0001,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010_floor_hardmin: + both floor and hard-min, ntb=0.01
    Candidate(
        name="objw05_ntb010_floor_hardmin",
        no_trade_band=0.01,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0002,
        validation_relative_floor_penalty=4.0,
        validation_hard_min_relative_return=0.0001,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010_floor30: higher floor target, ntb=0.01
    Candidate(
        name="objw05_ntb010_floor30",
        no_trade_band=0.01,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0003,
        validation_relative_floor_penalty=4.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb010_floor35: even higher floor target, ntb=0.01
    Candidate(
        name="objw05_ntb010_floor35",
        no_trade_band=0.01,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00035,
        validation_relative_floor_penalty=4.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor: floor constraint with promoted ntb=0.005
    Candidate(
        name="objw05_ntb005_floor",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0002,
        validation_relative_floor_penalty=4.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3: lower floor target and lower floor penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p3",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t20_p3: incumbent floor target with lower floor penalty
    Candidate(
        name="objw05_ntb005_floor_t20_p3",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0002,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t25_p3: higher floor target with lower floor penalty
    Candidate(
        name="objw05_ntb005_floor_t25_p3",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00025,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_pen26: lower turnover penalty sensitivity probe
    Candidate(
        name="objw05_ntb005_floor_t15_p3_pen26",
        no_trade_band=0.005,
        scale_turnover_penalty=2.6,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_pen30: near-current turnover penalty sensitivity probe
    Candidate(
        name="objw05_ntb005_floor_t15_p3_pen30",
        no_trade_band=0.005,
        scale_turnover_penalty=3.0,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_pen34: higher turnover penalty sensitivity probe
    Candidate(
        name="objw05_ntb005_floor_t15_p3_pen34",
        no_trade_band=0.005,
        scale_turnover_penalty=3.4,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailf25_pen15: looser tail floor, lighter penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailf25_pen15",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.0025,
        validation_tail_bootstrap_penalty=1.5,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailf25_pen20: looser tail floor, baseline penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailf25_pen20",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.0025,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailf35_pen20: stricter tail floor, baseline penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailf35_pen20",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.0035,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailf30_pen25: baseline tail floor, stronger penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailf30_pen25",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.5,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailw10: higher tail objective weight
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailw10",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=1.0,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailw15: much higher tail objective weight
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailw15",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=1.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailhm65: stricter tail hard minimum
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailhm65",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.0065,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailhm60: substantially stricter tail hard minimum
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailhm60",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.006,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr160: higher tail bootstrap fidelity
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr160",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=160,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr240: much higher tail bootstrap fidelity
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr240",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=240,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p20_tailr240: slightly weaker relative-floor penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p20_tailr240",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=2.0,
        validation_tail_bootstrap_reps=240,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p25_tailr240: mildly weaker relative-floor penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p25_tailr240",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=2.5,
        validation_tail_bootstrap_reps=240,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p35_tailr240: mildly stronger relative-floor penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p35_tailr240",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.5,
        validation_tail_bootstrap_reps=240,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p40_tailr240: stronger relative-floor penalty
    Candidate(
        name="objw05_ntb005_floor_t15_p40_tailr240",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=4.0,
        validation_tail_bootstrap_reps=240,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr160_q03: higher reps with lower-tail quantile
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr160_q03",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=160,
        validation_tail_bootstrap_quantile=0.03,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr160_q07: higher reps with less extreme quantile
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr160_q07",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=160,
        validation_tail_bootstrap_quantile=0.07,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr240_q03: strongest current reps with lower quantile
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr240_q03",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=240,
        validation_tail_bootstrap_quantile=0.03,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr240_q07: strongest current reps with less extreme quantile
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr240_q07",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=240,
        validation_tail_bootstrap_quantile=0.07,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr320: higher reps at default quantile
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr320_q03: higher reps with lower-tail quantile
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q03",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_quantile=0.03,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_t15_p3_tailr320_q07: higher reps with less extreme quantile
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q07",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_quantile=0.07,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # tailr320 sensitivity grid: quantile x block-size with seed10 follow-up intent
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q04_b12",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=12,
        validation_tail_bootstrap_quantile=0.04,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q05_b12",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=12,
        validation_tail_bootstrap_quantile=0.05,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q06_b12",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=12,
        validation_tail_bootstrap_quantile=0.06,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q04_b16",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=16,
        validation_tail_bootstrap_quantile=0.04,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q05_b16",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=16,
        validation_tail_bootstrap_quantile=0.05,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q06_b16",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=16,
        validation_tail_bootstrap_quantile=0.06,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q04_b20",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=20,
        validation_tail_bootstrap_quantile=0.04,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q05_b20",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=20,
        validation_tail_bootstrap_quantile=0.05,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    Candidate(
        name="objw05_ntb005_floor_t15_p3_tailr320_q06_b20",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.00015,
        validation_relative_floor_penalty=3.0,
        validation_tail_bootstrap_reps=320,
        validation_tail_bootstrap_block_size=20,
        validation_tail_bootstrap_quantile=0.06,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
    # objw05_ntb005_floor_hardmin: floor + hard-min with promoted ntb=0.005
    Candidate(
        name="objw05_ntb005_floor_hardmin",
        no_trade_band=0.005,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0002,
        validation_relative_floor_penalty=4.0,
        validation_hard_min_relative_return=0.0001,
        validation_tail_bootstrap_reps=80,
        validation_tail_bootstrap_floor_target=-0.003,
        validation_tail_bootstrap_penalty=2.0,
        validation_tail_bootstrap_hard_min=-0.007,
        validation_tail_bootstrap_objective_weight=0.5,
        path_bootstrap_reps=80,
    ),
]


def _prepare_returns_csv(
    *,
    returns_csv: str,
    date_column: str,
    start_date: str | None,
    end_date: str | None,
    label: str,
) -> tuple[str, int]:
    frame = pd.read_csv(returns_csv)
    if frame.empty:
        raise ValueError(f"Returns CSV '{returns_csv}' is empty.")
    if date_column not in frame.columns:
        raise ValueError(f"Date column '{date_column}' not found in '{returns_csv}'.")

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if frame[date_column].isna().any():
        raise ValueError(f"Date column '{date_column}' in '{returns_csv}' contains unparseable values.")

    filtered = frame.sort_values(date_column).copy()
    if start_date is not None:
        filtered = filtered.loc[filtered[date_column] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered.loc[filtered[date_column] <= pd.Timestamp(end_date)]
    if filtered.empty:
        raise ValueError("Date filtering removed all rows from the returns CSV.")

    if start_date is None and end_date is None:
        return str(Path(returns_csv).resolve()), int(len(filtered))

    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    tmp_dir = Path(tempfile.gettempdir()) / "wealth_first_main6"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{safe_label}_returns_slice.csv"
    filtered.to_csv(tmp_path, index=False)
    return str(tmp_path), int(len(filtered))


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _build_command(
    *,
    run_dir: Path,
    returns_csv: str,
    benchmark_column: str,
    seed: int,
    cost: int,
    scenario_block_size: int,
    scenario_seed: int,
    candidate: Candidate,
) -> list[str]:
    cmd = [
        str(PYTHON_BIN),
        "-m",
        "wealth_first.main6",
        "--returns-csv",
        returns_csv,
        "--benchmark-column",
        benchmark_column,
        "--gate",
        "001",
        "--gate-scale",
        "bps",
        "--n-folds",
        "5",
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
        "--transaction-cost-bps",
        str(cost),
        "--slippage-bps",
        str(cost),
        "--no-trade-band",
        str(candidate.no_trade_band),
        "--min-signal-scale",
        "-0.30",
        "--max-signal-scale",
        "0.30",
        "--n-scale-candidates",
        "51",
        "--action-smoothing",
        "0.9231",
        "--ridge-l2",
        "1.0",
        "--min-spy-weight",
        "0.80",
        "--max-spy-weight",
        "1.05",
        "--initial-spy-weight",
        "1.0",
        "--forward-horizon",
        "21",
        "--min-robust-min-relative",
        "-0.01",
        "--min-active-fraction",
        "0.01",
        "--max-dead-fold-fraction",
        "0.20",
        "--path-bootstrap-reps",
        "80",
        "--path-bootstrap-block-size",
        str(scenario_block_size),
        "--path-bootstrap-seed",
        str(scenario_seed + seed),
        "--no-fail-on-gate",
    ]

    # Emit all candidate knobs unconditionally to avoid silent default inheritance.
    # This ensures each candidate's true configuration intent is executed at runtime.
    cmd.extend(["--scale-turnover-penalty", str(candidate.scale_turnover_penalty)])
    cmd.extend(["--validation-relative-floor-penalty", str(candidate.validation_relative_floor_penalty)])
    if candidate.validation_relative_floor_target is not None:
        cmd.extend(["--validation-relative-floor-target", str(candidate.validation_relative_floor_target)])
    if candidate.validation_hard_min_relative_return is not None:
        cmd.extend(["--validation-hard-min-relative-return", str(candidate.validation_hard_min_relative_return)])
    cmd.extend(["--validation-tail-bootstrap-reps", str(candidate.validation_tail_bootstrap_reps)])
    if candidate.validation_tail_bootstrap_reps > 0:
        tail_block_size = (
            scenario_block_size
            if candidate.validation_tail_bootstrap_block_size is None
            else int(candidate.validation_tail_bootstrap_block_size)
        )
        cmd.extend(
            [
                "--validation-tail-bootstrap-block-size",
                str(tail_block_size),
                "--validation-tail-bootstrap-quantile",
                str(candidate.validation_tail_bootstrap_quantile),
                "--validation-tail-bootstrap-seed",
                str(scenario_seed + seed),
            ]
        )
    if candidate.validation_tail_bootstrap_floor_target is not None:
        cmd.extend(
            [
                "--validation-tail-bootstrap-floor-target",
                str(candidate.validation_tail_bootstrap_floor_target),
            ]
        )
    cmd.extend(["--validation-tail-bootstrap-penalty", str(candidate.validation_tail_bootstrap_penalty)])
    if candidate.validation_tail_bootstrap_hard_min is not None:
        cmd.extend(
            [
                "--validation-tail-bootstrap-hard-min",
                str(candidate.validation_tail_bootstrap_hard_min),
            ]
        )
    cmd.extend(
        [
            "--validation-tail-bootstrap-objective-weight",
            str(candidate.validation_tail_bootstrap_objective_weight),
        ]
    )
    cmd.extend(["--path-bootstrap-reps", str(candidate.path_bootstrap_reps)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-period canary investigation for main6")
    parser.add_argument("--label", type=str, default="canary")
    parser.add_argument("--returns-csv", type=str, default=DEFAULT_RETURNS_CSV)
    parser.add_argument("--benchmark-column", type=str, default=DEFAULT_BENCHMARK_COLUMN)
    parser.add_argument("--date-column", type=str, default=DEFAULT_DATE_COLUMN)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="7,17")
    parser.add_argument("--costs", type=str, default="5,10,15")
    parser.add_argument(
        "--candidates",
        type=str,
        default=",".join(candidate.name for candidate in CANDIDATES),
    )
    args = parser.parse_args()

    selected_names = {name.strip() for name in args.candidates.split(",") if name.strip()}
    selected_candidates = [candidate for candidate in CANDIDATES if candidate.name in selected_names]
    missing = sorted(selected_names - {candidate.name for candidate in selected_candidates})
    if missing:
        raise SystemExit(
            "Unknown candidates: "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(candidate.name for candidate in CANDIDATES)
        )

    returns_csv, row_count = _prepare_returns_csv(
        returns_csv=args.returns_csv,
        date_column=args.date_column,
        start_date=args.start_date,
        end_date=args.end_date,
        label=args.label,
    )
    seeds = _parse_csv_ints(args.seeds)
    costs = _parse_csv_ints(args.costs)

    run_root = RUN_ROOT / args.label
    run_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running main6 canary: {len(selected_candidates)} candidates x {len(DEFAULT_SCENARIOS)} scenarios x "
        f"{len(seeds)} seeds x {len(costs)} costs on {row_count} rows"
    )

    rows: list[dict[str, object]] = []
    for candidate in selected_candidates:
        for scenario_name, block_size, path_seed in DEFAULT_SCENARIOS:
            for seed in seeds:
                for cost in costs:
                    run_dir = run_root / candidate.name / f"{scenario_name}_s{seed}_c{cost}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    cmd = _build_command(
                        run_dir=run_dir,
                        returns_csv=returns_csv,
                        benchmark_column=args.benchmark_column,
                        seed=seed,
                        cost=cost,
                        scenario_block_size=block_size,
                        scenario_seed=path_seed,
                        candidate=candidate,
                    )
                    proc = subprocess.run(
                        cmd,
                        cwd=PROJECT_ROOT,
                        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    detail_path = run_dir / f"main6_gate001_f5_s{seed}_detailed.json"
                    if not detail_path.exists():
                        rows.append(
                            {
                                "candidate": candidate.name,
                                "scenario": scenario_name,
                                "seed": seed,
                                "cost": cost,
                                "breach": True,
                                "breach_reason": "missing_detail",
                                "failed_gate_checks": "",
                                "gate_passed": False,
                                "error": f"missing_detail_exit_{proc.returncode}",
                            }
                        )
                        continue

                    detail = json.loads(detail_path.read_text(encoding="utf-8"))
                    summary = detail.get("summary_metrics", {})
                    gate_checks = detail.get("gate_checks", {})
                    robust_min = float(summary.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))
                    dead_fraction = float(summary.get("dead_fold_fraction", float("nan")))
                    mean_test = float(summary.get("mean_test_relative_total_return", float("nan")))
                    gate_passed = bool(gate_checks.get("overall_passed", False))
                    failed_gate_checks: list[str] = []
                    for key, value in gate_checks.items():
                        if key == "overall_passed":
                            continue
                        if isinstance(value, dict) and value.get("passed") is False:
                            failed_gate_checks.append(key)
                        elif isinstance(value, bool) and value is False:
                            failed_gate_checks.append(key)

                    breach = (not gate_passed) or pd.isna(robust_min) or robust_min < -0.007
                    if pd.isna(robust_min):
                        breach_reason = "robust_min_nan"
                    elif robust_min < -0.007:
                        breach_reason = "robust_tail_floor_fail"
                    elif not gate_passed and failed_gate_checks:
                        breach_reason = "gate_fail:" + "|".join(sorted(failed_gate_checks))
                    elif not gate_passed:
                        breach_reason = "gate_fail_other"
                    else:
                        breach_reason = ""

                    rows.append(
                        {
                            "candidate": candidate.name,
                            "scenario": scenario_name,
                            "seed": seed,
                            "cost": cost,
                            "mean_test_relative": mean_test,
                            "robust_min_p05": robust_min,
                            "dead_fold_fraction": dead_fraction,
                            "gate_passed": gate_passed,
                            "breach": breach,
                            "breach_reason": breach_reason,
                            "failed_gate_checks": "|".join(sorted(failed_gate_checks)),
                            "error": None,
                        }
                    )

    summary_df = pd.DataFrame(rows)
    summary_path = run_root / f"{args.label}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    total = len(summary_df)
    breaches = int(summary_df["breach"].fillna(True).sum()) if total else 0
    print(f"Wrote {summary_path}")
    print(f"Breaches: {breaches}/{total}")
    if breaches > 0:
        reason_counts = (
            summary_df.loc[summary_df["breach"].fillna(True), "breach_reason"]
            .fillna("unknown")
            .replace("", "unknown")
            .value_counts()
        )
        print("Breach reasons:")
        print(reason_counts.to_string())
    if total:
        grouped = (
            summary_df.groupby("candidate", dropna=False)
            .agg(
                cases=("breach", "size"),
                breaches=("breach", "sum"),
                mean_test_relative=("mean_test_relative", "mean"),
                min_robust_min_p05=("robust_min_p05", "min"),
                max_dead_fold_fraction=("dead_fold_fraction", "max"),
            )
            .sort_values(["breaches", "min_robust_min_p05"], ascending=[True, False])
        )
        print(grouped.to_string(float_format=lambda value: f"{value: .6f}"))
    return 0 if breaches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())