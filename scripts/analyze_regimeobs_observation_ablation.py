from __future__ import annotations

from pathlib import Path

from wealth_first.ppo_analysis_common import ARTIFACTS, RULED_OUT_ARCHIVE, artifact_run_dir


CANDIDATE_PREFIX_BY_SPLIT = {
    "chrono": "ppo_frozen_live_baseline_regimeobs_chrono_seed_",
    "regime": "ppo_frozen_live_baseline_regimeobs_seed_",
}


def _artifact_dir(prefix_by_split: dict[str, str], row: dict[str, object]) -> Path:
    return artifact_run_dir(prefix_by_split, row, artifacts=ARTIFACTS, archive=RULED_OUT_ARCHIVE)
