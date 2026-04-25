from __future__ import annotations

from pathlib import Path

from wealth_first.ppo_analysis_common import ARTIFACTS, RULED_OUT_ARCHIVE, artifact_file


CANDIDATE_PREFIX_BY_SPLIT = {
    "chrono": "ppo_frozen_live_baseline_regimeobs_chrono_seed_",
    "regime": "ppo_frozen_live_baseline_regimeobs_seed_",
}


def _candidate_rollout_path(row: dict[str, object]) -> Path:
    return artifact_file(CANDIDATE_PREFIX_BY_SPLIT, row, "test_policy_rollout.csv", artifacts=ARTIFACTS, archive=RULED_OUT_ARCHIVE)


def _candidate_summary_path(row: dict[str, object]) -> Path:
    return artifact_file(CANDIDATE_PREFIX_BY_SPLIT, row, "test_policy_summary.json", artifacts=ARTIFACTS, archive=RULED_OUT_ARCHIVE)
