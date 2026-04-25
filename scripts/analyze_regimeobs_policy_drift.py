from __future__ import annotations

from pathlib import Path

from wealth_first.ppo_analysis_common import ARTIFACTS, RULED_OUT_ARCHIVE, artifact_file, preferred_artifact_path


CANDIDATE_PREFIX_BY_SPLIT = {
    "chrono": "ppo_frozen_live_baseline_regimeobs_chrono_seed_",
    "regime": "ppo_frozen_live_baseline_regimeobs_seed_",
}


def _preferred_artifact_path(filename: str) -> Path:
    return preferred_artifact_path(filename, artifacts=ARTIFACTS, archive=RULED_OUT_ARCHIVE)


def _rollout_path(prefix_by_split: dict[str, str], row: dict[str, object]) -> Path:
    return artifact_file(prefix_by_split, row, "test_policy_rollout.csv", artifacts=ARTIFACTS, archive=RULED_OUT_ARCHIVE)
