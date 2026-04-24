#!/bin/sh
set -eu

runtime_root="${WEALTH_FIRST_RUNTIME_ROOT:-/var/data}"
artifact_root="${WEALTH_FIRST_ARTIFACT_ROOT_PATH:-$runtime_root/artifacts}"
seed_artifact_root="${WEALTH_FIRST_SEED_ARTIFACT_ROOT:-/opt/wealth-first/seed-artifacts}"
promoted_config="${WEALTH_FIRST_PROMOTED_CONFIG:-/app/config/promoted_main4.env}"
event_log_path="${WEALTH_FIRST_EVENT_LOG_PATH:-$runtime_root/tradingview_events.jsonl}"
output_csv_path="${WEALTH_FIRST_OUTPUT_CSV_PATH:-$runtime_root/tradingview_truth.csv}"
execution_log_path="${WEALTH_FIRST_EXECUTION_LOG_PATH:-$runtime_root/tradingview_execution.jsonl}"
failure_log_path="${WEALTH_FIRST_FAILURE_LOG_PATH:-$runtime_root/tradingview_bridge_failures.jsonl}"
main2_compare_detail_csv_path="${WEALTH_FIRST_MAIN2_COMPARE_DETAIL_CSV_PATH:-$artifact_root/main2_2007_v8_best_detail.csv}"
python_bin="${WEALTH_FIRST_PYTHON_BIN:-python}"

mkdir -p "$runtime_root" "$artifact_root"

if [ ! -f "$promoted_config" ]; then
  echo "Missing promoted config: $promoted_config" >&2
  exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

export WEALTH_FIRST_RUNTIME_ROOT="$runtime_root"
export WEALTH_FIRST_ARTIFACT_ROOT_PATH="$artifact_root"
export WEALTH_FIRST_EVENT_LOG_PATH="$event_log_path"
export WEALTH_FIRST_OUTPUT_CSV_PATH="$output_csv_path"
export WEALTH_FIRST_EXECUTION_LOG_PATH="$execution_log_path"
export WEALTH_FIRST_FAILURE_LOG_PATH="$failure_log_path"
export WEALTH_FIRST_MAIN2_COMPARE_DETAIL_CSV_PATH="$main2_compare_detail_csv_path"

# shellcheck disable=SC1090
. "$promoted_config"
export WEALTH_FIRST_MAIN4_RIDGE_L2
export WEALTH_FIRST_MAIN4_ACTION_SMOOTHING
export WEALTH_FIRST_MAIN4_TRANSACTION_COST_BPS
export WEALTH_FIRST_MAIN4_SLIPPAGE_BPS
export WEALTH_FIRST_MAIN4_GATE
export WEALTH_FIRST_MAIN4_GATE_SCALE
export WEALTH_FIRST_MAIN4_MIN_ROBUST_MIN_RELATIVE
export WEALTH_FIRST_MAIN4_N_FOLDS
export WEALTH_FIRST_MAIN4_SEED
export WEALTH_FIRST_MAIN4_PROMOTION_DATE
export WEALTH_FIRST_MAIN4_PROMOTION_STATUS
export WEALTH_FIRST_MAIN4_PROMOTION_LABEL
export WEALTH_FIRST_MAIN4_PROMOTION_FRICTION_RANGE

if [ -d "$seed_artifact_root" ]; then
  find "$seed_artifact_root" -maxdepth 1 -type f | while IFS= read -r seed_file; do
    target_file="$artifact_root/$(basename "$seed_file")"
    if [ ! -e "$target_file" ]; then
      cp "$seed_file" "$target_file"
    fi
  done
fi

exec "$python_bin" -m wealth_first.tradingview_bridge serve-env