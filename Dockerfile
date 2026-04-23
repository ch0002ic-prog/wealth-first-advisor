FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY data /app/data
COPY artifacts /opt/wealth-first/seed-artifacts
COPY scripts/start_bridge_runtime.sh /app/scripts/start_bridge_runtime.sh

RUN python -m pip install --upgrade pip \
    && python -m pip install '.[rl]'

ENV WEALTH_FIRST_RUNTIME_ROOT=/var/data \
    WEALTH_FIRST_EVENT_LOG_PATH=/var/data/tradingview_events.jsonl \
    WEALTH_FIRST_OUTPUT_CSV_PATH=/var/data/tradingview_truth.csv \
    WEALTH_FIRST_EXECUTION_LOG_PATH=/var/data/tradingview_execution.jsonl \
    WEALTH_FIRST_FAILURE_LOG_PATH=/var/data/tradingview_bridge_failures.jsonl \
    WEALTH_FIRST_ARTIFACT_ROOT_PATH=/var/data/artifacts \
    WEALTH_FIRST_MAIN2_RETURNS_CSV_PATH=/app/data/demo_sleeves.csv \
    WEALTH_FIRST_MAIN2_COMPARE_DETAIL_CSV_PATH=/var/data/artifacts/main2_2007_v8_best_detail.csv \
    WEALTH_FIRST_EXECUTION_MODE=paper \
    WEALTH_FIRST_EXECUTION_PROBE_ON_STARTUP=false \
    WEALTH_FIRST_HOST=0.0.0.0 \
    WEALTH_FIRST_PORT=8000

EXPOSE 8000

CMD ["sh", "/app/scripts/start_bridge_runtime.sh"]