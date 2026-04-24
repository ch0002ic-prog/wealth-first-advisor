from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections import deque
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from wealth_first.execution import PaperExchangeAdapter, WebhookExchangeAdapter, build_execution_plan


logger = logging.getLogger(__name__)


NO_USABLE_RETURN_SIGNALS_ERROR = "TradingView event log does not contain any usable return signals."
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BridgeSettings:
    event_log_path: Path
    output_csv_path: Path | None = None
    webhook_token: str | None = None
    default_sleeve: str = "TRADINGVIEW"
    aggregate_freq: str = "D"
    base_equity: float | None = None
    normalize_on_ingest: bool = True
    execution_mode: str = "none"
    execution_webhook_url: str | None = None
    execution_webhook_token: str | None = None
    execution_log_path: Path | None = None
    execution_min_trade_weight: float = 0.0
    cash_symbol: str = "CASH"
    failure_log_path: Path | None = None
    worker_poll_interval_seconds: float = 0.25
    retry_delay_seconds: float = 2.0
    execution_probe_on_startup: bool = True
    execution_probe_url: str | None = None
    execution_probe_timeout: float = 5.0
    frontend_dist_path: Path | None = None
    artifact_root_path: Path | None = None
    python_executable_path: Path | None = None
    allowed_origins: tuple[str, ...] = ()
    allowed_origin_regex: str | None = None


def _env_string(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    raw_value = env.get(key)
    if raw_value is None:
        return default

    stripped = raw_value.strip()
    if stripped == "":
        return default
    return stripped


def _env_path(env: Mapping[str, str], key: str, default: Path | None = None) -> Path | None:
    raw_value = _env_string(env, key)
    if raw_value is None:
        return default
    return Path(raw_value)


def _env_float(env: Mapping[str, str], key: str, default: float | None = None) -> float | None:
    raw_value = _env_string(env, key)
    if raw_value is None:
        return default
    return float(raw_value)


def _env_int(env: Mapping[str, str], key: str, default: int | None = None) -> int | None:
    raw_value = _env_string(env, key)
    if raw_value is None:
        return default
    return int(raw_value)


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw_value = _env_string(env, key)
    if raw_value is None:
        return default

    normalized = raw_value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {key} must be a boolean value.")


def _env_csv(env: Mapping[str, str], key: str) -> tuple[str, ...]:
    raw_value = _env_string(env, key)
    if raw_value is None:
        return ()
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def load_bridge_settings_from_env(env: Mapping[str, str] | None = None) -> BridgeSettings:
    runtime_env = os.environ if env is None else env
    execution_mode = _env_string(runtime_env, "WEALTH_FIRST_EXECUTION_MODE", "none") or "none"
    if execution_mode not in {"none", "paper", "webhook"}:
        raise ValueError("WEALTH_FIRST_EXECUTION_MODE must be one of: none, paper, webhook.")

    return BridgeSettings(
        event_log_path=_env_path(runtime_env, "WEALTH_FIRST_EVENT_LOG_PATH", PROJECT_ROOT / "data" / "tradingview_events.jsonl") or PROJECT_ROOT / "data" / "tradingview_events.jsonl",
        output_csv_path=_env_path(runtime_env, "WEALTH_FIRST_OUTPUT_CSV_PATH", PROJECT_ROOT / "data" / "tradingview_truth.csv"),
        webhook_token=_env_string(runtime_env, "WEALTH_FIRST_WEBHOOK_TOKEN"),
        default_sleeve=_env_string(runtime_env, "WEALTH_FIRST_DEFAULT_SLEEVE", "TRADINGVIEW") or "TRADINGVIEW",
        aggregate_freq=_env_string(runtime_env, "WEALTH_FIRST_AGGREGATE_FREQ", "D") or "D",
        base_equity=_env_float(runtime_env, "WEALTH_FIRST_BASE_EQUITY"),
        normalize_on_ingest=_env_bool(runtime_env, "WEALTH_FIRST_NORMALIZE_ON_INGEST", True),
        execution_mode=execution_mode,
        execution_webhook_url=_env_string(runtime_env, "WEALTH_FIRST_EXECUTION_WEBHOOK_URL"),
        execution_webhook_token=_env_string(runtime_env, "WEALTH_FIRST_EXECUTION_WEBHOOK_TOKEN"),
        execution_log_path=_env_path(runtime_env, "WEALTH_FIRST_EXECUTION_LOG_PATH", PROJECT_ROOT / "data" / "tradingview_execution.jsonl"),
        execution_min_trade_weight=_env_float(runtime_env, "WEALTH_FIRST_EXECUTION_MIN_TRADE_WEIGHT", 0.0) or 0.0,
        cash_symbol=_env_string(runtime_env, "WEALTH_FIRST_CASH_SYMBOL", "CASH") or "CASH",
        failure_log_path=_env_path(runtime_env, "WEALTH_FIRST_FAILURE_LOG_PATH", PROJECT_ROOT / "data" / "tradingview_bridge_failures.jsonl"),
        worker_poll_interval_seconds=_env_float(runtime_env, "WEALTH_FIRST_WORKER_POLL_INTERVAL_SECONDS", 0.25) or 0.25,
        retry_delay_seconds=_env_float(runtime_env, "WEALTH_FIRST_RETRY_DELAY_SECONDS", 2.0) or 2.0,
        execution_probe_on_startup=_env_bool(runtime_env, "WEALTH_FIRST_EXECUTION_PROBE_ON_STARTUP", True),
        execution_probe_url=_env_string(runtime_env, "WEALTH_FIRST_EXECUTION_PROBE_URL"),
        execution_probe_timeout=_env_float(runtime_env, "WEALTH_FIRST_EXECUTION_PROBE_TIMEOUT", 5.0) or 5.0,
        frontend_dist_path=_env_path(runtime_env, "WEALTH_FIRST_FRONTEND_DIST_PATH"),
        artifact_root_path=_env_path(runtime_env, "WEALTH_FIRST_ARTIFACT_ROOT_PATH"),
        python_executable_path=_env_path(runtime_env, "WEALTH_FIRST_PYTHON_EXECUTABLE_PATH"),
        allowed_origins=_env_csv(runtime_env, "WEALTH_FIRST_ALLOWED_ORIGINS"),
        allowed_origin_regex=_env_string(runtime_env, "WEALTH_FIRST_ALLOWED_ORIGIN_REGEX"),
    )


def load_bridge_bind_from_env(env: Mapping[str, str] | None = None) -> tuple[str, int]:
    runtime_env = os.environ if env is None else env
    host = _env_string(runtime_env, "WEALTH_FIRST_HOST", _env_string(runtime_env, "HOST", "0.0.0.0")) or "0.0.0.0"
    port = _env_int(runtime_env, "WEALTH_FIRST_PORT")
    if port is None:
        port = _env_int(runtime_env, "PORT", 8000) or 8000
    return host, port


def _ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_first(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _extract_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    raw_value = _extract_first(payload, keys)
    if raw_value in (None, ""):
        return None
    return float(raw_value)


def _resolve_failure_log_path(settings: BridgeSettings) -> Path:
    return settings.failure_log_path or settings.event_log_path.parent / "tradingview_bridge_failures.jsonl"


def _coerce_weight_payload(raw_value: Any, field_name: str) -> pd.Series | None:
    if raw_value in (None, ""):
        return None
    parsed_value = raw_value
    if isinstance(raw_value, str):
        try:
            parsed_value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be a JSON object mapping symbols to weights.") from exc

    if not isinstance(parsed_value, dict):
        raise ValueError(f"{field_name} must be a mapping of symbols to weights.")
    if not parsed_value:
        raise ValueError(f"{field_name} cannot be empty.")

    series = pd.Series({str(symbol): float(weight) for symbol, weight in parsed_value.items()}, dtype=float)
    if series.empty:
        raise ValueError(f"{field_name} cannot be empty.")
    return series


def _extract_timestamp(payload: dict[str, Any]) -> str:
    raw_value = _extract_first(payload, ["timestamp", "timenow", "bar_time", "time"])
    if raw_value is None:
        return datetime.now(UTC).isoformat()

    if isinstance(raw_value, (int, float)):
        timestamp_value = float(raw_value)
        if timestamp_value > 10_000_000_000:
            timestamp_value /= 1_000.0
        return datetime.fromtimestamp(timestamp_value, tz=UTC).isoformat()

    parsed = pd.to_datetime(raw_value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse TradingView timestamp value '{raw_value}'.")
    return parsed.isoformat()


def normalize_tradingview_payload(
    payload: dict[str, Any],
    default_sleeve: str = "TRADINGVIEW",
) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "received_at": datetime.now(UTC).isoformat(),
        "timestamp": _extract_timestamp(payload),
        "event_type": str(_extract_first(payload, ["event_type", "type", "action"]) or "alert"),
        "strategy": str(_extract_first(payload, ["strategy", "strategy_name", "strategyName"]) or default_sleeve),
        "sleeve": str(_extract_first(payload, ["sleeve", "strategy", "strategy_name", "strategyName"]) or default_sleeve),
        "symbol": str(_extract_first(payload, ["symbol", "ticker", "tickerid"]) or ""),
        "side": _extract_first(payload, ["side", "direction", "order_side"]),
        "event_id": _extract_first(payload, ["event_id", "eventId", "alert_id", "alertId"]),
        "order_id": _extract_first(payload, ["order_id", "orderId", "strategy_order_id", "trade_id", "tradeId"]),
        "return_value": None,
        "realized_pnl": _extract_float(payload, ["realized_pnl", "realizedPnL", "profit", "pnl"]),
        "netprofit": _extract_float(payload, ["netprofit", "strategy_netprofit", "strategyNetprofit"]),
        "equity_before": _extract_float(payload, ["equity_before", "equityBefore", "account_equity", "accountEquity"]),
        "raw_payload": payload,
    }

    decimal_return = _extract_float(payload, ["return_value", "return_decimal", "return", "returnValue"])
    percent_return = _extract_float(payload, ["return_pct", "returnPercent", "return_percent"])
    if decimal_return is not None:
        normalized["return_value"] = decimal_return
    elif percent_return is not None:
        normalized["return_value"] = percent_return / 100.0

    return normalized


def compute_event_fingerprint(event: dict[str, Any]) -> str:
    raw_payload = event.get("raw_payload") if isinstance(event.get("raw_payload"), dict) else {}
    raw_timestamp = _extract_first(raw_payload, ["timestamp", "timenow", "bar_time", "time"])
    sanitized_raw_payload = {
        str(key): raw_payload[key]
        for key in sorted(raw_payload)
        if key not in {"token"}
    }
    fingerprint_payload = {
        "event_id": event.get("event_id"),
        "timestamp": raw_timestamp,
        "event_type": event.get("event_type"),
        "strategy": event.get("strategy"),
        "sleeve": event.get("sleeve"),
        "symbol": event.get("symbol"),
        "side": event.get("side"),
        "order_id": event.get("order_id"),
        "return_value": event.get("return_value"),
        "realized_pnl": event.get("realized_pnl"),
        "netprofit": event.get("netprofit"),
        "equity_before": event.get("equity_before"),
        "raw_payload": sanitized_raw_payload if not event.get("event_id") and not event.get("order_id") else None,
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_existing_event_fingerprints(event_log_path: str | Path) -> set[str]:
    target_path = Path(event_log_path)
    if not target_path.exists():
        return set()

    seen: set[str] = set()
    with target_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            seen.add(str(record.get("fingerprint") or compute_event_fingerprint(record)))
    return seen


def append_tradingview_event(event_log_path: str | Path, event: dict[str, Any]) -> None:
    target_path = Path(event_log_path)
    _ensure_parent_directory(target_path)
    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def append_execution_result(execution_log_path: str | Path, record: dict[str, Any]) -> None:
    target_path = Path(execution_log_path)
    _ensure_parent_directory(target_path)
    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_bridge_failure(failure_log_path: str | Path, record: dict[str, Any]) -> None:
    target_path = Path(failure_log_path)
    _ensure_parent_directory(target_path)
    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _dashboard_directory() -> Path:
    return Path(__file__).resolve().parent / "dashboard"


def _resolve_frontend_dist_path(settings: BridgeSettings) -> Path:
    return settings.frontend_dist_path or PROJECT_ROOT / "frontend" / "dist"


def _resolve_artifact_root_path(settings: BridgeSettings) -> Path:
    return settings.artifact_root_path or PROJECT_ROOT / "artifacts"


def _resolve_python_executable_path(settings: BridgeSettings) -> Path:
    return settings.python_executable_path or Path(sys.executable)


def _read_log_tail(log_path: str | Path | None, limit: int = 20) -> list[str]:
    if log_path is None:
        return []

    target_path = Path(log_path)
    if not target_path.exists():
        return []

    with target_path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=limit)]


def _coerce_progress_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _coerce_progress_int(value: Any) -> int | None:
    number = _coerce_progress_float(value)
    if number is None:
        return None
    return int(number)


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def _tail_jsonl_records(log_path: str | Path | None, limit: int) -> tuple[int, list[dict[str, Any]]]:
    if log_path is None:
        return 0, []

    target_path = Path(log_path)
    if not target_path.exists():
        return 0, []

    total_records = 0
    recent_records: deque[dict[str, Any]] = deque(maxlen=limit)
    with target_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_records += 1
            recent_records.append(json.loads(stripped))

    ordered_records = [_json_ready(record) for record in reversed(recent_records)]
    return total_records, ordered_records


def _read_returns_preview(output_csv_path: str | Path | None, limit: int) -> dict[str, Any]:
    if output_csv_path is None:
        return {"row_count": 0, "columns": [], "rows": []}

    target_path = Path(output_csv_path)
    if not target_path.exists():
        return {"row_count": 0, "columns": [], "rows": []}

    frame = pd.read_csv(target_path)
    preview = frame.tail(limit).copy()
    preview = preview.where(pd.notna(preview), None)
    rows = [
        {str(column_name): _json_ready(value) for column_name, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]
    return {
        "row_count": int(len(frame.index)),
        "columns": [str(column_name) for column_name in frame.columns],
        "rows": rows,
    }


def _read_returns_series(output_csv_path: str | Path | None, limit: int) -> dict[str, Any]:
    if output_csv_path is None:
        return {"row_count": 0, "date_column": "date", "asset_columns": [], "rows": []}

    target_path = Path(output_csv_path)
    if not target_path.exists():
        return {"row_count": 0, "date_column": "date", "asset_columns": [], "rows": []}

    frame = pd.read_csv(target_path)
    if frame.empty:
        return {"row_count": 0, "date_column": "date", "asset_columns": [], "rows": []}

    date_column = "date" if "date" in frame.columns else str(frame.columns[0])
    asset_columns = [str(column_name) for column_name in frame.columns if str(column_name) != date_column]
    if not asset_columns:
        return {"row_count": int(len(frame.index)), "date_column": date_column, "asset_columns": [], "rows": []}

    numeric_values = frame.loc[:, asset_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cumulative_values = (1.0 + numeric_values).cumprod() - 1.0
    preview = frame.tail(limit).copy()

    rows: list[dict[str, Any]] = []
    for row_index in preview.index:
        row: dict[str, Any] = {date_column: _json_ready(frame.at[row_index, date_column])}
        for column_name in asset_columns:
            row[column_name] = _json_ready(float(numeric_values.at[row_index, column_name]))
            row[f"cumulative_{column_name}"] = _json_ready(float(cumulative_values.at[row_index, column_name]))
        rows.append(row)

    return {
        "row_count": int(len(frame.index)),
        "date_column": date_column,
        "asset_columns": asset_columns,
        "rows": rows,
    }


def _read_execution_series(execution_log_path: str | Path | None, limit: int) -> dict[str, Any]:
    if execution_log_path is None:
        return {"row_count": 0, "rows": []}

    target_path = Path(execution_log_path)
    if not target_path.exists():
        return {"row_count": 0, "rows": []}

    total_records = 0
    recent_records: deque[dict[str, Any]] = deque(maxlen=limit)
    with target_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_records += 1
            recent_records.append(json.loads(stripped))

    rows: list[dict[str, Any]] = []
    for record in recent_records:
        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        orders = plan.get("orders") if isinstance(plan.get("orders"), list) else []
        rows.append(
            {
                "processed_at": _json_ready(record.get("processed_at")),
                "execution_mode": _json_ready(record.get("execution_mode")),
                "equity": _json_ready(plan.get("equity")),
                "order_count": len(orders),
                "target_cash_weight": _json_ready(plan.get("target_cash_weight")),
                "event_fingerprint": _json_ready(record.get("event_fingerprint")),
            }
        )

    return {"row_count": total_records, "rows": rows}


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _read_detail_table(detail_path: Path | None, limit: int | None = None) -> dict[str, Any]:
    if detail_path is None or not detail_path.exists():
        return {"row_count": 0, "columns": [], "rows": []}

    frame = pd.read_csv(detail_path)
    if limit is not None:
        frame = frame.head(limit)
    frame = frame.where(pd.notna(frame), None)
    rows = [
        {str(column_name): _json_ready(value) for column_name, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    return {
        "row_count": int(len(pd.read_csv(detail_path).index)),
        "columns": [str(column_name) for column_name in frame.columns],
        "rows": rows,
    }



def _build_health_payload(settings: BridgeSettings, app: FastAPI) -> dict[str, Any]:
    worker_thread = app.state.worker_thread
    return {
        "status": "ok",
        "event_log_path": str(settings.event_log_path),
        "output_csv_path": str(settings.output_csv_path) if settings.output_csv_path else None,
        "execution_mode": settings.execution_mode,
        "fingerprint_cache_size": len(app.state.seen_fingerprints),
        "worker_running": bool(worker_thread is not None and worker_thread.is_alive()),
        "execution_completed_count": len(app.state.execution_completed_fingerprints),
    }


def _build_dashboard_snapshot(settings: BridgeSettings, app: FastAPI, recent_limit: int, returns_limit: int) -> dict[str, Any]:
    event_count, recent_events = _tail_jsonl_records(settings.event_log_path, recent_limit)
    execution_count, recent_executions = _tail_jsonl_records(settings.execution_log_path, recent_limit)
    failure_log_path = _resolve_failure_log_path(settings)
    failure_count, recent_failures = _tail_jsonl_records(failure_log_path, recent_limit)
    returns_preview = _read_returns_preview(settings.output_csv_path, returns_limit)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "token_required": settings.webhook_token is not None,
        "health": _build_health_payload(settings, app),
        "counts": {
            "events": event_count,
            "executions": execution_count,
            "failures": failure_count,
            "returns_rows": returns_preview["row_count"],
        },
        "paths": {
            "event_log_path": str(settings.event_log_path),
            "output_csv_path": str(settings.output_csv_path) if settings.output_csv_path else None,
            "execution_log_path": str(settings.execution_log_path) if settings.execution_log_path else None,
            "failure_log_path": str(failure_log_path),
        },
        "recent_events": recent_events,
        "recent_executions": recent_executions,
        "recent_failures": recent_failures,
        "returns_preview": returns_preview,
    }


def load_tradingview_event_records(event_log_path: str | Path) -> list[dict[str, Any]]:
    target_path = Path(event_log_path)
    if not target_path.exists():
        raise FileNotFoundError(f"TradingView event log '{target_path}' does not exist.")

    records: list[dict[str, Any]] = []
    with target_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def load_existing_execution_fingerprints(execution_log_path: str | Path | None) -> set[str]:
    if execution_log_path is None:
        return set()

    target_path = Path(execution_log_path)
    if not target_path.exists():
        return set()

    seen: set[str] = set()
    with target_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            fingerprint = record.get("event_fingerprint")
            if fingerprint:
                seen.add(str(fingerprint))
    return seen


def load_tradingview_event_log(event_log_path: str | Path) -> pd.DataFrame:
    records = load_tradingview_event_records(event_log_path)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("TradingView event log is empty.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("TradingView event log contains unparseable timestamps.")
    return frame.sort_values(["sleeve", "timestamp"]).reset_index(drop=True)


def normalize_event_log_to_returns(
    event_log_path: str | Path,
    aggregate_freq: str = "D",
    base_equity: float | None = None,
    default_sleeve: str = "TRADINGVIEW",
) -> pd.DataFrame:
    events = load_tradingview_event_log(event_log_path)
    events["sleeve"] = events["sleeve"].fillna(default_sleeve).astype(str)

    if "return_value" not in events.columns:
        events["return_value"] = np.nan
    if "realized_pnl" not in events.columns:
        events["realized_pnl"] = np.nan
    if "equity_before" not in events.columns:
        events["equity_before"] = np.nan
    if "netprofit" not in events.columns:
        events["netprofit"] = np.nan

    event_returns = pd.to_numeric(events["return_value"], errors="coerce")
    pnl_returns = pd.Series(index=events.index, dtype=float)
    valid = event_returns.isna() & events["realized_pnl"].notna() & events["equity_before"].notna() & (events["equity_before"] != 0)
    pnl_returns.loc[valid] = pd.to_numeric(events.loc[valid, "realized_pnl"], errors="coerce") / pd.to_numeric(events.loc[valid, "equity_before"], errors="coerce")

    netprofit_returns = pd.Series(index=events.index, dtype=float)
    if events["netprofit"].notna().any():
        if base_equity is None or base_equity <= 0:
            raise ValueError("base_equity is required to normalize TradingView netprofit-only event logs.")
        netprofit_series = pd.to_numeric(events["netprofit"], errors="coerce")
        netprofit_deltas = netprofit_series.groupby(events["sleeve"], sort=False).diff().fillna(netprofit_series)
        valid = event_returns.isna() & pnl_returns.isna() & netprofit_deltas.notna()
        netprofit_returns.loc[valid] = netprofit_deltas.loc[valid] / float(base_equity)

    effective_returns = event_returns.fillna(pnl_returns).fillna(netprofit_returns)
    normalized = events.loc[effective_returns.notna(), ["timestamp", "sleeve"]].copy()
    if normalized.empty:
        raise ValueError(NO_USABLE_RETURN_SIGNALS_ERROR)

    normalized["event_return"] = effective_returns.loc[effective_returns.notna()].astype(float).to_numpy()
    normalized["bucket"] = normalized["timestamp"].dt.floor(aggregate_freq)

    aggregated = (
        normalized.groupby(["bucket", "sleeve"]) ["event_return"]
        .apply(lambda series: float((1.0 + series).prod() - 1.0))
        .unstack("sleeve")
        .sort_index()
        .fillna(0.0)
    )
    aggregated.index.name = "date"
    return aggregated


def write_normalized_returns_csv(
    event_log_path: str | Path,
    output_csv_path: str | Path,
    aggregate_freq: str = "D",
    base_equity: float | None = None,
    default_sleeve: str = "TRADINGVIEW",
) -> pd.DataFrame:
    returns = normalize_event_log_to_returns(
        event_log_path=event_log_path,
        aggregate_freq=aggregate_freq,
        base_equity=base_equity,
        default_sleeve=default_sleeve,
    )
    target_path = Path(output_csv_path)
    _ensure_parent_directory(target_path)
    returns.to_csv(target_path, index_label="date")
    return returns


def _parse_request_payload(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        raise ValueError("Webhook body is empty.")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Webhook body is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Webhook body must decode to a JSON object.")
    return payload


def prepare_execution_plan(payload: dict[str, Any], settings: BridgeSettings):
    target_weights = _coerce_weight_payload(payload.get("target_weights"), "target_weights")
    if target_weights is None:
        return None

    current_weights = _coerce_weight_payload(payload.get("current_weights"), "current_weights")
    equity = _extract_float(payload, ["equity", "account_equity", "accountEquity", "notional"]) or settings.base_equity or 1.0
    min_trade_weight = _extract_float(payload, ["min_trade_weight", "minTradeWeight"]) or settings.execution_min_trade_weight
    cash_symbol = str(_extract_first(payload, ["cash_symbol", "cashSymbol"]) or settings.cash_symbol)
    return build_execution_plan(
        target_weights=target_weights,
        current_weights=current_weights,
        equity=equity,
        min_trade_weight=min_trade_weight,
        cash_symbol=cash_symbol,
    )


def _event_log_modified_after(reference_path: Path | None, event_log_path: Path) -> bool:
    if not event_log_path.exists():
        return False
    if reference_path is None or not reference_path.exists():
        return True
    return event_log_path.stat().st_mtime > reference_path.stat().st_mtime


def _refresh_normalized_outputs(settings: BridgeSettings) -> None:
    if settings.output_csv_path is None:
        return
    write_normalized_returns_csv(
        event_log_path=settings.event_log_path,
        output_csv_path=settings.output_csv_path,
        aggregate_freq=settings.aggregate_freq,
        base_equity=settings.base_equity,
        default_sleeve=settings.default_sleeve,
    )


def _submit_execution_handoff(settings: BridgeSettings, plan, event_fingerprint: str) -> None:
    if settings.execution_mode == "paper":
        adapter = PaperExchangeAdapter()
    elif settings.execution_mode == "webhook":
        if not settings.execution_webhook_url:
            raise ValueError("execution_webhook_url is required when execution_mode is set to 'webhook'.")
        adapter = WebhookExchangeAdapter(settings.execution_webhook_url, token=settings.execution_webhook_token)
    else:
        return

    response = adapter.submit_allocation(plan)
    if settings.execution_log_path is not None:
        append_execution_result(
            settings.execution_log_path,
            {
                "processed_at": datetime.now(UTC).isoformat(),
                "event_fingerprint": event_fingerprint,
                "execution_mode": settings.execution_mode,
                "plan": plan.to_payload(),
                "response": response,
            },
        )


def _probe_execution_endpoint(settings: BridgeSettings) -> None:
    probe_url = settings.execution_probe_url or settings.execution_webhook_url
    if not probe_url:
        raise ValueError("execution_webhook_url is required when execution_mode is set to 'webhook'.")

    headers: dict[str, str] = {}
    if settings.execution_webhook_token:
        headers["Authorization"] = f"Bearer {settings.execution_webhook_token}"

    last_error: Exception | None = None
    for method in ("HEAD", "OPTIONS", "GET"):
        request = urllib_request.Request(probe_url, headers=headers, method=method)
        try:
            with urllib_request.urlopen(request, timeout=settings.execution_probe_timeout):
                return
        except urllib_error.HTTPError as exc:
            last_error = exc
            if exc.code in {401, 403, 404} or 500 <= exc.code < 600:
                raise ValueError(
                    f"Execution probe to '{probe_url}' failed with HTTP {exc.code}."
                ) from exc
            if exc.code == 405:
                continue
            return
        except urllib_error.URLError as exc:
            last_error = exc
            raise ValueError(f"Execution probe to '{probe_url}' failed: {exc.reason}.") from exc

    if last_error is not None:
        raise ValueError(f"Execution probe to '{probe_url}' failed: {last_error}.") from last_error


def _record_bridge_failure(settings: BridgeSettings, category: str, event_fingerprint: str | None, error: Exception, attempt: int | None = None) -> None:
    append_bridge_failure(
        _resolve_failure_log_path(settings),
        {
            "recorded_at": datetime.now(UTC).isoformat(),
            "category": category,
            "event_fingerprint": event_fingerprint,
            "attempt": attempt,
            "error": str(error),
        },
    )


def _process_normalization(settings: BridgeSettings, app: FastAPI) -> None:
    if settings.output_csv_path is None or not settings.normalize_on_ingest:
        return
    if not _event_log_modified_after(settings.output_csv_path, settings.event_log_path):
        return

    retry_state = app.state.normalization_retry_state
    event_log_mtime = settings.event_log_path.stat().st_mtime if settings.event_log_path.exists() else None
    if event_log_mtime is not None and retry_state.get("last_no_signal_event_log_mtime") == event_log_mtime:
        return

    now = datetime.now(UTC)
    next_retry_at = retry_state.get("next_retry_at")
    if next_retry_at is not None and now < next_retry_at:
        return

    try:
        _refresh_normalized_outputs(settings)
        retry_state.clear()
    except ValueError as exc:
        if str(exc) == NO_USABLE_RETURN_SIGNALS_ERROR:
            retry_state.clear()
            if event_log_mtime is not None:
                retry_state["last_no_signal_event_log_mtime"] = event_log_mtime
            return

        attempts = int(retry_state.get("attempts", 0)) + 1
        retry_state["attempts"] = attempts
        retry_state["next_retry_at"] = now + timedelta(seconds=settings.retry_delay_seconds * attempts)
        _record_bridge_failure(settings, "normalization", None, exc, attempts)
        logger.exception("Normalization worker failed")
    except Exception as exc:
        attempts = int(retry_state.get("attempts", 0)) + 1
        retry_state["attempts"] = attempts
        retry_state["next_retry_at"] = now + timedelta(seconds=settings.retry_delay_seconds * attempts)
        _record_bridge_failure(settings, "normalization", None, exc, attempts)
        logger.exception("Normalization worker failed")


def _process_execution_backlog(settings: BridgeSettings, app: FastAPI) -> None:
    if settings.execution_mode == "none":
        return

    try:
        event_records = load_tradingview_event_records(settings.event_log_path)
    except FileNotFoundError:
        return

    retry_schedule: dict[str, dict[str, Any]] = app.state.execution_retry_state
    completed_fingerprints: set[str] = app.state.execution_completed_fingerprints
    now = datetime.now(UTC)

    for record in event_records:
        fingerprint = str(record.get("fingerprint") or compute_event_fingerprint(record))
        if fingerprint in completed_fingerprints:
            continue

        raw_payload = record.get("raw_payload") if isinstance(record.get("raw_payload"), dict) else None
        if raw_payload is None or raw_payload.get("target_weights") is None:
            continue

        retry_state = retry_schedule.get(fingerprint)
        if retry_state is not None and now < retry_state["next_retry_at"]:
            continue

        try:
            plan = prepare_execution_plan(raw_payload, settings)
            if plan is None:
                continue
            _submit_execution_handoff(settings, plan, fingerprint)
            completed_fingerprints.add(fingerprint)
            retry_schedule.pop(fingerprint, None)
        except Exception as exc:
            attempts = int(retry_state["attempts"] if retry_state is not None else 0) + 1
            retry_schedule[fingerprint] = {
                "attempts": attempts,
                "next_retry_at": now + timedelta(seconds=settings.retry_delay_seconds * attempts),
            }
            _record_bridge_failure(settings, "execution", fingerprint, exc, attempts)
            logger.exception("Execution worker failed for fingerprint %s", fingerprint)


def _bridge_worker_loop(settings: BridgeSettings, app: FastAPI) -> None:
    while not app.state.worker_stop_event.is_set():
        _process_normalization(settings, app)
        _process_execution_backlog(settings, app)
        app.state.worker_stop_event.wait(settings.worker_poll_interval_seconds)


def validate_bridge_settings(settings: BridgeSettings) -> None:
    if settings.execution_mode not in {"none", "paper", "webhook"}:
        raise ValueError(f"Unsupported execution_mode '{settings.execution_mode}'.")
    if settings.execution_mode == "webhook" and not settings.execution_webhook_url:
        raise ValueError("execution_webhook_url is required when execution_mode is set to 'webhook'.")
    if settings.worker_poll_interval_seconds <= 0:
        raise ValueError("worker_poll_interval_seconds must be positive.")
    if settings.retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative.")
    if settings.execution_probe_timeout <= 0:
        raise ValueError("execution_probe_timeout must be positive.")
    if settings.execution_mode == "webhook" and settings.execution_probe_on_startup:
        _probe_execution_endpoint(settings)


def create_app(settings: BridgeSettings) -> FastAPI:
    validate_bridge_settings(settings)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker_thread = threading.Thread(
            target=_bridge_worker_loop,
            args=(settings, app),
            daemon=True,
            name="tradingview-bridge-worker",
        )
        worker_thread.start()
        app.state.worker_thread = worker_thread
        try:
            yield
        finally:
            app.state.worker_stop_event.set()
            worker_thread.join(timeout=max(1.0, settings.worker_poll_interval_seconds * 4.0))

    app = FastAPI(title="Wealth First TradingView Bridge", version="0.1.0", lifespan=lifespan)
    if settings.allowed_origins or settings.allowed_origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_origin_regex=settings.allowed_origin_regex,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.state.bridge_settings = settings
    app.state.seen_fingerprints = load_existing_event_fingerprints(settings.event_log_path)
    app.state.fingerprint_lock = threading.Lock()
    app.state.execution_completed_fingerprints = load_existing_execution_fingerprints(settings.execution_log_path)
    app.state.execution_retry_state = {}
    app.state.normalization_retry_state = {}
    app.state.worker_stop_event = threading.Event()
    app.state.worker_thread = None

    dashboard_directory = _dashboard_directory()
    dashboard_html_path = dashboard_directory / "index.html"
    dashboard_css_path = dashboard_directory / "dashboard.css"
    dashboard_js_path = dashboard_directory / "dashboard.js"
    frontend_dist_path = _resolve_frontend_dist_path(settings)
    frontend_index_path = frontend_dist_path / "index.html"
    frontend_assets_path = frontend_dist_path / "assets"

    if frontend_assets_path.exists():
        app.mount("/assets", StaticFiles(directory=frontend_assets_path), name="frontend-assets")

    def _validate_webhook_token(token: str | None, payload: dict[str, Any] | None = None) -> None:
        if settings.webhook_token is None:
            return

        body_token = payload.get("token") if payload is not None else None
        if token != settings.webhook_token and body_token != settings.webhook_token:
            raise HTTPException(status_code=401, detail="Invalid webhook token.")

    @app.get("/", include_in_schema=False)
    def dashboard_index() -> FileResponse:
        if frontend_index_path.exists():
            return FileResponse(frontend_index_path, media_type="text/html", headers={"Cache-Control": "no-store"})
        return FileResponse(dashboard_html_path, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/dashboard", include_in_schema=False)
    def dashboard_alias() -> FileResponse:
        return FileResponse(dashboard_html_path, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/app", include_in_schema=False)
    def frontend_app() -> FileResponse:
        if frontend_index_path.exists():
            return FileResponse(frontend_index_path, media_type="text/html", headers={"Cache-Control": "no-store"})
        return FileResponse(dashboard_html_path, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/dashboard.css", include_in_schema=False)
    def dashboard_stylesheet() -> FileResponse:
        return FileResponse(dashboard_css_path, media_type="text/css", headers={"Cache-Control": "no-store"})

    @app.get("/dashboard.js", include_in_schema=False)
    def dashboard_script() -> FileResponse:
        return FileResponse(dashboard_js_path, media_type="application/javascript", headers={"Cache-Control": "no-store"})

    @app.get("/api/dashboard")
    def dashboard_snapshot(
        recent_limit: int = Query(default=12, ge=1, le=100),
        returns_limit: int = Query(default=16, ge=1, le=100),
    ) -> dict[str, Any]:
        return _build_dashboard_snapshot(settings, app, recent_limit=recent_limit, returns_limit=returns_limit)

    @app.get("/api/dashboard/returns-series")
    def dashboard_returns_series(limit: int = Query(default=180, ge=1, le=1000)) -> dict[str, Any]:
        return _read_returns_series(settings.output_csv_path, limit=limit)

    @app.get("/api/dashboard/execution-series")
    def dashboard_execution_series(limit: int = Query(default=180, ge=1, le=1000)) -> dict[str, Any]:
        return _read_execution_series(settings.execution_log_path, limit=limit)

    @app.get("/healthz")
    def healthcheck() -> dict[str, Any]:
        return _build_health_payload(settings, app)

    @app.get("/webhook")
    async def webhook_reachability(token: str | None = Query(default=None)) -> dict[str, Any]:
        _validate_webhook_token(token)
        return {
            "status": "ok",
            "detail": "Webhook endpoint reachable. Send a POST request with a JSON body to submit TradingView alerts.",
            "method": "POST",
            "event_log_path": str(settings.event_log_path),
            "token_required": settings.webhook_token is not None,
        }

    @app.post("/webhook")
    async def receive_webhook(request: Request, token: str | None = Query(default=None)) -> dict[str, Any]:
        raw_body = await request.body()
        try:
            payload = _parse_request_payload(raw_body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _validate_webhook_token(token, payload)

        try:
            normalized_event = normalize_tradingview_payload(payload, default_sleeve=settings.default_sleeve)
            normalized_event["fingerprint"] = compute_event_fingerprint(normalized_event)
            execution_plan = prepare_execution_plan(payload, settings) if settings.execution_mode != "none" and payload.get("target_weights") is not None else None
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with app.state.fingerprint_lock:
            if normalized_event["fingerprint"] in app.state.seen_fingerprints:
                return {
                    "status": "duplicate",
                    "fingerprint": normalized_event["fingerprint"],
                    "sleeve": normalized_event["sleeve"],
                    "symbol": normalized_event["symbol"],
                    "event_type": normalized_event["event_type"],
                    "timestamp": normalized_event["timestamp"],
                }

            append_tradingview_event(settings.event_log_path, normalized_event)
            app.state.seen_fingerprints.add(normalized_event["fingerprint"])

        normalization_status = "queued" if settings.output_csv_path is not None and settings.normalize_on_ingest else "disabled"
        execution_status = "queued" if execution_plan is not None and settings.execution_mode != "none" else "disabled"
        if settings.execution_mode != "none" and execution_plan is None:
            execution_status = "no_target_weights"

        return {
            "status": "accepted",
            "fingerprint": normalized_event["fingerprint"],
            "sleeve": normalized_event["sleeve"],
            "symbol": normalized_event["symbol"],
            "event_type": normalized_event["event_type"],
            "timestamp": normalized_event["timestamp"],
            "normalization_status": normalization_status,
            "execution_status": execution_status,
        }

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TradingView webhook bridge and event-log normalizer for wealth-first comparison workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the TradingView webhook bridge server.")
    serve_parser.add_argument("--event-log", default="data/tradingview_events.jsonl", help="Path to the JSONL event log file.")
    serve_parser.add_argument("--output-csv", default="data/tradingview_truth.csv", help="Path to the normalized returns CSV updated after each event.")
    serve_parser.add_argument("--token", default=None, help="Optional shared secret accepted via query string or JSON field.")
    serve_parser.add_argument("--default-sleeve", default="TRADINGVIEW", help="Fallback sleeve name when the payload does not include one.")
    serve_parser.add_argument("--aggregate-freq", default="D", help="Aggregation frequency for normalized returns, such as D or W.")
    serve_parser.add_argument("--base-equity", type=float, default=None, help="Base equity used when payloads only provide cumulative netprofit.")
    serve_parser.add_argument("--no-normalize-on-ingest", action="store_true", help="Disable background regeneration of the normalized returns CSV on each accepted event.")
    serve_parser.add_argument("--execution-mode", choices=["none", "paper", "webhook"], default="none", help="Optional execution handoff mode for payloads that include target_weights.")
    serve_parser.add_argument("--execution-webhook-url", default=None, help="Destination URL for webhook execution handoff when execution-mode is webhook.")
    serve_parser.add_argument("--execution-webhook-token", default=None, help="Optional bearer token for the execution webhook adapter.")
    serve_parser.add_argument("--execution-log", default="data/tradingview_execution.jsonl", help="JSONL file used to record execution handoff results.")
    serve_parser.add_argument("--execution-min-trade-weight", type=float, default=0.0, help="Minimum absolute weight change required before an execution order is emitted.")
    serve_parser.add_argument("--failure-log", default="data/tradingview_bridge_failures.jsonl", help="JSONL file used to record normalization or execution worker failures.")
    serve_parser.add_argument("--worker-poll-interval", type=float, default=0.25, help="Polling interval in seconds for the durable bridge worker.")
    serve_parser.add_argument("--retry-delay", type=float, default=2.0, help="Base retry delay in seconds for failed normalization or execution attempts.")
    serve_parser.add_argument("--disable-execution-probe", action="store_true", help="Disable startup connectivity probing for webhook execution mode.")
    serve_parser.add_argument("--execution-probe-url", default=None, help="Optional health or probe endpoint used to validate the execution webhook at startup.")
    serve_parser.add_argument("--execution-probe-timeout", type=float, default=5.0, help="Timeout in seconds for the execution startup probe.")
    serve_parser.add_argument("--cash-symbol", default="CASH", help="Residual cash symbol used when building execution plans from TradingView payloads.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host for the local bridge server.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port for the local bridge server.")
    serve_parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Allowed browser origin for cross-site frontend requests. Repeat the flag for multiple origins.",
    )
    serve_parser.add_argument(
        "--allow-origin-regex",
        default=None,
        help="Optional regex used to allow browser origins such as Vercel preview domains.",
    )

    subparsers.add_parser(
        "serve-env",
        help="Run the bridge server using WEALTH_FIRST_* environment variables plus HOST/PORT.",
    )

    normalize_parser = subparsers.add_parser("normalize", help="Convert a TradingView event log into a dated returns CSV.")
    normalize_parser.add_argument("--event-log", required=True, help="Path to the JSONL event log file.")
    normalize_parser.add_argument("--output-csv", required=True, help="Path to the normalized returns CSV.")
    normalize_parser.add_argument("--aggregate-freq", default="D", help="Aggregation frequency for normalized returns, such as D or W.")
    normalize_parser.add_argument("--base-equity", type=float, default=None, help="Base equity used when payloads only provide cumulative netprofit.")
    normalize_parser.add_argument("--default-sleeve", default="TRADINGVIEW", help="Fallback sleeve name when the event log does not include one.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        settings = BridgeSettings(
            event_log_path=Path(args.event_log),
            output_csv_path=Path(args.output_csv) if args.output_csv else None,
            webhook_token=args.token,
            default_sleeve=args.default_sleeve,
            aggregate_freq=args.aggregate_freq,
            base_equity=args.base_equity,
            normalize_on_ingest=not args.no_normalize_on_ingest,
            execution_mode=args.execution_mode,
            execution_webhook_url=args.execution_webhook_url,
            execution_webhook_token=args.execution_webhook_token,
            execution_log_path=Path(args.execution_log) if args.execution_log else None,
            execution_min_trade_weight=args.execution_min_trade_weight,
            failure_log_path=Path(args.failure_log) if args.failure_log else None,
            worker_poll_interval_seconds=args.worker_poll_interval,
            retry_delay_seconds=args.retry_delay,
            execution_probe_on_startup=not args.disable_execution_probe,
            execution_probe_url=args.execution_probe_url,
            execution_probe_timeout=args.execution_probe_timeout,
            cash_symbol=args.cash_symbol,
            allowed_origins=tuple(origin.strip() for origin in args.allow_origin if origin.strip()),
            allowed_origin_regex=args.allow_origin_regex.strip() if isinstance(args.allow_origin_regex, str) and args.allow_origin_regex.strip() else None,
        )
        uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
        return 0

    if args.command == "serve-env":
        settings = load_bridge_settings_from_env()
        host, port = load_bridge_bind_from_env()
        uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
        return 0

    if args.command == "normalize":
        returns = write_normalized_returns_csv(
            event_log_path=args.event_log,
            output_csv_path=args.output_csv,
            aggregate_freq=args.aggregate_freq,
            base_equity=args.base_equity,
            default_sleeve=args.default_sleeve,
        )
        print(f"Saved {len(returns)} normalized rows to {args.output_csv}")
        return 0

    raise ValueError(f"Unsupported command '{args.command}'.")


if __name__ == "__main__":
    raise SystemExit(main())