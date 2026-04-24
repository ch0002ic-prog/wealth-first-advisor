from __future__ import annotations

import json
import re
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from wealth_first.tradingview_bridge import (
    BridgeSettings,
    create_app,
    load_bridge_bind_from_env,
    load_bridge_settings_from_env,
    normalize_event_log_to_returns,
    normalize_tradingview_payload,
)


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _ProbeHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class _ProbeServer:
    def __enter__(self) -> str:
        self.server = HTTPServer(("127.0.0.1", 0), _ProbeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/probe"

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2.0)
        self.server.server_close()


class TradingViewBridgeTests(unittest.TestCase):
    def test_load_bridge_settings_from_env_supports_deployment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = load_bridge_settings_from_env(
                {
                    "WEALTH_FIRST_EVENT_LOG_PATH": str(root / "runtime" / "events.jsonl"),
                    "WEALTH_FIRST_OUTPUT_CSV_PATH": str(root / "runtime" / "returns.csv"),
                    "WEALTH_FIRST_EXECUTION_LOG_PATH": str(root / "runtime" / "execution.jsonl"),
                    "WEALTH_FIRST_FAILURE_LOG_PATH": str(root / "runtime" / "failures.jsonl"),
                    "WEALTH_FIRST_ARTIFACT_ROOT_PATH": str(root / "runtime" / "artifacts"),
                    "WEALTH_FIRST_ALLOWED_ORIGINS": "https://wealth-first-advisor.vercel.app, https://preview.example.com ",
                    "WEALTH_FIRST_ALLOWED_ORIGIN_REGEX": r"https://wealth-first-advisor-.*\.vercel\.app",
                    "WEALTH_FIRST_NORMALIZE_ON_INGEST": "false",
                    "WEALTH_FIRST_EXECUTION_PROBE_ON_STARTUP": "no",
                    "WEALTH_FIRST_EXECUTION_MODE": "paper",
                    "WEALTH_FIRST_EXECUTION_MIN_TRADE_WEIGHT": "0.125",
                    "WEALTH_FIRST_DEFAULT_SLEEVE": "DEPLOYED_BRIDGE",
                }
            )

            self.assertEqual(settings.event_log_path, root / "runtime" / "events.jsonl")
            self.assertEqual(settings.output_csv_path, root / "runtime" / "returns.csv")
            self.assertEqual(settings.execution_log_path, root / "runtime" / "execution.jsonl")
            self.assertEqual(settings.failure_log_path, root / "runtime" / "failures.jsonl")
            self.assertEqual(settings.artifact_root_path, root / "runtime" / "artifacts")
            self.assertEqual(
                settings.allowed_origins,
                ("https://wealth-first-advisor.vercel.app", "https://preview.example.com"),
            )
            self.assertEqual(settings.allowed_origin_regex, r"https://wealth-first-advisor-.*\.vercel\.app")
            self.assertFalse(settings.normalize_on_ingest)
            self.assertFalse(settings.execution_probe_on_startup)
            self.assertEqual(settings.execution_mode, "paper")
            self.assertAlmostEqual(settings.execution_min_trade_weight, 0.125)
            self.assertEqual(settings.default_sleeve, "DEPLOYED_BRIDGE")

    def test_load_bridge_bind_from_env_prefers_platform_port_defaults(self) -> None:
        host, port = load_bridge_bind_from_env({"PORT": "9100"})
        self.assertEqual(host, "0.0.0.0")
        self.assertEqual(port, 9100)

        host, port = load_bridge_bind_from_env({"PORT": "9100", "WEALTH_FIRST_HOST": "127.0.0.1", "WEALTH_FIRST_PORT": "9200"})
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 9200)

    def test_api_routes_include_explicit_cors_origin_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                    allowed_origins=("https://wealth-first-advisor.vercel.app",),
                )
            )

            with TestClient(app) as client:
                response = client.options(
                    "/api/dashboard",
                    headers={
                        "Origin": "https://wealth-first-advisor.vercel.app",
                        "Access-Control-Request-Method": "GET",
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("access-control-allow-origin"), "https://wealth-first-advisor.vercel.app")

    def test_api_routes_include_cors_header_for_matching_origin_regex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                    allowed_origin_regex=r"https://wealth-first-advisor-.*\.vercel\.app",
                )
            )

            with TestClient(app) as client:
                response = client.options(
                    "/api/dashboard",
                    headers={
                        "Origin": "https://wealth-first-advisor-git-main-ch0002ic-prog.vercel.app",
                        "Access-Control-Request-Method": "GET",
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers.get("access-control-allow-origin"),
                "https://wealth-first-advisor-git-main-ch0002ic-prog.vercel.app",
            )
    def test_dashboard_root_prefers_frontend_dist_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            frontend_dist_path = Path(temp_dir) / "frontend-dist"
            frontend_dist_path.mkdir(parents=True, exist_ok=True)
            (frontend_dist_path / "index.html").write_text("<html><body><h1>Frontend Dist App</h1></body></html>", encoding="utf-8")

            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                    frontend_dist_path=frontend_dist_path,
                )
            )

            with TestClient(app) as client:
                response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Frontend Dist App", response.text)

    def test_dashboard_root_serves_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])
            self.assertIn("Wealth First Bridge Board", response.text)

    def test_webhook_get_returns_reachability_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    webhook_token="secret",
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                response = client.get("/webhook?token=secret")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")
            self.assertEqual(response.json()["method"], "POST")

    def test_webhook_get_requires_valid_token_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    webhook_token="secret",
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                response = client.get("/webhook")

            self.assertEqual(response.status_code, 401)

    def test_pine_bridge_example_avoids_nested_tradingview_placeholders(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "wealth_first_tv_bridge_example.pine"
        content = script_path.read_text(encoding="utf-8")

        self.assertTrue(script_path.exists())
        self.assertEqual(re.findall(r"\{\{[^}]+\}\}", content), ["{{strategy.order.alert_message}}"])
        self.assertIn("str.format_time(timenow", content)
        self.assertIn("strategy.equity", content)
        self.assertIn("strategy.netprofit", content)
        self.assertIn("syminfo.ticker", content)
        self.assertIn('input.string("LIVE_PROBE", "Strategy label")', content)
        self.assertIn('input.string("LIVE_PROBE", "Sleeve / truth-series column")', content)
        self.assertIn('input.bool(false, "Enable force test order")', content)
        self.assertIn('input.string("", "Force test nonce")', content)
        self.assertIn("lastForceTestNonce", content)
        self.assertIn("message := message + '\"strategy\":\"' + strategyName + '\",'", content)
        self.assertIn("message := message + '\"sleeve\":\"' + sleeveName + '\",'", content)

    def test_normalize_payload_supports_percent_returns(self) -> None:
        payload = {
            "strategy": "TREND_FOLLOWING",
            "ticker": "SPY",
            "event_type": "fill",
            "timestamp": "2024-01-03T21:00:00Z",
            "return_pct": 1.5,
        }

        normalized = normalize_tradingview_payload(payload)

        self.assertEqual(normalized["sleeve"], "TREND_FOLLOWING")
        self.assertEqual(normalized["symbol"], "SPY")
        self.assertAlmostEqual(float(normalized["return_value"]), 0.015)

    def test_normalize_event_log_compounds_intraday_returns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            event_log_path.write_text(
                "\n".join(
                    [
                        '{"timestamp": "2024-01-02T10:00:00Z", "sleeve": "TREND_FOLLOWING", "return_value": 0.01}',
                        '{"timestamp": "2024-01-02T15:30:00Z", "sleeve": "TREND_FOLLOWING", "return_value": -0.005}',
                        '{"timestamp": "2024-01-03T10:00:00Z", "sleeve": "TREND_FOLLOWING", "return_value": 0.02}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            normalized = normalize_event_log_to_returns(event_log_path)

        self.assertEqual(list(normalized.columns), ["TREND_FOLLOWING"])
        self.assertEqual(len(normalized), 2)
        self.assertAlmostEqual(float(normalized.iloc[0, 0]), (1.01 * 0.995) - 1.0)

    def test_normalize_event_log_supports_netprofit_with_base_equity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            event_log_path.write_text(
                "\n".join(
                    [
                        '{"timestamp": "2024-01-02T10:00:00Z", "sleeve": "MEAN_REVERSION", "netprofit": 100.0}',
                        '{"timestamp": "2024-01-03T10:00:00Z", "sleeve": "MEAN_REVERSION", "netprofit": 250.0}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            normalized = normalize_event_log_to_returns(event_log_path, base_equity=100_000.0)

        self.assertEqual(len(normalized), 2)
        self.assertAlmostEqual(float(normalized.iloc[0, 0]), 0.001)
        self.assertAlmostEqual(float(normalized.iloc[1, 0]), 0.0015)

    def test_webhook_app_writes_event_log_and_returns_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            output_csv_path = Path(temp_dir) / "returns.csv"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    output_csv_path=output_csv_path,
                    webhook_token="secret",
                    default_sleeve="TRADINGVIEW",
                    aggregate_freq="D",
                    worker_poll_interval_seconds=0.05,
                )
            )
            with TestClient(app) as client:
                response = client.post(
                    "/webhook?token=secret",
                    json={
                        "strategy": "HEDGE_OVERLAY",
                        "ticker": "SPY",
                        "event_type": "fill",
                        "timestamp": "2024-01-03T21:00:00Z",
                        "return_value": -0.0025,
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["normalization_status"], "queued")
                self.assertEqual(response.json()["execution_status"], "disabled")
                self.assertTrue(event_log_path.exists())
                self.assertTrue(_wait_for(output_csv_path.exists))

            returns = pd.read_csv(output_csv_path)
            self.assertIn("HEDGE_OVERLAY", returns.columns)

    def test_webhook_app_writes_returns_csv_from_netprofit_strategy_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            output_csv_path = Path(temp_dir) / "returns.csv"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    output_csv_path=output_csv_path,
                    webhook_token="secret",
                    default_sleeve="TRADINGVIEW",
                    aggregate_freq="D",
                    base_equity=100_000.0,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                first_response = client.post(
                    "/webhook?token=secret",
                    json={
                        "timestamp": "2024-01-02T21:00:00Z",
                        "event_id": "tv-order-1",
                        "event_type": "rebalance",
                        "strategy": "LIVE_PROBE",
                        "sleeve": "LIVE_PROBE",
                        "ticker": "SPY",
                        "equity": 100000,
                        "netprofit": 100.0,
                        "target_weights": {"SPY": 0.40, "CASH": 0.60},
                    },
                )
                second_response = client.post(
                    "/webhook?token=secret",
                    json={
                        "timestamp": "2024-01-03T21:00:00Z",
                        "event_id": "tv-order-2",
                        "event_type": "rebalance",
                        "strategy": "LIVE_PROBE",
                        "sleeve": "LIVE_PROBE",
                        "ticker": "SPY",
                        "equity": 100000,
                        "netprofit": 250.0,
                        "target_weights": {"SPY": 0.00, "CASH": 1.00},
                    },
                )

                self.assertEqual(first_response.status_code, 200)
                self.assertEqual(second_response.status_code, 200)
                self.assertTrue(_wait_for(lambda: output_csv_path.exists() and len(pd.read_csv(output_csv_path)) == 2))

            returns = pd.read_csv(output_csv_path)
            self.assertEqual(len(returns), 2)
            self.assertAlmostEqual(float(returns.loc[0, "LIVE_PROBE"]), 0.001)
            self.assertAlmostEqual(float(returns.loc[1, "LIVE_PROBE"]), 0.0015)

    def test_webhook_app_suppresses_duplicate_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    webhook_token="secret",
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                )
            )
            payload = {
                "strategy": "TREND_FOLLOWING",
                "ticker": "SPY",
                "event_type": "fill",
                "timestamp": "2024-01-03T21:00:00Z",
                "return_value": 0.01,
                "event_id": "abc-123",
            }

            with TestClient(app) as client:
                first_response = client.post("/webhook?token=secret", json=payload)
                second_response = client.post("/webhook?token=secret", json=payload)

                self.assertEqual(first_response.status_code, 200)
                self.assertEqual(first_response.json()["status"], "accepted")
                self.assertEqual(second_response.status_code, 200)
                self.assertEqual(second_response.json()["status"], "duplicate")

            with event_log_path.open("r", encoding="utf-8") as handle:
                self.assertEqual(len([line for line in handle if line.strip()]), 1)

    def test_webhook_app_suppresses_duplicate_events_without_source_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    webhook_token="secret",
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                )
            )
            payload = {
                "strategy": "TREND_FOLLOWING",
                "ticker": "SPY",
                "event_type": "fill",
                "return_value": 0.01,
            }

            with TestClient(app) as client:
                first_response = client.post("/webhook?token=secret", json=payload)
                second_response = client.post("/webhook?token=secret", json=payload)

                self.assertEqual(first_response.status_code, 200)
                self.assertEqual(second_response.status_code, 200)
                self.assertEqual(second_response.json()["status"], "duplicate")

            with event_log_path.open("r", encoding="utf-8") as handle:
                self.assertEqual(len([line for line in handle if line.strip()]), 1)

    def test_webhook_app_writes_execution_handoff_for_target_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            execution_log_path = Path(temp_dir) / "execution.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    webhook_token="secret",
                    normalize_on_ingest=False,
                    execution_mode="paper",
                    execution_log_path=execution_log_path,
                    execution_min_trade_weight=0.01,
                    worker_poll_interval_seconds=0.05,
                )
            )
            with TestClient(app) as client:
                response = client.post(
                    "/webhook?token=secret",
                    json={
                        "strategy": "ALLOCATOR",
                        "ticker": "SPY",
                        "event_type": "rebalance",
                        "timestamp": "2024-01-03T21:00:00Z",
                        "target_weights": {"SPY": 0.55, "TLT": 0.25, "CASH": 0.20},
                        "current_weights": {"SPY": 0.40, "TLT": 0.30, "CASH": 0.30},
                        "equity": 100000,
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["execution_status"], "queued")
                self.assertTrue(_wait_for(execution_log_path.exists))

            with execution_log_path.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["execution_mode"], "paper")
            self.assertEqual(record["response"]["adapter"], "paper")
            self.assertEqual(record["plan"]["cash_symbol"], "CASH")
            self.assertEqual(len(record["plan"]["orders"]), 2)

    def test_dashboard_snapshot_returns_recent_bridge_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            output_csv_path = Path(temp_dir) / "returns.csv"
            execution_log_path = Path(temp_dir) / "execution.jsonl"
            failure_log_path = Path(temp_dir) / "failures.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    output_csv_path=output_csv_path,
                    execution_mode="paper",
                    execution_log_path=execution_log_path,
                    failure_log_path=failure_log_path,
                    webhook_token="secret",
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                response = client.post(
                    "/webhook?token=secret",
                    json={
                        "strategy": "ALLOCATOR",
                        "sleeve": "ALLOCATOR",
                        "ticker": "SPY",
                        "event_type": "rebalance",
                        "timestamp": "2024-01-03T21:00:00Z",
                        "return_value": 0.0025,
                        "target_weights": {"SPY": 0.55, "TLT": 0.25, "CASH": 0.20},
                        "current_weights": {"SPY": 0.40, "TLT": 0.30, "CASH": 0.30},
                        "equity": 100000,
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertTrue(_wait_for(lambda: output_csv_path.exists() and execution_log_path.exists()))

                snapshot_response = client.get("/api/dashboard?recent_limit=5&returns_limit=5")

            self.assertEqual(snapshot_response.status_code, 200)
            snapshot = snapshot_response.json()
            self.assertTrue(snapshot["token_required"])
            self.assertEqual(snapshot["counts"]["events"], 1)
            self.assertEqual(snapshot["counts"]["executions"], 1)
            self.assertEqual(snapshot["counts"]["failures"], 0)
            self.assertEqual(snapshot["counts"]["returns_rows"], 1)
            self.assertEqual(snapshot["recent_events"][0]["sleeve"], "ALLOCATOR")
            self.assertEqual(snapshot["recent_executions"][0]["execution_mode"], "paper")
            self.assertEqual(snapshot["returns_preview"]["columns"][0], "date")
            self.assertEqual(len(snapshot["returns_preview"]["rows"]), 1)

    def test_dashboard_series_endpoints_return_chart_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            output_csv_path = Path(temp_dir) / "returns.csv"
            execution_log_path = Path(temp_dir) / "execution.jsonl"
            output_csv_path.write_text(
                "date,ALPHA,BETA\n2026-04-21,0.01,-0.02\n2026-04-22,0.03,0.01\n",
                encoding="utf-8",
            )
            execution_log_path.write_text(
                json.dumps(
                    {
                        "processed_at": "2026-04-22T10:00:00Z",
                        "execution_mode": "paper",
                        "event_fingerprint": "exec-1",
                        "plan": {
                            "equity": 100000,
                            "target_cash_weight": 0.4,
                            "orders": [{"symbol": "SPY"}, {"symbol": "TLT"}],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    output_csv_path=output_csv_path,
                    execution_log_path=execution_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                returns_response = client.get("/api/dashboard/returns-series?limit=5")
                execution_response = client.get("/api/dashboard/execution-series?limit=5")

            self.assertEqual(returns_response.status_code, 200)
            self.assertEqual(execution_response.status_code, 200)
            returns_payload = returns_response.json()
            execution_payload = execution_response.json()
            self.assertEqual(returns_payload["asset_columns"], ["ALPHA", "BETA"])
            self.assertAlmostEqual(returns_payload["rows"][1]["cumulative_ALPHA"], 0.0403)
            self.assertEqual(execution_payload["rows"][0]["order_count"], 2)
            self.assertEqual(execution_payload["rows"][0]["execution_mode"], "paper")


    def test_webhook_accepts_strategy_alert_payload_for_paper_trade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            execution_log_path = Path(temp_dir) / "execution.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    webhook_token="secret",
                    normalize_on_ingest=False,
                    execution_mode="paper",
                    execution_log_path=execution_log_path,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                response = client.post(
                    "/webhook?token=secret",
                    json={
                        "timestamp": "2026-04-14T02:00:00Z",
                        "event_id": "tv-order-1",
                        "event_type": "rebalance",
                        "strategy": "LIVE_PROBE",
                        "sleeve": "LIVE_PROBE",
                        "ticker": "SPY",
                        "equity": 100000,
                        "target_weights": {"SPY": 0.40, "CASH": 0.60},
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "accepted")
                self.assertEqual(response.json()["symbol"], "SPY")
                self.assertEqual(response.json()["event_type"], "rebalance")
                self.assertTrue(_wait_for(execution_log_path.exists))

            records = [json.loads(line) for line in execution_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 1)
            plan = records[0]["plan"]
            self.assertEqual(plan["equity"], 100000)
            self.assertEqual(plan["target_cash_weight"], 0.6)
            self.assertEqual(len(plan["orders"]), 1)
            self.assertEqual(plan["orders"][0]["symbol"], "SPY")
            self.assertEqual(plan["orders"][0]["target_notional"], 40000.0)

    def test_normalization_worker_skips_rebalance_only_events_until_returns_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            output_csv_path = Path(temp_dir) / "returns.csv"
            failure_log_path = Path(temp_dir) / "failures.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    output_csv_path=output_csv_path,
                    failure_log_path=failure_log_path,
                    webhook_token="secret",
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                rebalance_response = client.post(
                    "/webhook?token=secret",
                    json={
                        "timestamp": "2026-04-14T02:00:00Z",
                        "event_id": "tv-order-1",
                        "event_type": "rebalance",
                        "strategy": "LIVE_PROBE",
                        "sleeve": "LIVE_PROBE",
                        "ticker": "SPY",
                        "equity": 100000,
                        "target_weights": {"SPY": 0.40, "CASH": 0.60},
                    },
                )

                self.assertEqual(rebalance_response.status_code, 200)
                self.assertTrue(_wait_for(lambda: "last_no_signal_event_log_mtime" in app.state.normalization_retry_state))
                self.assertFalse(output_csv_path.exists())
                self.assertFalse(failure_log_path.exists())

                return_response = client.post(
                    "/webhook?token=secret",
                    json={
                        "timestamp": "2026-04-15T02:00:00Z",
                        "event_id": "tv-order-2",
                        "event_type": "fill",
                        "strategy": "LIVE_PROBE",
                        "sleeve": "LIVE_PROBE",
                        "ticker": "SPY",
                        "return_value": 0.01,
                    },
                )

                self.assertEqual(return_response.status_code, 200)
                self.assertTrue(_wait_for(output_csv_path.exists))

            returns = pd.read_csv(output_csv_path)
            self.assertIn("LIVE_PROBE", returns.columns)

    def test_worker_replays_persisted_event_log_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            output_csv_path = Path(temp_dir) / "returns.csv"
            event_log_path.write_text(
                '{"timestamp": "2024-01-03T21:00:00Z", "sleeve": "TRADINGVIEW", "return_value": 0.01, "fingerprint": "startup-1", "raw_payload": {"timestamp": "2024-01-03T21:00:00Z", "return_value": 0.01}}\n',
                encoding="utf-8",
            )

            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    output_csv_path=output_csv_path,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app):
                self.assertTrue(_wait_for(output_csv_path.exists))

            returns = pd.read_csv(output_csv_path)
            self.assertIn("TRADINGVIEW", returns.columns)

    def test_create_app_rejects_webhook_execution_mode_without_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"

            with self.assertRaisesRegex(ValueError, "execution_webhook_url"):
                create_app(
                    BridgeSettings(
                        event_log_path=event_log_path,
                        execution_mode="webhook",
                    )
                )

    def test_create_app_accepts_reachable_webhook_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, _ProbeServer() as probe_url:
            event_log_path = Path(temp_dir) / "events.jsonl"
            execution_log_path = Path(temp_dir) / "execution.jsonl"
            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    execution_mode="webhook",
                    execution_webhook_url=probe_url,
                    execution_log_path=execution_log_path,
                    worker_poll_interval_seconds=0.05,
                )
            )

            with TestClient(app) as client:
                response = client.get("/healthz")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["worker_running"])

    def test_create_app_rejects_unreachable_webhook_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                host, port = sock.getsockname()
            unreachable_url = f"http://{host}:{port}/probe"

            with self.assertRaisesRegex(ValueError, "Execution probe"):
                create_app(
                    BridgeSettings(
                        event_log_path=event_log_path,
                        execution_mode="webhook",
                        execution_webhook_url=unreachable_url,
                        worker_poll_interval_seconds=0.05,
                    )
                )


if __name__ == "__main__":
    unittest.main()