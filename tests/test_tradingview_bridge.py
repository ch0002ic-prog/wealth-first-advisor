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

# Import build_promotion_gate for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


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
                    "WEALTH_FIRST_EVENT_LOG_PATH": str(
                        root / "runtime" / "events.jsonl"
                    ),
                    "WEALTH_FIRST_OUTPUT_CSV_PATH": str(
                        root / "runtime" / "returns.csv"
                    ),
                    "WEALTH_FIRST_EXECUTION_LOG_PATH": str(
                        root / "runtime" / "execution.jsonl"
                    ),
                    "WEALTH_FIRST_FAILURE_LOG_PATH": str(
                        root / "runtime" / "failures.jsonl"
                    ),
                    "WEALTH_FIRST_ARTIFACT_ROOT_PATH": str(
                        root / "runtime" / "artifacts"
                    ),
                    "WEALTH_FIRST_MAIN2_RETURNS_CSV_PATH": str(
                        root / "data" / "demo.csv"
                    ),
                    "WEALTH_FIRST_MAIN2_COMPARE_DETAIL_CSV_PATH": str(
                        root / "runtime" / "artifacts" / "baseline_detail.csv"
                    ),
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
            self.assertEqual(
                settings.execution_log_path, root / "runtime" / "execution.jsonl"
            )
            self.assertEqual(
                settings.failure_log_path, root / "runtime" / "failures.jsonl"
            )
            self.assertEqual(
                settings.artifact_root_path, root / "runtime" / "artifacts"
            )
            self.assertEqual(
                settings.main2_returns_csv_path, root / "data" / "demo.csv"
            )
            self.assertEqual(
                settings.main2_compare_detail_csv_path,
                root / "runtime" / "artifacts" / "baseline_detail.csv",
            )
            self.assertEqual(
                settings.allowed_origins,
                (
                    "https://wealth-first-advisor.vercel.app",
                    "https://preview.example.com",
                ),
            )
            self.assertEqual(
                settings.allowed_origin_regex,
                r"https://wealth-first-advisor-.*\.vercel\.app",
            )
            self.assertFalse(settings.normalize_on_ingest)
            self.assertFalse(settings.execution_probe_on_startup)
            self.assertEqual(settings.execution_mode, "paper")
            self.assertAlmostEqual(settings.execution_min_trade_weight, 0.125)
            self.assertEqual(settings.default_sleeve, "DEPLOYED_BRIDGE")

    def test_load_bridge_bind_from_env_prefers_platform_port_defaults(self) -> None:
        host, port = load_bridge_bind_from_env({"PORT": "9100"})
        self.assertEqual(host, "0.0.0.0")
        self.assertEqual(port, 9100)

        host, port = load_bridge_bind_from_env(
            {
                "PORT": "9100",
                "WEALTH_FIRST_HOST": "127.0.0.1",
                "WEALTH_FIRST_PORT": "9200",
            }
        )
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
            self.assertEqual(
                response.headers.get("access-control-allow-origin"),
                "https://wealth-first-advisor.vercel.app",
            )

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
                    "/api/pipeline/launch",
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
            (frontend_dist_path / "index.html").write_text(
                "<html><body><h1>Frontend Dist App</h1></body></html>", encoding="utf-8"
            )

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

    def test_dashboard_alias_prefers_frontend_dist_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            frontend_dist_path = Path(temp_dir) / "frontend-dist"
            frontend_dist_path.mkdir(parents=True, exist_ok=True)
            (frontend_dist_path / "index.html").write_text(
                "<html><body><h1>Frontend Dist App</h1></body></html>", encoding="utf-8"
            )

            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                    frontend_dist_path=frontend_dist_path,
                )
            )

            with TestClient(app) as client:
                response = client.get("/dashboard")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Frontend Dist App", response.text)

    def test_dashboard_snapshot_reports_frontend_dist_mode_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            frontend_dist_path = Path(temp_dir) / "frontend-dist"
            frontend_dist_path.mkdir(parents=True, exist_ok=True)
            (frontend_dist_path / "index.html").write_text(
                "<html><body><h1>Frontend Dist App</h1></body></html>", encoding="utf-8"
            )

            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                    frontend_dist_path=frontend_dist_path,
                )
            )

            with TestClient(app) as client:
                snapshot_response = client.get("/api/dashboard")

            self.assertEqual(snapshot_response.status_code, 200)
            snapshot = snapshot_response.json()
            self.assertEqual(snapshot["frontend"]["mode"], "frontend_dist")
            self.assertTrue(snapshot["frontend"]["frontend_index_exists"])

    def test_dashboard_snapshot_reports_fallback_mode_when_dist_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            missing_dist_path = Path(temp_dir) / "missing-frontend-dist"

            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                    frontend_dist_path=missing_dist_path,
                )
            )

            with TestClient(app) as client:
                snapshot_response = client.get("/api/dashboard")

            self.assertEqual(snapshot_response.status_code, 200)
            snapshot = snapshot_response.json()
            self.assertEqual(snapshot["frontend"]["mode"], "fallback_dashboard")
            self.assertFalse(snapshot["frontend"]["frontend_index_exists"])

    def test_dashboard_static_mount_serves_local_assets(self) -> None:
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
                response = client.get("/dashboard-static/dashboard.css")

            self.assertEqual(response.status_code, 200)
            self.assertIn(":root", response.text)

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
            if '/dashboard-static/dashboard.css' in response.text:
                self.assertIn('/dashboard-static/dashboard.css', response.text)
                self.assertIn('/dashboard-static/dashboard.js', response.text)
            else:
                self.assertIn('/assets/', response.text)

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
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "wealth_first_tv_bridge_example.pine"
        )
        content = script_path.read_text(encoding="utf-8")

        self.assertTrue(script_path.exists())
        self.assertEqual(
            re.findall(r"\{\{[^}]+\}\}", content), ["{{strategy.order.alert_message}}"]
        )
        self.assertIn("str.format_time(timenow", content)
        self.assertIn("strategy.equity", content)
        self.assertIn("strategy.netprofit", content)
        self.assertIn("syminfo.ticker", content)
        self.assertIn('input.string("LIVE_PROBE", "Strategy label")', content)
        self.assertIn(
            'input.string("LIVE_PROBE", "Sleeve / truth-series column")', content
        )
        self.assertIn('input.bool(false, "Enable force test order")', content)
        self.assertIn('input.string("", "Force test nonce")', content)
        self.assertIn("lastForceTestNonce", content)
        self.assertIn(
            "message := message + '\"strategy\":\"' + strategyName + '\",'", content
        )
        self.assertIn(
            "message := message + '\"sleeve\":\"' + sleeveName + '\",'", content
        )

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

            normalized = normalize_event_log_to_returns(
                event_log_path, base_equity=100_000.0
            )

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

    def test_webhook_app_writes_returns_csv_from_netprofit_strategy_alerts(
        self,
    ) -> None:
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
                self.assertTrue(
                    _wait_for(
                        lambda: (
                            output_csv_path.exists()
                            and len(pd.read_csv(output_csv_path)) == 2
                        )
                    )
                )

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

    def test_webhook_app_suppresses_duplicate_events_without_source_timestamp(
        self,
    ) -> None:
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
                    frontend_dist_path=Path(temp_dir) / "missing-frontend-dist",
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
                self.assertTrue(
                    _wait_for(
                        lambda: output_csv_path.exists() and execution_log_path.exists()
                    )
                )

                snapshot_response = client.get(
                    "/api/dashboard?recent_limit=5&returns_limit=5"
                )

            self.assertEqual(snapshot_response.status_code, 200)
            snapshot = snapshot_response.json()
            self.assertTrue(snapshot["token_required"])
            self.assertEqual(snapshot["counts"]["events"], 1)
            self.assertEqual(snapshot["counts"]["executions"], 1)
            self.assertEqual(snapshot["counts"]["failures"], 0)
            self.assertEqual(snapshot["counts"]["returns_rows"], 1)
            self.assertEqual(snapshot["frontend"]["mode"], "fallback_dashboard")
            self.assertEqual(snapshot["recent_events"][0]["sleeve"], "ALLOCATOR")
            self.assertEqual(
                snapshot["recent_executions"][0]["execution_mode"], "paper"
            )
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
                execution_response = client.get(
                    "/api/dashboard/execution-series?limit=5"
                )

            self.assertEqual(returns_response.status_code, 200)
            self.assertEqual(execution_response.status_code, 200)
            returns_payload = returns_response.json()
            execution_payload = execution_response.json()
            self.assertEqual(returns_payload["asset_columns"], ["ALPHA", "BETA"])
            self.assertAlmostEqual(
                returns_payload["rows"][1]["cumulative_ALPHA"], 0.0403
            )
            self.assertEqual(execution_payload["rows"][0]["order_count"], 2)
            self.assertEqual(execution_payload["rows"][0]["execution_mode"], "paper")

    def test_pipeline_endpoints_return_main2_artifact_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            artifact_root_path = Path(temp_dir) / "artifacts"
            artifact_root_path.mkdir(parents=True, exist_ok=True)

            summary_payload = {
                "overall": {
                    "rows": 6,
                    "policy_beats_static_hold_rows": 4,
                    "policy_beats_optimizer_rows": 6,
                    "mean_policy_total_return": 0.21,
                    "mean_delta_total_return_vs_static_hold": 0.03,
                    "mean_delta_total_return_vs_optimizer": 0.19,
                },
                "weak_rows": {
                    "worst_vs_static_hold": [
                        {
                            "split": "chrono",
                            "seed": 7,
                            "fold": "fold_01",
                            "phase": "test",
                            "delta_total_return_vs_static_hold": -0.01,
                        }
                    ]
                },
            }
            comparison_payload = {
                "shared_rows": 6,
                "mean_policy_total_return_diff": 0.012,
                "main2_win_rows": 4,
                "current_win_rows": 2,
            }
            detail_csv = (
                "split,seed,fold,phase,policy_total_return,delta_total_return_vs_static_hold\n"
                "chrono,7,fold_01,test,0.2,-0.01\n"
                "regime,7,fold_01,validation,0.24,0.03\n"
            )

            (artifact_root_path / "main2_demo_summary.json").write_text(
                json.dumps(summary_payload), encoding="utf-8"
            )
            (artifact_root_path / "main2_demo_vs_current_best_summary.json").write_text(
                json.dumps(comparison_payload), encoding="utf-8"
            )
            (artifact_root_path / "main2_demo_detail.csv").write_text(
                detail_csv, encoding="utf-8"
            )

            app = create_app(
                BridgeSettings(
                    event_log_path=event_log_path,
                    normalize_on_ingest=False,
                    worker_poll_interval_seconds=0.05,
                    artifact_root_path=artifact_root_path,
                )
            )

            with TestClient(app) as client:
                experiments_response = client.get("/api/pipeline/experiments?limit=5")
                experiment_response = client.get(
                    "/api/pipeline/experiments/main2_demo?detail_limit=10"
                )

            self.assertEqual(experiments_response.status_code, 200)
            self.assertEqual(experiment_response.status_code, 200)
            experiments_payload = experiments_response.json()
            experiment_payload = experiment_response.json()
            self.assertEqual(experiments_payload["available_experiment_count"], 1)
            self.assertEqual(
                experiments_payload["recommended_experiment_id"], "main2_demo"
            )
            self.assertEqual(
                experiments_payload["experiments"][0]["comparison_metrics"][
                    "mean_policy_total_return_diff"
                ],
                0.012,
            )
            self.assertEqual(
                experiment_payload["experiment"]["artifact_id"], "main2_demo"
            )
            self.assertEqual(experiment_payload["detail"]["row_count"], 2)
            self.assertEqual(experiment_payload["detail"]["rows"][0]["split"], "chrono")

    def test_pipeline_launch_endpoints_start_and_report_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_log_path = Path(temp_dir) / "events.jsonl"
            artifact_root_path = Path(temp_dir) / "artifacts"
            returns_csv_path = Path(temp_dir) / "demo_sleeves.csv"
            artifact_root_path.mkdir(parents=True, exist_ok=True)
            returns_csv_path.write_text(
                "date,SPY_BENCHMARK\n2026-01-01,0.01\n2026-01-02,-0.02\n2026-01-05,0.015\n",
                encoding="utf-8",
            )

            class _FakeCompletedProcess:
                def __init__(self, command, cwd, stdout, stderr, text):
                    self.command = command
                    self.cwd = cwd
                    self.pid = 43210
                    self.returncode = None
                    stdout.write("fake main2 completed\n")
                    stdout.flush()

                    progress_log_path = Path(
                        command[command.index("--progress-log") + 1]
                    )
                    progress_log_path.write_text(
                        "\n".join(
                            [
                                json.dumps(
                                    {
                                        "event_type": "run_started",
                                        "timestamp": "2026-01-01T00:00:00+00:00",
                                        "total_folds": 2,
                                    }
                                ),
                                json.dumps(
                                    {
                                        "event_type": "fold_started",
                                        "timestamp": "2026-01-01T00:00:01+00:00",
                                        "split": "chrono",
                                        "seed": 7,
                                        "fold": "fold_01",
                                        "completed_folds": 0,
                                        "total_folds": 2,
                                    }
                                ),
                                json.dumps(
                                    {
                                        "event_type": "training_checkpoint",
                                        "timestamp": "2026-01-01T00:00:02+00:00",
                                        "split": "chrono",
                                        "seed": 7,
                                        "fold": "fold_01",
                                        "completed_folds": 0,
                                        "total_folds": 2,
                                        "step": 64,
                                        "train_steps": 128,
                                        "progress_fraction": 0.5,
                                        "validation_score": 0.11,
                                        "best_validation_score": 0.13,
                                        "epsilon": 0.41,
                                        "replay_size": 96,
                                        "episode_reset_count": 3,
                                        "validation_slice_scores": [0.07, 0.11, 0.15],
                                        "latest_loss": 0.02,
                                        "checkpoint_improved": False,
                                    }
                                ),
                                json.dumps(
                                    {
                                        "event_type": "fold_completed",
                                        "timestamp": "2026-01-01T00:00:03+00:00",
                                        "split": "chrono",
                                        "seed": 7,
                                        "fold": "fold_01",
                                        "completed_folds": 1,
                                        "total_folds": 2,
                                        "runtime_seconds": 1.5,
                                        "validation_total_return": 0.08,
                                        "test_total_return": 0.05,
                                        "validation_average_spy_weight": 0.62,
                                        "test_average_spy_weight": 0.59,
                                        "validation_average_turnover": 0.04,
                                        "test_average_turnover": 0.03,
                                    }
                                ),
                                json.dumps(
                                    {
                                        "event_type": "run_completed",
                                        "timestamp": "2026-01-01T00:00:04+00:00",
                                        "completed_folds": 2,
                                        "total_folds": 2,
                                    }
                                ),
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                def poll(self):
                    if self.returncode is None:
                        self.returncode = 0
                    return self.returncode

                def wait(self, timeout=None):
                    return self.poll()

                def terminate(self):
                    self.returncode = -15

                def kill(self):
                    self.returncode = -9

            with mock.patch(
                "wealth_first.tradingview_bridge.subprocess.Popen",
                side_effect=_FakeCompletedProcess,
            ) as popen_mock:
                app = create_app(
                    BridgeSettings(
                        event_log_path=event_log_path,
                        normalize_on_ingest=False,
                        worker_poll_interval_seconds=0.05,
                        artifact_root_path=artifact_root_path,
                        main2_returns_csv_path=returns_csv_path,
                    )
                )

                with TestClient(app) as client:
                    status_response = client.get("/api/pipeline/launch")
                    launch_response = client.post(
                        "/api/pipeline/launch",
                        json={
                            "preset_id": "quick-probe",
                            "artifact_label": "Smoke Check",
                        },
                    )
                    latest_response = client.get("/api/pipeline/launch")

            self.assertEqual(status_response.status_code, 200)
            self.assertEqual(launch_response.status_code, 200)
            self.assertTrue(popen_mock.called)

            launch_payload = launch_response.json()
            self.assertEqual(launch_payload["status"], "accepted")
            self.assertEqual(launch_payload["job"]["preset_id"], "quick-probe")
            self.assertTrue(
                launch_payload["job"]["artifact_id"].startswith("main2_smoke_check_")
            )
            self.assertIn("wealth_first.main2", launch_payload["job"]["command"])
            self.assertIn("--progress-log", launch_payload["job"]["command"])

            latest_payload = latest_response.json()
            self.assertTrue(latest_payload["can_launch"])
            self.assertEqual(latest_payload["latest_job"]["status"], "completed")
            self.assertEqual(latest_payload["latest_job"]["exit_code"], 0)
            self.assertEqual(latest_payload["presets"][0]["id"], "quick-probe")
            self.assertIn(
                "fake main2 completed",
                "\n".join(latest_payload["latest_job"]["log_tail"]),
            )
            self.assertTrue(latest_payload["latest_job"]["progress_log_exists"])
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["latest_event_type"],
                "run_completed",
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["completed_folds"], 2
            )
            self.assertEqual(latest_payload["latest_job"]["progress"]["total_folds"], 2)
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["overall_progress_fraction"],
                1.0,
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["active_fold"], "fold_01"
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["latest_validation_score"],
                0.11,
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["latest_epsilon"], 0.41
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["latest_replay_size"], 96
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["latest_episode_reset_count"],
                3,
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["validation_slice_scores"],
                [0.07, 0.11, 0.15],
            )
            self.assertEqual(
                len(latest_payload["latest_job"]["progress"]["checkpoint_history"]), 1
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["checkpoint_history"][0][
                    "step"
                ],
                64,
            )
            self.assertEqual(
                latest_payload["latest_job"]["progress"]["latest_fold_metrics"][
                    "test_total_return"
                ],
                0.05,
            )

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

            records = [
                json.loads(line)
                for line in execution_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            plan = records[0]["plan"]
            self.assertEqual(plan["equity"], 100000)
            self.assertEqual(plan["target_cash_weight"], 0.6)
            self.assertEqual(len(plan["orders"]), 1)
            self.assertEqual(plan["orders"][0]["symbol"], "SPY")
            self.assertEqual(plan["orders"][0]["target_notional"], 40000.0)

    def test_normalization_worker_skips_rebalance_only_events_until_returns_exist(
        self,
    ) -> None:
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
                self.assertTrue(
                    _wait_for(
                        lambda: (
                            "last_no_signal_event_log_mtime"
                            in app.state.normalization_retry_state
                        )
                    )
                )
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


class PromotionGateTests(unittest.TestCase):
    """Unit tests for phase25z promotion gate materiality thresholds."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import build_promotion_gate from phase25z script."""
        try:
            from run_phase25z_margin_lift_search import build_promotion_gate
            cls.build_promotion_gate = build_promotion_gate
        except ImportError as e:
            raise ImportError(
                "Could not import build_promotion_gate from run_phase25z_margin_lift_search. "
                "Ensure scripts/run_phase25z_margin_lift_search.py is available."
            ) from e
    def _get_build_promotion_gate(self):
        """Dynamically import build_promotion_gate from phase25z script."""
        try:
            from run_phase25z_margin_lift_search import build_promotion_gate
            return build_promotion_gate
        except ImportError as e:
            raise ImportError(
                "Could not import build_promotion_gate from run_phase25z_margin_lift_search. "
                "Ensure scripts/run_phase25z_margin_lift_search.py is available."
            ) from e

    def test_promotion_gate_rejects_new_variant_when_baseline_missing(self) -> None:
        """Test that promotion fails if v25z_ref not in finalists."""
        build_promotion_gate = self._get_build_promotion_gate()
        gate = build_promotion_gate(
            finalists=["v25z_ridge_hi", "v25z_band_lo"],  # baseline v25z_ref missing
            best_row={"variant": "v25z_ridge_hi", "breach_count": 0, "all_non_path": True},
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref={
                "delta_min_case_slack": 5e-5,  # Above threshold
                "delta_mean_case_slack": 5e-5,
                "delta_mean_test_relative": 5e-5,
            },
        )
        self.assertEqual(gate["status"], "KEEP_REFERENCE")
        self.assertEqual(gate["recommended_variant"], "v25z_ref")
        self.assertFalse(gate["checks"]["baseline_in_finalists"])

    def test_promotion_gate_rejects_when_materiality_delta_below_eps(self) -> None:
        """Test that promotion fails when delta is below materiality eps."""
        build_promotion_gate = self._get_build_promotion_gate()
        # Use eps-1e-6 (below the 1e-5 threshold)
        eps_minus = 1e-5 - 1e-6  # Should fail materiality check
        gate = build_promotion_gate(
            finalists=["v25z_ref", "v25z_ridge_hi", "v25z_band_lo"],
            best_row={"variant": "v25z_ridge_hi", "breach_count": 0, "all_non_path": True},
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref={
                "delta_min_case_slack": eps_minus,
                "delta_mean_case_slack": eps_minus,
                "delta_mean_test_relative": eps_minus,
            },
        )
        self.assertEqual(gate["status"], "KEEP_REFERENCE")
        self.assertEqual(gate["recommended_variant"], "v25z_ref")
        self.assertFalse(gate["checks"]["materiality_min_case_slack"])
        self.assertFalse(gate["checks"]["materiality_mean_case_slack"])
        self.assertFalse(gate["checks"]["materiality_mean_test_relative"])

    def test_promotion_gate_accepts_when_materiality_delta_above_eps(self) -> None:
        """Test that promotion passes when delta is above materiality eps."""
        build_promotion_gate = self._get_build_promotion_gate()
        # Use eps+1e-6 (above the 1e-5 threshold)
        eps_plus = 1e-5 + 1e-6  # Should pass materiality check
        gate = build_promotion_gate(
            finalists=["v25z_ref", "v25z_ridge_hi", "v25z_band_lo"],
            best_row={"variant": "v25z_ridge_hi", "breach_count": 0, "all_non_path": True},
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref={
                "delta_min_case_slack": eps_plus,
                "delta_mean_case_slack": eps_plus,
                "delta_mean_test_relative": eps_plus,
            },
        )
        self.assertEqual(gate["status"], "PROMOTE_NEW_VARIANT")
        self.assertEqual(gate["recommended_variant"], "v25z_ridge_hi")
        self.assertTrue(gate["checks"]["materiality_min_case_slack"])
        self.assertTrue(gate["checks"]["materiality_mean_case_slack"])
        self.assertTrue(gate["checks"]["materiality_mean_test_relative"])

    def test_promotion_gate_boundary_at_exact_eps(self) -> None:
        """Test behavior when delta equals materiality eps exactly."""
        build_promotion_gate = self._get_build_promotion_gate()
        gate = build_promotion_gate(
            finalists=["v25z_ref", "v25z_ridge_hi", "v25z_band_lo"],
            best_row={"variant": "v25z_ridge_hi", "breach_count": 0, "all_non_path": True},
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref={
                "delta_min_case_slack": 1e-5,  # Exactly at threshold
                "delta_mean_case_slack": 1e-5,
                "delta_mean_test_relative": 1e-5,
            },
        )
        # At exact boundary, >= check should pass
        self.assertEqual(gate["status"], "PROMOTE_NEW_VARIANT")
        self.assertEqual(gate["recommended_variant"], "v25z_ridge_hi")
        self.assertTrue(gate["checks"]["materiality_min_case_slack"])

    def test_promotion_gate_rejects_when_best_not_feasible(self) -> None:
        """Test that promotion fails if best variant is not feasible."""
        build_promotion_gate = self._get_build_promotion_gate()
        gate = build_promotion_gate(
            finalists=["v25z_ref", "v25z_ridge_hi"],
            best_row={
                "variant": "v25z_ridge_hi",
                "breach_count": 1,  # Not feasible
                "all_non_path": False,
            },
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref={
                "delta_min_case_slack": 5e-5,
                "delta_mean_case_slack": 5e-5,
                "delta_mean_test_relative": 5e-5,
            },
        )
        self.assertEqual(gate["status"], "KEEP_REFERENCE")
        self.assertFalse(gate["checks"]["best_is_feasible"])

    def test_promotion_gate_retains_ref_when_best_is_baseline(self) -> None:
        """Test that promotion keeps reference if best variant is v25z_ref itself."""
        build_promotion_gate = self._get_build_promotion_gate()
        gate = build_promotion_gate(
            finalists=["v25z_ref", "v25z_ridge_hi"],
            best_row={"variant": "v25z_ref", "breach_count": 0, "all_non_path": True},
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref={
                "delta_min_case_slack": 5e-5,
                "delta_mean_case_slack": 5e-5,
                "delta_mean_test_relative": 5e-5,
            },
        )
        self.assertEqual(gate["status"], "KEEP_REFERENCE")
        self.assertEqual(gate["recommended_variant"], "v25z_ref")

    def test_promotion_gate_preserves_improvement_metrics(self) -> None:
        """Test that gate preserves all improvement metric details."""
        build_promotion_gate = self._get_build_promotion_gate()
        improvement = {
            "delta_min_case_slack": 1.87e-9,
            "delta_mean_case_slack": 2.56e-9,
            "delta_mean_test_relative": -8.79e-10,
        }
        gate = build_promotion_gate(
            finalists=["v25z_ref", "v25z_ridge_hi"],
            best_row={"variant": "v25z_ridge_hi", "breach_count": 0, "all_non_path": True},
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref=improvement,
        )
        self.assertEqual(gate["improvement_vs_ref"], improvement)

    def test_promotion_gate_structure_completeness(self) -> None:
        """Test that gate output includes all required fields."""
        build_promotion_gate = self._get_build_promotion_gate()
        gate = build_promotion_gate(
            finalists=["v25z_ref", "v25z_ridge_hi"],
            best_row={"variant": "v25z_ridge_hi", "breach_count": 0, "all_non_path": True},
            ref_row={"variant": "v25z_ref"},
            improvement_vs_ref={
                "delta_min_case_slack": 1e-5,
                "delta_mean_case_slack": 1e-5,
                "delta_mean_test_relative": 1e-5,
            },
        )
        required_fields = [
            "status",
            "recommended_variant",
            "best_variant",
            "reference_variant",
            "materiality_eps",
            "checks",
            "improvement_vs_ref",
        ]
        for field in required_fields:
            self.assertIn(field, gate)
        required_checks = [
            "baseline_in_finalists",
            "full_ref_present",
            "best_is_feasible",
            "materiality_min_case_slack",
            "materiality_mean_case_slack",
            "materiality_mean_test_relative",
        ]
        for check in required_checks:
            self.assertIn(check, gate["checks"])


if __name__ == "__main__":

    unittest.main()
