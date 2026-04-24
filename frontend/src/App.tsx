import { FormEvent, Suspense, lazy, useEffect, useState } from "react";

const LIVE_POLL_MS = 2500;
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");

const ReturnsSeriesChart = lazy(() => import("./charts/ReturnsSeriesChart"));
const ExecutionActivityChart = lazy(() => import("./charts/ExecutionActivityChart"));

const samplePayload = {
  timestamp: "2026-04-23T14:30:00Z",
  event_id: "local-react-probe-001",
  event_type: "rebalance",
  strategy: "LOCAL_PROBE",
  sleeve: "LOCAL_PROBE",
  ticker: "SPY",
  equity: 100000,
  return_value: 0.0025,
  current_weights: {
    SPY: 0.45,
    CASH: 0.55,
  },
  target_weights: {
    SPY: 0.6,
    CASH: 0.4,
  },
};

function resolveApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path) || API_BASE_URL.length === 0) {
    return path;
  }

  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

type Counts = {
  events: number;
  executions: number;
  failures: number;
  returns_rows: number;
};

type Health = {
  status: string;
  worker_running: boolean;
  execution_mode: string;
  fingerprint_cache_size: number;
  execution_completed_count: number;
};

type DashboardSnapshot = {
  generated_at: string;
  token_required: boolean;
  counts: Counts;
  health: Health;
  paths: Record<string, string | null>;
  recent_events: Array<Record<string, unknown>>;
  recent_executions: Array<Record<string, unknown>>;
  recent_failures: Array<Record<string, unknown>>;
  returns_preview: {
    row_count: number;
    columns: string[];
    rows: Array<Record<string, unknown>>;
  };
};

type ReturnsSeries = {
  row_count: number;
  date_column: string;
  asset_columns: string[];
  rows: Array<Record<string, unknown>>;
};

type ExecutionSeries = {
  row_count: number;
  rows: Array<Record<string, unknown>>;
};

type PollingState<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
};

function usePollingJson<T>(url: string, intervalMs: number): PollingState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const response = await fetch(resolveApiUrl(url), { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
        const payload = (await response.json()) as T;
        if (!cancelled) {
          setData(payload);
          setError(null);
        }
      } catch (caughtError) {
        if (!cancelled) {
          setError(caughtError instanceof Error ? caughtError.message : "Request failed");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void load();
    const timer = window.setInterval(load, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [intervalMs, refreshTick, url]);

  return {
    data,
    error,
    loading,
    refresh: () => setRefreshTick((current) => current + 1),
  };
}

function formatCount(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function formatDecimal(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function formatPercentage(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(parsed);
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function renderRecord(record: Record<string, unknown>): string {
  return JSON.stringify(record, null, 2);
}

function App() {
  const dashboard = usePollingJson<DashboardSnapshot>("/api/dashboard?recent_limit=8&returns_limit=12", LIVE_POLL_MS);
  const returnsSeries = usePollingJson<ReturnsSeries>("/api/dashboard/returns-series?limit=180", LIVE_POLL_MS);
  const executionSeries = usePollingJson<ExecutionSeries>("/api/dashboard/execution-series?limit=120", LIVE_POLL_MS);

  const [token, setToken] = useState("");
  const [payloadText, setPayloadText] = useState(JSON.stringify(samplePayload, null, 2));
  const [submitStatus, setSubmitStatus] = useState("Ready");
  const [submitResponse, setSubmitResponse] = useState("{}");
  const [submitting, setSubmitting] = useState(false);

  const refreshAll = () => {
    dashboard.refresh();
    returnsSeries.refresh();
    executionSeries.refresh();
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setSubmitStatus("Sending webhook...");

    try {
      const payload = JSON.parse(payloadText);
      const query = token.trim() ? `?token=${encodeURIComponent(token.trim())}` : "";
      const response = await fetch(resolveApiUrl(`/webhook${query}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await response.json().catch(() => ({ detail: "Bridge did not return JSON." }));
      setSubmitResponse(JSON.stringify({ ok: response.ok, status: response.status, body }, null, 2));
      setSubmitStatus(response.ok ? "Webhook accepted." : "Webhook rejected.");
      refreshAll();
    } catch (caughtError) {
      setSubmitResponse(
        JSON.stringify(
          {
            ok: false,
            status: 0,
            body: {
              detail: caughtError instanceof Error ? caughtError.message : "Request failed",
            },
          },
          null,
          2,
        ),
      );
      setSubmitStatus("Request failed before the bridge handled it.");
    } finally {
      setSubmitting(false);
    }
  };

  const snapshot = dashboard.data;
  const counts = snapshot?.counts;
  const health = snapshot?.health;
  const returnsRows = returnsSeries.data?.rows ?? [];
  const assetColumns = returnsSeries.data?.asset_columns ?? [];
  const displayedAssets = assetColumns.slice(0, 4);
  const executionRows = (executionSeries.data?.rows ?? []).map((row) => ({
    processedAt: String(row.processed_at ?? "-"),
    equity: numberValue(row.equity) ?? 0,
    orderCount: numberValue(row.order_count) ?? 0,
    targetCashWeight: numberValue(row.target_cash_weight) ?? 0,
  }));
  const lastUpdated = snapshot ? formatDateTime(snapshot.generated_at) : "Waiting for first snapshot";

  return (
    <div className="app-shell">
      <div className="glow glow--amber" aria-hidden="true" />
      <div className="glow glow--teal" aria-hidden="true" />

      <header className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Local bridge control surface</p>
          <h1>Wealth First Live Operations Board</h1>
          <p className="hero__lede">
            Real-time webhook intake, normalized return monitoring, and execution telemetry for the current wealth-first workflow.
          </p>
        </div>

        <div className="hero__meta panel-card">
          <button className="button button--primary" type="button" onClick={refreshAll}>
            Refresh everything
          </button>
          <div className="hero__status-line">
            <span className={`status-pill ${health?.worker_running ? "status-pill--live" : "status-pill--idle"}`}>
              {health?.worker_running ? "bridge live" : "bridge idle"}
            </span>
            <span className="muted">{lastUpdated}</span>
          </div>
          <dl className="mini-stats">
            <div>
              <dt>Execution mode</dt>
              <dd>{health?.execution_mode ?? "-"}</dd>
            </div>
            <div>
              <dt>Token</dt>
              <dd>{snapshot?.token_required ? "required" : "optional"}</dd>
            </div>
            <div>
              <dt>Cache size</dt>
              <dd>{formatCount(health?.fingerprint_cache_size)}</dd>
            </div>
            <div>
              <dt>Completed handoffs</dt>
              <dd>{formatCount(health?.execution_completed_count)}</dd>
            </div>
          </dl>
        </div>
      </header>

      <main className="page-grid">
        <section className="panel-card panel-card--wide">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Bridge telemetry</p>
              <h2>Live counts</h2>
            </div>
            <p className="muted">Fast read on ingestion, normalization, execution, and failure state.</p>
          </div>
          <div className="metric-grid">
            <article className="metric-block metric-block--amber">
              <span>Accepted events</span>
              <strong>{formatCount(counts?.events)}</strong>
            </article>
            <article className="metric-block metric-block--teal">
              <span>Executions</span>
              <strong>{formatCount(counts?.executions)}</strong>
            </article>
            <article className="metric-block metric-block--ink">
              <span>Return rows</span>
              <strong>{formatCount(counts?.returns_rows)}</strong>
            </article>
            <article className="metric-block metric-block--coral">
              <span>Failures</span>
              <strong>{formatCount(counts?.failures)}</strong>
            </article>
          </div>
          <div className="path-grid">
            <div>
              <span className="path-grid__label">Event log</span>
              <code>{snapshot?.paths.event_log_path ?? "-"}</code>
            </div>
            <div>
              <span className="path-grid__label">Returns CSV</span>
              <code>{snapshot?.paths.output_csv_path ?? "-"}</code>
            </div>
            <div>
              <span className="path-grid__label">Execution log</span>
              <code>{snapshot?.paths.execution_log_path ?? "-"}</code>
            </div>
            <div>
              <span className="path-grid__label">Failure log</span>
              <code>{snapshot?.paths.failure_log_path ?? "-"}</code>
            </div>
          </div>
        </section>

        <section className="panel-card panel-card--chart">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Normalized stream</p>
              <h2>Cumulative return series</h2>
            </div>
            <p className="muted">The bridge output CSV transformed into a chartable live series.</p>
          </div>
          <div className="chart-shell">
            {displayedAssets.length > 0 ? (
              <Suspense fallback={<div className="empty-state">Loading chart module...</div>}>
                <ReturnsSeriesChart
                  assetColumns={displayedAssets}
                  dateColumn={returnsSeries.data?.date_column ?? "date"}
                  rows={returnsRows}
                />
              </Suspense>
            ) : (
              <div className="empty-state">{returnsSeries.loading ? "Loading return series..." : "No return series available yet."}</div>
            )}
          </div>
        </section>

        <section className="panel-card panel-card--chart">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Execution activity</p>
              <h2>Orders and cash posture</h2>
            </div>
            <p className="muted">Execution volume and target cash weight over the recent handoff stream.</p>
          </div>
          <div className="chart-shell">
            {executionRows.length > 0 ? (
              <Suspense fallback={<div className="empty-state">Loading chart module...</div>}>
                <ExecutionActivityChart rows={executionRows} />
              </Suspense>
            ) : (
              <div className="empty-state">{executionSeries.loading ? "Loading execution activity..." : "No execution history available yet."}</div>
            )}
          </div>
        </section>

        <section className="panel-card panel-card--composer">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Test intake</p>
              <h2>Webhook composer</h2>
            </div>
            <p className="muted">Use the browser to push a local payload through the same backend intake path.</p>
          </div>
          <form className="composer" onSubmit={handleSubmit}>
            <label>
              <span>Webhook token</span>
              <input value={token} onChange={(event) => setToken(event.target.value)} placeholder="Optional unless the bridge requires one" />
            </label>
            <label>
              <span>JSON payload</span>
              <textarea value={payloadText} onChange={(event) => setPayloadText(event.target.value)} spellCheck={false} />
            </label>
            <div className="composer__footer">
              <button className="button button--secondary" type="submit" disabled={submitting}>
                {submitting ? "Posting..." : "POST /webhook"}
              </button>
              <span className="muted">{submitStatus}</span>
            </div>
          </form>
          <pre className="response-shell">{submitResponse}</pre>
        </section>

        <section className="panel-card panel-card--wide">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Recent activity</p>
              <h2>Bridge event stream</h2>
            </div>
            <p className="muted">Latest webhook events, execution handoffs, and failures from the live bridge.</p>
          </div>
          <div className="comparison-grid">
            <article className="panel-card">
              <div className="section-heading section-heading--compact">
                <div>
                  <p className="eyebrow">Events</p>
                  <h3>Recent webhook intake</h3>
                </div>
              </div>
              <div className="launch-progress-events">
                {(snapshot?.recent_events ?? []).length > 0 ? (
                  snapshot?.recent_events.map((record, index) => (
                    <pre key={`event-${index}`} className="response-shell response-shell--short">{renderRecord(record)}</pre>
                  ))
                ) : (
                  <div className="empty-state empty-state--compact">No events recorded yet.</div>
                )}
              </div>
            </article>

            <article className="panel-card">
              <div className="section-heading section-heading--compact">
                <div>
                  <p className="eyebrow">Executions</p>
                  <h3>Recent handoffs</h3>
                </div>
              </div>
              <div className="launch-progress-events">
                {(snapshot?.recent_executions ?? []).length > 0 ? (
                  snapshot?.recent_executions.map((record, index) => (
                    <pre key={`execution-${index}`} className="response-shell response-shell--short">{renderRecord(record)}</pre>
                  ))
                ) : (
                  <div className="empty-state empty-state--compact">No execution handoffs recorded yet.</div>
                )}
              </div>
            </article>

            <article className="panel-card">
              <div className="section-heading section-heading--compact">
                <div>
                  <p className="eyebrow">Failures</p>
                  <h3>Recent worker issues</h3>
                </div>
              </div>
              <div className="launch-progress-events">
                {(snapshot?.recent_failures ?? []).length > 0 ? (
                  snapshot?.recent_failures.map((record, index) => (
                    <pre key={`failure-${index}`} className="response-shell response-shell--short">{renderRecord(record)}</pre>
                  ))
                ) : (
                  <div className="empty-state empty-state--compact">No recent failures.</div>
                )}
              </div>
            </article>
          </div>
        </section>
      </main>

      {(dashboard.error || returnsSeries.error || executionSeries.error) && (
        <aside className="error-banner">
          {dashboard.error || returnsSeries.error || executionSeries.error}
        </aside>
      )}
    </div>
  );
}

export default App;
