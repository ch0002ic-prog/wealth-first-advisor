const snapshotEndpoint = "/api/dashboard?recent_limit=12&returns_limit=16";
const refreshIntervalMs = 2500;

const samplePayload = {
  timestamp: "2026-04-23T14:30:00Z",
  event_id: "local-bridge-probe-001",
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

const elements = {
  eventsCount: document.getElementById("eventsCount"),
  executionsCount: document.getElementById("executionsCount"),
  failuresCount: document.getElementById("failuresCount"),
  returnsCount: document.getElementById("returnsCount"),
  workerPill: document.getElementById("workerPill"),
  lastUpdatedLabel: document.getElementById("lastUpdatedLabel"),
  pollingLabel: document.getElementById("pollingLabel"),
  eventLogPath: document.getElementById("eventLogPath"),
  returnsPath: document.getElementById("returnsPath"),
  executionPath: document.getElementById("executionPath"),
  failurePath: document.getElementById("failurePath"),
  frontendMode: document.getElementById("frontendMode"),
  tokenHint: document.getElementById("tokenHint"),
  tokenInput: document.getElementById("tokenInput"),
  payloadInput: document.getElementById("payloadInput"),
  submitButton: document.getElementById("submitButton"),
  submitStatus: document.getElementById("submitStatus"),
  submitResponse: document.getElementById("submitResponse"),
  refreshButton: document.getElementById("refreshButton"),
  eventsStream: document.getElementById("eventsStream"),
  executionsStream: document.getElementById("executionsStream"),
  failuresStream: document.getElementById("failuresStream"),
  returnsHead: document.getElementById("returnsHead"),
  returnsBody: document.getElementById("returnsBody"),
  webhookForm: document.getElementById("webhookForm"),
};

let refreshTimer = null;
let refreshInFlight = false;

function setText(node, value) {
  node.textContent = value == null || value === "" ? "-" : String(value);
}

function formatTimestamp(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }
  return parsed.toLocaleString();
}

function formatCompactNumber(value) {
  if (typeof value !== "number") {
    return String(value ?? 0);
  }
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
}

function formatOptionalNumber(value) {
  if (value == null || value === "") {
    return "-";
  }
  const numericValue = Number(value);
  if (Number.isNaN(numericValue)) {
    return String(value);
  }
  return numericValue.toFixed(4);
}

function shortFingerprint(value) {
  if (!value) {
    return "-";
  }
  const text = String(value);
  return `${text.slice(0, 8)}...${text.slice(-6)}`;
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function clearChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function createBadge(label, className = "") {
  const badge = document.createElement("span");
  badge.className = `badge ${className}`.trim();
  badge.textContent = label;
  return badge;
}

function createStreamCard({ title, subtitle, badges = [], details = null, stats = [] }) {
  const card = document.createElement("article");
  card.className = "stream-card";

  const head = document.createElement("div");
  head.className = "stream-card__head";

  const titleWrap = document.createElement("div");
  const titleNode = document.createElement("h3");
  titleNode.className = "stream-card__title";
  titleNode.textContent = title;
  titleWrap.appendChild(titleNode);

  const subtitleNode = document.createElement("p");
  subtitleNode.className = "muted";
  subtitleNode.textContent = subtitle;
  subtitleNode.style.margin = "0.35rem 0 0";
  titleWrap.appendChild(subtitleNode);
  head.appendChild(titleWrap);

  const badgeStrip = document.createElement("div");
  badgeStrip.className = "stream-card__meta";
  badges.forEach((badge) => badgeStrip.appendChild(badge));
  head.appendChild(badgeStrip);
  card.appendChild(head);

  if (stats.length > 0) {
    const statsRow = document.createElement("div");
    statsRow.className = "stream-card__stats";
    stats.forEach((item) => {
      statsRow.appendChild(createBadge(item.label, item.className || ""));
    });
    card.appendChild(statsRow);
  }

  if (details) {
    const detailsNode = document.createElement("details");
    const summaryNode = document.createElement("summary");
    summaryNode.textContent = details.label;
    detailsNode.appendChild(summaryNode);
    const pre = document.createElement("pre");
    pre.className = "payload-box";
    pre.textContent = prettyJson(details.value);
    detailsNode.appendChild(pre);
    card.appendChild(detailsNode);
  }

  return card;
}

function renderEmptyState(container, message) {
  clearChildren(container);
  container.className = "stream-list empty-state";
  container.textContent = message;
}

function renderEvents(events) {
  if (!events || events.length === 0) {
    renderEmptyState(elements.eventsStream, "No events yet.");
    return;
  }

  clearChildren(elements.eventsStream);
  elements.eventsStream.className = "stream-list";
  events.forEach((event) => {
    const badges = [
      createBadge(event.event_type || "event", "badge--warm"),
      createBadge(event.sleeve || "unknown sleeve", "badge--cool"),
      createBadge(event.symbol || "no symbol"),
    ];
    const stats = [
      { label: `return ${formatOptionalNumber(event.return_value)}` },
      { label: `net ${formatOptionalNumber(event.netprofit)}` },
      { label: `fingerprint ${shortFingerprint(event.fingerprint)}` },
    ];
    elements.eventsStream.appendChild(
      createStreamCard({
        title: formatTimestamp(event.timestamp),
        subtitle: `${event.strategy || "strategy not provided"}`,
        badges,
        stats,
        details: event.raw_payload ? { label: "Raw payload", value: event.raw_payload } : null,
      })
    );
  });
}

function renderExecutions(executions) {
  if (!executions || executions.length === 0) {
    renderEmptyState(elements.executionsStream, "No execution records yet.");
    return;
  }

  clearChildren(elements.executionsStream);
  elements.executionsStream.className = "stream-list";
  executions.forEach((record) => {
    const plan = record.plan || {};
    const response = record.response || {};
    const orders = Array.isArray(plan.orders) ? plan.orders.length : 0;
    const badges = [
      createBadge(record.execution_mode || "unknown mode", "badge--cool"),
      createBadge(`orders ${orders}`),
      createBadge(`cash ${formatOptionalNumber(plan.target_cash_weight)}`),
    ];
    const stats = [
      { label: `equity ${formatCompactNumber(plan.equity || 0)}` },
      { label: `adapter ${response.adapter || "-"}` },
      { label: `event ${shortFingerprint(record.event_fingerprint)}` },
    ];
    elements.executionsStream.appendChild(
      createStreamCard({
        title: formatTimestamp(record.processed_at),
        subtitle: response.status || "Execution response recorded",
        badges,
        stats,
        details: { label: "Execution record", value: record },
      })
    );
  });
}

function renderFailures(failures) {
  if (!failures || failures.length === 0) {
    renderEmptyState(elements.failuresStream, "No failures recorded.");
    return;
  }

  clearChildren(elements.failuresStream);
  elements.failuresStream.className = "stream-list";
  failures.forEach((record) => {
    const badges = [
      createBadge(record.category || "failure", "badge--danger"),
      createBadge(`attempt ${record.attempt ?? "-"}`),
      createBadge(shortFingerprint(record.event_fingerprint)),
    ];
    elements.failuresStream.appendChild(
      createStreamCard({
        title: formatTimestamp(record.recorded_at),
        subtitle: record.error || "Bridge worker failure",
        badges,
        details: { label: "Failure record", value: record },
      })
    );
  });
}

function renderReturnsTable(preview) {
  const columns = Array.isArray(preview?.columns) ? preview.columns : [];
  const rows = Array.isArray(preview?.rows) ? preview.rows : [];

  clearChildren(elements.returnsHead);
  clearChildren(elements.returnsBody);

  if (columns.length === 0 || rows.length === 0) {
    const headerCell = document.createElement("th");
    headerCell.textContent = "No data";
    elements.returnsHead.appendChild(headerCell);

    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.className = "empty-state";
    cell.colSpan = 1;
    cell.textContent = "No normalized return rows yet.";
    row.appendChild(cell);
    elements.returnsBody.appendChild(row);
    return;
  }

  columns.forEach((columnName) => {
    const headerCell = document.createElement("th");
    headerCell.textContent = columnName;
    elements.returnsHead.appendChild(headerCell);
  });

  rows.forEach((rowData) => {
    const row = document.createElement("tr");
    columns.forEach((columnName) => {
      const cell = document.createElement("td");
      const rawValue = rowData[columnName];
      cell.textContent = typeof rawValue === "number" ? formatOptionalNumber(rawValue) : String(rawValue ?? "-");
      row.appendChild(cell);
    });
    elements.returnsBody.appendChild(row);
  });
}

function updateWorkerPill(health) {
  const running = Boolean(health?.worker_running);
  elements.workerPill.className = `status-pill ${running ? "status-pill--live" : "status-pill--error"}`;
  elements.workerPill.textContent = running ? "worker live" : "worker offline";
}

function applySnapshot(snapshot) {
  const counts = snapshot.counts || {};
  setText(elements.eventsCount, counts.events ?? 0);
  setText(elements.executionsCount, counts.executions ?? 0);
  setText(elements.failuresCount, counts.failures ?? 0);
  setText(elements.returnsCount, counts.returns_rows ?? 0);

  const paths = snapshot.paths || {};
  setText(elements.eventLogPath, paths.event_log_path);
  setText(elements.returnsPath, paths.output_csv_path);
  setText(elements.executionPath, paths.execution_log_path);
  setText(elements.failurePath, paths.failure_log_path);
  setText(elements.frontendMode, snapshot.frontend?.mode);

  updateWorkerPill(snapshot.health);
  elements.lastUpdatedLabel.textContent = `Last updated ${formatTimestamp(snapshot.generated_at)}`;
  elements.tokenHint.textContent = snapshot.token_required
    ? "Bridge token is required for webhook POSTs."
    : "Bridge token is optional for webhook POSTs.";

  renderEvents(snapshot.recent_events || []);
  renderExecutions(snapshot.recent_executions || []);
  renderFailures(snapshot.recent_failures || []);
  renderReturnsTable(snapshot.returns_preview || {});
}

async function refreshSnapshot() {
  if (refreshInFlight) {
    return;
  }

  refreshInFlight = true;
  elements.refreshButton.disabled = true;
  try {
    const response = await fetch(snapshotEndpoint, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Snapshot request failed with ${response.status}`);
    }
    const snapshot = await response.json();
    applySnapshot(snapshot);
  } catch (error) {
    elements.workerPill.className = "status-pill status-pill--error";
    elements.workerPill.textContent = "snapshot failed";
    elements.lastUpdatedLabel.textContent = error instanceof Error ? error.message : "Snapshot failed";
  } finally {
    refreshInFlight = false;
    elements.refreshButton.disabled = false;
  }
}

async function submitWebhook(event) {
  event.preventDefault();
  elements.submitButton.disabled = true;
  elements.submitStatus.textContent = "Sending request...";

  try {
    const payload = JSON.parse(elements.payloadInput.value);
    const token = elements.tokenInput.value.trim();
    const query = token ? `?token=${encodeURIComponent(token)}` : "";
    const response = await fetch(`/webhook${query}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const responseBody = await response.json().catch(() => ({ detail: "No JSON response body." }));
    elements.submitResponse.textContent = prettyJson({
      ok: response.ok,
      status: response.status,
      body: responseBody,
    });
    elements.submitStatus.textContent = response.ok ? "Webhook accepted." : "Webhook rejected.";
    await refreshSnapshot();
  } catch (error) {
    elements.submitResponse.textContent = prettyJson({
      ok: false,
      status: 0,
      body: {
        detail: error instanceof Error ? error.message : "Request failed",
      },
    });
    elements.submitStatus.textContent = "Request failed before the bridge handled it.";
  } finally {
    elements.submitButton.disabled = false;
  }
}

function startPolling() {
  if (refreshTimer !== null) {
    window.clearInterval(refreshTimer);
  }
  refreshTimer = window.setInterval(refreshSnapshot, refreshIntervalMs);
}

function boot() {
  elements.payloadInput.value = prettyJson(samplePayload);
  elements.pollingLabel.textContent = `Polling every ${(refreshIntervalMs / 1000).toFixed(1)} seconds`;
  elements.refreshButton.addEventListener("click", refreshSnapshot);
  elements.webhookForm.addEventListener("submit", submitWebhook);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      refreshSnapshot();
    }
  });
  startPolling();
  refreshSnapshot();
}

boot();