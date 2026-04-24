import { FormEvent, Suspense, lazy, useEffect, useState } from "react";

const LIVE_POLL_MS = 2500;
const PIPELINE_POLL_MS = 8000;
const DETAIL_POLL_MS = 12000;
const PIPELINE_LAUNCH_POLL_MS = 3000;
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "")
  .trim()
  .replace(/\/+$/, "");

const ReturnsSeriesChart = lazy(() => import("./charts/ReturnsSeriesChart"));
const ExecutionActivityChart = lazy(
  () => import("./charts/ExecutionActivityChart"),
);
const ExperimentLeaderboardChart = lazy(
  () => import("./charts/ExperimentLeaderboardChart"),
);
const HardestWindowsChart = lazy(() => import("./charts/HardestWindowsChart"));
const LaunchProgressChart = lazy(() => import("./charts/LaunchProgressChart"));

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

type PipelineNode = {
  id: string;
  label: string;
  status: string;
  detail: string;
};

type ExperimentRecord = {
  artifact_id: string;
  label: string;
  modified_at: string;
  summary_path?: string;
  detail_path?: string | null;
  comparison_path?: string | null;
  leaderboard_metrics: {
    mean_policy_total_return: number | null;
    mean_delta_total_return_vs_static_hold: number | null;
    mean_delta_total_return_vs_optimizer: number | null;
    policy_beats_static_hold_rows: number;
    policy_beats_optimizer_rows: number;
    rows: number;
  };
  comparison_metrics: {
    mean_policy_total_return_diff: number | null;
    main2_win_rows: number | null;
    current_win_rows: number | null;
  };
  weak_rows_preview: Array<Record<string, unknown>>;
};

type ExperimentSortKey =
  | "comparison"
  | "static"
  | "optimizer"
  | "return"
  | "wins";
type ExperimentFilterMode =
  | "all"
  | "positive_current"
  | "majority_static"
  | "majority_optimizer";

type LaunchPreset = {
  id: string;
  label: string;
  description: string;
  estimated_runtime: string;
  default_artifact_label: string;
  recommended: boolean;
};

type PipelineLaunchProgressEvent = Record<string, unknown> & {
  event_type?: string;
  timestamp?: string;
  split?: string;
  seed?: number;
  fold?: string;
  step?: number;
  train_steps?: number;
  total_folds?: number;
  completed_folds?: number;
  progress_fraction?: number;
  validation_score?: number;
  best_validation_score?: number;
  test_total_return?: number;
  validation_total_return?: number;
  message?: string;
};

type PipelineLaunchProgress = {
  event_count: number;
  latest_event_type: string | null;
  latest_event_timestamp: string | null;
  stage_label: string;
  total_folds: number | null;
  completed_folds: number;
  active_split: string | null;
  active_seed: number | null;
  active_fold: string | null;
  active_step: number | null;
  train_steps: number | null;
  step_progress_fraction: number | null;
  overall_progress_fraction: number | null;
  latest_validation_score: number | null;
  best_validation_score: number | null;
  latest_loss: number | null;
  latest_epsilon: number | null;
  latest_replay_size: number | null;
  latest_episode_reset_count: number | null;
  validation_slice_scores: number[];
  checkpoint_improved: boolean | null;
  checkpoint_history: Array<{
    timestamp: string | null;
    split: string | null;
    seed: number | null;
    fold: string | null;
    step: number | null;
    train_steps: number | null;
    progress_fraction: number | null;
    validation_score: number | null;
    best_validation_score: number | null;
    latest_loss: number | null;
    epsilon: number | null;
    replay_size: number | null;
    episode_reset_count: number | null;
    checkpoint_improved: boolean | null;
  }>;
  latest_fold_metrics: {
    split: string | null;
    seed: number | null;
    fold: string | null;
    runtime_seconds: number | null;
    validation_total_return: number | null;
    test_total_return: number | null;
    validation_average_spy_weight: number | null;
    test_average_spy_weight: number | null;
    validation_average_turnover: number | null;
    test_average_turnover: number | null;
  } | null;
  recent_events: PipelineLaunchProgressEvent[];
};

type PipelineLaunchJob = {
  job_id: string;
  preset_id: string;
  preset_label: string;
  artifact_id: string;
  artifact_label: string;
  status: string;
  created_at: string;
  started_at: string;
  completed_at: string | null;
  exit_code: number | null;
  pid: number | null;
  command: string[];
  output_prefix: string;
  detail_path: string;
  summary_path: string;
  comparison_summary_path: string;
  log_path: string;
  progress_log_path: string;
  log_tail: string[];
  detail_exists: boolean;
  summary_exists: boolean;
  comparison_summary_exists: boolean;
  progress_log_exists: boolean;
  progress: PipelineLaunchProgress;
};

type PipelineLaunchState = {
  generated_at: string;
  artifact_root_path: string;
  returns_csv_path: string;
  compare_detail_csv_path: string;
  compare_detail_csv_exists: boolean;
  python_executable_path: string;
  can_launch: boolean;
  presets: LaunchPreset[];
  active_job: PipelineLaunchJob | null;
  latest_job: PipelineLaunchJob | null;
};

const DEFAULT_EXPERIMENT_SORT: ExperimentSortKey = "comparison";
const DEFAULT_EXPERIMENT_FILTER: ExperimentFilterMode = "all";
const EXPERIMENT_SORT_KEYS: ExperimentSortKey[] = [
  "comparison",
  "static",
  "optimizer",
  "return",
  "wins",
];
const EXPERIMENT_FILTER_MODES: ExperimentFilterMode[] = [
  "all",
  "positive_current",
  "majority_static",
  "majority_optimizer",
];

type PipelineOverview = {
  generated_at: string;
  approach: {
    id: string;
    label: string;
    module: string;
    model_family: string;
    description: string;
  };
  artifact_root_path: string;
  available_experiment_count: number;
  recommended_experiment_id: string | null;
  latest_experiment_id: string | null;
  pipeline_nodes: PipelineNode[];
  experiments: ExperimentRecord[];
};

type ExperimentDetail = {
  generated_at: string;
  approach: {
    id: string;
    label: string;
    module: string;
    model_family: string;
  };
  experiment: ExperimentRecord;
  summary: Record<string, unknown>;
  comparison_summary: Record<string, unknown> | null;
  detail: {
    row_count: number;
    columns: string[];
    rows: Array<Record<string, unknown>>;
  };
};

type PollingState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
};

function usePollingJson<T>(
  url: string | null,
  intervalMs: number,
): PollingState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(Boolean(url));
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!url) {
      setLoading(false);
      return;
    }

    let active = true;

    const load = async () => {
      try {
        setLoading(true);
        const response = await fetch(resolveApiUrl(url), { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
        const payload = (await response.json()) as T;
        if (active) {
          setData(payload);
          setError(null);
        }
      } catch (caughtError) {
        if (active) {
          setError(
            caughtError instanceof Error
              ? caughtError.message
              : "Request failed",
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    void load();
    const intervalId = window.setInterval(() => {
      void load();
    }, intervalMs);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [intervalMs, refreshKey, url]);

  return {
    data,
    loading,
    error,
    refresh: () => setRefreshKey((value) => value + 1),
  };
}

function numberValue(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function nullableNumber(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value !== "string" || value.trim().length === 0) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatPercentage(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSignedPercentage(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  const formatted = (value * 100).toFixed(digits);
  return `${value > 0 ? "+" : ""}${formatted}%`;
}

function formatCount(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  return new Intl.NumberFormat().format(value);
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatDecimal(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function shortId(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return value.length > 18
    ? `${value.slice(0, 10)}...${value.slice(-6)}`
    : value;
}

function compactLabel(value: string): string {
  return value.replace(/^2007 /, "");
}

function tooltipPercent(value: unknown): string {
  return formatPercentage(numberValue(value), 2);
}

function isExperimentSortKey(value: string | null): value is ExperimentSortKey {
  return (
    value !== null && EXPERIMENT_SORT_KEYS.includes(value as ExperimentSortKey)
  );
}

function isExperimentFilterMode(
  value: string | null,
): value is ExperimentFilterMode {
  return (
    value !== null &&
    EXPERIMENT_FILTER_MODES.includes(value as ExperimentFilterMode)
  );
}

function readExperimentViewUrlState(): {
  selectedExperimentId: string | null;
  comparisonExperimentId: string | null;
  experimentQuery: string;
  experimentSort: ExperimentSortKey;
  experimentFilter: ExperimentFilterMode;
} {
  if (typeof window === "undefined") {
    return {
      selectedExperimentId: null,
      comparisonExperimentId: null,
      experimentQuery: "",
      experimentSort: DEFAULT_EXPERIMENT_SORT,
      experimentFilter: DEFAULT_EXPERIMENT_FILTER,
    };
  }

  const params = new URLSearchParams(window.location.search);
  const sortParam = params.get("sort");
  const filterParam = params.get("filter");
  return {
    selectedExperimentId: params.get("exp"),
    comparisonExperimentId: params.get("cmp"),
    experimentQuery: params.get("q") ?? "",
    experimentSort: isExperimentSortKey(sortParam)
      ? sortParam
      : DEFAULT_EXPERIMENT_SORT,
    experimentFilter: isExperimentFilterMode(filterParam)
      ? filterParam
      : DEFAULT_EXPERIMENT_FILTER,
  };
}

function syncQueryParam(
  params: URLSearchParams,
  key: string,
  value: string | null | undefined,
  defaultValue?: string,
): void {
  const normalized = value?.trim();
  if (!normalized || normalized === defaultValue) {
    params.delete(key);
    return;
  }
  params.set(key, normalized);
}

function launchStatusClass(status: string | null | undefined): string {
  if (status === "running") {
    return "pipeline-node__state--live";
  }
  if (status === "completed") {
    return "pipeline-node__state--ready";
  }
  if (status === "failed" || status === "terminated") {
    return "pipeline-node__state--missing";
  }
  return "pipeline-node__state--idle";
}

function startCaseLabel(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return value
    .split("_")
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
}

function clampUnitInterval(value: number | null | undefined): number {
  if (value == null || !Number.isFinite(value)) {
    return 0;
  }
  return Math.min(Math.max(value, 0), 1);
}

function formatLaunchEventContext(event: PipelineLaunchProgressEvent): string {
  const parts: string[] = [];
  if (typeof event.split === "string") {
    parts.push(event.split);
  }
  if (typeof event.fold === "string") {
    parts.push(event.fold);
  }
  const seed = nullableNumber(event.seed);
  if (seed != null) {
    parts.push(`seed ${formatCount(seed)}`);
  }
  const step = nullableNumber(event.step);
  const trainSteps = nullableNumber(event.train_steps);
  if (step != null && trainSteps != null) {
    parts.push(`${formatCount(step)}/${formatCount(trainSteps)} steps`);
  }
  return parts.join(" · ") || "Run-level telemetry";
}

function formatLaunchEventDetail(event: PipelineLaunchProgressEvent): string {
  if (event.event_type === "training_checkpoint") {
    return `Validation ${formatPercentage(nullableNumber(event.validation_score))} · Best ${formatPercentage(nullableNumber(event.best_validation_score))}`;
  }
  if (event.event_type === "fold_completed") {
    return `Test ${formatPercentage(nullableNumber(event.test_total_return))} · Validation ${formatPercentage(nullableNumber(event.validation_total_return))}`;
  }
  if (event.event_type === "run_failed") {
    return typeof event.message === "string" ? event.message : "Run failed.";
  }
  const completedFolds = nullableNumber(event.completed_folds);
  const totalFolds = nullableNumber(event.total_folds);
  if (completedFolds != null && totalFolds != null) {
    return `${formatCount(completedFolds)} / ${formatCount(totalFolds)} folds`;
  }
  return "Adaptive RL pipeline event";
}

function ChartFallback({ message }: { message: string }) {
  return <div className="empty-state">{message}</div>;
}

function sortWeakRows(
  rows: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  return [...rows].sort(
    (left, right) =>
      numberValue(left.delta_total_return_vs_static_hold) -
      numberValue(right.delta_total_return_vs_static_hold),
  );
}

function differenceOrNull(
  left: number | null | undefined,
  right: number | null | undefined,
): number | null {
  if (
    left == null ||
    right == null ||
    !Number.isFinite(left) ||
    !Number.isFinite(right)
  ) {
    return null;
  }
  return left - right;
}

function buildWindowKey(row: Record<string, unknown>): string {
  return `${String(row.split ?? "-")}|${String(row.seed ?? "-")}|${String(row.fold ?? "-")}|${String(row.phase ?? "-")}`;
}

function buildWindowLabel(row: Record<string, unknown>): string {
  return `${String(row.split ?? "-")}/${String(row.seed ?? "-")}/${String(row.fold ?? "-")}/${String(row.phase ?? "-")}`;
}

function hasMajorityWins(wins: number, rows: number): boolean {
  return rows > 0 && wins >= Math.ceil(rows / 2);
}

function experimentSortMetric(
  experiment: ExperimentRecord,
  sortKey: ExperimentSortKey,
): number {
  switch (sortKey) {
    case "return":
      return (
        experiment.leaderboard_metrics.mean_policy_total_return ??
        Number.NEGATIVE_INFINITY
      );
    case "static":
      return (
        experiment.leaderboard_metrics.mean_delta_total_return_vs_static_hold ??
        Number.NEGATIVE_INFINITY
      );
    case "optimizer":
      return (
        experiment.leaderboard_metrics.mean_delta_total_return_vs_optimizer ??
        Number.NEGATIVE_INFINITY
      );
    case "wins":
      return experiment.leaderboard_metrics.rows > 0
        ? experiment.leaderboard_metrics.policy_beats_static_hold_rows /
            experiment.leaderboard_metrics.rows
        : Number.NEGATIVE_INFINITY;
    case "comparison":
    default:
      return (
        experiment.comparison_metrics.mean_policy_total_return_diff ??
        experiment.leaderboard_metrics.mean_delta_total_return_vs_static_hold ??
        Number.NEGATIVE_INFINITY
      );
  }
}

function deltaToneClass(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) {
    return "trend-value--neutral";
  }
  return value > 0 ? "trend-value--positive" : "trend-value--negative";
}

function App() {
  const dashboard = usePollingJson<DashboardSnapshot>(
    "/api/dashboard?recent_limit=8&returns_limit=12",
    LIVE_POLL_MS,
  );
  const returnsSeries = usePollingJson<ReturnsSeries>(
    "/api/dashboard/returns-series?limit=180",
    LIVE_POLL_MS,
  );
  const executionSeries = usePollingJson<ExecutionSeries>(
    "/api/dashboard/execution-series?limit=120",
    LIVE_POLL_MS,
  );
  const pipelineOverview = usePollingJson<PipelineOverview>(
    "/api/pipeline/experiments?limit=64",
    PIPELINE_POLL_MS,
  );
  const pipelineLaunch = usePollingJson<PipelineLaunchState>(
    "/api/pipeline/launch",
    PIPELINE_LAUNCH_POLL_MS,
  );

  const [initialExperimentView] = useState(readExperimentViewUrlState);

  const [selectedExperimentId, setSelectedExperimentId] = useState<
    string | null
  >(initialExperimentView.selectedExperimentId);
  const [comparisonExperimentId, setComparisonExperimentId] = useState<
    string | null
  >(initialExperimentView.comparisonExperimentId);
  const [experimentQuery, setExperimentQuery] = useState(
    initialExperimentView.experimentQuery,
  );
  const [experimentSort, setExperimentSort] = useState<ExperimentSortKey>(
    initialExperimentView.experimentSort,
  );
  const [experimentFilter, setExperimentFilter] =
    useState<ExperimentFilterMode>(initialExperimentView.experimentFilter);
  const [selectedLaunchPresetId, setSelectedLaunchPresetId] = useState("");
  const [launchArtifactLabel, setLaunchArtifactLabel] = useState("");
  const [launchSubmitStatus, setLaunchSubmitStatus] = useState("Ready");
  const [launchSubmitting, setLaunchSubmitting] = useState(false);
  const [processedLaunchSignature, setProcessedLaunchSignature] = useState<
    string | null
  >(null);
  const [token, setToken] = useState("");
  const [payloadText, setPayloadText] = useState(
    JSON.stringify(samplePayload, null, 2),
  );
  const [submitStatus, setSubmitStatus] = useState("Ready");
  const [submitResponse, setSubmitResponse] = useState("No request sent yet.");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const experiments = pipelineOverview.data?.experiments ?? [];
    if (!experiments.length) {
      return;
    }

    const selectionIsValid =
      selectedExperimentId !== null &&
      experiments.some(
        (experiment) => experiment.artifact_id === selectedExperimentId,
      );
    if (!selectionIsValid) {
      setSelectedExperimentId(
        pipelineOverview.data?.recommended_experiment_id ??
          experiments[0].artifact_id,
      );
    }
  }, [pipelineOverview.data, selectedExperimentId]);

  useEffect(() => {
    const experiments = pipelineOverview.data?.experiments ?? [];
    if (!experiments.length) {
      return;
    }

    const comparisonIsValid =
      comparisonExperimentId !== null &&
      comparisonExperimentId !== selectedExperimentId &&
      experiments.some(
        (experiment) => experiment.artifact_id === comparisonExperimentId,
      );

    if (comparisonIsValid) {
      return;
    }

    const fallbackComparison =
      experiments.find(
        (experiment) =>
          experiment.artifact_id !== selectedExperimentId &&
          experiment.artifact_id ===
            pipelineOverview.data?.recommended_experiment_id,
      ) ??
      experiments.find(
        (experiment) => experiment.artifact_id !== selectedExperimentId,
      );

    setComparisonExperimentId(fallbackComparison?.artifact_id ?? null);
  }, [comparisonExperimentId, pipelineOverview.data, selectedExperimentId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    syncQueryParam(params, "exp", selectedExperimentId);
    syncQueryParam(params, "cmp", comparisonExperimentId);
    syncQueryParam(params, "q", experimentQuery);
    syncQueryParam(params, "sort", experimentSort, DEFAULT_EXPERIMENT_SORT);
    syncQueryParam(
      params,
      "filter",
      experimentFilter,
      DEFAULT_EXPERIMENT_FILTER,
    );

    const nextSearch = params.toString();
    const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ""}`;
    const currentUrl = `${window.location.pathname}${window.location.search}`;
    if (nextUrl !== currentUrl) {
      window.history.replaceState(null, "", nextUrl);
    }
  }, [
    comparisonExperimentId,
    experimentFilter,
    experimentQuery,
    experimentSort,
    selectedExperimentId,
  ]);

  useEffect(() => {
    const presets = pipelineLaunch.data?.presets ?? [];
    if (!presets.length) {
      return;
    }

    const selectedPresetIsValid = presets.some(
      (preset) => preset.id === selectedLaunchPresetId,
    );
    if (!selectedPresetIsValid) {
      setSelectedLaunchPresetId(
        presets.find((preset) => preset.recommended)?.id ?? presets[0].id,
      );
    }
  }, [pipelineLaunch.data, selectedLaunchPresetId]);

  const experimentDetail = usePollingJson<ExperimentDetail>(
    selectedExperimentId
      ? `/api/pipeline/experiments/${selectedExperimentId}?detail_limit=120`
      : null,
    DETAIL_POLL_MS,
  );
  const comparisonDetail = usePollingJson<ExperimentDetail>(
    comparisonExperimentId
      ? `/api/pipeline/experiments/${comparisonExperimentId}?detail_limit=120`
      : null,
    DETAIL_POLL_MS,
  );

  const refreshAll = () => {
    dashboard.refresh();
    returnsSeries.refresh();
    executionSeries.refresh();
    pipelineOverview.refresh();
    pipelineLaunch.refresh();
    experimentDetail.refresh();
    comparisonDetail.refresh();
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setSubmitStatus("Sending webhook...");

    try {
      const payload = JSON.parse(payloadText);
      const query = token.trim()
        ? `?token=${encodeURIComponent(token.trim())}`
        : "";
      const response = await fetch(resolveApiUrl(`/webhook${query}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await response
        .json()
        .catch(() => ({ detail: "Bridge did not return JSON." }));
      setSubmitResponse(
        JSON.stringify(
          { ok: response.ok, status: response.status, body },
          null,
          2,
        ),
      );
      setSubmitStatus(response.ok ? "Webhook accepted." : "Webhook rejected.");
      refreshAll();
    } catch (caughtError) {
      setSubmitResponse(
        JSON.stringify(
          {
            ok: false,
            status: 0,
            body: {
              detail:
                caughtError instanceof Error
                  ? caughtError.message
                  : "Request failed",
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

  const handleLaunchMain2 = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedLaunchPresetId) {
      setLaunchSubmitStatus("Choose a launch preset first.");
      return;
    }

    setLaunchSubmitting(true);
    setLaunchSubmitStatus("Starting main2 run...");

    try {
      const response = await fetch(resolveApiUrl("/api/pipeline/launch"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preset_id: selectedLaunchPresetId,
          artifact_label: launchArtifactLabel.trim() || undefined,
        }),
      });
      const body = await response
        .json()
        .catch(() => ({ detail: "Launcher did not return JSON." }));
      if (!response.ok) {
        throw new Error(
          typeof body.detail === "string"
            ? body.detail
            : `Launch failed with status ${response.status}`,
        );
      }

      setLaunchSubmitStatus(
        `Started ${String(body.job?.artifact_id ?? selectedLaunchPresetId)}.`,
      );
      setLaunchArtifactLabel("");
      pipelineLaunch.refresh();
    } catch (caughtError) {
      setLaunchSubmitStatus(
        caughtError instanceof Error
          ? caughtError.message
          : "Launch request failed.",
      );
    } finally {
      setLaunchSubmitting(false);
    }
  };

  useEffect(() => {
    const latestJob = pipelineLaunch.data?.latest_job;
    if (!latestJob) {
      return;
    }

    const signature = `${latestJob.job_id}:${latestJob.status}`;
    if (signature === processedLaunchSignature) {
      return;
    }

    setProcessedLaunchSignature(signature);
    if (latestJob.status === "completed") {
      pipelineOverview.refresh();
    }
  }, [
    pipelineLaunch.data?.latest_job,
    pipelineOverview,
    processedLaunchSignature,
  ]);

  const snapshot = dashboard.data;
  const pipeline = pipelineOverview.data;
  const launchState = pipelineLaunch.data;
  const selectedDetail = experimentDetail.data;
  const comparisonExperimentDetail = comparisonDetail.data;
  const counts = snapshot?.counts;
  const health = snapshot?.health;
  const returnsRows = returnsSeries.data?.rows ?? [];
  const assetColumns = returnsSeries.data?.asset_columns ?? [];
  const displayedAssets = assetColumns.slice(0, 4);
  const allExperiments = pipeline?.experiments ?? [];
  const queryText = experimentQuery.trim().toLowerCase();
  const filteredExperiments = [...allExperiments]
    .filter((experiment) =>
      compactLabel(experiment.label).toLowerCase().includes(queryText),
    )
    .filter((experiment) => {
      if (experimentFilter === "positive_current") {
        return (
          (experiment.comparison_metrics.mean_policy_total_return_diff ??
            Number.NEGATIVE_INFINITY) > 0
        );
      }
      if (experimentFilter === "majority_static") {
        return hasMajorityWins(
          experiment.leaderboard_metrics.policy_beats_static_hold_rows,
          experiment.leaderboard_metrics.rows,
        );
      }
      if (experimentFilter === "majority_optimizer") {
        return hasMajorityWins(
          experiment.leaderboard_metrics.policy_beats_optimizer_rows,
          experiment.leaderboard_metrics.rows,
        );
      }
      return true;
    })
    .sort(
      (left, right) =>
        experimentSortMetric(right, experimentSort) -
        experimentSortMetric(left, experimentSort),
    );

  const executionRows = (executionSeries.data?.rows ?? []).map((row) => ({
    processedAt: String(row.processed_at ?? "-"),
    equity: numberValue(row.equity),
    orderCount: numberValue(row.order_count),
    targetCashWeight: numberValue(row.target_cash_weight),
  }));

  const leaderboardRows = filteredExperiments.map((experiment) => ({
    artifactId: experiment.artifact_id,
    label: compactLabel(experiment.label),
    comparisonScore:
      experiment.comparison_metrics.mean_policy_total_return_diff ??
      experiment.leaderboard_metrics.mean_delta_total_return_vs_static_hold ??
      0,
    policyTotalReturn:
      experiment.leaderboard_metrics.mean_policy_total_return ?? 0,
    beatsStatic: experiment.leaderboard_metrics.policy_beats_static_hold_rows,
  }));

  const detailRows = selectedDetail?.detail.rows ?? [];
  const comparisonRows = comparisonExperimentDetail?.detail.rows ?? [];
  const hardestWindows = sortWeakRows(detailRows)
    .slice(0, 10)
    .map((row) => ({
      label: buildWindowLabel(row),
      policy: numberValue(row.policy_total_return),
      staticHold: numberValue(row.static_hold_total_return),
      optimizer: numberValue(row.optimizer_total_return),
      deltaVsStatic: numberValue(row.delta_total_return_vs_static_hold),
    }));

  const weakRows =
    (
      selectedDetail?.summary.weak_rows as
        | { worst_vs_static_hold?: Array<Record<string, unknown>> }
        | undefined
    )?.worst_vs_static_hold ?? [];

  const comparisonRowMap = new Map(
    comparisonRows.map((row) => [buildWindowKey(row), row]),
  );
  const sharedWindowComparisons = detailRows
    .map((row) => {
      const comparisonRow = comparisonRowMap.get(buildWindowKey(row));
      if (!comparisonRow) {
        return null;
      }
      const selectedEdge = numberValue(row.delta_total_return_vs_static_hold);
      const comparisonEdge = numberValue(
        comparisonRow.delta_total_return_vs_static_hold,
      );
      return {
        key: buildWindowKey(row),
        label: buildWindowLabel(row),
        selectedEdge,
        comparisonEdge,
        edgeDelta: selectedEdge - comparisonEdge,
        selectedPolicy: numberValue(row.policy_total_return),
        comparisonPolicy: numberValue(comparisonRow.policy_total_return),
      };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null)
    .sort(
      (left, right) => Math.abs(right.edgeDelta) - Math.abs(left.edgeDelta),
    );

  const selectedComparisonWins = sharedWindowComparisons.filter(
    (row) => row.edgeDelta > 0,
  ).length;
  const comparisonComparisonWins = sharedWindowComparisons.filter(
    (row) => row.edgeDelta < 0,
  ).length;
  const meanComparisonEdge =
    sharedWindowComparisons.length > 0
      ? sharedWindowComparisons.reduce(
          (total, row) => total + row.edgeDelta,
          0,
        ) / sharedWindowComparisons.length
      : null;

  const comparisonMetricCards =
    selectedDetail && comparisonExperimentDetail
      ? [
          {
            label: "Policy return gap",
            value: differenceOrNull(
              selectedDetail.experiment.leaderboard_metrics
                .mean_policy_total_return,
              comparisonExperimentDetail.experiment.leaderboard_metrics
                .mean_policy_total_return,
            ),
          },
          {
            label: "Vs static gap",
            value: differenceOrNull(
              selectedDetail.experiment.leaderboard_metrics
                .mean_delta_total_return_vs_static_hold,
              comparisonExperimentDetail.experiment.leaderboard_metrics
                .mean_delta_total_return_vs_static_hold,
            ),
          },
          {
            label: "Vs optimizer gap",
            value: differenceOrNull(
              selectedDetail.experiment.leaderboard_metrics
                .mean_delta_total_return_vs_optimizer,
              comparisonExperimentDetail.experiment.leaderboard_metrics
                .mean_delta_total_return_vs_optimizer,
            ),
          },
          {
            label: "Vs current-best gap",
            value: differenceOrNull(
              selectedDetail.experiment.comparison_metrics
                .mean_policy_total_return_diff,
              comparisonExperimentDetail.experiment.comparison_metrics
                .mean_policy_total_return_diff,
            ),
          },
        ]
      : [];
  const selectedLaunchPreset =
    launchState?.presets.find(
      (preset) => preset.id === selectedLaunchPresetId,
    ) ?? null;
  const currentLaunchJob =
    launchState?.active_job ?? launchState?.latest_job ?? null;
  const launchProgress = currentLaunchJob?.progress ?? null;
  const recentLaunchEvents = launchProgress?.recent_events.slice(0, 6) ?? [];
  const checkpointHistory = launchProgress?.checkpoint_history ?? [];
  const launchCheckpointChartRows = checkpointHistory.map((checkpoint) => ({
    stepLabel:
      checkpoint.step != null && checkpoint.train_steps != null
        ? `${formatCount(checkpoint.step)}/${formatCount(checkpoint.train_steps)}`
        : formatDateTime(checkpoint.timestamp),
    validationScore: checkpoint.validation_score,
    bestValidationScore: checkpoint.best_validation_score,
    latestLoss: checkpoint.latest_loss,
    epsilon: checkpoint.epsilon,
    replaySize: checkpoint.replay_size,
    episodeResetCount: checkpoint.episode_reset_count,
  }));
  const overallLaunchProgress = clampUnitInterval(
    launchProgress?.overall_progress_fraction,
  );
  const currentFoldProgress = clampUnitInterval(
    launchProgress?.step_progress_fraction,
  );
  const launchProgressContext = [
    launchProgress?.active_split,
    launchProgress?.active_fold,
    launchProgress?.active_seed != null
      ? `seed ${formatCount(launchProgress.active_seed)}`
      : null,
  ]
    .filter((value): value is string => Boolean(value))
    .join(" · ");
  const launchLogText = currentLaunchJob?.log_tail.length
    ? currentLaunchJob.log_tail.join("\n")
    : "No launcher logs yet.";

  const lastUpdated = snapshot
    ? formatDateTime(snapshot.generated_at)
    : "Waiting for first snapshot";

  return (
    <div className="app-shell">
      <div className="glow glow--amber" aria-hidden="true" />
      <div className="glow glow--teal" aria-hidden="true" />

      <header className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Local bridge + adaptive pipeline</p>
          <h1>Wealth First Live Control Surface</h1>
          <p className="hero__lede">
            Real-time webhook intake, normalized truth monitoring, and
            experiment tracking for the adaptive Deep/Distributional RL
            pipeline.
          </p>
        </div>

        <div className="hero__meta panel-card">
          <button
            className="button button--primary"
            type="button"
            onClick={refreshAll}
          >
            Refresh everything
          </button>
          <div className="hero__status-line">
            <span
              className={`status-pill ${health?.worker_running ? "status-pill--live" : "status-pill--idle"}`}
            >
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
              <p className="eyebrow">Pipeline rail</p>
              <h2>How the live bridge links into adaptive DRL</h2>
            </div>
            <p className="muted">
              {pipeline?.approach.description ??
                "Loading adaptive pipeline metadata..."}
            </p>
          </div>
          <div className="pipeline-rail">
            {(pipeline?.pipeline_nodes ?? []).map((node) => (
              <article key={node.id} className="pipeline-node">
                <span
                  className={`pipeline-node__state pipeline-node__state--${node.status}`}
                >
                  {node.status}
                </span>
                <h3>{node.label}</h3>
                <p>{node.detail}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="panel-card panel-card--wide">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Main2 launcher</p>
              <h2>Run adaptive pipeline jobs</h2>
            </div>
            <p className="muted">
              Launch real <code>wealth_first.main2</code> runs from localhost
              and watch the board pick up the resulting artifacts.
            </p>
          </div>

          <form className="launcher-form" onSubmit={handleLaunchMain2}>
            <label>
              <span>Launch preset</span>
              <select
                value={selectedLaunchPresetId}
                onChange={(event) =>
                  setSelectedLaunchPresetId(event.target.value)
                }
              >
                {(launchState?.presets ?? []).map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Artifact label</span>
              <input
                value={launchArtifactLabel}
                onChange={(event) => setLaunchArtifactLabel(event.target.value)}
                placeholder={
                  selectedLaunchPreset?.default_artifact_label ??
                  "live_quick_probe"
                }
              />
            </label>

            <div className="launcher-form__actions">
              <button
                className="button button--secondary"
                type="submit"
                disabled={
                  launchSubmitting ||
                  launchState?.can_launch === false ||
                  !selectedLaunchPresetId
                }
              >
                {launchSubmitting ? "Launching..." : "Run main2 preset"}
              </button>
              <span className="muted">{launchSubmitStatus}</span>
            </div>
          </form>

          {selectedLaunchPreset && (
            <p className="muted launcher-description">
              {selectedLaunchPreset.description} Estimated runtime:{" "}
              {selectedLaunchPreset.estimated_runtime}.
            </p>
          )}

          <div className="artifact-grid artifact-grid--wide">
            <div>
              <span className="path-grid__label">Returns CSV</span>
              <code>{launchState?.returns_csv_path ?? "-"}</code>
            </div>
            <div>
              <span className="path-grid__label">Current-best detail</span>
              <code>{launchState?.compare_detail_csv_path ?? "-"}</code>
            </div>
            <div>
              <span className="path-grid__label">Python executable</span>
              <code>{launchState?.python_executable_path ?? "-"}</code>
            </div>
            <div>
              <span className="path-grid__label">Artifact root</span>
              <code>{launchState?.artifact_root_path ?? "-"}</code>
            </div>
          </div>

          {currentLaunchJob ? (
            <div className="comparison-shell comparison-shell--launch">
              <div className="comparison-shell__header">
                <div>
                  <h3>Latest pipeline job</h3>
                  <p className="muted">
                    {currentLaunchJob.artifact_id} · started{" "}
                    {formatDateTime(currentLaunchJob.started_at)}
                    {currentLaunchJob.completed_at
                      ? ` · finished ${formatDateTime(currentLaunchJob.completed_at)}`
                      : ""}
                  </p>
                </div>
                <span
                  className={`pipeline-node__state ${launchStatusClass(currentLaunchJob.status)}`}
                >
                  {currentLaunchJob.status}
                </span>
              </div>

              <div className="comparison-grid">
                <article className="metric-block metric-block--ink metric-block--delta">
                  <span>Preset</span>
                  <strong>{currentLaunchJob.preset_label}</strong>
                </article>
                <article className="metric-block metric-block--teal metric-block--delta">
                  <span>PID</span>
                  <strong>{formatCount(currentLaunchJob.pid)}</strong>
                </article>
                <article className="metric-block metric-block--amber metric-block--delta">
                  <span>Exit code</span>
                  <strong>{formatCount(currentLaunchJob.exit_code)}</strong>
                </article>
                <article className="metric-block metric-block--coral metric-block--delta">
                  <span>Artifacts ready</span>
                  <strong>
                    {formatCount(
                      [
                        currentLaunchJob.summary_exists,
                        currentLaunchJob.detail_exists,
                        currentLaunchJob.comparison_summary_exists,
                      ].filter(Boolean).length,
                    )}
                  </strong>
                </article>
              </div>

              <div className="launch-progress-shell">
                <div className="launch-progress__header">
                  <div>
                    <p className="eyebrow">Live DRL telemetry</p>
                    <h4>
                      {launchProgress?.stage_label ?? "Waiting for telemetry"}
                    </h4>
                  </div>
                  <p className="muted">
                    {launchProgressContext ||
                      "Structured fold and checkpoint events appear here while wealth_first.main2 trains."}
                  </p>
                </div>

                <div className="comparison-grid">
                  <article className="metric-block metric-block--teal metric-block--delta">
                    <span>Folds complete</span>
                    <strong>
                      {formatCount(launchProgress?.completed_folds)} /{" "}
                      {formatCount(launchProgress?.total_folds)}
                    </strong>
                  </article>
                  <article className="metric-block metric-block--ink metric-block--delta">
                    <span>Training step</span>
                    <strong>
                      {formatCount(launchProgress?.active_step)} /{" "}
                      {formatCount(launchProgress?.train_steps)}
                    </strong>
                  </article>
                  <article className="metric-block metric-block--amber metric-block--delta">
                    <span>Validation score</span>
                    <strong>
                      {formatPercentage(
                        launchProgress?.latest_validation_score,
                      )}
                    </strong>
                  </article>
                  <article className="metric-block metric-block--coral metric-block--delta">
                    <span>Best checkpoint</span>
                    <strong>
                      {formatPercentage(launchProgress?.best_validation_score)}
                    </strong>
                  </article>
                </div>

                <div className="comparison-grid launch-progress-grid--internals">
                  <article className="metric-block metric-block--ink metric-block--delta">
                    <span>Exploration epsilon</span>
                    <strong>
                      {formatPercentage(launchProgress?.latest_epsilon, 1)}
                    </strong>
                  </article>
                  <article className="metric-block metric-block--teal metric-block--delta">
                    <span>Replay buffer</span>
                    <strong>
                      {formatCount(launchProgress?.latest_replay_size)}
                    </strong>
                  </article>
                  <article className="metric-block metric-block--amber metric-block--delta">
                    <span>Episode resets</span>
                    <strong>
                      {formatCount(launchProgress?.latest_episode_reset_count)}
                    </strong>
                  </article>
                  <article className="metric-block metric-block--coral metric-block--delta">
                    <span>Latest loss</span>
                    <strong>
                      {formatDecimal(launchProgress?.latest_loss, 4)}
                    </strong>
                  </article>
                </div>

                <div className="launch-progress-bars">
                  <div className="launch-progress-bar">
                    <div className="launch-progress-bar__copy">
                      <span>Overall run progress</span>
                      <strong>
                        {formatPercentage(overallLaunchProgress, 1)}
                      </strong>
                    </div>
                    <div
                      className="launch-progress-bar__track"
                      aria-hidden="true"
                    >
                      <span
                        className="launch-progress-bar__fill"
                        style={{ width: `${overallLaunchProgress * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="launch-progress-bar">
                    <div className="launch-progress-bar__copy">
                      <span>Current fold progress</span>
                      <strong>
                        {formatPercentage(currentFoldProgress, 1)}
                      </strong>
                    </div>
                    <div
                      className="launch-progress-bar__track"
                      aria-hidden="true"
                    >
                      <span
                        className="launch-progress-bar__fill launch-progress-bar__fill--secondary"
                        style={{ width: `${currentFoldProgress * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="launch-progress-chart-shell">
                  <div className="section-heading section-heading--compact">
                    <div>
                      <p className="eyebrow">Checkpoint history</p>
                      <h4>Validation and loss trend</h4>
                    </div>
                    <p className="muted">
                      {launchProgress?.validation_slice_scores.length
                        ? `Latest validation slices: ${launchProgress.validation_slice_scores
                            .map((score) => formatPercentage(score, 1))
                            .join(" · ")}`
                        : "Each checkpoint captures validation score, best-so-far score, and current training loss."}
                    </p>
                  </div>
                  <div className="chart-shell chart-shell--launch-progress">
                    {launchCheckpointChartRows.length > 0 ? (
                      <Suspense
                        fallback={
                          <ChartFallback message="Loading checkpoint chart..." />
                        }
                      >
                        <LaunchProgressChart rows={launchCheckpointChartRows} />
                      </Suspense>
                    ) : (
                      <div className="empty-state empty-state--compact">
                        Waiting for checkpoint history...
                      </div>
                    )}
                  </div>
                </div>

                {launchProgress?.latest_fold_metrics && (
                  <div className="comparison-grid">
                    <article className="metric-block metric-block--ink metric-block--delta">
                      <span>Latest fold test return</span>
                      <strong>
                        {formatPercentage(
                          launchProgress.latest_fold_metrics.test_total_return,
                        )}
                      </strong>
                    </article>
                    <article className="metric-block metric-block--teal metric-block--delta">
                      <span>Latest fold validation return</span>
                      <strong>
                        {formatPercentage(
                          launchProgress.latest_fold_metrics
                            .validation_total_return,
                        )}
                      </strong>
                    </article>
                    <article className="metric-block metric-block--amber metric-block--delta">
                      <span>Avg SPY weight</span>
                      <strong>
                        {formatPercentage(
                          launchProgress.latest_fold_metrics
                            .test_average_spy_weight,
                        )}
                      </strong>
                    </article>
                    <article className="metric-block metric-block--coral metric-block--delta">
                      <span>Avg turnover</span>
                      <strong>
                        {formatPercentage(
                          launchProgress.latest_fold_metrics
                            .test_average_turnover,
                        )}
                      </strong>
                    </article>
                  </div>
                )}

                <div className="launch-progress-events">
                  {recentLaunchEvents.length > 0 ? (
                    recentLaunchEvents.map((event, index) => (
                      <article
                        key={`${String(event.timestamp ?? "event")}-${String(event.event_type ?? index)}`}
                        className="launch-progress-event"
                      >
                        <div className="launch-progress-event__header">
                          <span className="badge badge--ink">
                            {startCaseLabel(event.event_type)}
                          </span>
                          <span className="muted">
                            {formatDateTime(
                              typeof event.timestamp === "string"
                                ? event.timestamp
                                : null,
                            )}
                          </span>
                        </div>
                        <strong>{formatLaunchEventContext(event)}</strong>
                        <p className="muted">
                          {formatLaunchEventDetail(event)}
                        </p>
                      </article>
                    ))
                  ) : (
                    <div className="empty-state empty-state--compact">
                      Waiting for first structured DRL progress event...
                    </div>
                  )}
                </div>
              </div>

              <div className="artifact-grid artifact-grid--wide">
                <div>
                  <span className="path-grid__label">Output prefix</span>
                  <code>{currentLaunchJob.output_prefix}</code>
                </div>
                <div>
                  <span className="path-grid__label">Summary</span>
                  <code>{currentLaunchJob.summary_path}</code>
                </div>
                <div>
                  <span className="path-grid__label">Detail CSV</span>
                  <code>{currentLaunchJob.detail_path}</code>
                </div>
                <div>
                  <span className="path-grid__label">Launch log</span>
                  <code>{currentLaunchJob.log_path}</code>
                </div>
                <div>
                  <span className="path-grid__label">Progress log</span>
                  <code>{currentLaunchJob.progress_log_path}</code>
                </div>
              </div>

              <pre className="response-shell response-shell--short">
                {launchLogText}
              </pre>
            </div>
          ) : (
            <div className="empty-state empty-state--compact">
              {pipelineLaunch.loading
                ? "Loading launcher state..."
                : "No main2 launch has been started from the board yet."}
            </div>
          )}
        </section>

        <section className="panel-card panel-card--wide">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Bridge telemetry</p>
              <h2>Live counts</h2>
            </div>
            <p className="muted">
              Fast read on ingestion, normalization, execution, and failures.
            </p>
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
              <span className="path-grid__label">Artifact root</span>
              <code>{pipeline?.artifact_root_path ?? "-"}</code>
            </div>
          </div>
        </section>

        <section className="panel-card panel-card--chart">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Normalized stream</p>
              <h2>Cumulative return series</h2>
            </div>
            <p className="muted">
              The bridge output CSV transformed into a chartable live series.
            </p>
          </div>
          <div className="chart-shell">
            {displayedAssets.length > 0 ? (
              <Suspense
                fallback={<ChartFallback message="Loading chart module..." />}
              >
                <ReturnsSeriesChart
                  assetColumns={displayedAssets}
                  dateColumn={returnsSeries.data?.date_column ?? "date"}
                  rows={returnsRows}
                />
              </Suspense>
            ) : (
              <div className="empty-state">
                {returnsSeries.loading
                  ? "Loading return series..."
                  : "No return series available yet."}
              </div>
            )}
          </div>
        </section>

        <section className="panel-card panel-card--chart">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Execution activity</p>
              <h2>Orders and cash posture</h2>
            </div>
            <p className="muted">
              Execution volume and target cash weight over the recent handoff
              stream.
            </p>
          </div>
          <div className="chart-shell">
            {executionRows.length > 0 ? (
              <Suspense
                fallback={<ChartFallback message="Loading chart module..." />}
              >
                <ExecutionActivityChart rows={executionRows} />
              </Suspense>
            ) : (
              <div className="empty-state">
                {executionSeries.loading
                  ? "Loading execution activity..."
                  : "No execution history available yet."}
              </div>
            )}
          </div>
        </section>

        <section className="panel-card panel-card--composer">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Test intake</p>
              <h2>Webhook composer</h2>
            </div>
            <p className="muted">
              Use the browser to push a local payload through the same backend
              intake path.
            </p>
          </div>
          <form className="composer" onSubmit={handleSubmit}>
            <label>
              <span>Webhook token</span>
              <input
                value={token}
                onChange={(event) => setToken(event.target.value)}
                placeholder="Optional unless the bridge requires one"
              />
            </label>
            <label>
              <span>JSON payload</span>
              <textarea
                value={payloadText}
                onChange={(event) => setPayloadText(event.target.value)}
                spellCheck={false}
              />
            </label>
            <div className="composer__footer">
              <button
                className="button button--secondary"
                type="submit"
                disabled={submitting}
              >
                {submitting ? "Posting..." : "POST /webhook"}
              </button>
              <span className="muted">{submitStatus}</span>
            </div>
          </form>
          <pre className="response-shell">{submitResponse}</pre>
        </section>

        <section className="panel-card panel-card--stream">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Adaptive experiment board</p>
              <h2>Experiment leaderboard</h2>
            </div>
            <p className="muted">
              Filter, rank, and compare main2 distributional-RL experiments
              against the current best and each other.
            </p>
          </div>
          <div className="toolbar-grid">
            <label>
              <span>Search experiments</span>
              <input
                value={experimentQuery}
                onChange={(event) => setExperimentQuery(event.target.value)}
                placeholder="v6, longtrain, ntb02..."
              />
            </label>
            <label>
              <span>Filter</span>
              <select
                value={experimentFilter}
                onChange={(event) =>
                  setExperimentFilter(
                    event.target.value as ExperimentFilterMode,
                  )
                }
              >
                <option value="all">All experiments</option>
                <option value="positive_current">
                  Positive vs current best
                </option>
                <option value="majority_static">
                  Majority beat static hold
                </option>
                <option value="majority_optimizer">
                  Majority beat optimizer
                </option>
              </select>
            </label>
            <label>
              <span>Sort by</span>
              <select
                value={experimentSort}
                onChange={(event) =>
                  setExperimentSort(event.target.value as ExperimentSortKey)
                }
              >
                <option value="comparison">Vs current best</option>
                <option value="static">Vs static hold</option>
                <option value="optimizer">Vs optimizer</option>
                <option value="return">Mean policy return</option>
                <option value="wins">Static win rate</option>
              </select>
            </label>
          </div>
          <p className="muted toolbar-caption">
            Showing {formatCount(filteredExperiments.length)} of{" "}
            {formatCount(allExperiments.length)} discovered experiments.
          </p>
          <div className="chart-shell chart-shell--short">
            {leaderboardRows.length > 0 ? (
              <Suspense
                fallback={<ChartFallback message="Loading chart module..." />}
              >
                <ExperimentLeaderboardChart
                  recommendedExperimentId={
                    pipeline?.recommended_experiment_id ?? null
                  }
                  rows={leaderboardRows.slice(0, 10)}
                />
              </Suspense>
            ) : (
              <div className="empty-state">
                {pipelineOverview.loading
                  ? "Loading experiment board..."
                  : "No adaptive experiment artifacts were found."}
              </div>
            )}
          </div>

          <div className="selector-grid">
            <div className="selector-row">
              <label htmlFor="experimentSelect">Selected experiment</label>
              <select
                id="experimentSelect"
                value={selectedExperimentId ?? ""}
                onChange={(event) =>
                  setSelectedExperimentId(event.target.value)
                }
              >
                {allExperiments.map((experiment) => (
                  <option
                    key={experiment.artifact_id}
                    value={experiment.artifact_id}
                  >
                    {compactLabel(experiment.label)}
                  </option>
                ))}
              </select>
            </div>

            <div className="selector-row">
              <label htmlFor="comparisonExperimentSelect">
                Compare against
              </label>
              <select
                id="comparisonExperimentSelect"
                value={comparisonExperimentId ?? ""}
                onChange={(event) =>
                  setComparisonExperimentId(event.target.value || null)
                }
              >
                {allExperiments
                  .filter(
                    (experiment) =>
                      experiment.artifact_id !== selectedExperimentId,
                  )
                  .map((experiment) => (
                    <option
                      key={experiment.artifact_id}
                      value={experiment.artifact_id}
                    >
                      {compactLabel(experiment.label)}
                    </option>
                  ))}
              </select>
            </div>
          </div>

          {selectedDetail ? (
            <>
              <div className="metric-grid metric-grid--compact">
                <article className="metric-block metric-block--teal">
                  <span>Mean policy return</span>
                  <strong>
                    {formatPercentage(
                      selectedDetail.experiment.leaderboard_metrics
                        .mean_policy_total_return,
                    )}
                  </strong>
                </article>
                <article className="metric-block metric-block--amber">
                  <span>Vs static hold</span>
                  <strong>
                    {formatPercentage(
                      selectedDetail.experiment.leaderboard_metrics
                        .mean_delta_total_return_vs_static_hold,
                    )}
                  </strong>
                </article>
                <article className="metric-block metric-block--ink">
                  <span>Vs optimizer</span>
                  <strong>
                    {formatPercentage(
                      selectedDetail.experiment.leaderboard_metrics
                        .mean_delta_total_return_vs_optimizer,
                    )}
                  </strong>
                </article>
                <article className="metric-block metric-block--coral">
                  <span>Vs current best</span>
                  <strong>
                    {formatPercentage(
                      selectedDetail.experiment.comparison_metrics
                        .mean_policy_total_return_diff,
                    )}
                  </strong>
                </article>
              </div>

              <div className="comparison-shell">
                <div className="comparison-shell__header">
                  <div>
                    <h3>Artifact comparison</h3>
                    <p className="muted">
                      Stack the selected main2 run against a second artifact to
                      see which windows actually move and by how much.
                    </p>
                  </div>
                  <div className="comparison-shell__labels">
                    <span className="badge badge--teal">
                      selected {compactLabel(selectedDetail.experiment.label)}
                    </span>
                    <span className="badge badge--amber">
                      compare{" "}
                      {compactLabel(
                        comparisonExperimentDetail?.experiment.label ?? "none",
                      )}
                    </span>
                  </div>
                </div>

                {comparisonExperimentDetail ? (
                  <>
                    <div className="metric-grid metric-grid--compact">
                      {comparisonMetricCards.map((metric) => (
                        <article
                          key={metric.label}
                          className="metric-block metric-block--ink metric-block--delta"
                        >
                          <span>{metric.label}</span>
                          <strong className={deltaToneClass(metric.value)}>
                            {formatSignedPercentage(metric.value)}
                          </strong>
                        </article>
                      ))}
                    </div>

                    <div className="comparison-grid">
                      <article className="metric-block metric-block--ink metric-block--delta">
                        <span>Shared windows</span>
                        <strong>
                          {formatCount(sharedWindowComparisons.length)}
                        </strong>
                      </article>
                      <article className="metric-block metric-block--teal metric-block--delta">
                        <span>Selected wins</span>
                        <strong>{formatCount(selectedComparisonWins)}</strong>
                      </article>
                      <article className="metric-block metric-block--amber metric-block--delta">
                        <span>Comparison wins</span>
                        <strong>{formatCount(comparisonComparisonWins)}</strong>
                      </article>
                      <article className="metric-block metric-block--coral metric-block--delta">
                        <span>Mean edge delta</span>
                        <strong className={deltaToneClass(meanComparisonEdge)}>
                          {formatSignedPercentage(meanComparisonEdge)}
                        </strong>
                      </article>
                    </div>

                    <div className="weak-table-shell">
                      <table>
                        <thead>
                          <tr>
                            <th>Window</th>
                            <th>Selected vs static</th>
                            <th>Comparison vs static</th>
                            <th>Gap</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sharedWindowComparisons.slice(0, 8).map((row) => (
                            <tr key={row.key}>
                              <td>{row.label}</td>
                              <td>{formatPercentage(row.selectedEdge)}</td>
                              <td>{formatPercentage(row.comparisonEdge)}</td>
                              <td>
                                <span
                                  className={`trend-value ${deltaToneClass(row.edgeDelta)}`}
                                >
                                  {formatSignedPercentage(row.edgeDelta)}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    {comparisonDetail.loading
                      ? "Loading comparison experiment..."
                      : "Choose a second experiment to compare."}
                  </div>
                )}
              </div>

              <div className="chart-shell chart-shell--short">
                <Suspense
                  fallback={<ChartFallback message="Loading chart module..." />}
                >
                  <HardestWindowsChart rows={hardestWindows} />
                </Suspense>
              </div>

              <div className="weak-grid">
                <div>
                  <h3>Current selection</h3>
                  <p className="muted">
                    {selectedDetail.experiment.label} · updated{" "}
                    {formatDateTime(selectedDetail.experiment.modified_at)}
                  </p>
                  <p className="muted">
                    Quantile/distributional reinforcement learning in{" "}
                    <code>{selectedDetail.approach.module}</code> with
                    artifact-backed evaluation.
                  </p>
                  <div className="artifact-grid">
                    <div>
                      <span className="path-grid__label">Summary</span>
                      <code>
                        {selectedDetail.experiment.summary_path ?? "-"}
                      </code>
                    </div>
                    <div>
                      <span className="path-grid__label">Detail CSV</span>
                      <code>
                        {selectedDetail.experiment.detail_path ?? "-"}
                      </code>
                    </div>
                  </div>
                </div>
                <div className="weak-table-shell">
                  <table>
                    <thead>
                      <tr>
                        <th>Window</th>
                        <th>Vs static</th>
                        <th>Policy</th>
                        <th>Static</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortWeakRows(weakRows)
                        .slice(0, 8)
                        .map((row, index) => (
                          <tr
                            key={`${row.split}-${row.seed}-${row.fold}-${row.phase}-${index}`}
                          >
                            <td>{`${String(row.split ?? "-")}/${String(row.seed ?? "-")}/${String(row.fold ?? "-")}/${String(row.phase ?? "-")}`}</td>
                            <td>
                              {formatPercentage(
                                numberValue(
                                  row.delta_total_return_vs_static_hold,
                                ),
                              )}
                            </td>
                            <td>
                              {formatPercentage(
                                numberValue(row.policy_total_return),
                              )}
                            </td>
                            <td>
                              {formatPercentage(
                                numberValue(row.static_hold_total_return),
                              )}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state">
              {experimentDetail.loading
                ? "Loading selected experiment..."
                : "Choose an experiment to inspect."}
            </div>
          )}
        </section>

        <section className="panel-card panel-card--stream">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Recent bridge events</p>
              <h2>Live intake rail</h2>
            </div>
          </div>
          <div className="stream-list">
            {(snapshot?.recent_events ?? []).map((event, index) => (
              <article
                key={`${String(event.fingerprint ?? index)}`}
                className="stream-card"
              >
                <div className="stream-card__header">
                  <div>
                    <h3>{String(event.sleeve ?? "unknown sleeve")}</h3>
                    <p className="muted">
                      {formatDateTime(String(event.timestamp ?? ""))}
                    </p>
                  </div>
                  <span className="badge">
                    {String(event.event_type ?? "event")}
                  </span>
                </div>
                <div className="badge-row">
                  <span className="badge badge--teal">
                    symbol {String(event.symbol ?? "-")}
                  </span>
                  <span className="badge badge--amber">
                    return{" "}
                    {formatPercentage(numberValue(event.return_value), 2)}
                  </span>
                  <span className="badge">
                    {shortId(String(event.fingerprint ?? ""))}
                  </span>
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>

      {(dashboard.error ||
        pipelineOverview.error ||
        pipelineLaunch.error ||
        experimentDetail.error ||
        comparisonDetail.error) && (
        <footer className="error-strip">
          {dashboard.error ||
            pipelineOverview.error ||
            pipelineLaunch.error ||
            experimentDetail.error ||
            comparisonDetail.error}
        </footer>
      )}
    </div>
  );
}

export default App;
