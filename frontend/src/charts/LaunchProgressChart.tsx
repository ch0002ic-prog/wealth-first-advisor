import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type LaunchProgressCheckpointRow = {
  stepLabel: string;
  validationScore: number | null;
  bestValidationScore: number | null;
  latestLoss: number | null;
  epsilon: number | null;
  replaySize: number | null;
  episodeResetCount: number | null;
};

type LaunchProgressChartProps = {
  rows: LaunchProgressCheckpointRow[];
};

function numberValue(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  return null;
}

function formatPercentage(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function formatDecimal(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function formatCount(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  return new Intl.NumberFormat().format(value);
}

export default function LaunchProgressChart({ rows }: LaunchProgressChartProps) {
  const showLossAxis = rows.some((row) => row.latestLoss != null && Number.isFinite(row.latestLoss));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={rows} margin={{ left: 8, right: 8, top: 4 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
        <XAxis dataKey="stepLabel" tickLine={false} axisLine={false} />
        <YAxis
          yAxisId="score"
          tickFormatter={(value) => formatPercentage(numberValue(value), 0)}
          tickLine={false}
          axisLine={false}
        />
        {showLossAxis && <YAxis yAxisId="loss" orientation="right" tickLine={false} axisLine={false} />}
        <Tooltip
          formatter={(value, name, item) => {
            if (name === "Validation" || name === "Best") {
              return formatPercentage(numberValue(value), 2);
            }
            if (name === "Loss") {
              return formatDecimal(numberValue(value), 4);
            }

            const payload = item.payload as LaunchProgressCheckpointRow;
            return [
              `${formatDecimal(numberValue(value), 4)} | eps ${formatPercentage(payload.epsilon, 1)} | replay ${formatCount(payload.replaySize)} | resets ${formatCount(payload.episodeResetCount)}`,
              name,
            ];
          }}
        />
        <Legend />
        <Line
          yAxisId="score"
          type="monotone"
          dataKey="validationScore"
          name="Validation"
          stroke="#76c6d8"
          strokeWidth={2.5}
          dot={{ r: 3 }}
          activeDot={{ r: 5 }}
          connectNulls
        />
        <Line
          yAxisId="score"
          type="monotone"
          dataKey="bestValidationScore"
          name="Best"
          stroke="#f2b15a"
          strokeWidth={2.5}
          dot={{ r: 3 }}
          activeDot={{ r: 5 }}
          connectNulls
        />
        {showLossAxis && (
          <Line
            yAxisId="loss"
            type="monotone"
            dataKey="latestLoss"
            name="Loss"
            stroke="#ef7b68"
            strokeWidth={2}
            dot={{ r: 2.5 }}
            activeDot={{ r: 4 }}
            connectNulls
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}