import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type LeaderboardRow = {
  artifactId: string;
  label: string;
  comparisonScore: number;
};

type ExperimentLeaderboardChartProps = {
  recommendedExperimentId: string | null;
  rows: LeaderboardRow[];
};

function numberValue(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatPercentage(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export default function ExperimentLeaderboardChart({ recommendedExperimentId, rows }: ExperimentLeaderboardChartProps) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={rows} margin={{ left: 12, right: 12 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
        <XAxis dataKey="label" tickLine={false} axisLine={false} interval={0} angle={-18} textAnchor="end" height={76} />
        <YAxis tickFormatter={(value) => formatPercentage(numberValue(value), 0)} tickLine={false} axisLine={false} />
        <Tooltip formatter={(value) => formatPercentage(numberValue(value), 2)} />
        <Bar dataKey="comparisonScore" radius={[8, 8, 0, 0]}>
          {rows.map((row) => (
            <Cell key={row.artifactId} fill={row.artifactId === recommendedExperimentId ? "#4fd0c3" : "#7f9cff"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}