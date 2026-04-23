import {
  Bar,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ExecutionRow = {
  processedAt: string;
  equity: number;
  orderCount: number;
  targetCashWeight: number;
};

type ExecutionActivityChartProps = {
  rows: ExecutionRow[];
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

export default function ExecutionActivityChart({ rows }: ExecutionActivityChartProps) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={rows}>
        <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
        <XAxis dataKey="processedAt" tickFormatter={(value) => String(value).slice(11, 19)} tickLine={false} axisLine={false} minTickGap={20} />
        <YAxis yAxisId="left" tickLine={false} axisLine={false} allowDecimals={false} />
        <YAxis yAxisId="right" orientation="right" tickFormatter={(value) => formatPercentage(numberValue(value), 0)} tickLine={false} axisLine={false} />
        <Tooltip formatter={(value, name) => (name === "Target cash" ? formatPercentage(numberValue(value), 2) : String(value))} />
        <Legend />
        <Bar yAxisId="left" dataKey="orderCount" name="Order count" fill="#f2a541" radius={[8, 8, 0, 0]} />
        <Line yAxisId="right" type="monotone" dataKey="targetCashWeight" name="Target cash" stroke="#4fd0c3" strokeWidth={3} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}