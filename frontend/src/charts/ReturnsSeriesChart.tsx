import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ReturnsSeriesChartProps = {
  assetColumns: string[];
  dateColumn: string;
  rows: Array<Record<string, unknown>>;
};

const linePalette = ["#f2a541", "#4fd0c3", "#7f9cff", "#ff7b68"];
const fillPalette = ["url(#trendFillA)", "url(#trendFillB)", "url(#trendFillC)", "url(#trendFillD)"];

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

export default function ReturnsSeriesChart({ assetColumns, dateColumn, rows }: ReturnsSeriesChartProps) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <AreaChart data={rows}>
        <defs>
          <linearGradient id="trendFillA" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f2a541" stopOpacity={0.8} />
            <stop offset="95%" stopColor="#f2a541" stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="trendFillB" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#4fd0c3" stopOpacity={0.75} />
            <stop offset="95%" stopColor="#4fd0c3" stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="trendFillC" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#7f9cff" stopOpacity={0.7} />
            <stop offset="95%" stopColor="#7f9cff" stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="trendFillD" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ff7b68" stopOpacity={0.75} />
            <stop offset="95%" stopColor="#ff7b68" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
        <XAxis dataKey={dateColumn} tickLine={false} axisLine={false} minTickGap={26} />
        <YAxis tickFormatter={(value) => formatPercentage(numberValue(value), 0)} tickLine={false} axisLine={false} />
        <Tooltip formatter={(value) => formatPercentage(numberValue(value), 2)} />
        <Legend />
        {assetColumns.map((asset, index) => (
          <Area
            key={asset}
            type="monotone"
            dataKey={`cumulative_${asset}`}
            name={asset}
            stroke={linePalette[index % linePalette.length]}
            fill={fillPalette[index % fillPalette.length]}
            strokeWidth={2}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}