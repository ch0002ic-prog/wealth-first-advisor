import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type HardestWindowRow = {
  label: string;
  policy: number;
  staticHold: number;
  optimizer: number;
};

type HardestWindowsChartProps = {
  rows: HardestWindowRow[];
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

export default function HardestWindowsChart({ rows }: HardestWindowsChartProps) {
  return (
    <ResponsiveContainer width="100%" height={290}>
      <BarChart data={rows} margin={{ left: 8, right: 8 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
        <XAxis dataKey="label" tickLine={false} axisLine={false} interval={0} angle={-22} textAnchor="end" height={92} />
        <YAxis tickFormatter={(value) => formatPercentage(numberValue(value), 0)} tickLine={false} axisLine={false} />
        <Tooltip formatter={(value) => formatPercentage(numberValue(value), 2)} />
        <Legend />
        <Bar dataKey="policy" name="Policy" fill="#4fd0c3" radius={[6, 6, 0, 0]} />
        <Bar dataKey="staticHold" name="Static hold" fill="#f2a541" radius={[6, 6, 0, 0]} />
        <Bar dataKey="optimizer" name="Optimizer" fill="#7f9cff" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}