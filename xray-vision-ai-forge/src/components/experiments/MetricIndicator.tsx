import { memo } from "react";

interface MetricIndicatorProps {
  label: string;
  value: number;
  unit?: string;
  variant?: "primary" | "secondary" | "accent";
}

const variants = {
  primary: "from-[hsl(172_63%_22%)]/5 to-[hsl(172_63%_22%)]/0 border-[hsl(172_63%_22%)]/20",
  secondary: "from-[hsl(210_60%_40%)]/5 to-[hsl(210_60%_40%)]/0 border-[hsl(210_60%_40%)]/20",
  accent: "from-[hsl(152_60%_35%)]/5 to-[hsl(152_60%_35%)]/0 border-[hsl(152_60%_35%)]/20",
};

export const MetricIndicator = memo(({
  label,
  value,
  unit = "%",
  variant = "primary",
}: MetricIndicatorProps) => (
  <div className={`bg-gradient-to-br ${variants[variant]} rounded-lg p-2 border backdrop-blur-sm transition-all duration-300 hover:shadow-md`}>
    <p className="text-[9px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5">
      {label}
    </p>
    <div className="flex items-baseline gap-0.5">
      <p className="text-base font-semibold text-[hsl(172_43%_20%)]">
        {value.toFixed(1)}
      </p>
      <p className="text-[10px] text-[hsl(215_15%_50%)]">{unit}</p>
    </div>
  </div>
));

MetricIndicator.displayName = "MetricIndicator";
