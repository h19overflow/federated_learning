import React from "react";
import { Sparkles, Activity, HelpCircle } from "lucide-react";

// Metric explanations for educational tooltips
export const metricExplanations: Record<
  string,
  { description: string; relevance: string }
> = {
  "F1-Score": {
    description:
      "The harmonic mean of precision and recall (2 × (precision × recall) / (precision + recall)). It balances both false positives and false negatives into a single metric.",
    relevance:
      "For pneumonia detection, F1-Score ensures the model doesn't sacrifice recall (catching pneumonia cases) for precision (avoiding false alarms) or vice versa. A high F1-Score (>0.85) indicates reliable clinical performance with balanced error rates.",
  },
  "F1 Score": {
    description:
      "The harmonic mean of precision and recall (2 × (precision × recall) / (precision + recall)). It balances both false positives and false negatives into a single metric.",
    relevance:
      "For pneumonia detection, F1 Score ensures the model doesn't sacrifice recall (catching pneumonia cases) for precision (avoiding false alarms) or vice versa. A high F1 Score (>0.85) indicates reliable clinical performance with balanced error rates.",
  },
  AUC: {
    description:
      "Area Under the ROC Curve - measures the model's ability to distinguish between pneumonia and normal X-rays across all possible classification thresholds (0.0 = worst, 1.0 = perfect).",
    relevance:
      "In medical imaging, AUC (>0.90 is excellent, >0.95 is outstanding) shows how well the model separates normal X-rays from pneumonia cases. Unlike accuracy, it's robust to class imbalance, making it the gold standard for clinical validation and FDA approval.",
  },
  "AUC-ROC": {
    description:
      "Area Under the ROC Curve - measures the model's ability to distinguish between pneumonia and normal X-rays across all possible classification thresholds (0.0 = worst, 1.0 = perfect).",
    relevance:
      "In medical imaging, AUC-ROC (>0.90 is excellent, >0.95 is outstanding) shows how well the model separates normal X-rays from pneumonia cases. Unlike accuracy, it's robust to class imbalance, making it the gold standard for clinical validation and FDA approval.",
  },
  AUROC: {
    description:
      "Area Under the ROC Curve - measures the model's ability to distinguish between pneumonia and normal X-rays across all possible classification thresholds (0.0 = worst, 1.0 = perfect).",
    relevance:
      "In medical imaging, AUROC (>0.90 is excellent, >0.95 is outstanding) shows how well the model separates normal X-rays from pneumonia cases. Unlike accuracy, it's robust to class imbalance, making it the gold standard for clinical validation and FDA approval.",
  },
  Precision: {
    description:
      "Proportion of positive predictions that are actually correct (TP / (TP + FP)).",
    relevance:
      "High precision means fewer false alarms - when the model flags pneumonia, it's usually right. This reduces unnecessary follow-up tests and patient anxiety.",
  },
  Recall: {
    description:
      "Proportion of actual positive cases that are correctly identified (TP / (TP + FN)).",
    relevance:
      "High recall is critical in medical diagnosis - it means the model catches most pneumonia cases. Missing a pneumonia case (low recall) could delay life-saving treatment.",
  },
  Accuracy: {
    description:
      "Overall proportion of correct predictions ((TP + TN) / Total).",
    relevance:
      "While accuracy gives a general sense of performance, it can be misleading with imbalanced datasets. Use F1 Score and AUC-ROC for better clinical assessment.",
  },
};

interface MetricCardProps {
  name: string;
  value: number;
  index?: number;
  total?: number;
}

export const MetricCard = ({
  name,
  value,
  index = 0,
  total = 5,
}: MetricCardProps) => {
  const [showTooltip, setShowTooltip] = React.useState(false);
  const explanation = metricExplanations[name];

  // Determine tooltip position to prevent overflow
  const isFirst = index === 0;
  const isLast = index === total - 1;
  const tooltipPositionClass = isFirst
    ? "left-0"
    : isLast
      ? "right-0"
      : "left-1/2 -translate-x-1/2";

  const arrowPositionClass = isFirst
    ? "left-8"
    : isLast
      ? "right-8"
      : "left-1/2 -translate-x-1/2";

  // Gradient backgrounds for visual distinction
  const gradientMap: Record<string, string> = {
    "Accuracy": "from-[hsl(172_50%_96%)] to-[hsl(168_35%_93%)]",
    "Precision": "from-[hsl(168_40%_96%)] to-[hsl(168_35%_93%)]",
    "Recall": "from-[hsl(35_70%_96%)] to-[hsl(35_60%_93%)]",
    "F1-Score": "from-[hsl(172_45%_96%)] to-[hsl(172_40%_93%)]",
    "F1 Score": "from-[hsl(172_45%_96%)] to-[hsl(172_40%_93%)]",
    "AUC": "from-[hsl(210_60%_96%)] to-[hsl(210_50%_93%)]",
    "AUC-ROC": "from-[hsl(210_60%_96%)] to-[hsl(210_50%_93%)]",
    "AUROC": "from-[hsl(210_60%_96%)] to-[hsl(210_50%_93%)]",
  };

  const bgGradient = gradientMap[name] || "from-[hsl(172_50%_96%)] to-[hsl(168_35%_93%)]";

   return (
     <div
       className={`bg-gradient-to-br ${bgGradient} p-6 rounded-2xl border border-[hsl(210_15%_88%)] shadow-sm hover:shadow-lg transition-all duration-300 relative group`}
       style={{
         animation: `slideInUp 0.5s ease-out forwards`,
         animationDelay: `${index * 0.08}s`,
         opacity: 0,
       }}
       onMouseEnter={() => explanation && setShowTooltip(true)}
       onMouseLeave={() => setShowTooltip(false)}
     >
      {/* Subtle accent line on top */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-[hsl(172_63%_28%)] via-[hsl(210_60%_50%)] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-widest">
          {name}
        </p>
        {explanation && (
          <HelpCircle className="h-4 w-4 text-[hsl(172_63%_35%)] cursor-help opacity-60 group-hover:opacity-100 transition-opacity" />
        )}
      </div>

      {/* Metric value with refined typography */}
      <div className="mb-2">
        <p className="text-4xl font-bold text-[hsl(172_63%_22%)] tracking-tight">
          {(value * 100).toFixed(1)}<span className="text-lg font-semibold text-[hsl(172_63%_35%)] ml-1">%</span>
        </p>
      </div>

      {/* Subtle progress indicator */}
      <div className="h-1.5 bg-[hsl(210_15%_90%)] rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-[hsl(172_63%_28%)] to-[hsl(210_60%_50%)] rounded-full transition-all duration-700 ease-out"
          style={{ width: `${value * 100}%` }}
        />
      </div>

      {/* Educational Tooltip - positioned to avoid overflow */}
      {explanation && showTooltip && (
        <div
          className={`absolute z-50 w-72 p-4 bg-white border-2 border-[hsl(172_40%_85%)] rounded-xl shadow-2xl bottom-full mb-3 ${tooltipPositionClass} pointer-events-none`}
          style={{ animation: "fadeIn 0.2s ease-out" }}
        >
          <div className="space-y-3">
            <div>
              <p className="text-xs font-bold text-[hsl(172_63%_28%)] mb-1.5 flex items-center gap-1">
                <Sparkles className="h-3 w-3" />
                What is {name}?
              </p>
              <p className="text-xs text-[hsl(215_15%_40%)] leading-relaxed">
                {explanation.description}
              </p>
            </div>
            <div className="pt-2 border-t border-[hsl(210_15%_92%)]">
              <p className="text-xs font-bold text-[hsl(172_63%_28%)] mb-1.5 flex items-center gap-1">
                <Activity className="h-3 w-3" />
                Clinical Relevance
              </p>
              <p className="text-xs text-[hsl(215_15%_40%)] leading-relaxed">
                {explanation.relevance}
              </p>
            </div>
          </div>
           {/* Tooltip arrow pointing down */}
           <div
             className={`absolute bottom-full ${arrowPositionClass} border-8 border-transparent border-b-white`}
             style={{ marginBottom: "-2px" }}
           ></div>
           <div
             className={`absolute bottom-full ${arrowPositionClass} border-[9px] border-transparent border-b-[hsl(172_40%_85%)]`}
           ></div>
        </div>
      )}
    </div>
  );
};
