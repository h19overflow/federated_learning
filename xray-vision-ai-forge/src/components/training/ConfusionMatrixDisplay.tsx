import React from "react";
import { Grid3X3, HelpCircle, ChevronDown } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface ConfusionMatrixDisplayProps {
  matrix: number[][];
  title?: string;
}

/**
 * ConfusionMatrixDisplay Component
 * Displays confusion matrix with enhanced visual hierarchy
 */
const ConfusionMatrixDisplay = ({
  matrix,
  title,
}: ConfusionMatrixDisplayProps) => {
  const [isOpen, setIsOpen] = React.useState(false);

  // Calculate total and percentages for better context
  const total = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1];

  // Safe percentage calculation to prevent NaN
  const calculatePercent = (value: number): string => {
    if (!total || total === 0 || isNaN(value) || value === undefined) {
      return "0.0";
    }
    return ((value / total) * 100).toFixed(1);
  };

  const tnPercent = calculatePercent(matrix[0][0]);
  const fpPercent = calculatePercent(matrix[0][1]);
  const fnPercent = calculatePercent(matrix[1][0]);
  const tpPercent = calculatePercent(matrix[1][1]);

  return (
    <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(172_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
      {title && (
        <h4 className="text-sm font-semibold text-[hsl(172_43%_20%)] mb-5 flex items-center gap-2">
          <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)]">
            <Grid3X3 className="h-4 w-4 text-[hsl(172_63%_35%)]" />
          </div>
          {title}
        </h4>
      )}

      {/* Matrix Grid with enhanced styling */}
      <div className="grid grid-cols-2 gap-3 max-w-xs mx-auto mb-5">
        {/* True Negative - Success case */}
        <div
          className="bg-gradient-to-br from-[hsl(172_50%_96%)] to-[hsl(168_40%_94%)] p-4 rounded-xl border border-[hsl(172_40%_88%)] shadow-sm hover:shadow-md transition-all duration-300 group"
          style={{
            animation: "slideInUp 0.4s ease-out forwards",
            animationDelay: "0s",
            opacity: 0,
          }}
        >
          <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-wide mb-2">
            True Negative
          </p>
          <p className="text-3xl font-bold text-[hsl(172_63%_28%)] mb-1">
            {matrix[0][0]}
          </p>
          <p className="text-xs text-[hsl(215_15%_50%)]">
            {tnPercent}% of total
          </p>
        </div>

        {/* False Positive - Warning case */}
        <div
          className="bg-gradient-to-br from-[hsl(35_70%_96%)] to-[hsl(35_60%_94%)] p-4 rounded-xl border border-[hsl(35_60%_88%)] shadow-sm hover:shadow-md transition-all duration-300 group"
          style={{
            animation: "slideInUp 0.4s ease-out forwards",
            animationDelay: "0.08s",
            opacity: 0,
          }}
        >
          <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-wide mb-2">
            False Positive
          </p>
          <p className="text-3xl font-bold text-[hsl(35_65%_45%)] mb-1">
            {matrix[0][1]}
          </p>
          <p className="text-xs text-[hsl(215_15%_50%)]">
            {fpPercent}% of total
          </p>
        </div>

        {/* False Negative - Critical case */}
        <div
          className="bg-gradient-to-br from-[hsl(0_60%_96%)] to-[hsl(0_50%_94%)] p-4 rounded-xl border border-[hsl(0_60%_88%)] shadow-sm hover:shadow-md transition-all duration-300 group"
          style={{
            animation: "slideInUp 0.4s ease-out forwards",
            animationDelay: "0.16s",
            opacity: 0,
          }}
        >
          <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-wide mb-2">
            False Negative
          </p>
          <p className="text-3xl font-bold text-[hsl(0_72%_45%)] mb-1">
            {matrix[1][0]}
          </p>
          <p className="text-xs text-[hsl(215_15%_50%)]">
            {fnPercent}% of total
          </p>
        </div>

        {/* True Positive - Success case */}
        <div
          className="bg-gradient-to-br from-[hsl(152_60%_96%)] to-[hsl(152_50%_94%)] p-4 rounded-xl border border-[hsl(152_50%_88%)] shadow-sm hover:shadow-md transition-all duration-300 group"
          style={{
            animation: "slideInUp 0.4s ease-out forwards",
            animationDelay: "0.24s",
            opacity: 0,
          }}
        >
          <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-wide mb-2">
            True Positive
          </p>
          <p className="text-3xl font-bold text-[hsl(152_60%_40%)] mb-1">
            {matrix[1][1]}
          </p>
          <p className="text-xs text-[hsl(215_15%_50%)]">
            {tpPercent}% of total
          </p>
        </div>
      </div>

      {/* Educational Collapsible */}
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <button className="flex items-center gap-2 text-sm text-[hsl(172_63%_30%)] hover:text-[hsl(172_63%_25%)] transition-colors w-full justify-center py-2 rounded-lg hover:bg-[hsl(172_40%_94%)]">
            <HelpCircle className="h-4 w-4" />
            <span>Understanding the Confusion Matrix</span>
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
            />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3">
          <div className="bg-white rounded-lg p-4 border border-[hsl(168_20%_90%)] space-y-3">
            <p className="text-sm text-[hsl(215_15%_40%)]">
              A confusion matrix shows how well the model classifies X-rays as{" "}
              <strong>Normal</strong> or <strong>Pneumonia</strong>.
            </p>

            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-3 h-3 rounded-sm bg-[hsl(172_50%_85%)] mt-1 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-[hsl(172_43%_20%)]">
                    True Negative (TN)
                  </p>
                  <p className="text-xs text-[hsl(215_15%_50%)]">
                    Correctly identified as Normal (no pneumonia)
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <div className="w-3 h-3 rounded-sm bg-[hsl(172_50%_85%)] mt-1 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-[hsl(172_43%_20%)]">
                    True Positive (TP)
                  </p>
                  <p className="text-xs text-[hsl(215_15%_50%)]">
                    Correctly identified as Pneumonia
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <div className="w-3 h-3 rounded-sm bg-[hsl(0_60%_90%)] mt-1 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-[hsl(0_60%_40%)]">
                    False Positive (FP)
                  </p>
                  <p className="text-xs text-[hsl(215_15%_50%)]">
                    Incorrectly flagged as Pneumonia (false alarm)
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <div className="w-3 h-3 rounded-sm bg-[hsl(0_60%_90%)] mt-1 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-[hsl(0_60%_40%)]">
                    False Negative (FN)
                  </p>
                  <p className="text-xs text-[hsl(215_15%_50%)]">
                    Missed Pneumonia case (most critical error)
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-[hsl(168_20%_92%)]">
              <p className="text-xs text-[hsl(215_15%_45%)] italic">
                💡 In medical diagnosis, minimizing False Negatives is
                critical—missing a pneumonia case could delay treatment.
              </p>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

export default ConfusionMatrixDisplay;
