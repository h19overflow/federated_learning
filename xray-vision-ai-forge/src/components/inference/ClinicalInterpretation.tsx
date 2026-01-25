/**
 * ClinicalInterpretation Component
 *
 * Displays AI-generated clinical interpretation in a clean, professional format.
 * Includes risk assessment, recommendations, and disclaimer.
 * Medical UI standards with clear hierarchy and minimal animations.
 */

import React from "react";
import {
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
} from "lucide-react";
import { ClinicalInterpretation as ClinicalInterpretationType } from "@/types/inference";

interface ClinicalInterpretationProps {
  interpretation: ClinicalInterpretationType;
}

export const ClinicalInterpretation: React.FC<ClinicalInterpretationProps> = ({
  interpretation,
}) => {
  const {
    summary,
    confidence_explanation,
    risk_assessment,
    recommendations,
    disclaimer,
  } = interpretation;

  // Risk level styling
  const getRiskStyle = (riskLevel: string) => {
    const level = riskLevel.toUpperCase();
    switch (level) {
      case "LOW":
        return {
          bg: "hsl(152 50% 95%)",
          border: "hsl(152 50% 80%)",
          text: "hsl(152 60% 30%)",
          icon: CheckCircle2,
          accentBg: "hsl(152 60% 42%)",
        };
      case "MODERATE":
        return {
          bg: "hsl(45 60% 95%)",
          border: "hsl(45 60% 80%)",
          text: "hsl(45 70% 35%)",
          icon: AlertCircle,
          accentBg: "hsl(45 70% 50%)",
        };
      case "HIGH":
        return {
          bg: "hsl(35 60% 95%)",
          border: "hsl(35 60% 80%)",
          text: "hsl(35 70% 35%)",
          icon: AlertTriangle,
          accentBg: "hsl(35 70% 45%)",
        };
      case "CRITICAL":
        return {
          bg: "hsl(0 60% 95%)",
          border: "hsl(0 60% 80%)",
          text: "hsl(0 70% 40%)",
          icon: ShieldAlert,
          accentBg: "hsl(0 70% 50%)",
        };
      default:
        return {
          bg: "hsl(210 15% 95%)",
          border: "hsl(210 15% 80%)",
          text: "hsl(210 15% 40%)",
          icon: AlertCircle,
          accentBg: "hsl(210 15% 50%)",
        };
    }
  };

  const riskStyle = getRiskStyle(risk_assessment.risk_level);
  const RiskIcon = riskStyle.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
          <svg
            className="w-6 h-6 text-[hsl(172_63%_28%)]"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <div>
          <h2 className="text-xl font-semibold text-[hsl(172_43%_15%)]">
            Clinical Interpretation
          </h2>
          <p className="text-xs text-[hsl(215_15%_45%)] uppercase tracking-wide">
            AI-Assisted Analysis
          </p>
        </div>
      </div>

      {/* Summary section */}
      <div className="p-6 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-sm font-semibold text-[hsl(172_43%_20%)] uppercase tracking-widest mb-3">
          Summary
        </h3>
        <p className="text-[hsl(215_15%_40%)] leading-relaxed text-sm">
          {summary}
        </p>
      </div>

      {/* Confidence explanation */}
      <div className="p-6 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-sm font-semibold text-[hsl(172_43%_20%)] uppercase tracking-widest mb-3">
          Confidence Explanation
        </h3>
        <p className="text-[hsl(215_15%_40%)] leading-relaxed text-sm">
          {confidence_explanation}
        </p>
      </div>

      {/* Risk assessment card - prominent */}
      <div
        className="p-6 rounded-2xl border-l-4 shadow-md"
        style={{
          backgroundColor: riskStyle.bg,
          borderColor: riskStyle.accentBg,
        }}
      >
        <div className="flex items-start gap-4 mb-4">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ backgroundColor: "white" }}
          >
            <RiskIcon className="w-5 h-5" style={{ color: riskStyle.text }} />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <h3
                className="text-sm font-bold uppercase tracking-wide"
                style={{ color: riskStyle.text }}
              >
                Risk Assessment
              </h3>
              <span
                className="px-3 py-1 rounded-lg text-xs font-bold"
                style={{
                  backgroundColor: "white",
                  color: riskStyle.text,
                }}
              >
                {risk_assessment.risk_level}
              </span>
            </div>
            <p className="text-sm mb-3" style={{ color: riskStyle.text }}>
              False Negative Risk:{" "}
              <span className="font-semibold">
                {risk_assessment.false_negative_risk}
              </span>
            </p>
          </div>
        </div>

        {/* Risk factors */}
        {risk_assessment.factors.length > 0 && (
          <div className="pt-4 border-t" style={{ borderColor: riskStyle.border }}>
            <h4
              className="text-xs font-semibold uppercase tracking-widest mb-3"
              style={{ color: riskStyle.text }}
            >
              Contributing Factors
            </h4>
            <ul className="space-y-2">
              {risk_assessment.factors.map((factor, index) => (
                <li
                  key={index}
                  className="flex items-start gap-2 text-sm"
                  style={{ color: riskStyle.text }}
                >
                  <span
                    className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
                    style={{ backgroundColor: riskStyle.text }}
                  />
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="p-6 rounded-2xl bg-[hsl(172_40%_95%)] border border-[hsl(172_40%_85%)] shadow-sm">
          <h3 className="text-sm font-semibold text-[hsl(172_43%_20%)] uppercase tracking-widest mb-4 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            Recommendations
          </h3>
          <ul className="space-y-3">
            {recommendations.map((recommendation, index) => (
              <li
                key={index}
                className="flex items-start gap-3 text-[hsl(215_15%_40%)] text-sm"
              >
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-[hsl(172_63%_28%)] text-white text-xs font-bold flex items-center justify-center mt-0.5">
                  {index + 1}
                </span>
                <span className="leading-relaxed">{recommendation}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Disclaimer */}
      <div className="p-4 rounded-2xl bg-amber-50 border-l-4 border-amber-500 shadow-sm">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-semibold text-amber-900 mb-1">
              Medical Disclaimer
            </h4>
            <p className="text-xs text-amber-800 leading-relaxed">
              {disclaimer}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClinicalInterpretation;
