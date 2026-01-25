/**
 * HeroSection Component
 *
 * Clean, professional hero section with badge, title, subtitle, and mode toggle.
 * Minimal animations, medical-grade design with focus on clarity and usability.
 */

import React from "react";
import { Sparkles, ImageIcon, Layers } from "lucide-react";
import { motion } from "framer-motion";

export type AnalysisMode = "single" | "batch";

interface HeroSectionProps {
  mode: AnalysisMode;
  onModeChange: (mode: AnalysisMode) => void;
  title?: string;
  subtitle?: string;
  badgeText?: string;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  mode,
  onModeChange,
  title = "Chest X-Ray Analysis",
  subtitle = "Upload a chest X-ray image to detect pneumonia using our state-of-the-art AI model.",
  badgeText = "AI-Powered Diagnostics",
}) => {
  return (
    <section className="relative py-20 px-6 bg-gradient-to-b from-[hsl(210_20%_98%)] to-[hsl(168_25%_96%)] overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-5%] w-[40%] h-[40%] rounded-full bg-[hsl(172_63%_28%)] opacity-[0.03] blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-5%] w-[40%] h-[40%] rounded-full bg-[hsl(172_63%_28%)] opacity-[0.03] blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto text-center">
        {/* Badge - Subtle, professional */}
        <div className="hero-badge inline-flex items-center gap-2 px-3 py-1.5 mb-8 rounded-full bg-white border border-[hsl(172_30%_85%)] shadow-sm hover:shadow-md transition-shadow duration-200">
          <Sparkles className="w-4 h-4 text-[hsl(172_63%_28%)]" />
          <span className="text-xs font-semibold text-[hsl(172_43%_25%)] uppercase tracking-wide">
            {badgeText}
          </span>
        </div>

        {/* Title - Clear hierarchy */}
        <h1 className="hero-title text-5xl md:text-6xl font-semibold tracking-tight text-[hsl(172_43%_15%)] mb-6 leading-tight">
          {title}
        </h1>

        {/* Subtitle - Better readability */}
        <p className="hero-subtitle text-lg text-[hsl(215_15%_45%)] max-w-2xl mx-auto mb-12 leading-relaxed">
          {subtitle}
        </p>

        {/* Mode Toggle - Enhanced UX */}
        <div className="hero-mode-toggle inline-flex items-center p-1 rounded-2xl bg-white border border-[hsl(172_30%_85%)] shadow-md relative">
          <button
            onClick={() => onModeChange("single")}
            className={`relative z-10 flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-colors duration-300 ${
              mode === "single" ? "text-white" : "text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]"
            }`}
            aria-pressed={mode === "single"}
            aria-label="Single image analysis mode"
          >
            <ImageIcon className="w-5 h-5" />
            <span className="hidden sm:inline">Single Image</span>
            <span className="sm:hidden">Single</span>
          </button>
          <button
            onClick={() => onModeChange("batch")}
            className={`relative z-10 flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-colors duration-300 ${
              mode === "batch" ? "text-white" : "text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]"
            }`}
            aria-pressed={mode === "batch"}
            aria-label="Batch analysis mode"
          >
            <Layers className="w-5 h-5" />
            <span className="hidden sm:inline">Batch Analysis</span>
            <span className="sm:hidden">Batch</span>
          </button>

          {/* Animated Background Slider */}
          <motion.div
            className="absolute top-1 bottom-1 left-1 rounded-xl bg-[hsl(172_63%_28%)] shadow-md"
            initial={false}
            animate={{
              x: mode === "single" ? 0 : "100%",
              width: mode === "single" ? "calc(50% - 4px)" : "calc(50% - 4px)",
            }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            style={{ width: "calc(50% - 4px)" }}
          />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
