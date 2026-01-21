import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ArrowRight, ArrowLeft } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface WelcomeGuideProps {
  onClose: () => void;
  initialGuide?: "training" | "inference";
}

type GuideType = "training" | "inference";

interface Step {
  title: string;
  description: string;
  content: React.ReactNode;
  icon: React.ReactNode;
}

/**
 * Welcome Guide component for onboarding new users
 * Redesigned to match Landing page aesthetic - Apple-inspired, glass morphism, deep teal
 * Supports both Training and Inference guides
 */
const WelcomeGuide = ({
  onClose,
  initialGuide = "training",
}: WelcomeGuideProps) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [open, setOpen] = useState(true);
  const [activeGuide, setActiveGuide] = useState<GuideType>(initialGuide);
  const [direction, setDirection] = useState(1); // 1 = forward, -1 = backward

  // Custom SVG icons matching Landing page style
  const icons = {
    welcome: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <circle
          cx="20"
          cy="20"
          r="16"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          fill="hsl(168 40% 95%)"
        />
        <path
          d="M13 20c0-3.9 3.1-7 7-7s7 3.1 7 7-3.1 7-7 7"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="20" cy="20" r="3" fill="hsl(172 63% 22%)" />
        <path
          d="M20 7v3M20 30v3M7 20h3M30 20h3"
          stroke="hsl(172 40% 70%)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    ),
    upload: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <rect
          x="7"
          y="12"
          width="26"
          height="21"
          rx="3"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          fill="hsl(168 40% 95%)"
        />
        <path d="M7 18h26" stroke="hsl(172 63% 22%)" strokeWidth="2" />
        <circle cx="12" cy="15" r="1.5" fill="hsl(172 63% 35%)" />
        <circle cx="17" cy="15" r="1.5" fill="hsl(172 63% 35%)" />
        <path
          d="M20 24v6M17 27l3-3 3 3"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    configure: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <circle
          cx="20"
          cy="20"
          r="13"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          fill="hsl(168 40% 95%)"
        />
        <path
          d="M20 10v5m0 10v5m-10-10h5m10 0h5"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="20" cy="20" r="4" fill="hsl(172 63% 22%)" />
        <circle
          cx="20"
          cy="20"
          r="7"
          stroke="hsl(172 40% 70%)"
          strokeWidth="1"
          strokeDasharray="2 2"
        />
      </svg>
    ),
    train: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <path
          d="M7 30l8-8 6 6 14-14"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
        <circle cx="33" cy="14" r="4" fill="hsl(152 60% 42%)" />
        <path
          d="M7 33h26"
          stroke="hsl(172 30% 80%)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <circle
          cx="15"
          cy="22"
          r="2.5"
          fill="hsl(168 40% 95%)"
          stroke="hsl(172 63% 22%)"
          strokeWidth="1.5"
        />
      </svg>
    ),
    analyze: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <rect
          x="5"
          y="8"
          width="13"
          height="24"
          rx="2"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          fill="hsl(168 40% 95%)"
        />
        <rect
          x="22"
          y="8"
          width="13"
          height="24"
          rx="2"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          fill="hsl(168 40% 95%)"
        />
        <path
          d="M9 15h5M9 20h5M9 25h3"
          stroke="hsl(172 50% 50%)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M26 15h5M26 20h5M26 25h3"
          stroke="hsl(172 50% 50%)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M18 17l2 2 2-2M18 23l2-2 2 2"
          stroke="hsl(152 60% 42%)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    image: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <rect
          x="6"
          y="10"
          width="28"
          height="20"
          rx="3"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          fill="hsl(168 40% 95%)"
        />
        <circle
          cx="14"
          cy="17"
          r="3"
          fill="hsl(172 40% 85%)"
          stroke="hsl(172 63% 22%)"
          strokeWidth="1.5"
        />
        <path
          d="M6 26l8-6 6 4 8-6 6 5"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    stethoscope: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <path
          d="M12 8v10a8 8 0 0016 0V8"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
        <circle cx="12" cy="6" r="2" fill="hsl(172 63% 35%)" />
        <circle cx="28" cy="6" r="2" fill="hsl(172 63% 35%)" />
        <circle
          cx="20"
          cy="28"
          r="5"
          fill="hsl(168 40% 95%)"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
        />
        <circle cx="20" cy="28" r="2" fill="hsl(152 60% 42%)" />
      </svg>
    ),
    export: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <rect
          x="8"
          y="6"
          width="24"
          height="28"
          rx="3"
          stroke="hsl(172 63% 22%)"
          strokeWidth="2"
          fill="hsl(168 40% 95%)"
        />
        <path
          d="M14 15h12M14 21h8M14 27h10"
          stroke="hsl(172 50% 50%)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M20 35v-6M16 32l4 4 4-4"
          stroke="hsl(152 60% 42%)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
  };

  const trainingSteps: Step[] = [
    {
      title: "Train Your Model",
      description: "Build pneumonia detection models with your data",
      content: (
        <div className="space-y-5">
          <p className="text-[hsl(215_15%_40%)] leading-relaxed">
            Train deep learning models to detect pneumonia from chest X-rays.
            Choose between{" "}
            <span className="font-semibold text-[hsl(172_43%_25%)]">
              Centralized
            </span>{" "}
            or privacy-preserving{" "}
            <span className="font-semibold text-[hsl(172_43%_25%)]">
              Federated Learning
            </span>
            .
          </p>

          {/* Training mode cards */}
          <div className="grid grid-cols-2 gap-3">
            {[
              {
                title: "Centralized",
                badge: "Traditional",
                badgeColor: "hsl(210 60% 40%)",
                desc: "Training on one machine",
              },
              {
                title: "Federated",
                badge: "Privacy-First",
                badgeColor: "hsl(172 63% 28%)",
                desc: "Distributed privacy-preserving",
              },
            ].map((mode, i) => (
              <div key={i} className="relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-xl blur-lg opacity-25 group-hover:opacity-40 transition-opacity" />
                <div className="relative p-4 rounded-xl bg-white/70 backdrop-blur-sm border border-[hsl(172_30%_88%)] hover:shadow-md transition-all">
                  <div
                    className="inline-flex px-2 py-0.5 rounded-full text-[10px] font-medium mb-2"
                    style={{
                      backgroundColor: `${mode.badgeColor}15`,
                      color: mode.badgeColor,
                    }}
                  >
                    {mode.badge}
                  </div>
                  <h4 className="font-semibold text-[hsl(172_43%_18%)] text-sm">
                    {mode.title}
                  </h4>
                  <p className="text-xs text-[hsl(215_15%_50%)] mt-1">
                    {mode.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      ),
      icon: icons.welcome,
    },
    {
      title: "Upload & Configure",
      description: "Prepare your dataset and set parameters",
      content: (
        <div className="space-y-5">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[1.25rem] blur-xl opacity-35 group-hover:opacity-50 transition-opacity" />
            <div className="relative bg-white/75 backdrop-blur-sm rounded-[1.25rem] p-5 border border-[hsl(172_30%_88%)]">
              <div className="space-y-4">
                {[
                  {
                    icon: icons.upload,
                    title: "Upload Dataset",
                    desc: "ZIP file with X-ray images organized by class",
                  },
                  {
                    icon: icons.configure,
                    title: "Set Parameters",
                    desc: "Learning rate, batch size, epochs, train/val split",
                  },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="w-9 h-9 rounded-lg bg-[hsl(172_40%_94%)] flex items-center justify-center flex-shrink-0 [&>svg]:w-5 [&>svg]:h-5">
                      {item.icon}
                    </div>
                    <div>
                      <span className="font-medium text-[hsl(172_43%_18%)] text-sm">
                        {item.title}
                      </span>
                      <span className="text-[hsl(215_15%_50%)] text-xs block mt-0.5">
                        {item.desc}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-[hsl(35_60%_96%)] border border-[hsl(35_50%_85%)]">
            <p className="text-sm text-[hsl(35_50%_30%)]">
              <span className="font-semibold text-[hsl(35_70%_35%)]">Tip:</span>{" "}
              Default values work great for most cases!
            </p>
          </div>
        </div>
      ),
      icon: icons.upload,
    },
    {
      title: "Train & Analyze",
      description: "Monitor progress and view results",
      content: (
        <div className="space-y-5">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[1.25rem] blur-xl opacity-35 group-hover:opacity-50 transition-opacity" />
            <div className="relative bg-white/75 backdrop-blur-sm rounded-[1.25rem] p-5 border border-[hsl(172_30%_88%)]">
              <div className="space-y-4">
                {[
                  {
                    icon: icons.train,
                    title: "Real-time Monitoring",
                    desc: "Watch loss, accuracy update live via WebSocket",
                  },
                  {
                    icon: icons.analyze,
                    title: "Comprehensive Analytics",
                    desc: "Accuracy, precision, recall, F1, AUROC, confusion matrix",
                  },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="w-9 h-9 rounded-lg bg-[hsl(172_40%_94%)] flex items-center justify-center flex-shrink-0 [&>svg]:w-5 [&>svg]:h-5">
                      {item.icon}
                    </div>
                    <div>
                      <span className="font-medium text-[hsl(172_43%_18%)] text-sm">
                        {item.title}
                      </span>
                      <span className="text-[hsl(215_15%_50%)] text-xs block mt-0.5">
                        {item.desc}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* CTA card */}
          <div className="p-4 rounded-xl bg-[hsl(172_63%_22%)] text-white relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-[hsl(172_55%_28%)] to-transparent opacity-50" />
            <div className="relative flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                <svg
                  className="w-4 h-4 text-white"
                  viewBox="0 0 16 16"
                  fill="none"
                >
                  <path
                    d="M8 3v10M3 8h10"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
              <span className="font-medium text-sm">
                Ready to train your first model!
              </span>
            </div>
          </div>
        </div>
      ),
      icon: icons.analyze,
    },
  ];

  const inferenceSteps: Step[] = [
    {
      title: "Analyze X-Rays",
      description: "Get instant AI-powered pneumonia detection",
      content: (
        <div className="space-y-5">
          <p className="text-[hsl(215_15%_40%)] leading-relaxed">
            Upload chest X-ray images and get instant predictions. Supports{" "}
            <span className="font-semibold text-[hsl(172_43%_25%)]">
              single image
            </span>{" "}
            and
            <span className="font-semibold text-[hsl(172_43%_25%)]">
              {" "}
              batch analysis
            </span>{" "}
            modes.
          </p>

          {/* Mode cards */}
          <div className="grid grid-cols-2 gap-3">
            {[
              {
                title: "Single Image",
                badge: "Quick",
                badgeColor: "hsl(172 63% 28%)",
                desc: "Analyze one X-ray instantly",
              },
              {
                title: "Batch Mode",
                badge: "Up to 500",
                badgeColor: "hsl(210 60% 40%)",
                desc: "Process multiple images at once",
              },
            ].map((mode, i) => (
              <div key={i} className="relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-xl blur-lg opacity-25 group-hover:opacity-40 transition-opacity" />
                <div className="relative p-4 rounded-xl bg-white/70 backdrop-blur-sm border border-[hsl(172_30%_88%)] hover:shadow-md transition-all">
                  <div
                    className="inline-flex px-2 py-0.5 rounded-full text-[10px] font-medium mb-2"
                    style={{
                      backgroundColor: `${mode.badgeColor}15`,
                      color: mode.badgeColor,
                    }}
                  >
                    {mode.badge}
                  </div>
                  <h4 className="font-semibold text-[hsl(172_43%_18%)] text-sm">
                    {mode.title}
                  </h4>
                  <p className="text-xs text-[hsl(215_15%_50%)] mt-1">
                    {mode.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      ),
      icon: icons.image,
    },
    {
      title: "Upload & Predict",
      description: "Simple drag-and-drop interface",
      content: (
        <div className="space-y-5">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[1.25rem] blur-xl opacity-35 group-hover:opacity-50 transition-opacity" />
            <div className="relative bg-white/75 backdrop-blur-sm rounded-[1.25rem] p-5 border border-[hsl(172_30%_88%)]">
              <div className="space-y-4">
                {[
                  {
                    icon: icons.upload,
                    title: "Drag & Drop",
                    desc: "Simply drag X-ray image(s) onto the upload zone",
                  },
                  {
                    icon: icons.stethoscope,
                    title: "Clinical Interpretation",
                    desc: "Get AI-generated clinical analysis with recommendations",
                  },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="w-9 h-9 rounded-lg bg-[hsl(172_40%_94%)] flex items-center justify-center flex-shrink-0 [&>svg]:w-5 [&>svg]:h-5">
                      {item.icon}
                    </div>
                    <div>
                      <span className="font-medium text-[hsl(172_43%_18%)] text-sm">
                        {item.title}
                      </span>
                      <span className="text-[hsl(215_15%_50%)] text-xs block mt-0.5">
                        {item.desc}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-[hsl(35_60%_96%)] border border-[hsl(35_50%_85%)]">
            <p className="text-sm text-[hsl(35_50%_30%)]">
              <span className="font-semibold text-[hsl(35_70%_35%)]">
                Supported:
              </span>{" "}
              PNG, JPEG, DICOM formats
            </p>
          </div>
        </div>
      ),
      icon: icons.upload,
    },
    {
      title: "Review & Export",
      description: "Understand results and export data",
      content: (
        <div className="space-y-5">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[1.25rem] blur-xl opacity-35 group-hover:opacity-50 transition-opacity" />
            <div className="relative bg-white/75 backdrop-blur-sm rounded-[1.25rem] p-5 border border-[hsl(172_30%_88%)]">
              <div className="space-y-4">
                {[
                  {
                    icon: icons.analyze,
                    title: "Detailed Results",
                    desc: "Prediction confidence, class probabilities, clinical notes",
                  },
                  {
                    icon: icons.export,
                    title: "Export Options",
                    desc: "Download batch results as CSV or JSON",
                  },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="w-9 h-9 rounded-lg bg-[hsl(172_40%_94%)] flex items-center justify-center flex-shrink-0 [&>svg]:w-5 [&>svg]:h-5">
                      {item.icon}
                    </div>
                    <div>
                      <span className="font-medium text-[hsl(172_43%_18%)] text-sm">
                        {item.title}
                      </span>
                      <span className="text-[hsl(215_15%_50%)] text-xs block mt-0.5">
                        {item.desc}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* CTA card */}
          <div className="p-4 rounded-xl bg-[hsl(172_63%_22%)] text-white relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-[hsl(172_55%_28%)] to-transparent opacity-50" />
            <div className="relative flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                <svg
                  className="w-4 h-4 text-white"
                  viewBox="0 0 16 16"
                  fill="none"
                >
                  <path
                    d="M8 3v10M3 8h10"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
              <span className="font-medium text-sm">
                Ready to analyze your first X-ray!
              </span>
            </div>
          </div>
        </div>
      ),
      icon: icons.export,
    },
  ];

  const steps = activeGuide === "training" ? trainingSteps : inferenceSteps;
  const currentStepData = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isFirstStep = currentStep === 0;

  const handleNext = () => {
    if (isLastStep) {
      handleClose();
    } else {
      setDirection(1);
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setDirection(-1);
      setCurrentStep(currentStep - 1);
    }
  };

  const handleStepClick = (index: number) => {
    setDirection(index > currentStep ? 1 : -1);
    setCurrentStep(index);
  };

  const handleClose = () => {
    setOpen(false);
    setTimeout(onClose, 150);
  };

  const handleGuideSwitch = (guide: GuideType) => {
    setDirection(1);
    setActiveGuide(guide);
    setCurrentStep(0);
  };

  // Motion variants
  const contentVariants = {
    enter: (dir: number) => ({
      x: dir > 0 ? 30 : -30,
      opacity: 0,
    }),
    center: {
      x: 0,
      opacity: 1,
    },
    exit: (dir: number) => ({
      x: dir > 0 ? -30 : 30,
      opacity: 0,
    }),
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && handleClose()}>
      <DialogContent className="sm:max-w-lg max-h-[88vh] overflow-y-auto p-0 bg-transparent border-none shadow-none">
        {/* Outer glow effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_90%)] to-[hsl(168_40%_85%)] rounded-[1.75rem] blur-2xl opacity-45" />

        {/* Main dialog container - glass morphism */}
        <div className="relative bg-white/95 backdrop-blur-xl rounded-[1.75rem] border border-[hsl(168_20%_90%)] shadow-2xl shadow-[hsl(172_40%_70%)]/20 overflow-hidden">
          {/* Decorative background elements */}
          <div className="absolute top-0 right-0 w-52 h-52 bg-[hsl(172_40%_85%)] rounded-full blur-[80px] opacity-25 pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-40 h-40 bg-[hsl(210_60%_90%)] rounded-full blur-[60px] opacity-20 pointer-events-none" />

          {/* Guide Type Tabs - refined */}
          <div className="relative flex border-b border-[hsl(168_20%_90%)]">
            {[
              {
                key: "training" as GuideType,
                label: "Training Guide",
                icon: icons.configure,
              },
              {
                key: "inference" as GuideType,
                label: "Inference Guide",
                icon: icons.stethoscope,
              },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => handleGuideSwitch(tab.key)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3.5 text-sm font-medium transition-all ${
                  activeGuide === tab.key
                    ? "text-[hsl(172_63%_22%)] border-b-2 border-[hsl(172_63%_22%)] bg-[hsl(172_40%_98%)]"
                    : "text-[hsl(215_15%_50%)] hover:text-[hsl(172_43%_30%)] hover:bg-[hsl(168_25%_97%)]"
                }`}
              >
                <span className="[&>svg]:w-4 [&>svg]:h-4">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>

          <div className="relative p-6">
            <DialogHeader className="mb-4">
              <div className="flex items-center gap-4">
                {/* Icon container with glow */}
                <div className="relative">
                  <div className="absolute inset-0 bg-[hsl(172_50%_80%)] rounded-xl blur-md opacity-40" />
                  <div className="relative p-3 rounded-xl bg-white shadow-md border border-[hsl(168_20%_92%)]">
                    {currentStepData.icon}
                  </div>
                </div>
                <div>
                  <DialogTitle className="text-xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                    {currentStepData.title}
                  </DialogTitle>
                  <DialogDescription className="mt-0.5 text-[hsl(215_15%_45%)] text-sm">
                    {currentStepData.description}
                  </DialogDescription>
                </div>
              </div>
            </DialogHeader>

            {/* Step content with Motion animation */}
            <div className="py-2 min-h-[280px]">
              <AnimatePresence mode="wait" custom={direction}>
                <motion.div
                  key={`${activeGuide}-${currentStep}`}
                  custom={direction}
                  variants={contentVariants}
                  initial="enter"
                  animate="center"
                  exit="exit"
                  transition={{ duration: 0.25, ease: "easeOut" }}
                >
                  {currentStepData.content}
                </motion.div>
              </AnimatePresence>
            </div>

            {/* Progress indicator - pill style */}
            <div className="flex justify-center gap-2 py-5">
              {steps.map((_, index) => (
                <button
                  key={index}
                  onClick={() => handleStepClick(index)}
                  aria-label={`Go to step ${index + 1}`}
                  className={`rounded-full transition-all duration-300 ${
                    index === currentStep
                      ? "w-8 h-2 bg-[hsl(172_63%_22%)] shadow-md shadow-[hsl(172_63%_22%)]/30"
                      : index < currentStep
                        ? "w-2 h-2 bg-[hsl(152_60%_42%)]"
                        : "w-2 h-2 bg-[hsl(210_15%_85%)] hover:bg-[hsl(210_15%_75%)]"
                  }`}
                />
              ))}
            </div>

            <DialogFooter className="flex justify-between items-center sm:justify-between gap-3 pt-1">
              <div>
                {!isFirstStep && (
                  <Button
                    variant="ghost"
                    onClick={handlePrevious}
                    className="rounded-xl px-4 text-[hsl(172_43%_30%)] hover:bg-[hsl(168_25%_94%)] transition-all"
                  >
                    <ArrowLeft className="mr-1.5 h-4 w-4" />
                    Back
                  </Button>
                )}
              </div>
              <div className="flex gap-2">
                {!isLastStep && (
                  <Button
                    variant="ghost"
                    onClick={handleClose}
                    className="rounded-xl px-4 text-[hsl(215_15%_50%)] hover:bg-[hsl(168_25%_94%)] hover:text-[hsl(172_43%_25%)] transition-all"
                  >
                    Skip
                  </Button>
                )}
                <Button
                  onClick={handleNext}
                  className="rounded-xl px-5 py-4 bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white shadow-lg shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/35 hover:-translate-y-0.5"
                >
                  {isLastStep ? (
                    "Get Started"
                  ) : (
                    <>
                      Next <ArrowRight className="ml-1.5 h-4 w-4" />
                    </>
                  )}
                </Button>
              </div>
            </DialogFooter>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WelcomeGuide;
