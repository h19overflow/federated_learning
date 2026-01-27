import { CheckCircle2, Lock, Shield, Activity } from "lucide-react";
import { ReactNode } from "react";

export interface Feature {
  icon: ReactNode;
  title: string;
  description: string;
}

export interface Step {
  step: string;
  title: string;
  description: string;
  icon: ReactNode;
  color: string;
}

export interface ComparisonRow {
  feature: string;
  centralized: { text: string; status: "success" | "warning" };
  federated: { text: string; status: "success" | "warning" };
}

export const features: Feature[] = [
  {
    icon: (
      <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
        <circle cx="20" cy="20" r="18" stroke="hsl(172 63% 22%)" strokeWidth="2" fill="hsl(168 40% 95%)" />
        <path d="M12 20c0-4.4 3.6-8 8-8s8 3.6 8 8-3.6 8-8 8" stroke="hsl(172 63% 22%)" strokeWidth="2" strokeLinecap="round" />
        <circle cx="20" cy="20" r="3" fill="hsl(172 63% 22%)" />
      </svg>
    ),
    title: "Advanced AI Models",
    description: "Pre-trained ResNet50 V2 architecture, fine-tuned for medical imaging with exceptional accuracy.",
  },
  {
    icon: <Shield className="w-10 h-10 text-[hsl(172_63%_22%)]" />,
    title: "Privacy-Preserving",
    description: "Federated learning ensures patient data remains private and secure, never leaving local devices.",
  },
  {
    icon: <Activity className="w-10 h-10 text-[hsl(172_63%_22%)]" />,
    title: "Real-Time Monitoring",
    description: "Watch training metrics and performance live with elegant, interactive visualizations.",
  },
];

export const centralizedBenefits = [
  "Faster training time",
  "Simpler implementation",
  "Easier debugging",
  "Ideal for small datasets",
];

export const centralizedConsiderations = [
  "Requires data centralization",
  "Privacy concerns with sensitive data",
  "Single point of failure",
];

export const federatedBenefits = [
  "Data privacy preserved",
  "HIPAA/GDPR compliant",
  "Distributed computation",
  "Highly scalable",
];

export const federatedConsiderations = [
  "Longer training time",
  "More complex setup",
  "Network overhead",
];

export const comparisonData: ComparisonRow[] = [
  { feature: "Data Privacy", centralized: { text: "Data must be shared", status: "warning" }, federated: { text: "Data stays local", status: "success" } },
  { feature: "Training Speed", centralized: { text: "Fast", status: "success" }, federated: { text: "Slower", status: "warning" } },
  { feature: "Setup Complexity", centralized: { text: "Simple", status: "success" }, federated: { text: "Complex", status: "warning" } },
  { feature: "Compliance", centralized: { text: "Requires safeguards", status: "warning" }, federated: { text: "Built-in privacy", status: "success" } },
  { feature: "Model Quality", centralized: { text: "Excellent", status: "success" }, federated: { text: "Comparable", status: "success" } },
  { feature: "Scalability", centralized: { text: "Limited by server", status: "warning" }, federated: { text: "Highly scalable", status: "success" } },
];

export const howItWorksSteps: Step[] = [
  {
    step: "01",
    title: "Upload Dataset",
    description: "Upload chest X-ray images with NORMAL/PNEUMONIA folder structure.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
        <rect x="5" y="10" width="30" height="25" rx="4" stroke="currentColor" strokeWidth="2" />
        <path d="M5 17h30" stroke="currentColor" strokeWidth="2" />
        <circle cx="11" cy="13.5" r="1.5" fill="currentColor" />
        <circle cx="17" cy="13.5" r="1.5" fill="currentColor" />
        <path d="M15 27l5-5 4 4 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    color: "hsl(172 63% 35%)",
  },
  {
    step: "02",
    title: "Configure Training",
    description: "Set hyperparameters and choose Centralized or Federated mode.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
        <circle cx="20" cy="20" r="14" stroke="currentColor" strokeWidth="2" />
        <path d="M20 10v5m0 10v5m-10-10h5m10 0h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        <circle cx="20" cy="20" r="4" fill="currentColor" />
      </svg>
    ),
    color: "hsl(200 70% 45%)",
  },
  {
    step: "03",
    title: "Train Model",
    description: "PyTorch Lightning training with real-time metrics via WebSocket.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
        <path d="M5 30l8-8 6 6 16-16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="33" cy="14" r="4" fill="currentColor" />
        <path d="M5 35h30" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.3" />
      </svg>
    ),
    color: "hsl(152 60% 40%)",
  },
  {
    step: "04",
    title: "Compare Results",
    description: "Analyze Centralized vs Federated performance with detailed metrics.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
        <rect x="5" y="8" width="13" height="24" rx="2" stroke="currentColor" strokeWidth="2" />
        <rect x="22" y="8" width="13" height="24" rx="2" stroke="currentColor" strokeWidth="2" />
        <path d="M9 14h5M9 19h5M9 24h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M26 14h5M26 19h5M26 24h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M18 16l2 2 2-2M18 24l2-2 2 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    color: "hsl(260 60% 55%)",
  },
];
