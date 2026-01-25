/**
 * EmptyState Component
 *
 * Clean empty state display with icon, heading, and description.
 * Minimal design for clear visual hierarchy and user guidance.
 */

import React from "react";
import { LucideIcon } from "lucide-react";

interface EmptyStateProps {
  icon?: LucideIcon | React.ReactNode;
  title: string;
  description: string;
  minHeight?: string;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  minHeight = "min-h-[400px]",
  action,
}) => {
  // Default upload icon if none provided
  const DefaultIcon = () => (
    <svg
      className="w-8 h-8 text-[hsl(172_63%_28%)]"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );

  const renderIcon = () => {
    if (!icon) return <DefaultIcon />;
    if (React.isValidElement(icon)) return icon;
    const IconComponent = icon as LucideIcon;
    return <IconComponent className="w-8 h-8 text-[hsl(172_63%_28%)]" />;
  };

  return (
    <div className={`flex flex-col items-center justify-center ${minHeight} space-y-5 px-4`}>
      {/* Icon - subtle background */}
      <div className="w-16 h-16 rounded-xl bg-[hsl(172_30%_92%)] flex items-center justify-center">
        {renderIcon()}
      </div>

      {/* Text content */}
      <div className="text-center space-y-2 max-w-sm">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
          {title}
        </h3>
        <p className="text-sm text-[hsl(215_15%_50%)] leading-relaxed">
          {description}
        </p>
      </div>

      {/* Optional action */}
      {action && (
        <div className="mt-4">
          {action}
        </div>
      )}
    </div>
  );
};

export default EmptyState;
