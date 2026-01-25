/**
 * SectionHeader Component
 *
 * Clean section header with professional typography and spacing.
 * Consistent visual hierarchy for page sections.
 */

import React from "react";

interface SectionHeaderProps {
  title: string;
  description?: string;
  className?: string;
  badge?: React.ReactNode;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  description,
  className = "mb-6",
  badge,
}) => {
  return (
    <div className={className}>
      <div className="flex items-center gap-3 mb-2">
        <h2 className="text-xl font-semibold text-[hsl(172_43%_15%)]">
          {title}
        </h2>
        {badge && <div>{badge}</div>}
      </div>
      {description && (
        <p className="text-sm text-[hsl(215_15%_50%)] leading-relaxed">
          {description}
        </p>
      )}
    </div>
  );
};

export default SectionHeader;
