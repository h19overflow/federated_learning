/**
 * MetadataDisplay Component
 *
 * Purpose: Display experiment metadata in a clean, intuitive format
 * Dependencies: lucide-react for icons, shadcn/ui components
 * Role: Renders metadata key-value pairs with proper formatting and icons
 * Redesigned with Clinical Clarity theme - teal accents, soft backgrounds
 */

import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Calendar,
  Clock,
  Database,
  Layers,
  Settings,
  Hash,
  FileText,
  CheckCircle2,
  XCircle,
  Info,
  User,
  Tag,
  Activity,
  BarChart3,
  Cpu,
  HardDrive,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface MetadataDisplayProps {
  metadata: Record<string, any>;
  title?: string;
}

/**
 * Maps metadata keys to appropriate icons and labels
 */
const getMetadataIcon = (key: string): React.ReactNode => {
  const keyLower = key.toLowerCase();

  if (
    keyLower.includes("date") ||
    keyLower.includes("created") ||
    keyLower.includes("timestamp")
  ) {
    return <Calendar className="h-4 w-4" />;
  }
  if (keyLower.includes("time") || keyLower.includes("duration")) {
    return <Clock className="h-4 w-4" />;
  }
  if (keyLower.includes("epoch") || keyLower.includes("iteration")) {
    return <Hash className="h-4 w-4" />;
  }
  if (keyLower.includes("batch") || keyLower.includes("size")) {
    return <Layers className="h-4 w-4" />;
  }
  if (keyLower.includes("dataset") || keyLower.includes("data")) {
    return <Database className="h-4 w-4" />;
  }
  if (keyLower.includes("model") || keyLower.includes("architecture")) {
    return <Cpu className="h-4 w-4" />;
  }
  if (
    keyLower.includes("optimizer") ||
    keyLower.includes("learning") ||
    keyLower.includes("config")
  ) {
    return <Settings className="h-4 w-4" />;
  }
  if (
    keyLower.includes("metric") ||
    keyLower.includes("score") ||
    keyLower.includes("accuracy")
  ) {
    return <BarChart3 className="h-4 w-4" />;
  }
  if (
    keyLower.includes("client") ||
    keyLower.includes("node") ||
    keyLower.includes("device")
  ) {
    return <Activity className="h-4 w-4" />;
  }
  if (keyLower.includes("user") || keyLower.includes("author")) {
    return <User className="h-4 w-4" />;
  }
  if (
    keyLower.includes("tag") ||
    keyLower.includes("label") ||
    keyLower.includes("category")
  ) {
    return <Tag className="h-4 w-4" />;
  }
  if (
    keyLower.includes("storage") ||
    keyLower.includes("path") ||
    keyLower.includes("directory")
  ) {
    return <HardDrive className="h-4 w-4" />;
  }

  return <Info className="h-4 w-4" />;
};

/**
 * Formats a metadata key for display (converts snake_case to Title Case)
 */
const formatKey = (key: string): string => {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

/**
 * Formats different value types appropriately
 */
const formatValue = (value: any): React.ReactNode => {
  if (value === null || value === undefined) {
    return <span className="text-[hsl(215_15%_55%)] italic">Not set</span>;
  }

  if (typeof value === "boolean") {
    return value ? (
      <Badge
        variant="outline"
        className="bg-[hsl(172_40%_95%)] text-[hsl(172_63%_25%)] border-[hsl(172_40%_80%)]"
      >
        <CheckCircle2 className="h-3 w-3 mr-1" />
        True
      </Badge>
    ) : (
      <Badge
        variant="outline"
        className="bg-[hsl(0_50%_97%)] text-[hsl(0_60%_45%)] border-[hsl(0_40%_80%)]"
      >
        <XCircle className="h-3 w-3 mr-1" />
        False
      </Badge>
    );
  }

  if (typeof value === "number") {
    // Format numbers with appropriate precision
    if (Number.isInteger(value)) {
      return (
        <span className="font-mono font-semibold text-[hsl(172_43%_20%)]">
          {value.toLocaleString()}
        </span>
      );
    }
    return (
      <span className="font-mono font-semibold text-[hsl(172_43%_20%)]">
        {value.toFixed(4)}
      </span>
    );
  }

  if (typeof value === "string") {
    // Check if it's a date string
    const datePattern = /^\d{4}-\d{2}-\d{2}(T|\s)/;
    if (datePattern.test(value)) {
      try {
        const date = new Date(value);
        return (
          <span className="text-sm text-[hsl(172_43%_25%)]">
            {date.toLocaleDateString()} {date.toLocaleTimeString()}
          </span>
        );
      } catch {
        return <span className="text-sm text-[hsl(172_43%_25%)]">{value}</span>;
      }
    }

    // Check if it's a long string (truncate if needed)
    if (value.length > 100) {
      return (
        <span
          className="text-sm break-words text-[hsl(172_43%_25%)]"
          title={value}
        >
          {value.substring(0, 100)}...
        </span>
      );
    }

    return <span className="text-sm text-[hsl(172_43%_25%)]">{value}</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return (
        <span className="text-[hsl(215_15%_55%)] italic">Empty array</span>
      );
    }

    // If array contains primitives, show as badges
    if (value.every((v) => typeof v === "string" || typeof v === "number")) {
      return (
        <div className="flex flex-wrap gap-1.5">
          {value.map((item, idx) => (
            <Badge
              key={idx}
              variant="secondary"
              className="text-xs bg-[hsl(168_25%_94%)] text-[hsl(172_43%_25%)] border-0"
            >
              {String(item)}
            </Badge>
          ))}
        </div>
      );
    }

    // For complex arrays, show count
    return (
      <Badge
        variant="outline"
        className="text-xs border-[hsl(168_20%_85%)] text-[hsl(215_15%_45%)]"
      >
        {value.length} items
      </Badge>
    );
  }

  if (typeof value === "object") {
    // Render nested object
    return <NestedMetadata data={value} depth={1} />;
  }

  return (
    <span className="text-sm text-[hsl(172_43%_25%)]">{String(value)}</span>
  );
};

/**
 * Renders nested metadata objects
 */
const NestedMetadata: React.FC<{
  data: Record<string, any>;
  depth: number;
}> = ({ data, depth }) => {
  // Limit nesting depth to prevent overly deep rendering
  if (depth > 2) {
    return (
      <span className="text-[10px] text-[hsl(215_15%_60%)] italic">
        Complex object
      </span>
    );
  }

  return (
    <div
      className={cn(
        "space-y-1 w-full",
        depth > 0 && "pl-3 border-l border-[hsl(172_30%_90%)]",
      )}
    >
      {Object.entries(data).map(([key, value]) => (
        <div
          key={key}
          className="flex items-center justify-between gap-4 text-[11px]"
        >
          <span className="text-[hsl(215_15%_55%)] font-medium whitespace-nowrap">
            {formatKey(key)}
          </span>
          <span className="text-[hsl(172_43%_25%)]">
            {formatValue(value)}
          </span>
        </div>
      ))}
    </div>
  );
};

/**
 * Groups metadata into categories for better organization
 */
const categorizeMetadata = (metadata: Record<string, any>) => {
  const categories: Record<string, Record<string, any>> = {
    general: {},
    training: {},
    model: {},
    data: {},
    performance: {},
    other: {},
  };

  Object.entries(metadata).forEach(([key, value]) => {
    const keyLower = key.toLowerCase();

    if (
      keyLower.includes("epoch") ||
      keyLower.includes("batch") ||
      keyLower.includes("learning") ||
      keyLower.includes("optimizer") ||
      keyLower.includes("loss")
    ) {
      categories.training[key] = value;
    } else if (
      keyLower.includes("model") ||
      keyLower.includes("architecture") ||
      keyLower.includes("layer")
    ) {
      categories.model[key] = value;
    } else if (
      keyLower.includes("dataset") ||
      keyLower.includes("data") ||
      keyLower.includes("sample")
    ) {
      categories.data[key] = value;
    } else if (
      keyLower.includes("accuracy") ||
      keyLower.includes("precision") ||
      keyLower.includes("recall") ||
      keyLower.includes("f1") ||
      keyLower.includes("auc") ||
      keyLower.includes("metric")
    ) {
      categories.performance[key] = value;
    } else if (
      keyLower.includes("date") ||
      keyLower.includes("time") ||
      keyLower.includes("id") ||
      keyLower.includes("name") ||
      keyLower.includes("version")
    ) {
      categories.general[key] = value;
    } else {
      categories.other[key] = value;
    }
  });

  // Filter out empty categories
  return Object.entries(categories).filter(
    ([_, values]) => Object.keys(values).length > 0,
  );
};

/**
 * Main MetadataDisplay component
 */
export const MetadataDisplay: React.FC<MetadataDisplayProps> = ({
  metadata,
  title = "Experiment Metadata",
}) => {
  if (!metadata || Object.keys(metadata).length === 0) {
    return (
      <Card className="rounded-2xl border-[hsl(210_15%_92%)]">
        <CardContent className="p-8 text-center">
          <div className="p-3 rounded-2xl bg-[hsl(168_25%_95%)] w-fit mx-auto mb-4">
            <FileText className="h-10 w-10 text-[hsl(215_15%_55%)]" />
          </div>
          <p className="text-[hsl(215_15%_55%)]">No metadata available</p>
        </CardContent>
      </Card>
    );
  }

  const categorizedData = categorizeMetadata(metadata);
  const categoryLabels: Record<string, string> = {
    general: "General Information",
    training: "Training Configuration",
    model: "Model Details",
    data: "Dataset Information",
    performance: "Performance Metrics",
    other: "Additional Details",
  };

  return (
    <Card className="overflow-hidden rounded-xl border-[hsl(210_15%_92%)] shadow-sm bg-white">
      <div className="bg-gradient-to-r from-[hsl(168_25%_98%)] to-white px-5 py-3 border-b border-[hsl(168_20%_92%)]">
        <h3 className="text-sm font-bold text-[hsl(172_43%_18%)] flex items-center gap-2">
          <Settings className="h-4 w-4 text-[hsl(172_63%_35%)]" />
          {title}
        </h3>
      </div>
      <CardContent className="p-6 space-y-8">
        {categorizedData.map(([category, values]) => (
          <div key={category} className="space-y-4">
            {/* Category Header */}
            <div className="flex items-center gap-3">
              <div className="h-1.5 w-1.5 rounded-full bg-[hsl(172_63%_45%)]" />
              <h4 className="text-[11px] font-bold text-[hsl(172_63%_30%)] uppercase tracking-widest">
                {categoryLabels[category] || formatKey(category)}
              </h4>
              <div className="h-[1px] flex-1 bg-gradient-to-r from-[hsl(172_30%_90%)] to-transparent" />
            </div>

            {/* Grid of Metadata Items */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(values).map(([key, value]) => (
                <div
                  key={key}
                  className="group flex items-start gap-3 p-3 rounded-xl border border-[hsl(168_20%_92%)] bg-[hsl(168_25%_99%)]/50 hover:bg-white hover:border-[hsl(172_40%_80%)] hover:shadow-sm transition-all duration-200"
                >
                  <div className="flex-shrink-0 mt-0.5 p-2 rounded-lg bg-white border border-[hsl(168_20%_92%)] text-[hsl(172_40%_45%)] group-hover:text-[hsl(172_63%_35%)] group-hover:border-[hsl(172_40%_85%)] transition-colors shadow-sm">
                    {getMetadataIcon(key)}
                  </div>
                  <div className="flex flex-col min-w-0 flex-1">
                    <span className="text-[10px] font-bold text-[hsl(215_15%_50%)] uppercase tracking-wider mb-1">
                      {formatKey(key)}
                    </span>
                    <div className="text-sm font-semibold text-[hsl(172_43%_18%)] break-words">
                      {formatValue(value)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default MetadataDisplay;
