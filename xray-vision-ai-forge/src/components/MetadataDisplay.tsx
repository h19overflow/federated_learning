/**
 * MetadataDisplay Component
 *
 * Purpose: Display experiment metadata in a clean, intuitive format
 * Dependencies: lucide-react for icons, shadcn/ui components
 * Role: Renders metadata key-value pairs with proper formatting and icons
 */

import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
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
  HardDrive
} from 'lucide-react';

interface MetadataDisplayProps {
  metadata: Record<string, any>;
  title?: string;
}

/**
 * Maps metadata keys to appropriate icons and labels
 */
const getMetadataIcon = (key: string): React.ReactNode => {
  const keyLower = key.toLowerCase();

  if (keyLower.includes('date') || keyLower.includes('created') || keyLower.includes('timestamp')) {
    return <Calendar className="h-4 w-4" />;
  }
  if (keyLower.includes('time') || keyLower.includes('duration')) {
    return <Clock className="h-4 w-4" />;
  }
  if (keyLower.includes('epoch') || keyLower.includes('iteration')) {
    return <Hash className="h-4 w-4" />;
  }
  if (keyLower.includes('batch') || keyLower.includes('size')) {
    return <Layers className="h-4 w-4" />;
  }
  if (keyLower.includes('dataset') || keyLower.includes('data')) {
    return <Database className="h-4 w-4" />;
  }
  if (keyLower.includes('model') || keyLower.includes('architecture')) {
    return <Cpu className="h-4 w-4" />;
  }
  if (keyLower.includes('optimizer') || keyLower.includes('learning') || keyLower.includes('config')) {
    return <Settings className="h-4 w-4" />;
  }
  if (keyLower.includes('metric') || keyLower.includes('score') || keyLower.includes('accuracy')) {
    return <BarChart3 className="h-4 w-4" />;
  }
  if (keyLower.includes('client') || keyLower.includes('node') || keyLower.includes('device')) {
    return <Activity className="h-4 w-4" />;
  }
  if (keyLower.includes('user') || keyLower.includes('author')) {
    return <User className="h-4 w-4" />;
  }
  if (keyLower.includes('tag') || keyLower.includes('label') || keyLower.includes('category')) {
    return <Tag className="h-4 w-4" />;
  }
  if (keyLower.includes('storage') || keyLower.includes('path') || keyLower.includes('directory')) {
    return <HardDrive className="h-4 w-4" />;
  }

  return <Info className="h-4 w-4" />;
};

/**
 * Formats a metadata key for display (converts snake_case to Title Case)
 */
const formatKey = (key: string): string => {
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

/**
 * Formats different value types appropriately
 */
const formatValue = (value: any): React.ReactNode => {
  if (value === null || value === undefined) {
    return <span className="text-muted-foreground italic">Not set</span>;
  }

  if (typeof value === 'boolean') {
    return value ? (
      <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
        <CheckCircle2 className="h-3 w-3 mr-1" />
        True
      </Badge>
    ) : (
      <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
        <XCircle className="h-3 w-3 mr-1" />
        False
      </Badge>
    );
  }

  if (typeof value === 'number') {
    // Format numbers with appropriate precision
    if (Number.isInteger(value)) {
      return <span className="font-mono font-semibold text-medical-dark">{value.toLocaleString()}</span>;
    }
    return <span className="font-mono font-semibold text-medical-dark">{value.toFixed(4)}</span>;
  }

  if (typeof value === 'string') {
    // Check if it's a date string
    const datePattern = /^\d{4}-\d{2}-\d{2}(T|\s)/;
    if (datePattern.test(value)) {
      try {
        const date = new Date(value);
        return (
          <span className="text-sm">
            {date.toLocaleDateString()} {date.toLocaleTimeString()}
          </span>
        );
      } catch {
        return <span className="text-sm">{value}</span>;
      }
    }

    // Check if it's a long string (truncate if needed)
    if (value.length > 100) {
      return (
        <span className="text-sm break-words" title={value}>
          {value.substring(0, 100)}...
        </span>
      );
    }

    return <span className="text-sm">{value}</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-muted-foreground italic">Empty array</span>;
    }

    // If array contains primitives, show as badges
    if (value.every(v => typeof v === 'string' || typeof v === 'number')) {
      return (
        <div className="flex flex-wrap gap-1">
          {value.map((item, idx) => (
            <Badge key={idx} variant="secondary" className="text-xs">
              {String(item)}
            </Badge>
          ))}
        </div>
      );
    }

    // For complex arrays, show count
    return (
      <Badge variant="outline" className="text-xs">
        {value.length} items
      </Badge>
    );
  }

  if (typeof value === 'object') {
    // Render nested object
    return <NestedMetadata data={value} depth={1} />;
  }

  return <span className="text-sm">{String(value)}</span>;
};

/**
 * Renders nested metadata objects
 */
const NestedMetadata: React.FC<{ data: Record<string, any>; depth: number }> = ({ data, depth }) => {
  // Limit nesting depth to prevent overly deep rendering
  if (depth > 2) {
    return <span className="text-xs text-muted-foreground italic">Complex object</span>;
  }

  return (
    <div className={`space-y-2 ${depth > 0 ? 'pl-4 border-l-2 border-gray-200' : ''}`}>
      {Object.entries(data).map(([key, value]) => (
        <div key={key} className="flex items-start justify-between gap-3 text-sm">
          <span className="text-muted-foreground font-medium min-w-[120px]">
            {formatKey(key)}:
          </span>
          <span className="flex-1 text-right">{formatValue(value)}</span>
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
    other: {}
  };

  Object.entries(metadata).forEach(([key, value]) => {
    const keyLower = key.toLowerCase();

    if (keyLower.includes('epoch') || keyLower.includes('batch') || keyLower.includes('learning') ||
        keyLower.includes('optimizer') || keyLower.includes('loss')) {
      categories.training[key] = value;
    } else if (keyLower.includes('model') || keyLower.includes('architecture') || keyLower.includes('layer')) {
      categories.model[key] = value;
    } else if (keyLower.includes('dataset') || keyLower.includes('data') || keyLower.includes('sample')) {
      categories.data[key] = value;
    } else if (keyLower.includes('accuracy') || keyLower.includes('precision') || keyLower.includes('recall') ||
               keyLower.includes('f1') || keyLower.includes('auc') || keyLower.includes('metric')) {
      categories.performance[key] = value;
    } else if (keyLower.includes('date') || keyLower.includes('time') || keyLower.includes('id') ||
               keyLower.includes('name') || keyLower.includes('version')) {
      categories.general[key] = value;
    } else {
      categories.other[key] = value;
    }
  });

  // Filter out empty categories
  return Object.entries(categories).filter(([_, values]) => Object.keys(values).length > 0);
};

/**
 * Main MetadataDisplay component
 */
export const MetadataDisplay: React.FC<MetadataDisplayProps> = ({
  metadata,
  title = "Experiment Metadata"
}) => {
  if (!metadata || Object.keys(metadata).length === 0) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <FileText className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
          <p className="text-muted-foreground">No metadata available</p>
        </CardContent>
      </Card>
    );
  }

  const categorizedData = categorizeMetadata(metadata);
  const categoryLabels: Record<string, string> = {
    general: 'General Information',
    training: 'Training Configuration',
    model: 'Model Details',
    data: 'Dataset Information',
    performance: 'Performance Metrics',
    other: 'Additional Details'
  };

  // If there's only one category or fewer than 4 items, show flat view
  const shouldUseFlatView = categorizedData.length === 1 || Object.keys(metadata).length < 4;

  if (shouldUseFlatView) {
    return (
      <Card className="overflow-hidden">
        <CardContent className="p-0">
          <div className="bg-gradient-to-r from-medical/10 to-medical-dark/10 px-6 py-4 border-b">
            <h3 className="text-lg font-semibold text-medical-dark flex items-center">
              <Settings className="h-5 w-5 mr-2" />
              {title}
            </h3>
          </div>
          <div className="p-6 space-y-4">
            {Object.entries(metadata).map(([key, value]) => (
              <div key={key} className="flex items-start gap-4 pb-3 border-b border-gray-100 last:border-0">
                <div className="text-medical-dark/70 mt-1">
                  {getMetadataIcon(key)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-700 mb-1">{formatKey(key)}</p>
                  <div className="text-gray-900">{formatValue(value)}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  // Categorized view for larger metadata
  return (
    <div className="space-y-4">
      {categorizedData.map(([category, values]) => (
        <Card key={category} className="overflow-hidden">
          <CardContent className="p-0">
            <div className="bg-gradient-to-r from-medical/10 to-medical-dark/10 px-6 py-3 border-b">
              <h4 className="text-md font-semibold text-medical-dark">
                {categoryLabels[category] || formatKey(category)}
              </h4>
            </div>
            <div className="p-6 space-y-4">
              {Object.entries(values).map(([key, value]) => (
                <div key={key} className="flex items-start gap-4 pb-3 border-b border-gray-100 last:border-0">
                  <div className="text-medical-dark/70 mt-1">
                    {getMetadataIcon(key)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-700 mb-1">{formatKey(key)}</p>
                    <div className="text-gray-900">{formatValue(value)}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default MetadataDisplay;
