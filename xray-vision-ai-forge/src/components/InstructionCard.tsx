import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Info, 
  AlertCircle, 
  CheckCircle2, 
  Lightbulb, 
  BookOpen,
  Sparkles 
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface InstructionCardProps {
  title?: string;
  children: React.ReactNode;
  variant?: 'default' | 'info' | 'warning' | 'success' | 'tip' | 'guide';
  icon?: React.ReactNode;
  className?: string;
}

/**
 * Instruction card component for displaying helpful information and guidance
 */
const InstructionCard = ({
  title,
  children,
  variant = 'default',
  icon,
  className,
}: InstructionCardProps) => {
  // Choose icon based on variant
  const getIcon = () => {
    if (icon) return icon;
    
    switch (variant) {
      case 'info':
        return <Info className="h-5 w-5" />;
      case 'warning':
        return <AlertCircle className="h-5 w-5" />;
      case 'success':
        return <CheckCircle2 className="h-5 w-5" />;
      case 'tip':
        return <Lightbulb className="h-5 w-5" />;
      case 'guide':
        return <BookOpen className="h-5 w-5" />;
      default:
        return <Sparkles className="h-5 w-5" />;
    }
  };

  // Get colors based on variant
  const getVariantStyles = () => {
    switch (variant) {
      case 'info':
        return 'bg-blue-50 border-blue-200 text-blue-900';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200 text-yellow-900';
      case 'success':
        return 'bg-green-50 border-green-200 text-green-900';
      case 'tip':
        return 'bg-purple-50 border-purple-200 text-purple-900';
      case 'guide':
        return 'bg-teal-50 border-teal-200 text-teal-900';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const getIconColor = () => {
    switch (variant) {
      case 'info':
        return 'text-blue-600';
      case 'warning':
        return 'text-yellow-600';
      case 'success':
        return 'text-green-600';
      case 'tip':
        return 'text-purple-600';
      case 'guide':
        return 'text-teal-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <Alert className={cn(getVariantStyles(), 'border-2', className)}>
      <div className="flex items-start gap-3">
        <div className={cn('flex-shrink-0 mt-0.5', getIconColor())}>
          {getIcon()}
        </div>
        <div className="flex-1 space-y-1">
          {title && (
            <div className="font-semibold text-sm mb-2">{title}</div>
          )}
          <AlertDescription className="text-sm leading-relaxed">
            {children}
          </AlertDescription>
        </div>
      </div>
    </Alert>
  );
};

export default InstructionCard;

