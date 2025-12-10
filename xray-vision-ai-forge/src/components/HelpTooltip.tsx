import React from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { HelpCircle, Info } from 'lucide-react';
import { cn } from '@/lib/utils';

interface HelpTooltipProps {
  content: React.ReactNode;
  title?: string;
  icon?: 'help' | 'info';
  side?: 'top' | 'right' | 'bottom' | 'left';
  className?: string;
  iconClassName?: string;
}

/**
 * Reusable help tooltip component for providing contextual information
 */
const HelpTooltip = ({
  content,
  title,
  icon = 'help',
  side = 'top',
  className,
  iconClassName,
}: HelpTooltipProps) => {
  const IconComponent = icon === 'help' ? HelpCircle : Info;

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className={cn(
              'inline-flex items-center justify-center',
              'text-muted-foreground hover:text-foreground',
              'transition-colors',
              'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
              'rounded-full',
              className
            )}
            onClick={(e) => e.preventDefault()}
          >
            <IconComponent className={cn('h-4 w-4', iconClassName)} />
            <span className="sr-only">Help information</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side={side} className="max-w-xs p-4">
          {title && (
            <div className="font-semibold mb-2 text-sm">{title}</div>
          )}
          <div className="text-sm text-muted-foreground">{content}</div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default HelpTooltip;

