/**
 * Progress Indicator Component
 * Quick Win #3: Show loading states with progress bars
 */

import { Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils';

interface ProgressIndicatorProps {
  message?: string;
  progress?: number; // 0-100
  stage?: string;
  estimatedTime?: string;
  variant?: 'default' | 'minimal' | 'detailed';
  className?: string;
}

export function ProgressIndicator({
  message = 'Processing...',
  progress,
  stage,
  estimatedTime,
  variant = 'default',
  className,
}: ProgressIndicatorProps) {
  if (variant === 'minimal') {
    return (
      <div className={cn('flex items-center gap-2', className)}>
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        <span className="text-sm text-muted-foreground">{message}</span>
      </div>
    );
  }

  if (variant === 'detailed') {
    return (
      <div className={cn('space-y-3 p-4 rounded-lg border border-border bg-muted/30', className)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <span className="font-medium text-sm">{message}</span>
          </div>
          {progress !== undefined && (
            <span className="text-sm font-semibold text-primary">{progress}%</span>
          )}
        </div>

        {progress !== undefined && (
          <div className="w-full bg-secondary rounded-full h-2 overflow-hidden">
            <div
              className="bg-primary h-full transition-all duration-300 ease-out rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {(stage || estimatedTime) && (
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            {stage && <span>{stage}</span>}
            {estimatedTime && <span>{estimatedTime}</span>}
          </div>
        )}
      </div>
    );
  }

  // Default variant
  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        <span className="text-sm text-muted-foreground">{message}</span>
        {progress !== undefined && (
          <span className="text-sm font-medium text-primary ml-auto">{progress}%</span>
        )}
      </div>

      {progress !== undefined && (
        <div className="w-full bg-secondary rounded-full h-1.5 overflow-hidden">
          <div
            className="bg-primary h-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
}

/**
 * Loading Skeleton Component
 * For use in empty states while data is loading
 */
interface LoadingSkeletonProps {
  lines?: number;
  className?: string;
}

export function LoadingSkeleton({ lines = 3, className }: LoadingSkeletonProps) {
  return (
    <div className={cn('space-y-3', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-4 bg-muted rounded animate-pulse"
          style={{ width: `${100 - i * 10}%` }}
        />
      ))}
    </div>
  );
}

/**
 * Inline Loading Indicator
 * For use within buttons or small spaces
 */
interface InlineLoadingProps {
  text?: string;
  className?: string;
}

export function InlineLoading({ text = 'Loading...', className }: InlineLoadingProps) {
  return (
    <span className={cn('inline-flex items-center gap-2', className)}>
      <Loader2 className="h-3 w-3 animate-spin" />
      <span className="text-sm">{text}</span>
    </span>
  );
}

