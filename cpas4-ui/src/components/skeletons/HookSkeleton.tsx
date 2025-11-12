/**
 * HookSkeleton - Skeleton loader for hook cards
 * Displays animated placeholder while hooks are loading
 */

export function HookSkeleton() {
  return (
    <div className="p-3 border border-border rounded-md bg-card animate-pulse">
      {/* Hook Header */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            {/* Hook Icon */}
            <div className="h-4 w-4 bg-muted rounded"></div>
            {/* Hook Name */}
            <div className="h-4 bg-muted rounded w-32"></div>
            {/* Enabled Badge */}
            <div className="h-4 bg-muted rounded-full w-16"></div>
          </div>
          {/* Hook Details */}
          <div className="flex items-center gap-4 mt-2">
            <div className="h-3 bg-muted rounded w-24"></div>
            <div className="h-3 bg-muted rounded w-20"></div>
          </div>
        </div>
        {/* Action Buttons */}
        <div className="flex gap-2">
          <div className="h-8 w-8 bg-muted rounded"></div>
          <div className="h-8 w-8 bg-muted rounded"></div>
        </div>
      </div>
    </div>
  );
}

/**
 * HookSkeletonList - Renders multiple skeleton cards
 */
export function HookSkeletonList({ count = 3 }: { count?: number }) {
  return (
    <div className="grid gap-2">
      {Array.from({ length: count }).map((_, index) => (
        <HookSkeleton key={index} />
      ))}
    </div>
  );
}

/**
 * PluginSkeleton - Skeleton loader for plugin cards
 */
export function PluginSkeleton() {
  return (
    <div className="p-3 bg-muted rounded-md animate-pulse">
      <div className="flex items-center gap-2 mb-1">
        {/* Plugin Icon */}
        <div className="h-4 w-4 bg-background rounded"></div>
        {/* Plugin Name */}
        <div className="h-4 bg-background rounded w-28"></div>
        {/* Version */}
        <div className="h-3 bg-background rounded w-12"></div>
      </div>
      {/* Description */}
      <div className="h-3 bg-background rounded w-full mt-2"></div>
    </div>
  );
}

/**
 * EventTypeSkeleton - Skeleton loader for event type cards
 */
export function EventTypeSkeleton() {
  return (
    <div className="p-2 bg-muted rounded-md animate-pulse">
      <div className="flex items-center gap-2 mb-1">
        {/* Event Icon */}
        <div className="h-3 w-3 bg-background rounded"></div>
        {/* Event Name */}
        <div className="h-3 bg-background rounded w-24"></div>
      </div>
      {/* Description */}
      <div className="h-2 bg-background rounded w-full mt-1"></div>
    </div>
  );
}
