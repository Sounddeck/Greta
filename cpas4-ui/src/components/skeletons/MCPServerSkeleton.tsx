/**
 * MCPServerSkeleton - Skeleton loader for MCP server cards
 * Displays animated placeholder while MCP servers are loading
 */

export function MCPServerSkeleton() {
  return (
    <div className="p-4 border border-border rounded-lg bg-card animate-pulse">
      {/* Server Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            {/* Server Icon */}
            <div className="h-5 w-5 bg-muted rounded"></div>
            {/* Server Name */}
            <div className="h-5 bg-muted rounded w-40"></div>
            {/* Status Badge */}
            <div className="h-5 bg-muted rounded-full w-20"></div>
          </div>
          {/* Server URL */}
          <div className="h-4 bg-muted rounded w-56 ml-7"></div>
        </div>
        {/* Action Buttons */}
        <div className="flex gap-2">
          <div className="h-8 w-8 bg-muted rounded"></div>
          <div className="h-8 w-8 bg-muted rounded"></div>
        </div>
      </div>

      {/* Server Description */}
      <div className="mt-3">
        <div className="h-4 bg-muted rounded w-full mb-2"></div>
        <div className="h-4 bg-muted rounded w-3/4"></div>
      </div>

      {/* Server Stats */}
      <div className="flex items-center gap-4 mt-4 pt-3 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded w-16"></div>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded w-20"></div>
        </div>
      </div>
    </div>
  );
}

/**
 * MCPServerSkeletonList - Renders multiple skeleton cards
 */
export function MCPServerSkeletonList({ count = 3 }: { count?: number }) {
  return (
    <div className="grid gap-3">
      {Array.from({ length: count }).map((_, index) => (
        <MCPServerSkeleton key={index} />
      ))}
    </div>
  );
}
