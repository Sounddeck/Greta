/**
 * TableSkeleton - Generic skeleton loader for table rows
 * Configurable number of rows and columns with animated shimmer effect
 */

interface TableSkeletonProps {
  rows?: number;
  columns?: number;
  showHeader?: boolean;
}

export function TableSkeleton({ 
  rows = 5, 
  columns = 4, 
  showHeader = true 
}: TableSkeletonProps) {
  return (
    <div className="w-full border border-border rounded-lg overflow-hidden">
      {/* Table Header */}
      {showHeader && (
        <div className="bg-muted border-b border-border">
          <div className="flex items-center gap-4 p-3">
            {Array.from({ length: columns }).map((_, index) => (
              <div
                key={`header-${index}`}
                className="h-4 bg-background rounded animate-pulse"
                style={{ width: `${100 / columns}%` }}
              ></div>
            ))}
          </div>
        </div>
      )}

      {/* Table Rows */}
      <div className="divide-y divide-border">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={`row-${rowIndex}`} className="flex items-center gap-4 p-3 animate-pulse">
            {Array.from({ length: columns }).map((_, colIndex) => (
              <div
                key={`cell-${rowIndex}-${colIndex}`}
                className="h-4 bg-muted rounded"
                style={{ 
                  width: `${100 / columns}%`,
                  // Vary the width slightly for more realistic appearance
                  maxWidth: colIndex === 0 ? '80%' : '90%'
                }}
              ></div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * ListSkeleton - Generic skeleton loader for list items
 */
interface ListSkeletonProps {
  items?: number;
  showIcon?: boolean;
}

export function ListSkeleton({ items = 5, showIcon = true }: ListSkeletonProps) {
  return (
    <div className="space-y-2">
      {Array.from({ length: items }).map((_, index) => (
        <div key={index} className="flex items-center gap-3 p-3 border border-border rounded-md animate-pulse">
          {showIcon && <div className="h-5 w-5 bg-muted rounded"></div>}
          <div className="flex-1 space-y-2">
            <div className="h-4 bg-muted rounded w-3/4"></div>
            <div className="h-3 bg-muted rounded w-1/2"></div>
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * CardSkeleton - Generic skeleton loader for card layouts
 */
export function CardSkeleton() {
  return (
    <div className="p-4 border border-border rounded-lg bg-card animate-pulse">
      {/* Card Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="h-5 bg-muted rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-muted rounded w-3/4"></div>
        </div>
        <div className="h-8 w-8 bg-muted rounded"></div>
      </div>

      {/* Card Content */}
      <div className="space-y-3">
        <div className="h-4 bg-muted rounded w-full"></div>
        <div className="h-4 bg-muted rounded w-5/6"></div>
        <div className="h-4 bg-muted rounded w-4/6"></div>
      </div>

      {/* Card Footer */}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-border">
        <div className="h-3 bg-muted rounded w-20"></div>
        <div className="h-3 bg-muted rounded w-24"></div>
      </div>
    </div>
  );
}
