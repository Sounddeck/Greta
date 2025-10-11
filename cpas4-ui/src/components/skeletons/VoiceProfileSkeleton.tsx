/**
 * VoiceProfileSkeleton - Skeleton loader for voice profile cards
 * Displays animated placeholder while voice profiles are loading
 */

export function VoiceProfileSkeleton() {
  return (
    <div className="p-4 border border-border rounded-lg bg-card animate-pulse">
      {/* Profile Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          {/* Profile Name */}
          <div className="h-5 bg-muted rounded w-32 mb-2"></div>
          {/* Language Badge */}
          <div className="h-4 bg-muted rounded w-16"></div>
        </div>
        {/* Action Buttons */}
        <div className="flex gap-2">
          <div className="h-8 w-8 bg-muted rounded"></div>
          <div className="h-8 w-8 bg-muted rounded"></div>
        </div>
      </div>

      {/* Profile Details */}
      <div className="space-y-2 mt-4">
        {/* Voice */}
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded w-24"></div>
        </div>
        {/* Speed */}
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded w-20"></div>
        </div>
        {/* Pitch */}
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded w-20"></div>
        </div>
        {/* Volume */}
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded w-20"></div>
        </div>
      </div>
    </div>
  );
}

/**
 * VoiceProfileSkeletonList - Renders multiple skeleton cards
 */
export function VoiceProfileSkeletonList({ count = 3 }: { count?: number }) {
  return (
    <div className="grid gap-3">
      {Array.from({ length: count }).map((_, index) => (
        <VoiceProfileSkeleton key={index} />
      ))}
    </div>
  );
}
