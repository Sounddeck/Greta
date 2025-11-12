/**
 * Dataset Card Component
 * Displays a training dataset with status and actions
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import type { TrainingDataset, DatasetStatus } from '../../types/training';
import { formatDistanceToNow } from 'date-fns';

interface DatasetCardProps {
  dataset: TrainingDataset;
  onView?: (dataset: TrainingDataset) => void;
  onDelete?: (dataset: TrainingDataset) => void;
}

const statusConfig: Record<DatasetStatus, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline' }> = {
  pending: { label: '⏳ Pending', variant: 'secondary' },
  processing: { label: '🔄 Processing', variant: 'default' },
  ready: { label: '✅ Ready', variant: 'outline' },
  error: { label: '❌ Error', variant: 'destructive' },
};

export function DatasetCard({ dataset, onView, onDelete }: DatasetCardProps) {
  const statusInfo = statusConfig[dataset.status];

  const formatFileSize = (mb: number) => {
    if (mb < 1) return `${(mb * 1024).toFixed(1)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-lg flex items-center gap-2">
              📊 {dataset.name}
            </CardTitle>
            {dataset.description && (
              <CardDescription className="mt-1">{dataset.description}</CardDescription>
            )}
          </div>
          <Badge variant={statusInfo.variant}>{statusInfo.label}</Badge>
        </div>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-500 dark:text-gray-400">Samples</p>
            <p className="font-semibold">{dataset.total_samples.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Size</p>
            <p className="font-semibold">{formatFileSize(dataset.file_size_mb)}</p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Sources</p>
            <p className="font-semibold">{dataset.source_collections.length} collection(s)</p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Created</p>
            <p className="font-semibold">
              {formatDistanceToNow(new Date(dataset.created_at), { addSuffix: true })}
            </p>
          </div>
        </div>

        {dataset.error_message && (
          <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-800 dark:text-red-200">{dataset.error_message}</p>
          </div>
        )}

        {dataset.filters && Object.keys(dataset.filters).length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Filters Applied:</p>
            <div className="flex flex-wrap gap-1">
              {dataset.filters.min_rating && (
                <Badge variant="outline" className="text-xs">
                  Rating ≥ {dataset.filters.min_rating}
                </Badge>
              )}
              {dataset.filters.min_confidence && (
                <Badge variant="outline" className="text-xs">
                  Confidence ≥ {dataset.filters.min_confidence}
                </Badge>
              )}
              {dataset.filters.success_only && (
                <Badge variant="outline" className="text-xs">
                  Success Only
                </Badge>
              )}
              {dataset.filters.exclude_duplicates && (
                <Badge variant="outline" className="text-xs">
                  No Duplicates
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>

      <CardFooter className="flex gap-2">
        {onView && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onView(dataset)}
            disabled={dataset.status === 'processing'}
          >
            View Details
          </Button>
        )}
        {onDelete && (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => onDelete(dataset)}
            disabled={dataset.status === 'processing'}
          >
            Delete
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

