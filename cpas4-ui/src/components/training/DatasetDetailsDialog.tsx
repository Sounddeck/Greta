/**
 * Dataset Details Dialog Component
 * Shows detailed information about a training dataset
 */

import React from 'react';
import { Dialog } from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import type { TrainingDataset } from '../../types/training';
import { formatDistanceToNow } from 'date-fns';

interface DatasetDetailsDialogProps {
  dataset: TrainingDataset;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function DatasetDetailsDialog({ dataset, open, onOpenChange }: DatasetDetailsDialogProps) {
  const formatFileSize = (mb: number) => {
    if (mb < 1) return `${(mb * 1024).toFixed(1)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <div className="fixed inset-0 z-50 bg-black/50" onClick={() => onOpenChange(false)} />
      <div className="fixed left-[50%] top-[50%] z-50 w-full max-w-2xl translate-x-[-50%] translate-y-[-50%] bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">📊 {dataset.name}</h2>

        <div className="space-y-6">
          {/* Status */}
          <div>
            <h3 className="font-semibold mb-2">Status</h3>
            <Badge>{dataset.status}</Badge>
          </div>

          {/* Description */}
          {dataset.description && (
            <div>
              <h3 className="font-semibold mb-2">Description</h3>
              <p className="text-gray-600 dark:text-gray-400">{dataset.description}</p>
            </div>
          )}

          {/* Statistics */}
          <div>
            <h3 className="font-semibold mb-2">Statistics</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500">Total Samples</p>
                <p className="text-lg font-semibold">{dataset.total_samples.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">File Size</p>
                <p className="text-lg font-semibold">{formatFileSize(dataset.file_size_mb)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Created</p>
                <p className="text-lg font-semibold">
                  {formatDistanceToNow(new Date(dataset.created_at), { addSuffix: true })}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">File Path</p>
                <p className="text-sm font-mono break-all">{dataset.file_path}</p>
              </div>
            </div>
          </div>

          {/* Source Collections */}
          <div>
            <h3 className="font-semibold mb-2">Source Collections</h3>
            <div className="flex flex-wrap gap-2">
              {dataset.source_collections.map((collection) => (
                <Badge key={collection} variant="outline">
                  {collection}
                </Badge>
              ))}
            </div>
          </div>

          {/* Filters */}
          {dataset.filters && Object.keys(dataset.filters).length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Applied Filters</h3>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md space-y-2">
                {dataset.filters.min_rating && (
                  <p className="text-sm">
                    <span className="font-medium">Min Rating:</span> {dataset.filters.min_rating}
                  </p>
                )}
                {dataset.filters.min_confidence && (
                  <p className="text-sm">
                    <span className="font-medium">Min Confidence:</span> {dataset.filters.min_confidence}
                  </p>
                )}
                {dataset.filters.success_only && (
                  <p className="text-sm">
                    <span className="font-medium">Success Only:</span> Yes
                  </p>
                )}
                {dataset.filters.exclude_duplicates && (
                  <p className="text-sm">
                    <span className="font-medium">Exclude Duplicates:</span> Yes
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Error Message */}
          {dataset.error_message && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">Error</h3>
              <p className="text-sm text-red-700 dark:text-red-300">{dataset.error_message}</p>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 mt-6 pt-4 border-t">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

