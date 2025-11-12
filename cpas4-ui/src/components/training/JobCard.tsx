/**
 * Job Card Component
 * Displays individual training job with progress bar, status, and metrics
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import type { TrainingJob, TrainingJobStatus } from '../../types/training';
import { formatDistanceToNow } from 'date-fns';

interface JobCardProps {
  job: TrainingJob;
  onViewLogs?: (job: TrainingJob) => void;
  onCancel?: (job: TrainingJob) => void;
  onViewDetails?: (job: TrainingJob) => void;
}

export function JobCard({ job, onViewLogs, onCancel, onViewDetails }: JobCardProps) {
  const statusConfig: Record<TrainingJobStatus, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline' }> = {
    pending: { label: '⏳ Pending', variant: 'secondary' },
    running: { label: '🔄 Running', variant: 'default' },
    completed: { label: '✅ Completed', variant: 'outline' },
    failed: { label: '❌ Failed', variant: 'destructive' },
    cancelled: { label: '🚫 Cancelled', variant: 'secondary' },
  };

  const config = statusConfig[job.status];

  // Calculate progress percentage
  const getProgressPercentage = () => {
    if (job.status === 'completed') return 100;
    if (job.status === 'failed' || job.status === 'cancelled') return 0;
    if (!job.progress?.current_epoch || !job.config.epochs) return 0;
    return Math.min(100, (job.progress.current_epoch / job.config.epochs) * 100);
  };

  const progressPercentage = getProgressPercentage();

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-lg">{job.job_name}</CardTitle>
            <CardDescription className="mt-1">
              Dataset: {job.dataset_id.substring(0, 8)}...
            </CardDescription>
          </div>
          <Badge variant={config.variant}>{config.label}</Badge>
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-4">
          {/* Progress Bar (for running/completed jobs) */}
          {(job.status === 'running' || job.status === 'completed') && (
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-gray-600 dark:text-gray-400">Progress</span>
                <span className="font-semibold">{progressPercentage.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progressPercentage}%` }}
                />
              </div>
            </div>
          )}

          {/* Training Metrics (for running/completed jobs) */}
          {job.progress && (
            <div className="grid grid-cols-2 gap-3 text-sm">
              {job.progress.loss !== undefined && (
                <div>
                  <p className="text-gray-500 dark:text-gray-400">Loss</p>
                  <p className="font-semibold">{job.progress.loss.toFixed(4)}</p>
                </div>
              )}
              {job.progress.learning_rate !== undefined && (
                <div>
                  <p className="text-gray-500 dark:text-gray-400">Learning Rate</p>
                  <p className="font-semibold">{job.progress.learning_rate.toExponential(2)}</p>
                </div>
              )}
              {job.progress.current_epoch !== undefined && (
                <div>
                  <p className="text-gray-500 dark:text-gray-400">Epoch</p>
                  <p className="font-semibold">
                    {job.progress.current_epoch} / {job.config.epochs}
                  </p>
                </div>
              )}
              {job.progress.epoch !== undefined && (
                <div>
                  <p className="text-gray-500 dark:text-gray-400">Current Epoch</p>
                  <p className="font-semibold">{job.progress.epoch.toFixed(2)}</p>
                </div>
              )}
            </div>
          )}

          {/* Training Config */}
          <div className="border-t pt-3">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Base Model:</span>
                <p className="font-mono truncate">{job.base_model}</p>
              </div>
              <div>
                <span className="text-gray-500">LoRA Rank:</span>
                <p className="font-semibold">{job.config.lora_rank}</p>
              </div>
              <div>
                <span className="text-gray-500">Learning Rate:</span>
                <p className="font-semibold">{job.config.learning_rate}</p>
              </div>
              <div>
                <span className="text-gray-500">Batch Size:</span>
                <p className="font-semibold">{job.config.batch_size}</p>
              </div>
            </div>
          </div>

          {/* Timestamps */}
          <div className="text-xs text-gray-500 dark:text-gray-400">
            <p>Created {formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}</p>
            {job.started_at && (
              <p>Started {formatDistanceToNow(new Date(job.started_at), { addSuffix: true })}</p>
            )}
            {job.completed_at && (
              <p>Completed {formatDistanceToNow(new Date(job.completed_at), { addSuffix: true })}</p>
            )}
          </div>

          {/* Error Message */}
          {job.error_message && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-2">
              <p className="text-xs text-red-700 dark:text-red-300">{job.error_message}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            {onViewDetails && (
              <Button variant="outline" size="sm" onClick={() => onViewDetails(job)} className="flex-1">
                View Details
              </Button>
            )}
            {onViewLogs && (
              <Button variant="outline" size="sm" onClick={() => onViewLogs(job)} className="flex-1">
                View Logs
              </Button>
            )}
            {job.status === 'running' && onCancel && (
              <Button variant="destructive" size="sm" onClick={() => onCancel(job)} className="flex-1">
                Cancel
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

