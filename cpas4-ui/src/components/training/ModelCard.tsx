/**
 * Model Card Component
 * Displays model version with deployment status, quantization type, and size
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import type { ModelVersion, DeploymentStatus } from '../../types/training';
import { formatDistanceToNow } from 'date-fns';

interface ModelCardProps {
  model: ModelVersion;
  onDeploy?: (model: ModelVersion) => void;
  onRollback?: (model: ModelVersion) => void;
  onDelete?: (model: ModelVersion) => void;
  onViewDetails?: (model: ModelVersion) => void;
}

export function ModelCard({ model, onDeploy, onRollback, onDelete, onViewDetails }: ModelCardProps) {
  const statusConfig: Record<DeploymentStatus, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline' }> = {
    testing: { label: '🧪 Testing', variant: 'secondary' },
    deployed: { label: '🚀 Deployed', variant: 'default' },
    archived: { label: '📦 Archived', variant: 'outline' },
  };

  const config = statusConfig[model.deployment_status];

  const formatFileSize = (mb: number) => {
    if (mb < 1) return `${(mb * 1024).toFixed(1)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

  return (
    <Card className={`hover:shadow-md transition-shadow ${model.is_active ? 'border-2 border-green-500' : ''}`}>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-lg">{model.version_name}</CardTitle>
              {model.is_active && (
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-300">
                  ✓ Active
                </Badge>
              )}
            </div>
            <CardDescription className="mt-1">
              Base: {model.base_model}
            </CardDescription>
          </div>
          <Badge variant={config.variant}>{config.label}</Badge>
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-4">
          {/* Model Info */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-gray-500 dark:text-gray-400">Quantization</p>
              <p className="font-semibold">{model.quantization_type || 'F16'}</p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400">Model Size</p>
              <p className="font-semibold">{formatFileSize(model.model_size_mb)}</p>
            </div>
          </div>

          {/* Training Config */}
          {model.training_config && (
            <div className="border-t pt-3">
              <p className="text-xs text-gray-500 mb-2">Training Configuration</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500">LoRA Rank:</span>
                  <span className="ml-1 font-semibold">{model.training_config.lora_rank}</span>
                </div>
                <div>
                  <span className="text-gray-500">Learning Rate:</span>
                  <span className="ml-1 font-semibold">{model.training_config.learning_rate}</span>
                </div>
                <div>
                  <span className="text-gray-500">Epochs:</span>
                  <span className="ml-1 font-semibold">{model.training_config.epochs}</span>
                </div>
                <div>
                  <span className="text-gray-500">Batch Size:</span>
                  <span className="ml-1 font-semibold">{model.training_config.batch_size}</span>
                </div>
              </div>
            </div>
          )}

          {/* Performance Metrics */}
          {model.performance_metrics && (
            <div className="border-t pt-3">
              <p className="text-xs text-gray-500 mb-2">Performance Metrics</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {model.performance_metrics.final_loss !== undefined && (
                  <div>
                    <span className="text-gray-500">Final Loss:</span>
                    <span className="ml-1 font-semibold">{model.performance_metrics.final_loss.toFixed(4)}</span>
                  </div>
                )}
                {model.performance_metrics.training_time_minutes !== undefined && (
                  <div>
                    <span className="text-gray-500">Training Time:</span>
                    <span className="ml-1 font-semibold">{model.performance_metrics.training_time_minutes.toFixed(1)} min</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Timestamps */}
          <div className="text-xs text-gray-500 dark:text-gray-400">
            <p>Created {formatDistanceToNow(new Date(model.created_at), { addSuffix: true })}</p>
            {model.deployed_at && (
              <p>Deployed {formatDistanceToNow(new Date(model.deployed_at), { addSuffix: true })}</p>
            )}
          </div>

          {/* Tags */}
          {model.tags && model.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {model.tags.map((tag) => (
                <Badge key={tag} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            {onViewDetails && (
              <Button variant="outline" size="sm" onClick={() => onViewDetails(model)} className="flex-1">
                Details
              </Button>
            )}
            {!model.is_active && model.deployment_status !== 'deployed' && onDeploy && (
              <Button variant="default" size="sm" onClick={() => onDeploy(model)} className="flex-1">
                Deploy
              </Button>
            )}
            {model.is_active && onRollback && (
              <Button variant="outline" size="sm" onClick={() => onRollback(model)} className="flex-1">
                Rollback
              </Button>
            )}
            {!model.is_active && onDelete && (
              <Button variant="destructive" size="sm" onClick={() => onDelete(model)} className="flex-1">
                Delete
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

