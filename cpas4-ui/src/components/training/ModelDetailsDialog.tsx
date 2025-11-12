/**
 * Model Details Dialog Component
 * View detailed model information
 */

import React from 'react';
import { Dialog } from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import type { ModelVersion } from '../../types/training';
import { formatDistanceToNow } from 'date-fns';

interface ModelDetailsDialogProps {
  model: ModelVersion;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ModelDetailsDialog({ model, open, onOpenChange }: ModelDetailsDialogProps) {
  const formatFileSize = (mb: number) => {
    if (mb < 1) return `${(mb * 1024).toFixed(1)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <div className="fixed inset-0 z-50 bg-black/50" onClick={() => onOpenChange(false)} />
      <div className="fixed left-[50%] top-[50%] z-50 w-full max-w-3xl translate-x-[-50%] translate-y-[-50%] bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-h-[90vh] overflow-y-auto">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold">🤖 {model.version_name}</h2>
            {model.is_active && (
              <Badge variant="outline" className="mt-2 bg-green-50 text-green-700 border-green-300">
                ✓ Active Model
              </Badge>
            )}
          </div>
          <Badge>{model.deployment_status}</Badge>
        </div>

        <div className="space-y-6">
          {/* Description */}
          {model.description && (
            <div>
              <h3 className="font-semibold mb-2">Description</h3>
              <p className="text-gray-600 dark:text-gray-400">{model.description}</p>
            </div>
          )}

          {/* Model Information */}
          <div>
            <h3 className="font-semibold mb-2">Model Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500">Base Model</p>
                <p className="font-mono text-sm">{model.base_model}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Quantization Type</p>
                <p className="font-semibold">{model.quantization_type || 'F16'}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Model Size</p>
                <p className="font-semibold">{formatFileSize(model.model_size_mb)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Job ID</p>
                <p className="font-mono text-xs">{model.job_id.substring(0, 16)}...</p>
              </div>
            </div>
          </div>

          {/* Model Paths */}
          <div>
            <h3 className="font-semibold mb-2">Model Paths</h3>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md space-y-2">
              <div>
                <p className="text-xs text-gray-500">Merged Model (FP16)</p>
                <p className="font-mono text-xs break-all">{model.merged_model_path}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">GGUF F16</p>
                <p className="font-mono text-xs break-all">{model.gguf_f16_path}</p>
              </div>
              {model.quantized_model_paths && Object.keys(model.quantized_model_paths).length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 mb-1">Quantized Models</p>
                  {Object.entries(model.quantized_model_paths).map(([type, path]) => (
                    <div key={type} className="ml-2 mb-1">
                      <Badge variant="outline" className="text-xs mr-2">{type}</Badge>
                      <span className="font-mono text-xs break-all">{path}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Training Configuration */}
          {model.training_config && (
            <div>
              <h3 className="font-semibold mb-2">Training Configuration</h3>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-500">LoRA Rank:</span>
                    <span className="ml-2 font-semibold">{model.training_config.lora_rank}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">LoRA Alpha:</span>
                    <span className="ml-2 font-semibold">{model.training_config.lora_alpha}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Learning Rate:</span>
                    <span className="ml-2 font-semibold">{model.training_config.learning_rate}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Epochs:</span>
                    <span className="ml-2 font-semibold">{model.training_config.epochs}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Batch Size:</span>
                    <span className="ml-2 font-semibold">{model.training_config.batch_size}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Context Size:</span>
                    <span className="ml-2 font-semibold">{model.training_config.ctx_size}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Performance Metrics */}
          {model.performance_metrics && (
            <div>
              <h3 className="font-semibold mb-2">Performance Metrics</h3>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  {model.performance_metrics.final_loss !== undefined && (
                    <div>
                      <span className="text-gray-500">Final Loss:</span>
                      <span className="ml-2 font-semibold">{model.performance_metrics.final_loss.toFixed(4)}</span>
                    </div>
                  )}
                  {model.performance_metrics.training_time_minutes !== undefined && (
                    <div>
                      <span className="text-gray-500">Training Time:</span>
                      <span className="ml-2 font-semibold">{model.performance_metrics.training_time_minutes.toFixed(1)} minutes</span>
                    </div>
                  )}
                  {model.performance_metrics.samples_per_second !== undefined && (
                    <div>
                      <span className="text-gray-500">Samples/Second:</span>
                      <span className="ml-2 font-semibold">{model.performance_metrics.samples_per_second.toFixed(2)}</span>
                    </div>
                  )}
                  {model.performance_metrics.gpu_memory_used_gb !== undefined && (
                    <div>
                      <span className="text-gray-500">GPU Memory:</span>
                      <span className="ml-2 font-semibold">{model.performance_metrics.gpu_memory_used_gb.toFixed(2)} GB</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Tags */}
          {model.tags && model.tags.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {model.tags.map((tag) => (
                  <Badge key={tag} variant="outline">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Timestamps */}
          <div>
            <h3 className="font-semibold mb-2">Timeline</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-gray-500">Created:</span>
                <span>{formatDistanceToNow(new Date(model.created_at), { addSuffix: true })}</span>
              </div>
              {model.deployed_at && (
                <div className="flex items-center gap-2">
                  <span className="text-gray-500">Deployed:</span>
                  <span>{formatDistanceToNow(new Date(model.deployed_at), { addSuffix: true })}</span>
                </div>
              )}
              {model.archived_at && (
                <div className="flex items-center gap-2">
                  <span className="text-gray-500">Archived:</span>
                  <span>{formatDistanceToNow(new Date(model.archived_at), { addSuffix: true })}</span>
                </div>
              )}
            </div>
          </div>
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

