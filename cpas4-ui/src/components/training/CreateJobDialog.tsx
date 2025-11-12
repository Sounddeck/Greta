/**
 * Create Job Dialog Component
 * Form for creating new training jobs
 */

import React, { useState, useEffect } from 'react';
import { Dialog } from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select } from '../ui/select';
import { useTrainingStore } from '../../stores/trainingStore';
import type { CreateJobRequest } from '../../types/training';
import { toast } from 'sonner';

interface CreateJobDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

export function CreateJobDialog({ open, onOpenChange, onSuccess }: CreateJobDialogProps) {
  const { datasets, fetchDatasets, createJob } = useTrainingStore();
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState<CreateJobRequest>({
    job_name: '',
    dataset_id: '',
    base_model: 'unsloth/llama-3-8b-bnb-4bit',
    config: {
      lora_rank: 16,
      lora_alpha: 16,
      learning_rate: 0.0002,
      epochs: 3,
      batch_size: 2,
      ctx_size: 2048,
    },
  });

  useEffect(() => {
    if (open) {
      fetchDatasets();
    }
  }, [open, fetchDatasets]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      await createJob(formData);
      toast.success('Training job created successfully');
      onSuccess?.();
      // Reset form
      setFormData({
        job_name: '',
        dataset_id: '',
        base_model: 'unsloth/llama-3-8b-bnb-4bit',
        config: {
          lora_rank: 16,
          lora_alpha: 16,
          learning_rate: 0.0002,
          epochs: 3,
          batch_size: 2,
          ctx_size: 2048,
        },
      });
    } catch (error: any) {
      toast.error(`Failed to create job: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const readyDatasets = datasets.filter((d) => d.status === 'ready');

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <div className="fixed inset-0 z-50 bg-black/50" onClick={() => onOpenChange(false)} />
      <div className="fixed left-[50%] top-[50%] z-50 w-full max-w-2xl translate-x-[-50%] translate-y-[-50%] bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Create Training Job</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Job Name */}
          <div>
            <Label htmlFor="job_name">Job Name *</Label>
            <Input
              id="job_name"
              value={formData.job_name}
              onChange={(e) => setFormData({ ...formData, job_name: e.target.value })}
              placeholder="e.g., customer-support-training-v1"
              required
            />
          </div>

          {/* Dataset Selection */}
          <div>
            <Label htmlFor="dataset_id">Dataset *</Label>
            <select
              id="dataset_id"
              value={formData.dataset_id}
              onChange={(e) => setFormData({ ...formData, dataset_id: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
              required
            >
              <option value="">Select a dataset...</option>
              {readyDatasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.total_samples} samples)
                </option>
              ))}
            </select>
            {readyDatasets.length === 0 && (
              <p className="text-xs text-red-500 mt-1">No ready datasets available. Create a dataset first.</p>
            )}
          </div>

          {/* Base Model */}
          <div>
            <Label htmlFor="base_model">Base Model *</Label>
            <select
              id="base_model"
              value={formData.base_model}
              onChange={(e) => setFormData({ ...formData, base_model: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
              required
            >
              <option value="unsloth/llama-3-8b-bnb-4bit">Llama 3 8B (4-bit)</option>
              <option value="unsloth/llama-3-70b-bnb-4bit">Llama 3 70B (4-bit)</option>
              <option value="unsloth/mistral-7b-bnb-4bit">Mistral 7B (4-bit)</option>
              <option value="unsloth/phi-3-mini-4k-instruct-bnb-4bit">Phi-3 Mini (4-bit)</option>
            </select>
          </div>

          {/* Training Configuration */}
          <div className="border-t pt-4">
            <h3 className="font-semibold mb-4">Training Configuration</h3>

            <div className="grid grid-cols-2 gap-4">
              {/* LoRA Rank */}
              <div>
                <Label htmlFor="lora_rank">LoRA Rank</Label>
                <Input
                  id="lora_rank"
                  type="number"
                  value={formData.config.lora_rank}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, lora_rank: parseInt(e.target.value) || 16 },
                    })
                  }
                  min={4}
                  max={128}
                />
                <p className="text-xs text-gray-500 mt-1">Recommended: 8-32</p>
              </div>

              {/* LoRA Alpha */}
              <div>
                <Label htmlFor="lora_alpha">LoRA Alpha</Label>
                <Input
                  id="lora_alpha"
                  type="number"
                  value={formData.config.lora_alpha}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, lora_alpha: parseInt(e.target.value) || 16 },
                    })
                  }
                  min={4}
                  max={128}
                />
                <p className="text-xs text-gray-500 mt-1">Usually same as rank</p>
              </div>

              {/* Learning Rate */}
              <div>
                <Label htmlFor="learning_rate">Learning Rate</Label>
                <Input
                  id="learning_rate"
                  type="number"
                  step="0.00001"
                  value={formData.config.learning_rate}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, learning_rate: parseFloat(e.target.value) || 0.0002 },
                    })
                  }
                  min={0.00001}
                  max={0.01}
                />
                <p className="text-xs text-gray-500 mt-1">Recommended: 0.0001-0.0003</p>
              </div>

              {/* Epochs */}
              <div>
                <Label htmlFor="epochs">Epochs</Label>
                <Input
                  id="epochs"
                  type="number"
                  value={formData.config.epochs}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, epochs: parseInt(e.target.value) || 3 },
                    })
                  }
                  min={1}
                  max={20}
                />
                <p className="text-xs text-gray-500 mt-1">Recommended: 1-5</p>
              </div>

              {/* Batch Size */}
              <div>
                <Label htmlFor="batch_size">Batch Size</Label>
                <Input
                  id="batch_size"
                  type="number"
                  value={formData.config.batch_size}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, batch_size: parseInt(e.target.value) || 2 },
                    })
                  }
                  min={1}
                  max={32}
                />
                <p className="text-xs text-gray-500 mt-1">Depends on GPU memory</p>
              </div>

              {/* Context Size */}
              <div>
                <Label htmlFor="ctx_size">Context Size</Label>
                <Input
                  id="ctx_size"
                  type="number"
                  value={formData.config.ctx_size}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      config: { ...formData.config, ctx_size: parseInt(e.target.value) || 2048 },
                    })
                  }
                  min={512}
                  max={8192}
                  step={512}
                />
                <p className="text-xs text-gray-500 mt-1">Max sequence length</p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-4 border-t">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={loading}>
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={loading || !formData.job_name || !formData.dataset_id || readyDatasets.length === 0}
            >
              {loading ? 'Creating...' : 'Create Job'}
            </Button>
          </div>
        </form>
      </div>
    </Dialog>
  );
}

