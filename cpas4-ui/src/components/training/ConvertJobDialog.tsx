/**
 * Convert Job Dialog Component
 * Form for converting completed training jobs to GGUF models
 */

import React, { useState, useEffect } from 'react';
import { Dialog } from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { useTrainingStore } from '../../stores/trainingStore';
import type { ModelConversionRequest, QuantizationType } from '../../types/training';
import { toast } from 'sonner';

interface ConvertJobDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

export function ConvertJobDialog({ open, onOpenChange, onSuccess }: ConvertJobDialogProps) {
  const { jobs, fetchJobs, convertJobToModel } = useTrainingStore();
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState<ModelConversionRequest>({
    job_id: '',
    output_name: '',
    description: '',
    quantization_types: ['Q4_K_M'],
  });

  useEffect(() => {
    if (open) {
      fetchJobs();
    }
  }, [open, fetchJobs]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      await convertJobToModel(formData);
      toast.success('Model conversion started successfully');
      onSuccess?.();
      // Reset form
      setFormData({
        job_id: '',
        output_name: '',
        description: '',
        quantization_types: ['Q4_K_M'],
      });
    } catch (error: any) {
      toast.error(`Failed to start conversion: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const completedJobs = jobs.filter((j) => j.status === 'completed');

  const toggleQuantizationType = (type: QuantizationType) => {
    if (formData.quantization_types.includes(type)) {
      setFormData({
        ...formData,
        quantization_types: formData.quantization_types.filter((t) => t !== type),
      });
    } else {
      setFormData({
        ...formData,
        quantization_types: [...formData.quantization_types, type],
      });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <div className="fixed inset-0 z-50 bg-black/50" onClick={() => onOpenChange(false)} />
      <div className="fixed left-[50%] top-[50%] z-50 w-full max-w-2xl translate-x-[-50%] translate-y-[-50%] bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Convert Job to Model</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Job Selection */}
          <div>
            <Label htmlFor="job_id">Completed Training Job *</Label>
            <select
              id="job_id"
              value={formData.job_id}
              onChange={(e) => setFormData({ ...formData, job_id: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
              required
            >
              <option value="">Select a completed job...</option>
              {completedJobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {job.job_name} (Loss: {job.progress?.loss?.toFixed(4) || 'N/A'})
                </option>
              ))}
            </select>
            {completedJobs.length === 0 && (
              <p className="text-xs text-red-500 mt-1">No completed jobs available. Complete a training job first.</p>
            )}
          </div>

          {/* Output Name */}
          <div>
            <Label htmlFor="output_name">Model Name *</Label>
            <Input
              id="output_name"
              value={formData.output_name}
              onChange={(e) => setFormData({ ...formData, output_name: e.target.value })}
              placeholder="e.g., customer-support-model-v1"
              required
            />
          </div>

          {/* Description */}
          <div>
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Describe this model version..."
              rows={3}
            />
          </div>

          {/* Quantization Types */}
          <div>
            <Label>Quantization Types *</Label>
            <p className="text-xs text-gray-500 mb-3">
              Select one or more quantization types. Each type will create a separate model file.
            </p>
            <div className="space-y-3">
              {/* Q4_K_M */}
              <label className="flex items-start gap-3 p-3 border rounded-md cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700">
                <input
                  type="checkbox"
                  checked={formData.quantization_types.includes('Q4_K_M')}
                  onChange={() => toggleQuantizationType('Q4_K_M')}
                  className="mt-1"
                />
                <div className="flex-1">
                  <div className="font-semibold">Q4_K_M (Recommended)</div>
                  <div className="text-xs text-gray-500">
                    ~4.5GB for 8B model • Good quality • Fast inference • Best balance
                  </div>
                </div>
              </label>

              {/* Q5_K_M */}
              <label className="flex items-start gap-3 p-3 border rounded-md cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700">
                <input
                  type="checkbox"
                  checked={formData.quantization_types.includes('Q5_K_M')}
                  onChange={() => toggleQuantizationType('Q5_K_M')}
                  className="mt-1"
                />
                <div className="flex-1">
                  <div className="font-semibold">Q5_K_M</div>
                  <div className="text-xs text-gray-500">
                    ~5.5GB for 8B model • Better quality • Slightly slower • Good for production
                  </div>
                </div>
              </label>

              {/* Q8_0 */}
              <label className="flex items-start gap-3 p-3 border rounded-md cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700">
                <input
                  type="checkbox"
                  checked={formData.quantization_types.includes('Q8_0')}
                  onChange={() => toggleQuantizationType('Q8_0')}
                  className="mt-1"
                />
                <div className="flex-1">
                  <div className="font-semibold">Q8_0</div>
                  <div className="text-xs text-gray-500">
                    ~8.5GB for 8B model • Best quality • Slower inference • For evaluation
                  </div>
                </div>
              </label>
            </div>
            {formData.quantization_types.length === 0 && (
              <p className="text-xs text-red-500 mt-2">Select at least one quantization type</p>
            )}
          </div>

          {/* Info Box */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Conversion Process</h4>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
              <li>1. Merge LoRA adapter with base model (FP16)</li>
              <li>2. Convert to GGUF format (F16)</li>
              <li>3. Quantize to selected types (Q4_K_M, Q5_K_M, Q8_0)</li>
              <li>4. Create model version entry</li>
            </ul>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-2">
              This process may take 10-30 minutes depending on model size and quantization types.
            </p>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-4 border-t">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={loading}>
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={
                loading ||
                !formData.job_id ||
                !formData.output_name ||
                formData.quantization_types.length === 0 ||
                completedJobs.length === 0
              }
            >
              {loading ? 'Starting Conversion...' : 'Start Conversion'}
            </Button>
          </div>
        </form>
      </div>
    </Dialog>
  );
}

