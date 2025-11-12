/**
 * Create Dataset Dialog Component
 * Form for creating new training datasets
 */

import React, { useState } from 'react';
import { Dialog } from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Slider } from '../ui/slider';
import { Switch } from '../ui/switch';
import { useTrainingStore } from '../../stores/trainingStore';
import type { CreateDatasetRequest } from '../../types/training';
import { toast } from 'sonner';

interface CreateDatasetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

export function CreateDatasetDialog({ open, onOpenChange, onSuccess }: CreateDatasetDialogProps) {
  const { createDataset } = useTrainingStore();
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState<CreateDatasetRequest>({
    name: '',
    description: '',
    source_collections: ['agent_conversations'],
    filters: {
      min_rating: 4.0,
      min_confidence: 0.7,
      success_only: true,
      exclude_duplicates: true,
    },
    max_samples: 1000,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      await createDataset(formData);
      toast.success('Dataset creation started');
      onSuccess?.();
      // Reset form
      setFormData({
        name: '',
        description: '',
        source_collections: ['agent_conversations'],
        filters: {
          min_rating: 4.0,
          min_confidence: 0.7,
          success_only: true,
          exclude_duplicates: true,
        },
        max_samples: 1000,
      });
    } catch (error: any) {
      toast.error(`Failed to create dataset: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <div className="fixed inset-0 z-50 bg-black/50" onClick={() => onOpenChange(false)} />
      <div className="fixed left-[50%] top-[50%] z-50 w-full max-w-2xl translate-x-[-50%] translate-y-[-50%] bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Create Training Dataset</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Name */}
          <div>
            <Label htmlFor="name">Dataset Name *</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g., customer-support-v1"
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
              placeholder="Describe this dataset..."
              rows={3}
            />
          </div>

          {/* Source Collections */}
          <div>
            <Label>Source Collections</Label>
            <div className="space-y-2 mt-2">
              {['agent_conversations', 'pai_learning_interactions', 'interactions'].map((collection) => (
                <label key={collection} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={formData.source_collections.includes(collection)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setFormData({
                          ...formData,
                          source_collections: [...formData.source_collections, collection],
                        });
                      } else {
                        setFormData({
                          ...formData,
                          source_collections: formData.source_collections.filter((c) => c !== collection),
                        });
                      }
                    }}
                    className="rounded"
                  />
                  <span className="text-sm">{collection}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Filters */}
          <div className="border-t pt-4">
            <h3 className="font-semibold mb-4">Data Quality Filters</h3>

            {/* Min Rating */}
            <div className="mb-4">
              <Label>Minimum Rating: {formData.filters?.min_rating?.toFixed(1) || 0}</Label>
              <Slider
                value={[formData.filters?.min_rating || 0]}
                onValueChange={([value]) =>
                  setFormData({
                    ...formData,
                    filters: { ...formData.filters, min_rating: value },
                  })
                }
                min={0}
                max={5}
                step={0.1}
                className="mt-2"
              />
            </div>

            {/* Min Confidence */}
            <div className="mb-4">
              <Label>Minimum Confidence: {formData.filters?.min_confidence?.toFixed(2) || 0}</Label>
              <Slider
                value={[formData.filters?.min_confidence || 0]}
                onValueChange={([value]) =>
                  setFormData({
                    ...formData,
                    filters: { ...formData.filters, min_confidence: value },
                  })
                }
                min={0}
                max={1}
                step={0.05}
                className="mt-2"
              />
            </div>

            {/* Success Only */}
            <div className="flex items-center justify-between mb-4">
              <Label>Success Only</Label>
              <Switch
                checked={formData.filters?.success_only || false}
                onCheckedChange={(checked) =>
                  setFormData({
                    ...formData,
                    filters: { ...formData.filters, success_only: checked },
                  })
                }
              />
            </div>

            {/* Exclude Duplicates */}
            <div className="flex items-center justify-between">
              <Label>Exclude Duplicates</Label>
              <Switch
                checked={formData.filters?.exclude_duplicates || false}
                onCheckedChange={(checked) =>
                  setFormData({
                    ...formData,
                    filters: { ...formData.filters, exclude_duplicates: checked },
                  })
                }
              />
            </div>
          </div>

          {/* Max Samples */}
          <div>
            <Label htmlFor="max_samples">Maximum Samples</Label>
            <Input
              id="max_samples"
              type="number"
              value={formData.max_samples}
              onChange={(e) => setFormData({ ...formData, max_samples: parseInt(e.target.value) || 1000 })}
              min={1}
              max={100000}
            />
            <p className="text-xs text-gray-500 mt-1">Leave blank for no limit</p>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-4 border-t">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={loading}>
              Cancel
            </Button>
            <Button type="submit" disabled={loading || !formData.name || formData.source_collections.length === 0}>
              {loading ? 'Creating...' : 'Create Dataset'}
            </Button>
          </div>
        </form>
      </div>
    </Dialog>
  );
}

