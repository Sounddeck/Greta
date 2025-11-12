/**
 * Dataset List Component
 * Displays list of training datasets with create and manage actions
 */

import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { DatasetCard } from './DatasetCard';
import { CreateDatasetDialog } from './CreateDatasetDialog';
import { DatasetDetailsDialog } from './DatasetDetailsDialog';
import { Button } from '../ui/button';
import type { TrainingDataset } from '../../types/training';
import { toast } from 'sonner';

export function DatasetList() {
  const {
    datasets,
    datasetsLoading,
    datasetsError,
    fetchDatasets,
    deleteDataset,
  } = useTrainingStore();

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<TrainingDataset | null>(null);
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  const handleDelete = async (dataset: TrainingDataset) => {
    if (!confirm(`Are you sure you want to delete dataset "${dataset.name}"?`)) {
      return;
    }

    try {
      await deleteDataset(dataset.dataset_id);
      toast.success('Dataset deleted successfully');
    } catch (error: any) {
      toast.error(`Failed to delete dataset: ${error.message}`);
    }
  };

  const handleView = (dataset: TrainingDataset) => {
    setSelectedDataset(dataset);
    setShowDetailsDialog(true);
  };

  const handleCreateSuccess = () => {
    setShowCreateDialog(false);
    toast.success('Dataset created successfully');
    fetchDatasets();
  };

  if (datasetsLoading && datasets.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading datasets...</p>
        </div>
      </div>
    );
  }

  if (datasetsError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400">Error loading datasets</p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">{datasetsError}</p>
          <Button onClick={() => fetchDatasets()} className="mt-4" variant="outline">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold">Training Datasets</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {datasets.length} dataset{datasets.length !== 1 ? 's' : ''} available
          </p>
        </div>
        <Button onClick={() => setShowCreateDialog(true)}>
          ➕ Create Dataset
        </Button>
      </div>

      {/* Dataset Grid */}
      {datasets.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl mb-4">📊</div>
            <h3 className="text-lg font-semibold mb-2">No datasets yet</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Create your first training dataset to get started
            </p>
            <Button onClick={() => setShowCreateDialog(true)}>
              Create Dataset
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto">
          {datasets.map((dataset) => (
            <DatasetCard
              key={dataset.dataset_id}
              dataset={dataset}
              onView={handleView}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}

      {/* Create Dataset Dialog */}
      <CreateDatasetDialog
        open={showCreateDialog}
        onOpenChange={setShowCreateDialog}
        onSuccess={handleCreateSuccess}
      />

      {/* Dataset Details Dialog */}
      {selectedDataset && (
        <DatasetDetailsDialog
          dataset={selectedDataset}
          open={showDetailsDialog}
          onOpenChange={setShowDetailsDialog}
        />
      )}
    </div>
  );
}

