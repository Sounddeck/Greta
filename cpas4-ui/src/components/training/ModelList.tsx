/**
 * Model List Component
 * Grid layout for model versions with filtering by deployment status
 */

import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { ModelCard } from './ModelCard';
import { ConvertJobDialog } from './ConvertJobDialog';
import { ModelDetailsDialog } from './ModelDetailsDialog';
import { useTrainingStore } from '../../stores/trainingStore';
import type { ModelVersion, DeploymentStatus } from '../../types/training';
import { toast } from 'sonner';

export function ModelList() {
  const { models, modelsLoading, modelsError, fetchModels, deployModel, rollbackModel, deleteModel } = useTrainingStore();
  const [convertDialogOpen, setConvertDialogOpen] = useState(false);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelVersion | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // Initial fetch
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleConvertSuccess = () => {
    setConvertDialogOpen(false);
    fetchModels();
  };

  const handleViewDetails = (model: ModelVersion) => {
    setSelectedModel(model);
    setDetailsDialogOpen(true);
  };

  const handleDeploy = async (model: ModelVersion) => {
    if (!confirm(`Deploy model "${model.version_name}"? This will make it the active model.`)) {
      return;
    }

    try {
      await deployModel(model.id);
      toast.success('Model deployed successfully');
      fetchModels();
    } catch (error: any) {
      toast.error(`Failed to deploy model: ${error.message}`);
    }
  };

  const handleRollback = async (model: ModelVersion) => {
    if (!confirm(`Rollback to model "${model.version_name}"? This will replace the current active model.`)) {
      return;
    }

    try {
      await rollbackModel(model.id);
      toast.success('Model rollback successful');
      fetchModels();
    } catch (error: any) {
      toast.error(`Failed to rollback model: ${error.message}`);
    }
  };

  const handleDelete = async (model: ModelVersion) => {
    if (!confirm(`Delete model "${model.version_name}"? This will remove all model files.`)) {
      return;
    }

    try {
      await deleteModel(model.id, true);
      toast.success('Model deleted successfully');
      fetchModels();
    } catch (error: any) {
      toast.error(`Failed to delete model: ${error.message}`);
    }
  };

  // Filter models
  const filteredModels = filterStatus === 'all' 
    ? models 
    : models.filter((model) => model.deployment_status === filterStatus);

  // Sort models (active first, then by created date)
  const sortedModels = [...filteredModels].sort((a, b) => {
    if (a.is_active && !b.is_active) return -1;
    if (!a.is_active && b.is_active) return 1;
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

  if (modelsLoading && models.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading models...</p>
        </div>
      </div>
    );
  }

  if (modelsError) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-6xl mb-4">❌</div>
          <h3 className="text-lg font-semibold mb-2">Error Loading Models</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{modelsError}</p>
          <Button onClick={() => fetchModels()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold">Model Versions</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {models.length} total models
            {models.some((m) => m.is_active) && (
              <span className="ml-2 text-green-500">• 1 active</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Filter */}
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
          >
            <option value="all">All Models</option>
            <option value="testing">Testing</option>
            <option value="deployed">Deployed</option>
            <option value="archived">Archived</option>
          </select>
          <Button onClick={() => setConvertDialogOpen(true)}>Convert Job to Model</Button>
        </div>
      </div>

      {/* Model Grid */}
      {sortedModels.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl mb-4">🤖</div>
            <h3 className="text-lg font-semibold mb-2">
              {filterStatus === 'all' ? 'No Model Versions' : `No ${filterStatus} Models`}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {filterStatus === 'all' 
                ? 'Convert a completed training job to create your first model'
                : `No models with status "${filterStatus}"`}
            </p>
            {filterStatus === 'all' && (
              <Button onClick={() => setConvertDialogOpen(true)}>Convert Job to Model</Button>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {sortedModels.map((model) => (
              <ModelCard
                key={model.id}
                model={model}
                onDeploy={handleDeploy}
                onRollback={handleRollback}
                onDelete={handleDelete}
                onViewDetails={handleViewDetails}
              />
            ))}
          </div>
        </div>
      )}

      {/* Dialogs */}
      <ConvertJobDialog
        open={convertDialogOpen}
        onOpenChange={setConvertDialogOpen}
        onSuccess={handleConvertSuccess}
      />

      {selectedModel && (
        <ModelDetailsDialog
          model={selectedModel}
          open={detailsDialogOpen}
          onOpenChange={setDetailsDialogOpen}
        />
      )}
    </div>
  );
}

