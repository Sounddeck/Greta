/**
 * Training API Client
 * API client for LLM training automation endpoints
 */

import { api } from '../lib/api';
import type {
  CreateDatasetRequest,
  CreateJobRequest,
  ModelConversionRequest,
  DatasetListResponse,
  JobListResponse,
  ModelVersionListResponse,
  TrainingDataset,
  TrainingJob,
  ModelVersion,
  TrainingStats,
  TrainingHealth,
  DeploymentStatus,
} from '../types/training';

const BASE_PATH = '/api/v1/llm-training';

export const trainingAPI = {
  // Dataset Endpoints
  datasets: {
    /**
     * List all training datasets
     */
    list: async (): Promise<DatasetListResponse> => {
      const response = await api.get(`${BASE_PATH}/datasets`);
      return response.data;
    },

    /**
     * Get dataset by ID
     */
    get: async (datasetId: string): Promise<TrainingDataset> => {
      const response = await api.get(`${BASE_PATH}/datasets/${datasetId}`);
      return response.data;
    },

    /**
     * Create new training dataset
     */
    create: async (data: CreateDatasetRequest): Promise<TrainingDataset> => {
      const response = await api.post(`${BASE_PATH}/datasets`, data);
      return response.data;
    },

    /**
     * Delete dataset
     */
    delete: async (datasetId: string): Promise<void> => {
      await api.delete(`${BASE_PATH}/datasets/${datasetId}`);
    },
  },

  // Training Job Endpoints
  jobs: {
    /**
     * List all training jobs
     */
    list: async (): Promise<JobListResponse> => {
      const response = await api.get(`${BASE_PATH}/jobs`);
      return response.data;
    },

    /**
     * Get job by ID
     */
    get: async (jobId: string): Promise<TrainingJob> => {
      const response = await api.get(`${BASE_PATH}/jobs/${jobId}`);
      return response.data;
    },

    /**
     * Create new training job
     */
    create: async (data: CreateJobRequest): Promise<TrainingJob> => {
      const response = await api.post(`${BASE_PATH}/jobs`, data);
      return response.data;
    },

    /**
     * Cancel running job
     */
    cancel: async (jobId: string): Promise<void> => {
      await api.post(`${BASE_PATH}/jobs/${jobId}/cancel`);
    },
  },

  // Model Version Endpoints
  models: {
    /**
     * List all model versions
     */
    list: async (status?: DeploymentStatus): Promise<ModelVersionListResponse> => {
      const params = status ? { status } : {};
      const response = await api.get(`${BASE_PATH}/models`, { params });
      return response.data;
    },

    /**
     * Get model version by ID
     */
    get: async (versionId: string): Promise<ModelVersion> => {
      const response = await api.get(`${BASE_PATH}/models/${versionId}`);
      return response.data;
    },

    /**
     * Get active model version
     */
    getActive: async (): Promise<ModelVersion> => {
      const response = await api.get(`${BASE_PATH}/models/active`);
      return response.data;
    },

    /**
     * Convert training job to deployable model
     */
    convert: async (data: ModelConversionRequest): Promise<ModelVersion> => {
      const response = await api.post(`${BASE_PATH}/models/convert`, data);
      return response.data;
    },

    /**
     * Deploy model version
     */
    deploy: async (versionId: string): Promise<{ success: boolean; message: string; note: string }> => {
      const response = await api.post(`${BASE_PATH}/models/${versionId}/deploy`);
      return response.data;
    },

    /**
     * Rollback to previous model version
     */
    rollback: async (versionId: string): Promise<{ success: boolean; message: string; note: string }> => {
      const response = await api.post(`${BASE_PATH}/models/${versionId}/rollback`);
      return response.data;
    },

    /**
     * Delete model version
     */
    delete: async (versionId: string, deleteFiles: boolean = false): Promise<{ success: boolean; message: string }> => {
      const response = await api.delete(`${BASE_PATH}/models/${versionId}`, {
        params: { delete_files: deleteFiles },
      });
      return response.data;
    },
  },

  // Statistics & Health
  /**
   * Get training statistics
   */
  getStats: async (): Promise<TrainingStats> => {
    const response = await api.get(`${BASE_PATH}/stats`);
    return response.data;
  },

  /**
   * Get training system health
   */
  getHealth: async (): Promise<TrainingHealth> => {
    const response = await api.get(`${BASE_PATH}/health`);
    return response.data;
  },
};

