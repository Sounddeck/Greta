/**
 * Training Store
 * Zustand store for LLM training automation state management
 */

import { create } from 'zustand';
import { trainingAPI } from '../api/trainingApi';
import type {
  TrainingDataset,
  TrainingJob,
  ModelVersion,
  TrainingStats,
  TrainingHealth,
  CreateDatasetRequest,
  CreateJobRequest,
  ModelConversionRequest,
  DeploymentStatus,
} from '../types/training';

interface TrainingStore {
  // Datasets
  datasets: TrainingDataset[];
  datasetsLoading: boolean;
  datasetsError: string | null;

  // Jobs
  jobs: TrainingJob[];
  jobsLoading: boolean;
  jobsError: string | null;

  // Models
  models: ModelVersion[];
  modelsLoading: boolean;
  modelsError: string | null;
  activeModel: ModelVersion | null;

  // Stats
  stats: TrainingStats | null;
  statsLoading: boolean;
  statsError: string | null;

  // Health
  health: TrainingHealth | null;
  healthLoading: boolean;

  // Dataset Actions
  fetchDatasets: () => Promise<void>;
  createDataset: (data: CreateDatasetRequest) => Promise<TrainingDataset>;
  deleteDataset: (datasetId: string) => Promise<void>;
  getDataset: (datasetId: string) => Promise<TrainingDataset>;

  // Job Actions
  fetchJobs: () => Promise<void>;
  createJob: (data: CreateJobRequest) => Promise<TrainingJob>;
  cancelJob: (jobId: string) => Promise<void>;
  getJob: (jobId: string) => Promise<TrainingJob>;

  // Model Actions
  fetchModels: (status?: DeploymentStatus) => Promise<void>;
  fetchActiveModel: () => Promise<void>;
  convertJobToModel: (data: ModelConversionRequest) => Promise<ModelVersion>;
  deployModel: (versionId: string) => Promise<void>;
  rollbackModel: (versionId: string) => Promise<void>;
  deleteModel: (versionId: string, deleteFiles?: boolean) => Promise<void>;

  // Stats & Health Actions
  fetchStats: () => Promise<void>;
  fetchHealth: () => Promise<void>;

  // Utility Actions
  reset: () => void;
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  // Initial State
  datasets: [],
  datasetsLoading: false,
  datasetsError: null,

  jobs: [],
  jobsLoading: false,
  jobsError: null,

  models: [],
  modelsLoading: false,
  modelsError: null,
  activeModel: null,

  stats: null,
  statsLoading: false,
  statsError: null,

  health: null,
  healthLoading: false,

  // Dataset Actions
  fetchDatasets: async () => {
    set({ datasetsLoading: true, datasetsError: null });
    try {
      const response = await trainingAPI.datasets.list();
      set({ datasets: response.datasets, datasetsLoading: false });
    } catch (error: any) {
      set({ datasetsError: error.message, datasetsLoading: false });
      throw error;
    }
  },

  createDataset: async (data: CreateDatasetRequest) => {
    try {
      const dataset = await trainingAPI.datasets.create(data);
      set((state) => ({ datasets: [dataset, ...state.datasets] }));
      return dataset;
    } catch (error: any) {
      throw error;
    }
  },

  deleteDataset: async (datasetId: string) => {
    try {
      await trainingAPI.datasets.delete(datasetId);
      set((state) => ({
        datasets: state.datasets.filter((d) => d.dataset_id !== datasetId),
      }));
    } catch (error: any) {
      throw error;
    }
  },

  getDataset: async (datasetId: string) => {
    try {
      const dataset = await trainingAPI.datasets.get(datasetId);
      // Update in store if exists
      set((state) => ({
        datasets: state.datasets.map((d) =>
          d.dataset_id === datasetId ? dataset : d
        ),
      }));
      return dataset;
    } catch (error: any) {
      throw error;
    }
  },

  // Job Actions
  fetchJobs: async () => {
    set({ jobsLoading: true, jobsError: null });
    try {
      const response = await trainingAPI.jobs.list();
      set({ jobs: response.jobs, jobsLoading: false });
    } catch (error: any) {
      set({ jobsError: error.message, jobsLoading: false });
      throw error;
    }
  },

  createJob: async (data: CreateJobRequest) => {
    try {
      const job = await trainingAPI.jobs.create(data);
      set((state) => ({ jobs: [job, ...state.jobs] }));
      return job;
    } catch (error: any) {
      throw error;
    }
  },

  cancelJob: async (jobId: string) => {
    try {
      await trainingAPI.jobs.cancel(jobId);
      // Refresh job to get updated status
      await get().getJob(jobId);
    } catch (error: any) {
      throw error;
    }
  },

  getJob: async (jobId: string) => {
    try {
      const job = await trainingAPI.jobs.get(jobId);
      // Update in store if exists
      set((state) => ({
        jobs: state.jobs.map((j) => (j.job_id === jobId ? job : j)),
      }));
      return job;
    } catch (error: any) {
      throw error;
    }
  },

  // Model Actions
  fetchModels: async (status?: DeploymentStatus) => {
    set({ modelsLoading: true, modelsError: null });
    try {
      const response = await trainingAPI.models.list(status);
      set({ models: response.versions, modelsLoading: false });
    } catch (error: any) {
      set({ modelsError: error.message, modelsLoading: false });
      throw error;
    }
  },

  fetchActiveModel: async () => {
    try {
      const model = await trainingAPI.models.getActive();
      set({ activeModel: model });
    } catch (error: any) {
      // No active model is not an error
      if (error.response?.status === 404) {
        set({ activeModel: null });
      } else {
        throw error;
      }
    }
  },

  convertJobToModel: async (data: ModelConversionRequest) => {
    try {
      const model = await trainingAPI.models.convert(data);
      set((state) => ({ models: [model, ...state.models] }));
      return model;
    } catch (error: any) {
      throw error;
    }
  },

  deployModel: async (versionId: string) => {
    try {
      await trainingAPI.models.deploy(versionId);
      // Refresh models to get updated status
      await get().fetchModels();
      await get().fetchActiveModel();
    } catch (error: any) {
      throw error;
    }
  },

  rollbackModel: async (versionId: string) => {
    try {
      await trainingAPI.models.rollback(versionId);
      // Refresh models to get updated status
      await get().fetchModels();
      await get().fetchActiveModel();
    } catch (error: any) {
      throw error;
    }
  },

  deleteModel: async (versionId: string, deleteFiles: boolean = false) => {
    try {
      await trainingAPI.models.delete(versionId, deleteFiles);
      set((state) => ({
        models: state.models.filter((m) => m.version_id !== versionId),
      }));
    } catch (error: any) {
      throw error;
    }
  },

  // Stats & Health Actions
  fetchStats: async () => {
    set({ statsLoading: true, statsError: null });
    try {
      const stats = await trainingAPI.getStats();
      set({ stats, statsLoading: false });
    } catch (error: any) {
      set({ statsError: error.message, statsLoading: false });
      throw error;
    }
  },

  fetchHealth: async () => {
    set({ healthLoading: true });
    try {
      const health = await trainingAPI.getHealth();
      set({ health, healthLoading: false });
    } catch (error: any) {
      set({ healthLoading: false });
      throw error;
    }
  },

  // Utility Actions
  reset: () => {
    set({
      datasets: [],
      datasetsLoading: false,
      datasetsError: null,
      jobs: [],
      jobsLoading: false,
      jobsError: null,
      models: [],
      modelsLoading: false,
      modelsError: null,
      activeModel: null,
      stats: null,
      statsLoading: false,
      statsError: null,
      health: null,
      healthLoading: false,
    });
  },
}));

