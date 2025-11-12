/**
 * Training Types
 * TypeScript types for LLM training automation
 */

// Enums
export enum DatasetStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  READY = 'ready',
  ERROR = 'error',
}

export enum TrainingJobStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export enum QuantizationType {
  Q4_K_M = 'Q4_K_M',
  Q5_K_M = 'Q5_K_M',
  Q8_0 = 'Q8_0',
  NONE = 'none',
}

export enum DeploymentStatus {
  TESTING = 'testing',
  DEPLOYED = 'deployed',
  ARCHIVED = 'archived',
}

// Dataset Types
export interface DatasetFilters {
  min_rating?: number;
  min_confidence?: number;
  success_only?: boolean;
  date_range?: {
    start?: string;
    end?: string;
  };
  agent_ids?: string[];
  exclude_duplicates?: boolean;
}

export interface TrainingDataset {
  dataset_id: string;
  name: string;
  description?: string;
  source_collections: string[];
  filters: DatasetFilters;
  total_samples: number;
  file_path: string;
  file_size_mb: number;
  status: DatasetStatus;
  created_at: string;
  error_message?: string;
}

export interface CreateDatasetRequest {
  name: string;
  description?: string;
  source_collections: string[];
  filters?: DatasetFilters;
  max_samples?: number;
}

// Training Job Types
export interface TrainingConfig {
  lora_rank?: number;
  lora_alpha?: number;
  learning_rate?: number;
  epochs?: number;
  batch_size?: number;
  ctx_size?: number;
  warmup_steps?: number;
  max_steps?: number;
}

export interface TrainingProgress {
  current_step?: number;
  total_steps?: number;
  current_epoch?: number;
  total_epochs?: number;
  loss?: number;
  learning_rate?: number;
  samples_per_second?: number;
  eta_seconds?: number;
}

export interface TrainingJob {
  job_id: string;
  dataset_id: string;
  base_model: string;
  output_dir: string;
  training_config: TrainingConfig;
  status: TrainingJobStatus;
  progress: TrainingProgress;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  logs: string[];
  created_at: string;
}

export interface CreateJobRequest {
  dataset_id: string;
  base_model: string;
  training_config?: TrainingConfig;
}

// Model Version Types
export interface ModelVersion {
  version_id: string;
  job_id: string;
  dataset_id: string;
  name: string;
  description?: string;
  base_model: string;
  model_paths: Record<string, string>;
  model_sizes: Record<string, number>;
  training_config: Record<string, any>;
  performance_metrics: Record<string, any>;
  deployment_status: DeploymentStatus;
  is_active: boolean;
  deployed_at?: string;
  created_at: string;
  created_by?: string;
  tags: string[];
}

export interface ModelConversionRequest {
  job_id: string;
  output_name: string;
  description?: string;
  quantization_types?: QuantizationType[];
}

// Statistics Types
export interface TrainingStats {
  success: boolean;
  total_datasets: number;
  total_training_samples: number;
  total_jobs: number;
  active_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_models: number;
  active_model?: string;
  disk_usage_mb: number;
}

// Health Check Types
export interface TrainingHealth {
  status: string;
  components: {
    data_collection: string;
    training_pipeline: string;
    model_management: string;
  };
  dependencies: {
    unsloth: boolean;
    torch: boolean;
    llamacpp_convert?: boolean;
    llamacpp_quantize?: boolean;
  };
  version: string;
}

// API Response Types
export interface DatasetListResponse {
  success: boolean;
  datasets: TrainingDataset[];
  total: number;
}

export interface JobListResponse {
  success: boolean;
  jobs: TrainingJob[];
  total: number;
}

export interface ModelVersionListResponse {
  success: boolean;
  versions: ModelVersion[];
  total: number;
}

