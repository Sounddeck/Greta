import axios, { AxiosError, AxiosRequestConfig } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000; // Initial delay
const RETRY_BACKOFF_MULTIPLIER = 2; // Exponential backoff

// Retryable HTTP status codes
const RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504];

// Retry delay calculator with exponential backoff
const getRetryDelay = (retryCount: number): number => {
  return RETRY_DELAY_MS * Math.pow(RETRY_BACKOFF_MULTIPLIER, retryCount);
};

// Sleep utility
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Add request interceptor for correlation IDs
api.interceptors.request.use(
  (config) => {
    // Add correlation ID for request tracing
    const correlationId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    config.headers['X-Correlation-ID'] = correlationId;
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor with retry logic and error handling
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const config = error.config as AxiosRequestConfig & { _retryCount?: number };

    // Initialize retry count
    if (!config._retryCount) {
      config._retryCount = 0;
    }

    // Check if error is retryable
    const isRetryable =
      error.response &&
      RETRYABLE_STATUS_CODES.includes(error.response.status) &&
      config._retryCount < MAX_RETRIES;

    // Retry logic for transient errors
    if (isRetryable) {
      config._retryCount++;
      const delay = getRetryDelay(config._retryCount);

      console.warn(
        `Request failed with status ${error.response?.status}. ` +
        `Retrying (${config._retryCount}/${MAX_RETRIES}) after ${delay}ms...`,
        {
          url: config.url,
          correlationId: config.headers?.['X-Correlation-ID'],
        }
      );

      await sleep(delay);
      return api.request(config);
    }

    // Handle specific error cases
    if (error.response?.status === 404) {
      // Silently pass through 404 errors - they're handled by stores
      return Promise.reject(error);
    }

    if (error.response?.status === 401) {
      // Authentication error - could trigger re-login
      console.error('Authentication required. Please log in again.');
    }

    if (error.response?.status === 403) {
      // Authorization error
      console.error('Access denied. You do not have permission to perform this action.');
    }

    if (error.code === 'ECONNABORTED') {
      // Timeout error
      console.error('Request timed out. Please check your connection and try again.');
    }

    if (!error.response) {
      // Network error
      console.error('Network error. Please check your internet connection.');
    }

    // Log error with correlation ID for debugging
    console.error('API Error:', {
      url: config.url,
      method: config.method,
      status: error.response?.status,
      message: error.message,
      correlationId: config.headers?.['X-Correlation-ID'],
      retryCount: config._retryCount,
    });

    return Promise.reject(error);
  }
);

// Agent endpoints
export const agentAPI = {
  list: () => api.get('/agents'),
  get: (id: string) => api.get(`/agents/${id}`),
  create: (data: any) => api.post('/agents', data),
  delete: (id: string) => api.delete(`/agents/${id}`),
  update: (id: string, data: any) => api.put(`/agents/${id}`, data),
};


// Agent Builder endpoints
export const agentBuilderAPI = {
  // Templates
  createTemplate: (data: any) => api.post('/agent-builder/templates', data),
  getTemplate: (id: string) => api.get(`/agent-builder/templates/${id}`),
  updateTemplate: (id: string, data: any) => api.patch(`/agent-builder/templates/${id}`, data),
  deleteTemplate: (id: string) => api.delete(`/agent-builder/templates/${id}`),
  listTemplates: (params?: any) => api.get('/agent-builder/templates', { params }),

  // Agents
  createAgent: (data: any) => api.post('/agent-builder/agents', data),
  getAgent: (agentId: string) => api.get(`/agent-builder/agents/${agentId}`),
  updateAgent: (id: string, data: any) => api.patch(`/agent-builder/agents/${id}`, data),
  deleteAgent: (id: string) => api.delete(`/agent-builder/agents/${id}`),
  listAgents: (params?: any) => api.get('/agent-builder/agents', { params }),
  cloneAgent: (id: string, newName: string) => api.post(`/agent-builder/agents/${id}/clone`, null, { params: { new_name: newName } }),

  // Versions
  createVersion: (data: any) => api.post('/agent-builder/versions', data),
  getVersion: (versionId: string) => api.get(`/agent-builder/versions/${versionId}`),
  listVersions: (agentId: string, limit?: number) => api.get(`/agent-builder/agents/${agentId}/versions`, { params: { limit } }),
  compareVersions: (versionIdA: string, versionIdB: string) => api.get(`/agent-builder/versions/${versionIdA}/compare/${versionIdB}`),
  rollbackVersion: (agentId: string, versionId: string) => api.post(`/agent-builder/agents/${agentId}/rollback`, null, { params: { version_id: versionId } }),

  // Deployments
  deployAgent: (data: any) => api.post('/agent-builder/deployments', data),
  getDeployment: (id: string) => api.get(`/agent-builder/deployments/${id}`),
  listDeployments: (params?: any) => api.get('/agent-builder/deployments', { params }),

  // Executions
  executeAgent: (data: any) => api.post('/agent-builder/executions', data),
  getExecution: (id: string) => api.get(`/agent-builder/executions/${id}`),
  listExecutions: (params?: any) => api.get('/agent-builder/executions', { params }),

  // Code Generation
  generateCode: (agentId: string, includeTools?: boolean, includeTests?: boolean) =>
    api.post(`/agent-builder/agents/${agentId}/generate-code`, null, { params: { include_tools: includeTools, include_tests: includeTests } }),

  // Validation
  validateAgent: (agentId: string) => api.post(`/agent-builder/agents/${agentId}/validate`),
  validateEnhanced: (agentId?: string | null, draftId?: string | null) =>
    api.post('/agent-builder/validate-enhanced', null, { params: { agent_id: agentId, draft_id: draftId } }),

  // Drafts (Autosave)
  createDraft: (data: any) => api.post('/agent-builder/drafts', data),
  getDraft: (draftId: string) => api.get(`/agent-builder/drafts/${draftId}`),
  updateDraft: (draftId: string, data: any) => api.patch(`/agent-builder/drafts/${draftId}`, data),
  deleteDraft: (draftId: string) => api.delete(`/agent-builder/drafts/${draftId}`),
  listDrafts: (limit?: number) => api.get('/agent-builder/drafts', { params: { limit } }),
  convertDraftToAgent: (draftId: string) => api.post(`/agent-builder/drafts/${draftId}/convert`),
  cleanupExpiredDrafts: () => api.post('/agent-builder/drafts/cleanup'),

  // Statistics
  getStats: () => api.get('/agent-builder/stats'),

  // Models
  getAvailableModels: () => api.get('/agent-builder/models'),

  // MCP integrations for agent
  listMCP: (agentId: string) => api.get(`/agent-builder/agents/${agentId}/mcp-config`),
  addMCP: (agentId: string, data: any) => api.post(`/agent-builder/agents/${agentId}/mcp-config`, data),
  updateMCP: (agentId: string, integrationId: string, data: any) => api.put(`/agent-builder/agents/${agentId}/mcp-config/${integrationId}`, data),
  deleteMCP: (agentId: string, integrationId: string) => api.delete(`/agent-builder/agents/${agentId}/mcp-config/${integrationId}`),

  // Workflows
  createWorkflow: (data: any) => api.post('/agent-builder/workflows', data),
  executeWorkflow: (workflowId: string, inputData: any) => api.post(`/agent-builder/workflows/${workflowId}/execute`, inputData),
  getWorkflowExecution: (workflowId: string, executionId: string) => api.get(`/agent-builder/workflows/${workflowId}/executions/${executionId}`),
  getWorkflowMetrics: () => api.get('/agent-builder/workflows/metrics'),

  // Legacy workflow endpoints (for backward compatibility)
  validateWorkflow: (workflow: any) => api.post('/agent-builder/workflows/validate', workflow),
  deployWorkflow: (workflow: any) => api.post('/agent-builder/workflows/deploy', workflow),
};

// Agent Runs endpoints
export const agentRunAPI = {
  start: (agentId: string, data: { prompt: string; input?: any }) =>
    api.post(`/agent-builder/agents/${agentId}/runs`, data),
  get: (runId: string) => api.get(`/agent-builder/runs/${runId}`),
};


// Agent Versions endpoints
export const agentVersionAPI = {
  list: (agentId: string) => api.get(`/agent-builder/agents/${agentId}/versions`),
  clone: (agentId: string) => api.post(`/agent-builder/agents/${agentId}/versions/clone`, {}),
  activate: (versionId: string) => api.post(`/agent-builder/versions/${versionId}/activate`, {}),
};


// Task endpoints (Master Agent)
export const taskAPI = {
  execute: (data: any) => {
    // Add timeout of 60 seconds for task execution
    return api.post('/tasks/execute', data, { timeout: 60000 });
  },
  get: (id: string) => api.get(`/tasks/${id}`),
  list: () => api.get('/tasks'),
  cancel: (id: string) => api.post(`/tasks/${id}/cancel`),
};

// Report endpoints
export const reportAPI = {
  generate: (data: any) => api.post('/reports/generate', data),
  get: (id: string) => api.get(`/reports/${id}`),
  list: () => api.get('/reports'),
  download: (id: string) => api.get(`/reports/${id}/download`, { responseType: 'blob' }),
  statistics: () => api.get('/reports/statistics'),
};

// Prompt endpoints
export const promptAPI = {
  enhance: (data: any) => {
    // Use simple enhancement endpoint (LLM-based, no CoT templates)
    return api.post('/prompts/enhance', {
      prompt: data.prompt,
      task_type: data.agent_type || 'general',
      custom_template: data.custom_template || null
    });
  },
  enhanceWithCoT: (data: any) => {
    // Use CoT enhancement endpoint for step-by-step reasoning
    return api.post('/prompts/enhance-cot', {
      prompt: data.prompt,
      task_type: data.agent_type || 'general',
      custom_template: data.custom_template || null
    });
  },
  enhanceAsync: (data: any) => api.post('/prompts/enhance/async', data),
  get: (id: string) => api.get(`/prompts/enhance/${id}`),
  feedback: (id: string, data: any) => api.post(`/prompts/enhance/${id}/feedback`, data),
  statistics: () => api.get('/prompts/statistics'),
};

// Knowledge endpoints
export const knowledgeAPI = {
  upload: (formData: FormData) => api.post('/knowledge/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  }),
  search: (query: string, params?: any) => api.post('/knowledge/search', { query, ...params }),
  list: () => api.get('/knowledge/documents'),
  delete: (id: string) => api.delete(`/knowledge/documents/${id}`),
};

// Prompt Template endpoints
export const promptTemplateAPI = {
  initialize: () => api.post('/prompts/templates/initialize'),
  list: (params?: any) => api.get('/prompts/templates', { params }),
  search: (query: string, category?: string) => api.get('/prompts/templates/search', { params: { q: query, category } }),
  get: (id: string) => api.get(`/prompts/templates/${id}`),
  create: (data: any) => api.post('/prompts/templates', data),
  update: (id: string, data: any) => api.put(`/prompts/templates/${id}`, data),
  delete: (id: string) => api.delete(`/prompts/templates/${id}`),
  render: (templateId: string, variables: Record<string, string>) =>
    api.post('/prompts/templates/render', { template_id: templateId, variables }),
  rate: (templateId: string, rating: number, feedback?: string) =>
    api.post('/prompts/templates/rate', { template_id: templateId, rating, feedback }),
};

// Scraping endpoints
export const scrapingAPI = {
  createJob: (data: any) => api.post('/scraping/jobs', data),
  getJob: (id: string) => api.get(`/scraping/jobs/${id}`),
  listJobs: () => api.get('/scraping/jobs'),
  cancelJob: (id: string) => api.post(`/scraping/jobs/${id}/cancel`),
};

export default api;


// Voice endpoints
export const voiceApi = {
  getVoiceProfiles: () => api.get('/voice/profiles').then(res => res.data),
  createVoiceProfile: (data: any) => api.post('/voice/profiles', data).then(res => res.data),
  updateVoiceProfile: (id: string, data: any) => api.put(`/voice/profiles/${id}`, data).then(res => res.data),
  deleteVoiceProfile: (id: string) => api.delete(`/voice/profiles/${id}`).then(res => res.data),
  getSupportedLanguages: () => api.get('/voice/languages').then(res => res.data),
  getSupportedVoices: (language?: string) => api.get('/voice/voices', { params: { language } }).then(res => res.data),
  speechToText: (audioBlob: Blob, language?: string) => {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    if (language) formData.append('language', language);
    return api.post('/voice/stt', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(res => res.data);
  },
  textToSpeech: (text: string, voice?: string) => {
    return api.post('/voice/tts', { text, voice }, {
      responseType: 'blob',
    }).then(res => res.data);
  },

  // Wake word detection
  startWakeWordDetection: (keywords: string[], sensitivities: number[]) => {
    return api.post('/voice/wake-word/start', { keywords, sensitivities }).then(res => res.data);
  },
  stopWakeWordDetection: () => {
    return api.post('/voice/wake-word/stop').then(res => res.data);
  },
  getWakeWordStatus: () => {
    return api.get('/voice/wake-word/status').then(res => res.data);
  },
};

// MCP endpoints
export const mcpApi = {
  getServers: () => api.get('/mcp/servers').then(res => res.data),
  createServer: (data: any) => api.post('/mcp/servers', data).then(res => res.data),
  updateServer: (id: string, data: any) => api.put(`/mcp/servers/${id}`, data).then(res => res.data),
  deleteServer: (id: string) => api.delete(`/mcp/servers/${id}`).then(res => res.data),
  testConnection: (id: string) => api.post(`/mcp/servers/${id}/test`).then(res => res.data),
  getTools: (serverId?: string) => api.get('/mcp/tools', { params: { serverId } }).then(res => res.data),
  executeTool: (toolId: string, parameters?: any) => api.post(`/mcp/tools/${toolId}/execute`, { parameters }).then(res => res.data),
};

// Hook endpoints
export const hookApi = {
  getHooks: () => api.get('/hooks/hooks').then(res => res.data),
  registerHook: (data: any) => api.post('/hooks/hooks', data).then(res => res.data),
  updateHook: (id: string, data: any) => api.put(`/hooks/hooks/${id}`, data).then(res => res.data),
  deleteHook: (id: string) => api.delete(`/hooks/hooks/${id}`).then(res => res.data),
  testHook: (id: string, payload?: any) => api.post(`/hooks/hooks/${id}/test`, { payload }).then(res => res.data),
  getPlugins: () => api.get('/hooks/plugins').then(res => res.data),
  getEventTypes: () => api.get('/hooks/events').then(res => res.data),
};
