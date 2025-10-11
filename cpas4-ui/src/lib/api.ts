import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Agent endpoints
export const agentAPI = {
  list: () => api.get('/agents'),
  get: (id: string) => api.get(`/agents/${id}`),
  create: (data: any) => api.post('/agents', data),
  delete: (id: string) => api.delete(`/agents/${id}`),
  update: (id: string, data: any) => api.put(`/agents/${id}`, data),
};

// Task endpoints
export const taskAPI = {
  execute: (data: any) => api.post('/tasks/execute', data),
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
  enhance: (data: any) => api.post('/prompts/enhance', data),
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
