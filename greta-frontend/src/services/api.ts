// API service for connecting to Greta backend
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  public async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      const data = await response.json();

      return {
        data: response.ok ? data : undefined,
        error: response.ok ? undefined : data.detail || 'An error occurred',
        status: response.status,
      };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Network error',
        status: 0,
      };
    }
  }

  // System endpoints
  async getSystemStatus() {
    return this.request('/');
  }

  async getSystemHealth() {
    return this.request('/api/v1/system/health');
  }

  // LLM endpoints
  async getAvailableModels() {
    return this.request('/api/v1/llm/models');
  }

  async sendChatMessage(message: string, model?: string) {
    return this.request('/api/v1/llm/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        model: model || 'greta-enhanced',
        max_tokens: 2000,
        temperature: 0.7,
      }),
    });
  }

  async generateCompletion(prompt: string, model?: string) {
    return this.request('/api/v1/llm/completion', {
      method: 'POST',
      body: JSON.stringify({
        prompt,
        model: model || 'greta-enhanced',
        max_tokens: 1000,
        temperature: 0.7,
      }),
    });
  }

  // Agent endpoints
  async getAgents() {
    return this.request('/api/v1/agents');
  }

  async getAgentStatus(agentId: string) {
    return this.request(`/api/v1/agents/${agentId}/status`);
  }

  async startAgent(agentId: string) {
    return this.request(`/api/v1/agents/${agentId}/start`, {
      method: 'POST',
    });
  }

  async stopAgent(agentId: string) {
    return this.request(`/api/v1/agents/${agentId}/stop`, {
      method: 'POST',
    });
  }

  // Master Agent endpoints
  async getMasterAgentStatus() {
    return this.request('/api/v1/master-agent/status');
  }

  async executeMasterTask(task: string, parameters?: any) {
    return this.request('/api/v1/master-agent/execute', {
      method: 'POST',
      body: JSON.stringify({
        task,
        parameters: parameters || {},
      }),
    });
  }

  async createAgent(spec: any) {
    return this.request('/api/v1/master-agent/agent/create', {
      method: 'POST',
      body: JSON.stringify({ spec }),
    });
  }

  async manageAgent(action: string, agentName: string, parameters?: any) {
    return this.request('/api/v1/master-agent/agent/manage', {
      method: 'POST',
      body: JSON.stringify({
        action,
        agent_name: agentName,
        parameters: parameters || {},
      }),
    });
  }

  // Workflow endpoints
  async getWorkflows() {
    return this.request('/api/v1/workflows');
  }

  async executeWorkflow(workflowId: string, parameters?: any) {
    return this.request(`/api/v1/workflows/${workflowId}/execute`, {
      method: 'POST',
      body: JSON.stringify({ parameters: parameters || {} }),
    });
  }

  // Memory endpoints
  async getMemoryStatus() {
    return this.request('/api/v1/memory/status');
  }

  async searchMemory(query: string) {
    return this.request('/api/v1/memory/search', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }

  // Voice endpoints
  async synthesizeVoice(text: string, voice?: string) {
    return this.request('/api/v1/voice/synthesize', {
      method: 'POST',
      body: JSON.stringify({
        text,
        voice_model: voice || 'en_US-lessac-medium',
        speed: 1.0,
        caching: true
      }),
    });
  }

  // Training endpoints
  async getTrainingLessons() {
    return this.request('/api/training/lessons');
  }

  async startLesson(lessonId: string) {
    return this.request(`/api/training/lesson/${lessonId}/start`, {
      method: 'POST',
    });
  }

  // Web commands (for existing web UI compatibility)
  async executeWebCommand(command: string, context?: string) {
    return this.request('/api/web-commands/execute-command', {
      method: 'POST',
      body: JSON.stringify({
        command,
        context: context || '',
        agent_type: 'auto',
        priority: 'normal',
      }),
    });
  }

  async submitPrompt(prompt: string, agent?: string) {
    return this.request('/api/web-commands/submit-prompt', {
      method: 'POST',
      body: JSON.stringify({
        prompt,
        agent: agent || 'auto',
        return_format: 'formatted',
      }),
    });
  }

  // WebSocket connection for real-time updates
  createWebSocket(endpoint: string): WebSocket | null {
    try {
      const wsUrl = this.baseUrl.replace('http', 'ws') + endpoint;
      return new WebSocket(wsUrl);
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      return null;
    }
  }
}

// Create and export a singleton instance
export const apiService = new ApiService();

// Export types for use in components
export type { ApiResponse };

// Helper function to handle API responses in React Query
export const handleApiResponse = <T>(response: ApiResponse<T>) => {
  if (response.error) {
    throw new Error(response.error);
  }
  return response.data;
};

// Mock data generators for development (when backend is not available)
export const mockApiService = {
  getSystemStatus: () => Promise.resolve({
    data: {
      message: "GRETA4: The World's Most Advanced Personal AI System",
      version: "2.2.0",
      status: "running",
      capabilities: [
        "🧠 Enhanced Learning System with Auto Fine-tuning",
        "🦙 Local llama.cpp Processing (No Cloud Dependencies)",
        "🎭 Multi-Agent Coordination System",
        "🔧 Real-time Performance Optimization"
      ]
    },
    status: 200
  }),

  getAvailableModels: () => Promise.resolve({
    data: {
      models: [
        { name: 'greta-enhanced', displayName: 'GRETA Enhanced', description: 'Optimized for personal assistance' },
        { name: 'llama3-local', displayName: 'Llama 3 Local', description: 'Local processing, privacy-first' },
        { name: 'mock-model', displayName: 'Mock Model', description: 'For testing purposes' }
      ]
    },
    status: 200
  }),

  sendChatMessage: (message: string) => Promise.resolve({
    data: {
      response: `I understand you said: "${message}". As GRETA, your Personal AI Assistant, I'm here to help you with any questions or tasks. How can I assist you further?`,
      model: 'greta-enhanced',
      processing_time: 1.2
    },
    status: 200
  })
};

export default apiService;
