/**
 * CPAS4 API Service
 * Handles communication with CPAS4 backend routers (tasks, prompts, knowledge)
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}

// ============================================================================
// CPAS4 Task Types
// ============================================================================

export interface TaskExecuteRequest {
  agent_id?: string;
  description: string;
  input_data?: Record<string, any>;
  max_iterations?: number;
}

export interface TaskResponse {
  id: string;
  status: string;
  result?: string;
  output?: string;
  error?: string;
  created_at: string;
  completed_at?: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: string;
  progress?: number;
  current_step?: string;
  result?: any;
  error?: string;
}

// ============================================================================
// CPAS4 Prompt Types
// ============================================================================

export interface PromptEnhanceRequest {
  prompt: string;
  context?: string;
  enhancement_type?: 'clarity' | 'specificity' | 'structure' | 'all';
}

export interface PromptEnhanceResponse {
  original_prompt: string;
  enhanced_prompt: string;
  improvements: string[];
  confidence_score: number;
}

export interface PromptTemplateRequest {
  category: string;
  use_case: string;
  variables?: Record<string, string>;
}

export interface PromptTemplate {
  id: string;
  name: string;
  category: string;
  template: string;
  variables: string[];
  description: string;
}

// ============================================================================
// CPAS4 Knowledge Types
// ============================================================================

export interface KnowledgeQueryRequest {
  query: string;
  top_k?: number;
  filters?: Record<string, any>;
}

export interface KnowledgeQueryResponse {
  results: KnowledgeItem[];
  total_results: number;
  query_time: number;
}

export interface KnowledgeItem {
  id: string;
  content: string;
  metadata: Record<string, any>;
  relevance_score: number;
  source?: string;
}

export interface KnowledgeAddRequest {
  content: string;
  metadata?: Record<string, any>;
  source?: string;
  category?: string;
}

// ============================================================================
// CPAS4 API Service Class
// ============================================================================

class CPAS4ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
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

  // ==========================================================================
  // Tasks API
  // ==========================================================================

  /**
   * Execute a task using CPAS4
   */
  async executeTask(request: TaskExecuteRequest): Promise<ApiResponse<TaskResponse>> {
    return this.request<TaskResponse>('/api/v1/tasks/execute', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get task status
   */
  async getTaskStatus(taskId: string): Promise<ApiResponse<TaskStatusResponse>> {
    return this.request<TaskStatusResponse>(`/api/v1/tasks/${taskId}/status`);
  }

  /**
   * Cancel a running task
   */
  async cancelTask(taskId: string): Promise<ApiResponse<{ success: boolean }>> {
    return this.request(`/api/v1/tasks/${taskId}/cancel`, {
      method: 'POST',
    });
  }

  /**
   * Get task history
   */
  async getTaskHistory(limit: number = 10): Promise<ApiResponse<TaskResponse[]>> {
    return this.request<TaskResponse[]>(`/api/v1/tasks/history?limit=${limit}`);
  }

  // ==========================================================================
  // Prompts API
  // ==========================================================================

  /**
   * Enhance a prompt using AI
   */
  async enhancePrompt(request: PromptEnhanceRequest): Promise<ApiResponse<PromptEnhanceResponse>> {
    return this.request<PromptEnhanceResponse>('/api/v1/prompts/enhance', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get prompt templates
   */
  async getPromptTemplates(category?: string): Promise<ApiResponse<PromptTemplate[]>> {
    const endpoint = category 
      ? `/api/v1/prompts/templates?category=${category}`
      : '/api/v1/prompts/templates';
    return this.request<PromptTemplate[]>(endpoint);
  }

  /**
   * Get a specific prompt template
   */
  async getPromptTemplate(templateId: string): Promise<ApiResponse<PromptTemplate>> {
    return this.request<PromptTemplate>(`/api/v1/prompts/templates/${templateId}`);
  }

  /**
   * Generate prompt from template
   */
  async generateFromTemplate(request: PromptTemplateRequest): Promise<ApiResponse<{ prompt: string }>> {
    return this.request<{ prompt: string }>('/api/v1/prompts/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Analyze prompt quality
   */
  async analyzePrompt(prompt: string): Promise<ApiResponse<{
    quality_score: number;
    suggestions: string[];
    strengths: string[];
    weaknesses: string[];
  }>> {
    return this.request('/api/v1/prompts/analyze', {
      method: 'POST',
      body: JSON.stringify({ prompt }),
    });
  }

  // ==========================================================================
  // Knowledge API
  // ==========================================================================

  /**
   * Query the knowledge base
   */
  async queryKnowledge(request: KnowledgeQueryRequest): Promise<ApiResponse<KnowledgeQueryResponse>> {
    return this.request<KnowledgeQueryResponse>('/api/v1/knowledge/query', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Add knowledge to the knowledge base
   */
  async addKnowledge(request: KnowledgeAddRequest): Promise<ApiResponse<{ id: string; success: boolean }>> {
    return this.request('/api/v1/knowledge/add', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get knowledge item by ID
   */
  async getKnowledgeItem(itemId: string): Promise<ApiResponse<KnowledgeItem>> {
    return this.request<KnowledgeItem>(`/api/v1/knowledge/${itemId}`);
  }

  /**
   * Update knowledge item
   */
  async updateKnowledgeItem(
    itemId: string,
    updates: Partial<KnowledgeItem>
  ): Promise<ApiResponse<{ success: boolean }>> {
    return this.request(`/api/v1/knowledge/${itemId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  /**
   * Delete knowledge item
   */
  async deleteKnowledgeItem(itemId: string): Promise<ApiResponse<{ success: boolean }>> {
    return this.request(`/api/v1/knowledge/${itemId}`, {
      method: 'DELETE',
    });
  }

  /**
   * Get knowledge statistics
   */
  async getKnowledgeStats(): Promise<ApiResponse<{
    total_items: number;
    categories: Record<string, number>;
    recent_additions: number;
  }>> {
    return this.request('/api/v1/knowledge/stats');
  }

  // ==========================================================================
  // Health Check
  // ==========================================================================

  /**
   * Check CPAS4 service health
   */
  async healthCheck(): Promise<ApiResponse<{
    status: string;
    services: {
      tasks: boolean;
      prompts: boolean;
      knowledge: boolean;
    };
  }>> {
    return this.request('/api/v1/health');
  }
}

// Create and export singleton instance
export const cpas4Api = new CPAS4ApiService();

// Helper function to handle API responses
export const handleCPAS4Response = <T>(response: ApiResponse<T>): T => {
  if (response.error) {
    throw new Error(response.error);
  }
  if (!response.data) {
    throw new Error('No data received from API');
  }
  return response.data;
};

export default cpas4Api;
