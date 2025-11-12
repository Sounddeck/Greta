/**
 * OSINT Store - Zustand State Management
 * Manages OSINT tools, executions, results, and metrics
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { api } from '../lib/api';
import { toast } from 'sonner';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Extract error message from various error response formats
 */
function extractErrorMessage(error: any, defaultMessage: string): string {
  // Handle different error response formats
  if (error.response?.data) {
    const data = error.response.data;

    // Check for common error message fields
    if (typeof data === 'string') {
      return data;
    }

    if (data.message && typeof data.message === 'string') {
      return data.message;
    }

    if (data.error && typeof data.error === 'string') {
      return data.error;
    }

    if (data.detail && typeof data.detail === 'string') {
      return data.detail;
    }

    // If data is an object with message property that's also an object
    if (data.message && typeof data.message === 'object') {
      return JSON.stringify(data.message);
    }
  }

  // Fallback to error.message
  if (error.message && typeof error.message === 'string') {
    return error.message;
  }

  // Return default message
  return defaultMessage;
}

// ============================================================================
// Types
// ============================================================================

export interface OSINTTool {
  tool_id: string;
  name: string;
  description: string;
  category: string;
  permission_level: string;
  parameters: Record<string, any>;
  returns: Record<string, any>;
  rate_limit?: number;
  timeout?: number;
  tags: string[];
  metadata?: Record<string, any>;
  is_available: boolean;
}

export interface OSINTExecution {
  execution_id: string;
  tool_id: string;
  tool_name?: string;
  agent_id: string;
  parameters: Record<string, any>;
  result?: any;
  error?: string;
  success: boolean;
  execution_time: number;
  timestamp: string;
}

export interface OSINTMetrics {
  total_executions: number;
  successful_executions: number;
  failed_executions: number;
  avg_execution_time: number;
  tools_used: number;
  executions_by_tool: Record<string, number>;
  executions_by_category: Record<string, number>;
}

export interface ToolExecutionRequest {
  tool_id: string;
  parameters: Record<string, any>;
  agent_id?: string;
}

// ============================================================================
// Initial State
// ============================================================================

const initialState = {
  tools: [] as OSINTTool[],
  executions: [] as OSINTExecution[],
  metrics: null as OSINTMetrics | null,
  selectedTool: null as OSINTTool | null,
  isLoading: false,
  isExecuting: false,
  error: null as string | null,
  filterCategory: 'all' as string,
  searchQuery: '' as string,
};

// ============================================================================
// Store Interface
// ============================================================================

interface OSINTState {
  // State
  tools: OSINTTool[];
  executions: OSINTExecution[];
  metrics: OSINTMetrics | null;
  selectedTool: OSINTTool | null;
  isLoading: boolean;
  isExecuting: boolean;
  error: string | null;
  filterCategory: string;
  searchQuery: string;

  // Actions
  fetchTools: () => Promise<void>;
  fetchExecutions: (limit?: number) => Promise<void>;
  fetchMetrics: () => Promise<void>;
  executeTool: (request: ToolExecutionRequest) => Promise<OSINTExecution>;
  setSelectedTool: (tool: OSINTTool | null) => void;
  setFilterCategory: (category: string) => void;
  setSearchQuery: (query: string) => void;
  clearError: () => void;
  reset: () => void;
}

// ============================================================================
// Store Implementation
// ============================================================================

export const useOSINTStore = create<OSINTState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Fetch available OSINT tools
      fetchTools: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/tools/discover', {
            params: {
              category: 'osint',  // lowercase to match backend enum
              limit: 100
            }
          });

          const tools = response.data.tools || [];
          set({ tools, isLoading: false });
        } catch (error: any) {
          const errorMessage = extractErrorMessage(error, 'Failed to fetch OSINT tools');
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      // Fetch execution history
      fetchExecutions: async (limit = 50) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/tools/executions', {
            params: {
              category: 'osint',  // lowercase to match backend enum
              limit
            }
          });

          const executions = response.data.executions || [];
          set({ executions, isLoading: false });
        } catch (error: any) {
          const errorMessage = extractErrorMessage(error, 'Failed to fetch execution history');
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      // Fetch usage metrics
      fetchMetrics: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/tools/metrics', {
            params: {
              category: 'osint'  // lowercase to match backend enum
            }
          });

          const metrics = response.data.metrics || null;
          set({ metrics, isLoading: false });
        } catch (error: any) {
          const errorMessage = extractErrorMessage(error, 'Failed to fetch metrics');
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      // Execute a tool
      executeTool: async (request: ToolExecutionRequest) => {
        set({ isExecuting: true, error: null });
        try {
          const response = await api.post('/tools/execute', {
            tool_id: request.tool_id,
            parameters: request.parameters,
            agent_id: request.agent_id || 'osint_dashboard'
          });

          const execution: OSINTExecution = {
            execution_id: response.data.execution_id,
            tool_id: request.tool_id,
            tool_name: get().tools.find(t => t.tool_id === request.tool_id)?.name,
            agent_id: request.agent_id || 'osint_dashboard',
            parameters: request.parameters,
            result: response.data.result,
            error: response.data.error,
            success: response.data.success,
            execution_time: response.data.execution_time,
            timestamp: response.data.timestamp || new Date().toISOString()
          };

          // Add to executions list
          set((state) => ({
            executions: [execution, ...state.executions].slice(0, 100), // Keep last 100
            isExecuting: false
          }));

          if (execution.success) {
            toast.success(`Tool executed successfully in ${execution.execution_time.toFixed(2)}s`);
          } else {
            // Ensure error is a string
            const errorMsg = typeof execution.error === 'string'
              ? execution.error
              : JSON.stringify(execution.error || 'Unknown error');
            toast.error(`Tool execution failed: ${errorMsg}`);
          }

          // Refresh metrics
          get().fetchMetrics();

          return execution;
        } catch (error: any) {
          const errorMessage = extractErrorMessage(error, 'Tool execution failed');
          set({ error: errorMessage, isExecuting: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      // Set selected tool
      setSelectedTool: (tool) => {
        set({ selectedTool: tool });
      },

      // Set filter category
      setFilterCategory: (category) => {
        set({ filterCategory: category });
      },

      // Set search query
      setSearchQuery: (query) => {
        set({ searchQuery: query });
      },

      // Clear error
      clearError: () => set({ error: null }),

      // Reset store
      reset: () => set(initialState),
    }),
    {
      name: 'osint-storage',
      partialize: (state) => ({
        // Only persist selected tool and filters
        selectedTool: state.selectedTool,
        filterCategory: state.filterCategory,
      }),
    }
  )
);

