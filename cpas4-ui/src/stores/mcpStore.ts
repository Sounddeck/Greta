/**
 * MCP Store - Zustand State Management
 * Manages MCP servers, tools, and execution state
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { mcpApi } from '../lib/api';
import { toast } from 'sonner';

export interface MCPServer {
  id: string;
  name: string;
  url: string;
  description?: string;
  status: 'connected' | 'disconnected' | 'error';
  isEnabled: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface MCPTool {
  id: string;
  serverId: string;
  name: string;
  description?: string;
  parameters?: Record<string, any>;
  isAvailable: boolean;
}

export interface MCPState {
  // State
  servers: MCPServer[];
  tools: MCPTool[];
  selectedServer: MCPServer | null;
  isLoading: boolean;
  isExecuting: boolean;
  error: string | null;

  // Actions
  fetchServers: () => Promise<void>;
  fetchTools: (serverId?: string) => Promise<void>;
  createServer: (server: Omit<MCPServer, 'id' | 'status' | 'createdAt' | 'updatedAt'>) => Promise<void>;
  updateServer: (id: string, updates: Partial<MCPServer>) => Promise<void>;
  deleteServer: (id: string) => Promise<void>;
  testConnection: (id: string) => Promise<boolean>;
  executeTool: (toolId: string, parameters?: Record<string, any>) => Promise<any>;
  setSelectedServer: (server: MCPServer | null) => void;
  clearError: () => void;
  reset: () => void;
}

const initialState = {
  servers: [],
  tools: [],
  selectedServer: null,
  isLoading: false,
  isExecuting: false,
  error: null,
};

export const useMCPStore = create<MCPState>()(
  persist(
    (set, get) => ({
      ...initialState,

      fetchServers: async () => {
        set({ isLoading: true, error: null });
        try {
          const servers = await mcpApi.getServers();
          set({ servers, isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to fetch MCP servers';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      fetchTools: async (serverId) => {
        set({ isLoading: true, error: null });
        try {
          const tools = await mcpApi.getTools(serverId);
          set({ tools, isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to fetch MCP tools';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      createServer: async (serverData) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
          const tempId = `temp-${Date.now()}`;
          const optimisticServer: MCPServer = {
            ...serverData,
            id: tempId,
            status: 'disconnected',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          };
          set((state) => ({ servers: [...state.servers, optimisticServer] }));

          // API call
          const newServer = await mcpApi.createServer(serverData);
          
          // Replace optimistic update with real data
          set((state) => ({
            servers: state.servers.map((s) => (s.id === tempId ? newServer : s)),
            isLoading: false,
          }));
          
          toast.success('MCP server created successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set((state) => ({
            servers: state.servers.filter((s) => !s.id.startsWith('temp-')),
            error: error.response?.data?.message || 'Failed to create server',
            isLoading: false,
          }));
          toast.error(error.response?.data?.message || 'Failed to create server');
        }
      },

      updateServer: async (id, updates) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
// //           const previousServers = get().servers;
          set((state) => ({
            servers: state.servers.map((s) =>
              s.id === id ? { ...s, ...updates, updatedAt: new Date().toISOString() } : s
            ),
          }));

          // API call
          const updatedServer = await mcpApi.updateServer(id, updates);
          
          // Update with real data
          set((state) => ({
            servers: state.servers.map((s) => (s.id === id ? updatedServer : s)),
            isLoading: false,
          }));
          
          toast.success('MCP server updated successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set({ servers: get().servers, error: error.response?.data?.message || 'Failed to update server', isLoading: false });
          toast.error(error.response?.data?.message || 'Failed to update server');
        }
      },

      deleteServer: async (id) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
// //           const previousServers = get().servers;
          set((state) => ({
            servers: state.servers.filter((s) => s.id !== id),
          }));

          // API call
          await mcpApi.deleteServer(id);
          
          set({ isLoading: false });
          toast.success('MCP server deleted successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set({ servers: get().servers, error: error.response?.data?.message || 'Failed to delete server', isLoading: false });
          toast.error(error.response?.data?.message || 'Failed to delete server');
        }
      },

      testConnection: async (id) => {
        set({ isLoading: true, error: null });
        try {
          const result = await mcpApi.testConnection(id);
          
          // Update server status
          set((state) => ({
            servers: state.servers.map((s) =>
              s.id === id ? { ...s, status: result.success ? 'connected' : 'error' } : s
            ),
            isLoading: false,
          }));
          
          if (result.success) {
            toast.success('Connection successful');
          } else {
            toast.error('Connection failed');
          }
          
          return result.success;
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Connection test failed';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
          return false;
        }
      },

      executeTool: async (toolId, parameters) => {
        set({ isExecuting: true, error: null });
        try {
          const result = await mcpApi.executeTool(toolId, parameters);
          set({ isExecuting: false });
          toast.success('Tool executed successfully');
          return result;
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Tool execution failed';
          set({ error: errorMessage, isExecuting: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      setSelectedServer: (server) => {
        set({ selectedServer: server });
      },

      clearError: () => set({ error: null }),

      reset: () => set(initialState),
    }),
    {
      name: 'mcp-storage',
      partialize: (state) => ({
        servers: state.servers,
        selectedServer: state.selectedServer,
      }),
    }
  )
);
