import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from 'sonner';
import { agentBuilderAPI } from '../lib/api';

// Types
export type AgentType = 'GENERAL' | 'RESEARCH' | 'CODING' | 'DATA_ANALYSIS' | 'CREATIVE' | 'CUSTOMER_SERVICE' | 'SALES' | 'MARKETING';
export type AgentStatus = 'DRAFT' | 'ACTIVE' | 'DEPLOYED' | 'ARCHIVED';
export type DeploymentStatus = 'PENDING' | 'DEPLOYING' | 'DEPLOYED' | 'FAILED' | 'ROLLED_BACK';
export type ExecutionStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface AgentTemplate {
  template_id: string;
  name: string;
  description: string;
  agent_type: AgentType;
  capabilities: string[];
  default_tools: string[];
  default_prompt: string;
  configuration: Record<string, any>;
  tags: string[];
  usage_count: number;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface CustomAgent {
  agent_id: string;
  name: string;
  description: string;
  agent_type: AgentType;
  capabilities: string[];
  system_prompt: string;
  configuration: Record<string, any>;
  tools: string[];
  tags: string[];
  status: AgentStatus;
  version: number;
  current_version_id?: string;
  template_id?: string;
  generated_code?: string;
  execution_count: number;
  success_count: number;
  failure_count: number;
  average_execution_time: number;
  last_executed?: string;
  created_at: string;
  updated_at: string;
  deployed_at?: string;
  created_by: string;
}

export interface AgentExecution {
  execution_id: string;
  agent_id: string;
  input_data: Record<string, any>;
  output_data?: Record<string, any>;
  status: ExecutionStatus;
  error_message?: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  tokens_used?: number;
  cost?: number;
  configuration?: Record<string, any>;
}

export interface Deployment {
  deployment_id: string;
  agent_id: string;
  version_id: string;
  environment: string;
  status: DeploymentStatus;
  configuration: Record<string, any>;
  deployed_at?: string;
  deployed_by: string;
  error_message?: string;
}

export interface AgentBuilderStats {
  total_templates: number;
  total_agents: number;
  total_deployments: number;
  total_executions: number;
  active_agents: number;
  deployed_agents: number;
  success_rate: number;
  average_execution_time: number;
}

interface AgentBuilderState {
  // Templates
  templates: AgentTemplate[];
  selectedTemplate: AgentTemplate | null;
  isLoadingTemplates: boolean;
  
  // Agents
  agents: CustomAgent[];
  selectedAgent: CustomAgent | null;
  isLoadingAgents: boolean;
  
  // Executions
  executions: AgentExecution[];
  isLoadingExecutions: boolean;
  
  // Deployments
  deployments: Deployment[];
  isLoadingDeployments: boolean;
  
  // Statistics
  stats: AgentBuilderStats | null;
  isLoadingStats: boolean;
  
  // Actions - Templates
  fetchTemplates: (filters?: { agent_type?: AgentType; tags?: string[]; limit?: number }) => Promise<void>;
  fetchTemplate: (id: string) => Promise<void>;
  createTemplate: (data: Partial<AgentTemplate>) => Promise<AgentTemplate | null>;
  updateTemplate: (id: string, data: Partial<AgentTemplate>) => Promise<void>;
  deleteTemplate: (id: string) => Promise<void>;
  setSelectedTemplate: (template: AgentTemplate | null) => void;
  
  // Actions - Agents
  fetchAgents: (filters?: { status?: AgentStatus; agent_type?: AgentType; tags?: string[]; limit?: number }) => Promise<void>;
  fetchAgent: (id: string) => Promise<void>;
  createAgent: (data: Partial<CustomAgent>) => Promise<CustomAgent | null>;
  updateAgent: (id: string, data: Partial<CustomAgent>) => Promise<void>;
  deleteAgent: (id: string) => Promise<void>;
  cloneAgent: (id: string, newName: string) => Promise<CustomAgent | null>;
  setSelectedAgent: (agent: CustomAgent | null) => void;
  
  // Actions - Executions
  fetchExecutions: (filters?: { agent_id?: string; status?: ExecutionStatus; limit?: number }) => Promise<void>;
  executeAgent: (agentId: string, inputData: Record<string, any>) => Promise<AgentExecution | null>;
  
  // Actions - Deployments
  fetchDeployments: (filters?: { agent_id?: string; status?: DeploymentStatus; limit?: number }) => Promise<void>;
  deployAgent: (agentId: string, environment: string, configuration?: Record<string, any>) => Promise<Deployment | null>;
  
  // Actions - Code Generation
  generateCode: (agentId: string, includeTools?: boolean, includeTests?: boolean) => Promise<string | null>;
  
  // Actions - Validation
  validateAgent: (agentId: string) => Promise<any>;
  
  // Actions - Statistics
  fetchStats: () => Promise<void>;
}

export const useAgentBuilderStore = create<AgentBuilderState>()(
  persist(
    (set, get) => ({
      // Initial state
      templates: [],
      selectedTemplate: null,
      isLoadingTemplates: false,
      
      agents: [],
      selectedAgent: null,
      isLoadingAgents: false,
      
      executions: [],
      isLoadingExecutions: false,
      
      deployments: [],
      isLoadingDeployments: false,
      
      stats: null,
      isLoadingStats: false,
      
      // Template actions
      fetchTemplates: async (filters) => {
        console.log('[AgentBuilderStore] 🚀 fetchTemplates START', { filters });
        set({ isLoadingTemplates: true });
        try {
          console.log('[AgentBuilderStore] 📡 Making API call to listTemplates...');
          const response = await agentBuilderAPI.listTemplates(filters);
          console.log('[AgentBuilderStore] ✅ API Response received:', {
            status: response.status,
            dataType: typeof response.data,
            isArray: Array.isArray(response.data),
            count: response.data?.length || 0,
            fullResponse: response,
            firstTemplate: response.data?.[0]
          });

          console.log('[AgentBuilderStore] 💾 Setting templates in store...');
          set({ templates: response.data, isLoadingTemplates: false });

          console.log('[AgentBuilderStore] ✅ Store updated successfully');

          // Verify store state after update
          const currentState = get();
          console.log('[AgentBuilderStore] 🔍 Current store state after update:', {
            templatesCount: currentState.templates?.length || 0,
            isLoadingTemplates: currentState.isLoadingTemplates,
            templates: currentState.templates
          });
        } catch (error: any) {
          console.error('[AgentBuilderStore] ❌ Failed to fetch templates:', {
            error,
            message: error.message,
            response: error.response,
            stack: error.stack
          });
          toast.error('Failed to load templates');
          set({ isLoadingTemplates: false });
        }
      },
      
      fetchTemplate: async (id) => {
        try {
          const response = await agentBuilderAPI.getTemplate(id);
          set({ selectedTemplate: response.data });
        } catch (error: any) {
          console.error('Failed to fetch template:', error);
          toast.error('Failed to load template');
        }
      },
      
      createTemplate: async (data) => {
        try {
          const response = await agentBuilderAPI.createTemplate(data);
          set((state) => ({ templates: [...state.templates, response.data] }));
          toast.success('Template created successfully');
          return response.data;
        } catch (error: any) {
          console.error('Failed to create template:', error);
          toast.error('Failed to create template');
          return null;
        }
      },
      
      updateTemplate: async (id, data) => {
        try {
          const response = await agentBuilderAPI.updateTemplate(id, data);
          set((state) => ({
            templates: state.templates.map((t) => (t.template_id === id ? response.data : t)),
            selectedTemplate: state.selectedTemplate?.template_id === id ? response.data : state.selectedTemplate,
          }));
          toast.success('Template updated successfully');
        } catch (error: any) {
          console.error('Failed to update template:', error);
          toast.error('Failed to update template');
        }
      },
      
      deleteTemplate: async (id) => {
        try {
          await agentBuilderAPI.deleteTemplate(id);
          set((state) => ({
            templates: state.templates.filter((t) => t.template_id !== id),
            selectedTemplate: state.selectedTemplate?.template_id === id ? null : state.selectedTemplate,
          }));
          toast.success('Template deleted successfully');
        } catch (error: any) {
          console.error('Failed to delete template:', error);
          toast.error('Failed to delete template');
        }
      },
      
      setSelectedTemplate: (template) => set({ selectedTemplate: template }),
      
      // Agent actions
      fetchAgents: async (filters) => {
        console.log('[AgentBuilderStore] 🚀 fetchAgents START', { filters });
        set({ isLoadingAgents: true });
        try {
          console.log('[AgentBuilderStore] 📡 Making API call to listAgents...');
          const response = await agentBuilderAPI.listAgents(filters);
          console.log('[AgentBuilderStore] ✅ API Response received:', {
            status: response.status,
            dataType: typeof response.data,
            isArray: Array.isArray(response.data),
            count: response.data?.length || 0,
            fullResponse: response,
            firstAgent: response.data?.[0]
          });

          console.log('[AgentBuilderStore] 💾 Setting agents in store...');
          set({ agents: response.data, isLoadingAgents: false });

          console.log('[AgentBuilderStore] ✅ Store updated successfully');

          // Verify store state after update
          const currentState = get();
          console.log('[AgentBuilderStore] 🔍 Current store state after update:', {
            agentsCount: currentState.agents?.length || 0,
            isLoadingAgents: currentState.isLoadingAgents,
            agents: currentState.agents
          });
        } catch (error: any) {
          console.error('[AgentBuilderStore] ❌ Failed to fetch agents:', {
            error,
            message: error.message,
            response: error.response,
            stack: error.stack
          });
          toast.error('Failed to load agents');
          set({ isLoadingAgents: false });
        }
      },
      
      fetchAgent: async (id) => {
        try {
          const response = await agentBuilderAPI.getAgent(id);
          set({ selectedAgent: response.data });
        } catch (error: any) {
          console.error('Failed to fetch agent:', error);
          toast.error('Failed to load agent');
        }
      },
      
      createAgent: async (data) => {
        try {
          const response = await agentBuilderAPI.createAgent(data);
          set((state) => ({ agents: [...state.agents, response.data] }));
          toast.success('Agent created successfully');
          return response.data;
        } catch (error: any) {
          console.error('Failed to create agent:', error);
          toast.error('Failed to create agent');
          return null;
        }
      },
      
      updateAgent: async (id, data) => {
        try {
          const response = await agentBuilderAPI.updateAgent(id, data);
          set((state) => ({
            agents: state.agents.map((a) => (a.agent_id === id ? response.data : a)),
            selectedAgent: state.selectedAgent?.agent_id === id ? response.data : state.selectedAgent,
          }));
          toast.success('Agent updated successfully');
        } catch (error: any) {
          console.error('Failed to update agent:', error);
          toast.error('Failed to update agent');
        }
      },
      
      deleteAgent: async (id) => {
        try {
          await agentBuilderAPI.deleteAgent(id);
          set((state) => ({
            agents: state.agents.filter((a) => a.agent_id !== id),
            selectedAgent: state.selectedAgent?.agent_id === id ? null : state.selectedAgent,
          }));
          toast.success('Agent deleted successfully');
        } catch (error: any) {
          console.error('Failed to delete agent:', error);
          toast.error('Failed to delete agent');
        }
      },
      
      cloneAgent: async (id, newName) => {
        try {
          const response = await agentBuilderAPI.cloneAgent(id, newName);
          set((state) => ({ agents: [...state.agents, response.data] }));
          toast.success(`Agent cloned as "${newName}"`);
          return response.data;
        } catch (error: any) {
          console.error('Failed to clone agent:', error);
          toast.error('Failed to clone agent');
          return null;
        }
      },
      
      setSelectedAgent: (agent) => set({ selectedAgent: agent }),
      
      // Execution actions
      fetchExecutions: async (filters) => {
        set({ isLoadingExecutions: true });
        try {
          const response = await agentBuilderAPI.listExecutions(filters);
          set({ executions: response.data, isLoadingExecutions: false });
        } catch (error: any) {
          console.error('Failed to fetch executions:', error);
          toast.error('Failed to load executions');
          set({ isLoadingExecutions: false });
        }
      },
      
      executeAgent: async (agentId, inputData) => {
        try {
          const response = await agentBuilderAPI.executeAgent({ agent_id: agentId, input_data: inputData });
          set((state) => ({ executions: [response.data, ...state.executions] }));
          toast.success('Agent execution started');
          return response.data;
        } catch (error: any) {
          console.error('Failed to execute agent:', error);
          toast.error('Failed to execute agent');
          return null;
        }
      },
      
      // Deployment actions
      fetchDeployments: async (filters) => {
        set({ isLoadingDeployments: true });
        try {
          const response = await agentBuilderAPI.listDeployments(filters);
          set({ deployments: response.data, isLoadingDeployments: false });
        } catch (error: any) {
          console.error('Failed to fetch deployments:', error);
          toast.error('Failed to load deployments');
          set({ isLoadingDeployments: false });
        }
      },
      
      deployAgent: async (agentId, environment, configuration) => {
        try {
          const response = await agentBuilderAPI.deployAgent({ agent_id: agentId, environment, configuration });
          set((state) => ({ deployments: [response.data, ...state.deployments] }));
          toast.success(`Agent deployed to ${environment}`);
          return response.data;
        } catch (error: any) {
          console.error('Failed to deploy agent:', error);
          toast.error('Failed to deploy agent');
          return null;
        }
      },
      
      // Code generation
      generateCode: async (agentId, includeTools = true, includeTests = false) => {
        try {
          const response = await agentBuilderAPI.generateCode(agentId, includeTools, includeTests);
          toast.success('Code generated successfully');
          return response.data.generated_code;
        } catch (error: any) {
          console.error('Failed to generate code:', error);
          toast.error('Failed to generate code');
          return null;
        }
      },
      
      // Validation
      validateAgent: async (agentId) => {
        try {
          const response = await agentBuilderAPI.validateAgent(agentId);
          if (response.data.is_valid) {
            toast.success('Agent configuration is valid');
          } else {
            toast.warning(`Validation found ${response.data.errors?.length || 0} errors`);
          }
          return response.data;
        } catch (error: any) {
          console.error('Failed to validate agent:', error);
          toast.error('Failed to validate agent');
          return null;
        }
      },
      
      // Statistics
      fetchStats: async () => {
        set({ isLoadingStats: true });
        try {
          const response = await agentBuilderAPI.getStats();
          set({ stats: response.data, isLoadingStats: false });
        } catch (error: any) {
          console.error('Failed to fetch stats:', error);
          toast.error('Failed to load statistics');
          set({ isLoadingStats: false });
        }
      },
    }),
    {
      name: 'agent-builder-storage',
      partialize: (state) => ({
        // Only persist selected items, not full lists
        selectedTemplate: state.selectedTemplate,
        selectedAgent: state.selectedAgent,
      }),
    }
  )
);

