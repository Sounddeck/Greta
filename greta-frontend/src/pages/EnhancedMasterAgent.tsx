import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Crown, 
  Play, 
  Pause, 
  Square, 
  Settings, 
  Plus,
  Trash2,
  Activity,
  Users,
  Zap,
  CheckCircle,
  Clock,
  Brain,
  Target,
  BarChart3,
  Cpu,
  HardDrive,
  TrendingUp,
  Workflow
} from 'lucide-react';
import { apiService } from '../services/api';

interface Agent {
  agent_id: string;
  name: string;
  type: string;
  status: string;
  capabilities: string[];
  llm_model: string;
  created_at: string;
  deployed_at?: string;
}

interface SystemStatus {
  success: boolean;
  status: string;
  system_resources: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_gb: number;
    disk_percent: number;
  };
  agents: {
    total: number;
    running: number;
    stopped: number;
    deployed: number;
  };
  modules: {
    total: number;
    active: number;
    ready: number;
  };
  workflows: {
    active: number;
    completed: number;
  };
  metrics: {
    total_requests: number;
    successful_operations: number;
    failed_operations: number;
    agents_created: number;
    workflows_executed: number;
    llm_interactions: number;
    learning_cycles: number;
    average_response_time: number;
  };
}

interface Workflow {
  workflow_id: string;
  name: string;
  description: string;
  status: string;
  created_at: string;
  completed_at?: string;
  results: any[];
}

const EnhancedMasterAgent: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [showCreateAgent, setShowCreateAgent] = useState(false);
  const [showCreateWorkflow, setShowCreateWorkflow] = useState(false);
  const [newAgentConfig, setNewAgentConfig] = useState({
    name: '',
    type: 'research',
    llm_model: 'llama-3.1-8b'
  });
  const [llmQuery, setLlmQuery] = useState('');

  const queryClient = useQueryClient();

  // Fetch system status
  const { data: systemStatus, isLoading: statusLoading } = useQuery<SystemStatus>({
    queryKey: ['master-agent-status'],
    queryFn: async (): Promise<SystemStatus> => {
      try {
        const response = await apiService.request<SystemStatus>('/api/v1/master/status', { method: 'GET' });
        return response.data || {
          success: false,
          status: 'unknown',
          system_resources: { cpu_percent: 0, memory_percent: 0, memory_available_gb: 0, disk_percent: 0 },
          agents: { total: 0, running: 0, stopped: 0, deployed: 0 },
          modules: { total: 12, active: 1, ready: 11 },
          workflows: { active: 0, completed: 0 },
          metrics: {
            total_requests: 0,
            successful_operations: 0,
            failed_operations: 0,
            agents_created: 0,
            workflows_executed: 0,
            llm_interactions: 0,
            learning_cycles: 0,
            average_response_time: 0
          }
        };
      } catch (error) {
        return {
          success: false,
          status: 'error',
          system_resources: { cpu_percent: 0, memory_percent: 0, memory_available_gb: 0, disk_percent: 0 },
          agents: { total: 0, running: 0, stopped: 0, deployed: 0 },
          modules: { total: 12, active: 1, ready: 11 },
          workflows: { active: 0, completed: 0 },
          metrics: {
            total_requests: 0,
            successful_operations: 0,
            failed_operations: 0,
            agents_created: 0,
            workflows_executed: 0,
            llm_interactions: 0,
            learning_cycles: 0,
            average_response_time: 0
          }
        };
      }
    },
    refetchInterval: 5000
  });

  // Fetch agents
  const { data: agentsData, isLoading: agentsLoading } = useQuery<{ success: boolean; agents: Agent[]; count: number }>({
    queryKey: ['master-agents'],
    queryFn: async (): Promise<{ success: boolean; agents: Agent[]; count: number }> => {
      try {
        const response = await apiService.request<{ success: boolean; agents: Agent[]; count: number }>('/api/v1/master/agents', { method: 'GET' });
        return response.data || { success: false, agents: [], count: 0 };
      } catch (error) {
        return { success: false, agents: [], count: 0 };
      }
    },
    refetchInterval: 10000
  });

  // Fetch workflows
  const { data: workflowsData, isLoading: workflowsLoading } = useQuery<{ success: boolean; workflows: Workflow[]; count: number }>({
    queryKey: ['master-workflows'],
    queryFn: async (): Promise<{ success: boolean; workflows: Workflow[]; count: number }> => {
      try {
        const response = await apiService.request<{ success: boolean; workflows: Workflow[]; count: number }>('/api/v1/master/workflows', { method: 'GET' });
        return response.data || { success: false, workflows: [], count: 0 };
      } catch (error) {
        return { success: false, workflows: [], count: 0 };
      }
    },
    refetchInterval: 10000
  });

  // Build agent mutation
  const buildAgentMutation = useMutation({
    mutationFn: async (config: typeof newAgentConfig) => {
      const response = await apiService.request('/api/v1/master/agent/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['master-agents'] });
      queryClient.invalidateQueries({ queryKey: ['master-agent-status'] });
      setShowCreateAgent(false);
      setNewAgentConfig({ name: '', type: 'research', llm_model: 'llama-3.1-8b' });
    }
  });

  // Deploy agent mutation
  const deployAgentMutation = useMutation({
    mutationFn: async (agentId: string) => {
      const response = await apiService.request('/api/v1/master/agent/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId, deployment_config: {} })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['master-agents'] });
      queryClient.invalidateQueries({ queryKey: ['master-agent-status'] });
    }
  });

  // Manage agent mutation
  const manageAgentMutation = useMutation({
    mutationFn: async ({ agentId, action }: { agentId: string; action: string }) => {
      const response = await apiService.request('/api/v1/master/agent/manage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId, action })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['master-agents'] });
      queryClient.invalidateQueries({ queryKey: ['master-agent-status'] });
    }
  });

  // Query LLM mutation
  const queryLLMMutation = useMutation<{ result?: { response?: string } }, Error, string>({
    mutationFn: async (query: string): Promise<{ result?: { response?: string } }> => {
      const response = await apiService.request('/api/v1/master/llm/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context: {} })
      });
      return response.data as { result?: { response?: string } };
    }
  });

  // Optimize system mutation
  const optimizeSystemMutation = useMutation({
    mutationFn: async (type: string) => {
      const response = await apiService.request('/api/v1/master/system/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['master-agent-status'] });
    }
  });

  const handleCreateAgent = () => {
    if (newAgentConfig.name && newAgentConfig.type) {
      buildAgentMutation.mutate(newAgentConfig);
    }
  };

  const handleDeployAgent = (agentId: string) => {
    deployAgentMutation.mutate(agentId);
  };

  const handleManageAgent = (agentId: string, action: string) => {
    manageAgentMutation.mutate({ agentId, action });
  };

  const handleQueryLLM = () => {
    if (llmQuery.trim()) {
      queryLLMMutation.mutate(llmQuery);
    }
  };

  const handleOptimizeSystem = (type: string) => {
    optimizeSystemMutation.mutate(type);
  };

  const agents = agentsData?.agents || [];
  const workflows = workflowsData?.workflows || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8 shadow-2xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-blue-500/20 p-4 rounded-xl backdrop-blur-sm">
                <Crown className="w-12 h-12 text-white" />
              </div>
              <div>
                <h1 className="text-4xl font-bold text-white mb-2">Enhanced Master Agent</h1>
                <p className="text-purple-100">Complete System Controller - Controls All Greta Modules</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`px-4 py-2 rounded-lg ${
                systemStatus?.status === 'active' ? 'bg-green-500' : 'bg-yellow-500'
              }`}>
                <span className="text-white font-semibold">
                  {systemStatus?.status?.toUpperCase() || 'UNKNOWN'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* System Resources */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Cpu className="w-8 h-8 text-blue-400" />
              <span className="text-2xl font-bold text-white">
                {systemStatus?.system_resources?.cpu_percent?.toFixed(1) || 0}%
              </span>
            </div>
            <p className="text-gray-400">CPU Usage</p>
          </div>

          <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <HardDrive className="w-8 h-8 text-green-400" />
              <span className="text-2xl font-bold text-white">
                {systemStatus?.system_resources?.memory_percent?.toFixed(1) || 0}%
              </span>
            </div>
            <p className="text-gray-400">Memory Usage</p>
          </div>

          <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Users className="w-8 h-8 text-purple-400" />
              <span className="text-2xl font-bold text-white">
                {systemStatus?.agents?.total || 0}
              </span>
            </div>
            <p className="text-gray-400">Total Agents</p>
          </div>

          <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Workflow className="w-8 h-8 text-pink-400" />
              <span className="text-2xl font-bold text-white">
                {systemStatus?.workflows?.completed || 0}
              </span>
            </div>
            <p className="text-gray-400">Workflows Completed</p>
          </div>
        </div>

        {/* System Metrics */}
        <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-blue-400" />
            System Metrics
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-800 hover:bg-blue-900 rounded-lg p-4">
              <p className="text-gray-400 text-sm mb-1">Total Requests</p>
              <p className="text-2xl font-bold text-white">{systemStatus?.metrics?.total_requests || 0}</p>
            </div>
            <div className="bg-blue-800 hover:bg-blue-900 rounded-lg p-4">
              <p className="text-gray-400 text-sm mb-1">Success Rate</p>
              <p className="text-2xl font-bold text-green-400">
                {systemStatus?.metrics?.total_requests
                  ? ((systemStatus.metrics.successful_operations / systemStatus.metrics.total_requests) * 100).toFixed(1)
                  : 0}%
              </p>
            </div>
            <div className="bg-blue-800 hover:bg-blue-900 rounded-lg p-4">
              <p className="text-gray-400 text-sm mb-1">LLM Interactions</p>
              <p className="text-2xl font-bold text-purple-400">{systemStatus?.metrics?.llm_interactions || 0}</p>
            </div>
            <div className="bg-blue-800 hover:bg-blue-900 rounded-lg p-4">
              <p className="text-gray-400 text-sm mb-1">Learning Cycles</p>
              <p className="text-2xl font-bold text-pink-400">{systemStatus?.metrics?.learning_cycles || 0}</p>
            </div>
          </div>
        </div>

        {/* Agents Management */}
        <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white flex items-center">
              <Users className="w-6 h-6 mr-2 text-purple-400" />
              Agents ({agents.length})
            </h2>
            <button
              onClick={() => setShowCreateAgent(true)}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
            >
              <Plus className="w-5 h-5" />
              <span>Create Agent</span>
            </button>
          </div>

          {agentsLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
              <p className="text-gray-400 mt-4">Loading agents...</p>
            </div>
          ) : agents.length === 0 ? (
            <div className="text-center py-8">
              <Users className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No agents created yet</p>
              <button
                onClick={() => setShowCreateAgent(true)}
                className="mt-4 bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg transition-colors"
              >
                Create Your First Agent
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {agents.map((agent) => (
                <div key={agent.agent_id} className="bg-blue-800 hover:bg-blue-900 rounded-lg p-4 border border-gray-600">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
                    <div className={`px-2 py-1 rounded text-xs font-semibold ${
                      agent.status === 'running' ? 'bg-green-500 text-white' :
                      agent.status === 'deployed' ? 'bg-blue-500 text-white' :
                      agent.status === 'stopped' ? 'bg-blue-600 text-white' :
                      'bg-yellow-500 text-white'
                    }`}>
                      {agent.status}
                    </div>
                  </div>
                  <p className="text-gray-400 text-sm mb-2">Type: {agent.type}</p>
                  <p className="text-gray-400 text-sm mb-2">Model: {agent.llm_model}</p>
                  <div className="flex flex-wrap gap-1 mb-3">
                    {agent.capabilities.map((cap, idx) => (
                      <span key={idx} className="bg-purple-600 text-white text-xs px-2 py-1 rounded">
                        {cap}
                      </span>
                    ))}
                  </div>
                  <div className="flex space-x-2">
                    {agent.status === 'created' && (
                      <button
                        onClick={() => handleDeployAgent(agent.agent_id)}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm transition-colors"
                      >
                        Deploy
                      </button>
                    )}
                    {agent.status === 'deployed' && (
                      <button
                        onClick={() => handleManageAgent(agent.agent_id, 'start')}
                        className="flex-1 bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm transition-colors"
                      >
                        Start
                      </button>
                    )}
                    {agent.status === 'running' && (
                      <button
                        onClick={() => handleManageAgent(agent.agent_id, 'stop')}
                        className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-1 rounded text-sm transition-colors"
                      >
                        Stop
                      </button>
                    )}
                    <button
                      onClick={() => handleManageAgent(agent.agent_id, 'delete')}
                      className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* LLM Query Interface */}
        <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
            <Brain className="w-6 h-6 mr-2 text-pink-400" />
            Query Local LLM
          </h2>
          <>
            <textarea
              value={llmQuery}
              onChange={(e) => setLlmQuery(e.target.value)}
              placeholder="Ask the local LLM anything..."
              className="w-full bg-blue-800 hover:bg-blue-900 text-white rounded-lg p-4 border border-gray-600 focus:border-purple-500 focus:outline-none min-h-[100px]"
            />
            <button
              onClick={handleQueryLLM}
              disabled={queryLLMMutation.isPending || !llmQuery.trim()}
              className="bg-purple-600 hover:bg-purple-700 disabled:bg-blue-700 hover:bg-blue-800 text-white px-6 py-2 rounded-lg transition-colors mt-4"
            >
              {queryLLMMutation.isPending ? 'Querying...' : 'Query LLM'}
            </button>
            {queryLLMMutation.data && (
              <div className="bg-blue-800 hover:bg-blue-900 rounded-lg p-4 border border-gray-600 mt-4">
                <p className="text-white whitespace-pre-wrap">{queryLLMMutation.data.result?.response || 'No response'}</p>
              </div>
            )}
          </>
        </div>

        {/* System Optimization */}
        <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-6 border border-gray-700">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
            <TrendingUp className="w-6 h-6 mr-2 text-green-400" />
            System Optimization
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <button
              onClick={() => handleOptimizeSystem('memory')}
              disabled={optimizeSystemMutation.isPending}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-700 hover:bg-blue-800 text-white px-4 py-3 rounded-lg transition-colors"
            >
              Optimize Memory
            </button>
            <button
              onClick={() => handleOptimizeSystem('performance')}
              disabled={optimizeSystemMutation.isPending}
              className="bg-green-600 hover:bg-green-700 disabled:bg-blue-700 hover:bg-blue-800 text-white px-4 py-3 rounded-lg transition-colors"
            >
              Optimize Performance
            </button>
            <button
              onClick={() => handleOptimizeSystem('resources')}
              disabled={optimizeSystemMutation.isPending}
              className="bg-purple-600 hover:bg-purple-700 disabled:bg-blue-700 hover:bg-blue-800 text-white px-4 py-3 rounded-lg transition-colors"
            >
              Optimize Resources
            </button>
            <button
              onClick={() => handleOptimizeSystem('general')}
              disabled={optimizeSystemMutation.isPending}
              className="bg-pink-600 hover:bg-pink-700 disabled:bg-blue-700 hover:bg-blue-800 text-white px-4 py-3 rounded-lg transition-colors"
            >
              Full Optimization
            </button>
          </div>
        </div>

        {/* Create Agent Modal */}
        {showCreateAgent && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-blue-900/40 hover:bg-blue-900/60 rounded-xl p-8 max-w-md w-full border border-gray-700">
              <h3 className="text-2xl font-bold text-white mb-6">Create New Agent</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-gray-400 mb-2">Agent Name</label>
                  <input
                    type="text"
                    value={newAgentConfig.name}
                    onChange={(e) => setNewAgentConfig({ ...newAgentConfig, name: e.target.value })}
                    className="w-full bg-blue-800 hover:bg-blue-900 text-white rounded-lg p-3 border border-gray-600 focus:border-purple-500 focus:outline-none"
                    placeholder="My Research Agent"
                  />
                </div>
                <div>
                  <label className="block text-gray-400 mb-2">Agent Type</label>
                  <select
                    value={newAgentConfig.type}
                    onChange={(e) => setNewAgentConfig({ ...newAgentConfig, type: e.target.value })}
                    className="w-full bg-blue-800 hover:bg-blue-900 text-white rounded-lg p-3 border border-gray-600 focus:border-purple-500 focus:outline-none"
                  >
                    <option value="research">Research Agent</option>
                    <option value="code">Code Agent</option>
                    <option value="creative">Creative Agent</option>
                    <option value="data">Data Agent</option>
                    <option value="osint">OSINT Agent</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-400 mb-2">LLM Model</label>
                  <select
                    value={newAgentConfig.llm_model}
                    onChange={(e) => setNewAgentConfig({ ...newAgentConfig, llm_model: e.target.value })}
                    className="w-full bg-blue-800 hover:bg-blue-900 text-white rounded-lg p-3 border border-gray-600 focus:border-purple-500 focus:outline-none"
                  >
                    <option value="llama-3.1-8b">Llama 3.1 8B</option>
                    <option value="mistral-7b">Mistral 7B</option>
                  </select>
                </div>
                <div className="flex space-x-4 mt-6">
                  <button
                    onClick={handleCreateAgent}
                    disabled={buildAgentMutation.isPending || !newAgentConfig.name}
                    className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-blue-700 hover:bg-blue-800 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {buildAgentMutation.isPending ? 'Creating...' : 'Create Agent'}
                  </button>
                  <button
                    onClick={() => setShowCreateAgent(false)}
                    className="flex-1 bg-blue-700 hover:bg-blue-800 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedMasterAgent;
