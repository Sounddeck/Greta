import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Crown, 
  Play, 
  Pause, 
  Square, 
  Settings, 
  Plus,
  Trash2,
  Edit,
  Activity,
  Users,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  Brain,
  Target,
  BarChart3
} from 'lucide-react';
import { apiService } from '../services/api';

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'idle' | 'error' | 'stopped';
  capabilities: string[];
  performance: number;
  tasks_completed: number;
  created_at: string;
  last_activity: string;
}

interface MasterAgentStatus {
  status: string;
  active_agents: Agent[];
  total_tasks: number;
  success_rate: number;
  system_load: number;
  coordination_efficiency: number;
}

export const MasterAgent: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [showCreateAgent, setShowCreateAgent] = useState(false);
  const [newAgentConfig, setNewAgentConfig] = useState({
    name: '',
    type: 'general',
    capabilities: [] as string[]
  });

  const queryClient = useQueryClient();

  const { data: masterStatus, isLoading } = useQuery({
    queryKey: ['master-agent-status'],
    queryFn: async () => {
      try {
        const response = await apiService.request('/api/v1/master-agent/status');
        return response.data as MasterAgentStatus;
      } catch (error) {
        // Mock data for development
        return {
          status: 'operational',
          active_agents: [
            {
              id: 'agent-001',
              name: 'Research Assistant',
              type: 'research',
              status: 'active',
              capabilities: ['web_search', 'data_analysis', 'report_generation'],
              performance: 94,
              tasks_completed: 156,
              created_at: '2024-01-15T10:00:00Z',
              last_activity: new Date(Date.now() - 300000).toISOString()
            },
            {
              id: 'agent-002',
              name: 'Content Creator',
              type: 'creative',
              status: 'active',
              capabilities: ['writing', 'editing', 'content_strategy'],
              performance: 87,
              tasks_completed: 89,
              created_at: '2024-01-16T14:30:00Z',
              last_activity: new Date(Date.now() - 600000).toISOString()
            },
            {
              id: 'agent-003',
              name: 'Data Analyst',
              type: 'analytical',
              status: 'idle',
              capabilities: ['data_processing', 'visualization', 'insights'],
              performance: 91,
              tasks_completed: 203,
              created_at: '2024-01-14T09:15:00Z',
              last_activity: new Date(Date.now() - 1800000).toISOString()
            }
          ],
          total_tasks: 448,
          success_rate: 96.2,
          system_load: 67,
          coordination_efficiency: 89
        } as MasterAgentStatus;
      }
    },
    refetchInterval: 10000
  });

  const createAgentMutation = useMutation({
    mutationFn: async (agentConfig: typeof newAgentConfig) => {
      const response = await apiService.request('/api/v1/master-agent/create', {
        method: 'POST',
        body: JSON.stringify(agentConfig)
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['master-agent-status'] });
      setShowCreateAgent(false);
      setNewAgentConfig({ name: '', type: 'general', capabilities: [] });
    }
  });

  const controlAgentMutation = useMutation({
    mutationFn: async ({ agentId, action }: { agentId: string; action: string }) => {
      const response = await apiService.request(`/api/v1/master-agent/control/${agentId}`, {
        method: 'POST',
        body: JSON.stringify({ action })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['master-agent-status'] });
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'idle': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      case 'stopped': return 'text-gray-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Play className="w-4 h-4" />;
      case 'idle': return <Pause className="w-4 h-4" />;
      case 'error': return <AlertTriangle className="w-4 h-4" />;
      case 'stopped': return <Square className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const handleCreateAgent = () => {
    if (newAgentConfig.name && newAgentConfig.type) {
      createAgentMutation.mutate(newAgentConfig);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Crown className="w-8 h-8 mr-3 text-yellow-400" />
            Master Agent Control
          </h1>
          <p className="text-gray-400 mt-2">
            Orchestrate and manage all AI agents in the GRETA ecosystem
          </p>
        </div>
        <button
          onClick={() => setShowCreateAgent(true)}
          className="btn-primary flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>Create Agent</span>
        </button>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Agents</p>
              <p className="text-2xl font-bold text-white">
                {masterStatus?.active_agents.filter(a => a.status === 'active').length || 0}
              </p>
            </div>
            <Users className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Tasks</p>
              <p className="text-2xl font-bold text-white">{masterStatus?.total_tasks || 0}</p>
            </div>
            <Target className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Success Rate</p>
              <p className="text-2xl font-bold text-white">{masterStatus?.success_rate || 0}%</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">System Load</p>
              <p className="text-2xl font-bold text-white">{masterStatus?.system_load || 0}%</p>
            </div>
            <BarChart3 className="w-8 h-8 text-yellow-400" />
          </div>
        </div>
      </div>

      {/* Agent Management */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Agent List */}
        <div className="lg:col-span-2 glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-blue-400" />
            Agent Fleet
          </h3>
          <div className="space-y-4">
            {masterStatus?.active_agents.map((agent) => (
              <div 
                key={agent.id}
                className={`p-4 bg-blue-500/10 rounded-lg border-l-4 cursor-pointer transition-all hover:bg-blue-500/15 ${
                  selectedAgent?.id === agent.id ? 'border-purple-500 bg-blue-500/15' : 'border-gray-600'
                }`}
                onClick={() => setSelectedAgent(agent)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`${getStatusColor(agent.status)}`}>
                      {getStatusIcon(agent.status)}
                    </div>
                    <div>
                      <h4 className="font-semibold text-white">{agent.name}</h4>
                      <p className="text-sm text-gray-400 capitalize">{agent.type} Agent</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="text-sm text-gray-400">Performance</p>
                      <p className="font-semibold text-white">{agent.performance}%</p>
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          controlAgentMutation.mutate({ 
                            agentId: agent.id, 
                            action: agent.status === 'active' ? 'pause' : 'start' 
                          });
                        }}
                        className="p-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors"
                      >
                        {agent.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          // Handle edit
                        }}
                        className="p-2 bg-yellow-500/20 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors"
                      >
                        <Edit className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {agent.capabilities.slice(0, 3).map((capability, index) => (
                    <span key={index} className="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded-full">
                      {capability}
                    </span>
                  ))}
                  {agent.capabilities.length > 3 && (
                    <span className="px-2 py-1 bg-blue-600/20 text-gray-400 text-xs rounded-full">
                      +{agent.capabilities.length - 3} more
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Agent Details */}
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Brain className="w-5 h-5 mr-2 text-purple-400" />
            Agent Details
          </h3>
          {selectedAgent ? (
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-white">{selectedAgent.name}</h4>
                <p className="text-sm text-gray-400 capitalize">{selectedAgent.type} Agent</p>
              </div>
              
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-400">Status</p>
                  <div className={`flex items-center space-x-2 ${getStatusColor(selectedAgent.status)}`}>
                    {getStatusIcon(selectedAgent.status)}
                    <span className="capitalize">{selectedAgent.status}</span>
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-gray-400">Performance</p>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                        style={{ width: `${selectedAgent.performance}%` }}
                      ></div>
                    </div>
                    <span className="text-white text-sm">{selectedAgent.performance}%</span>
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-gray-400">Tasks Completed</p>
                  <p className="text-white font-semibold">{selectedAgent.tasks_completed}</p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-400">Last Activity</p>
                  <p className="text-white text-sm">
                    {new Date(selectedAgent.last_activity).toLocaleString()}
                  </p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-400 mb-2">Capabilities</p>
                  <div className="space-y-1">
                    {selectedAgent.capabilities.map((capability, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <CheckCircle className="w-3 h-3 text-green-400" />
                        <span className="text-sm text-gray-300">{capability}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Select an agent to view details</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Agent Modal */}
      {showCreateAgent && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass p-6 rounded-xl max-w-md w-full mx-4">
            <h3 className="text-xl font-semibold text-white mb-4">Create New Agent</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Agent Name</label>
                <input
                  type="text"
                  value={newAgentConfig.name}
                  onChange={(e) => setNewAgentConfig({ ...newAgentConfig, name: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
                  placeholder="Enter agent name"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Agent Type</label>
                <select
                  value={newAgentConfig.type}
                  onChange={(e) => setNewAgentConfig({ ...newAgentConfig, type: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
                >
                  <option value="general">General Assistant</option>
                  <option value="research">Research Specialist</option>
                  <option value="creative">Creative Assistant</option>
                  <option value="analytical">Data Analyst</option>
                  <option value="technical">Technical Expert</option>
                </select>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={() => setShowCreateAgent(false)}
                  className="flex-1 px-4 py-2 bg-blue-700 text-white rounded-lg hover:bg-blue-800 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateAgent}
                  disabled={!newAgentConfig.name || createAgentMutation.isPending}
                  className="flex-1 btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {createAgentMutation.isPending ? 'Creating...' : 'Create'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MasterAgent;
