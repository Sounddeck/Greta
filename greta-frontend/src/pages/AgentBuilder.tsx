import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Plus,
  Users,
  Settings,
  Code,
  Brain,
  Zap,
  Save,
  Play,
  Pause,
  Trash2,
  Edit,
  Copy,
  CheckCircle,
  AlertCircle,
  Clock,
  Activity,
  Target,
  Wrench,
  Database,
  MessageSquare,
  Search,
  Globe,
  Shield
} from 'lucide-react';
import { apiService } from '../services/api';

interface Agent {
  id: string;
  name: string;
  description: string;
  type: 'research' | 'analysis' | 'communication' | 'automation' | 'custom';
  status: 'active' | 'inactive' | 'training' | 'error';
  capabilities: string[];
  created_at: string;
  last_used?: string;
  usage_count: number;
  success_rate: number;
  configuration: {
    model: string;
    temperature: number;
    max_tokens: number;
    tools: string[];
    prompt_template: string;
  };
}

interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  type: string;
  capabilities: string[];
  default_config: any;
}

const AgentBuilder: React.FC = () => {
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newAgent, setNewAgent] = useState<{
    name: string;
    description: string;
    type: 'research' | 'analysis' | 'communication' | 'automation' | 'custom';
    prompt_template: string;
  }>({
    name: '',
    description: '',
    type: 'custom',
    prompt_template: ''
  });
  const queryClient = useQueryClient();

  const { data: agents, isLoading } = useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: async () => {
      try {
        const response = await apiService.request<Agent[]>('/api/v1/agents');
        return response.data || getMockAgents();
      } catch (error) {
        return getMockAgents();
      }
    },
    refetchInterval: 30000
  });

  const { data: templates } = useQuery<AgentTemplate[]>({
    queryKey: ['agent-templates'],
    queryFn: async () => {
      try {
        const response = await apiService.request<AgentTemplate[]>('/api/v1/agents/templates');
        return response.data || getMockTemplates();
      } catch (error) {
        return getMockTemplates();
      }
    }
  });

  const createAgentMutation = useMutation({
    mutationFn: async (agentData: any) => {
      const response = await apiService.request('/api/v1/agents', {
        method: 'POST',
        body: JSON.stringify(agentData)
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agents'] });
      setShowCreateModal(false);
      setNewAgent({ name: '', description: '', type: 'custom', prompt_template: '' });
    }
  });

  const getMockAgents = (): Agent[] => [
    {
      id: 'research-agent-1',
      name: 'Research Assistant',
      description: 'Advanced research and information gathering agent',
      type: 'research',
      status: 'active',
      capabilities: ['web_search', 'data_analysis', 'report_generation'],
      created_at: new Date(Date.now() - 86400000).toISOString(),
      last_used: new Date(Date.now() - 3600000).toISOString(),
      usage_count: 156,
      success_rate: 94.2,
      configuration: {
        model: 'gpt-4',
        temperature: 0.3,
        max_tokens: 2000,
        tools: ['web_search', 'arxiv_search'],
        prompt_template: 'You are a research assistant...'
      }
    },
    {
      id: 'analysis-agent-1',
      name: 'Data Analyst',
      description: 'Specialized in data analysis and visualization',
      type: 'analysis',
      status: 'active',
      capabilities: ['data_processing', 'statistical_analysis', 'visualization'],
      created_at: new Date(Date.now() - 172800000).toISOString(),
      last_used: new Date(Date.now() - 7200000).toISOString(),
      usage_count: 89,
      success_rate: 91.7,
      configuration: {
        model: 'gpt-4',
        temperature: 0.1,
        max_tokens: 1500,
        tools: ['analytics_engine', 'data_visualization'],
        prompt_template: 'You are a data analyst...'
      }
    },
    {
      id: 'comm-agent-1',
      name: 'Communication Manager',
      description: 'Handles email, messaging, and communication tasks',
      type: 'communication',
      status: 'inactive',
      capabilities: ['email_management', 'scheduling', 'follow_ups'],
      created_at: new Date(Date.now() - 259200000).toISOString(),
      usage_count: 45,
      success_rate: 88.9,
      configuration: {
        model: 'gpt-3.5-turbo',
        temperature: 0.7,
        max_tokens: 1000,
        tools: ['gmail_integration', 'calendar_sync'],
        prompt_template: 'You are a communication manager...'
      }
    }
  ];

  const getMockTemplates = (): AgentTemplate[] => [
    {
      id: 'research-template',
      name: 'Research Agent',
      description: 'Template for creating research and information gathering agents',
      type: 'research',
      capabilities: ['web_search', 'data_analysis', 'report_generation'],
      default_config: {
        model: 'gpt-4',
        temperature: 0.3,
        max_tokens: 2000,
        tools: ['web_search', 'arxiv_search']
      }
    },
    {
      id: 'analysis-template',
      name: 'Analysis Agent',
      description: 'Template for data analysis and processing agents',
      type: 'analysis',
      capabilities: ['data_processing', 'statistical_analysis', 'visualization'],
      default_config: {
        model: 'gpt-4',
        temperature: 0.1,
        max_tokens: 1500,
        tools: ['analytics_engine']
      }
    },
    {
      id: 'automation-template',
      name: 'Automation Agent',
      description: 'Template for workflow automation and task management',
      type: 'automation',
      capabilities: ['task_automation', 'workflow_management', 'scheduling'],
      default_config: {
        model: 'gpt-3.5-turbo',
        temperature: 0.5,
        max_tokens: 1000,
        tools: ['task_manager', 'scheduler']
      }
    }
  ];

  const getAgentTypeIcon = (type: string) => {
    switch (type) {
      case 'research': return <Search className="w-5 h-5" />;
      case 'analysis': return <Database className="w-5 h-5" />;
      case 'communication': return <MessageSquare className="w-5 h-5" />;
      case 'automation': return <Zap className="w-5 h-5" />;
      default: return <Brain className="w-5 h-5" />;
    }
  };

  const getAgentTypeColor = (type: string) => {
    switch (type) {
      case 'research': return 'text-blue-400 bg-blue-400/20';
      case 'analysis': return 'text-green-400 bg-green-400/20';
      case 'communication': return 'text-purple-400 bg-purple-400/20';
      case 'automation': return 'text-orange-400 bg-orange-400/20';
      default: return 'text-blue-400 bg-blue-400/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'inactive': return <Pause className="w-5 h-5 text-gray-400" />;
      case 'training': return <Activity className="w-5 h-5 text-blue-400 animate-spin" />;
      case 'error': return <AlertCircle className="w-5 h-5 text-red-400" />;
      default: return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const handleCreateAgent = () => {
    if (newAgent.name && newAgent.description) {
      createAgentMutation.mutate({
        ...newAgent,
        template_id: selectedTemplate || undefined
      });
    }
  };

  const handleCopyAgent = async (agentId: string) => {
    try {
      const agent = agents?.find(a => a.id === agentId);
      if (agent) {
        setNewAgent({
          name: `${agent.name} (Copy)`,
          description: agent.description,
          type: agent.type,
          prompt_template: agent.configuration.prompt_template
        });
        setShowCreateModal(true);
      }
    } catch (error) {
      console.error('Error copying agent:', error);
    }
  };

  const handleEditAgent = async (agentId: string) => {
    try {
      const agent = agents?.find(a => a.id === agentId);
      if (agent) {
        setNewAgent({
          name: agent.name,
          description: agent.description,
          type: agent.type,
          prompt_template: agent.configuration.prompt_template
        });
        setShowCreateModal(true);
      }
    } catch (error) {
      console.error('Error editing agent:', error);
    }
  };

  const handleDeleteAgent = async (agentId: string) => {
    if (window.confirm('Are you sure you want to delete this agent?')) {
      try {
        await apiService.request(`/api/v1/agents/${agentId}`, {
          method: 'DELETE'
        });
        queryClient.invalidateQueries({ queryKey: ['agents'] });
      } catch (error) {
        console.error('Error deleting agent:', error);
      }
    }
  };

  const handleToggleAgent = async (agentId: string, currentStatus: string) => {
    try {
      const newStatus = currentStatus === 'active' ? 'inactive' : 'active';
      await apiService.request(`/api/v1/agents/${agentId}/status`, {
        method: 'PATCH',
        body: JSON.stringify({ status: newStatus })
      });
      queryClient.invalidateQueries({ queryKey: ['agents'] });
    } catch (error) {
      console.error('Error toggling agent:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Wrench className="w-8 h-8 mr-3 text-blue-400" />
            Agent Builder
          </h1>
          <p className="text-gray-400 mt-2">
            Create and customize AI agents for specific tasks
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>Create Agent</span>
        </button>
      </div>

      {/* Agent Templates */}
      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2 text-blue-400" />
          Agent Templates
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(templates || []).map((template) => (
            <div
              key={template.id}
              onClick={() => setSelectedTemplate(template.id)}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${
                selectedTemplate === template.id
                  ? 'border-blue-500 bg-blue-500/20'
                  : 'border-white/20 bg-blue-500/10 hover:bg-blue-500/15'
              }`}
            >
              <div className="flex items-center space-x-3 mb-3">
                <div className={`p-2 rounded-lg ${getAgentTypeColor(template.type)}`}>
                  {getAgentTypeIcon(template.type)}
                </div>
                <div>
                  <h4 className="text-white font-medium">{template.name}</h4>
                  <p className="text-gray-400 text-sm">{template.type}</p>
                </div>
              </div>
              <p className="text-gray-300 text-sm mb-3">{template.description}</p>
              <div className="flex flex-wrap gap-1">
                {template.capabilities.slice(0, 2).map((capability, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded-full"
                  >
                    {capability.replace('_', ' ')}
                  </span>
                ))}
                {template.capabilities.length > 2 && (
                  <span className="px-2 py-1 bg-blue-600/20 text-gray-300 text-xs rounded-full">
                    +{template.capabilities.length - 2}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Existing Agents */}
      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Users className="w-5 h-5 mr-2 text-green-400" />
          Your Agents ({(agents || []).length})
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {(agents || []).map((agent) => (
            <div key={agent.id} className="bg-blue-500/10 p-6 rounded-xl hover:bg-blue-500/15 transition-all">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getAgentTypeColor(agent.type)}`}>
                    {getAgentTypeIcon(agent.type)}
                  </div>
                  <div>
                    <h4 className="text-white font-semibold">{agent.name}</h4>
                    <p className="text-gray-400 text-sm">{agent.type}</p>
                  </div>
                </div>
                {getStatusIcon(agent.status)}
              </div>

              <p className="text-gray-300 text-sm mb-4">{agent.description}</p>

              <div className="space-y-2 mb-4">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Usage Count</span>
                  <span className="text-white">{agent.usage_count}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Success Rate</span>
                  <span className="text-green-400">{agent.success_rate}%</span>
                </div>
                {agent.last_used && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Last Used</span>
                    <span className="text-white">
                      {new Date(agent.last_used).toLocaleDateString()}
                    </span>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-1 mb-4">
                {agent.capabilities.slice(0, 3).map((capability, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded-full"
                  >
                    {capability.replace('_', ' ')}
                  </span>
                ))}
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => handleEditAgent(agent.id)}
                  className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
                >
                  <Edit className="w-4 h-4 inline mr-1" />
                  Edit
                </button>
                <button
                  onClick={() => handleToggleAgent(agent.id, agent.status)}
                  className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
                  title={agent.status === 'active' ? 'Pause Agent' : 'Activate Agent'}
                >
                  {agent.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => handleCopyAgent(agent.id)}
                  className="px-3 py-2 bg-blue-700 hover:bg-blue-800 text-white text-sm rounded-lg transition-colors"
                  title="Copy Agent"
                >
                  <Copy className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleDeleteAgent(agent.id)}
                  className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  title="Delete Agent"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Create Agent Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-blue-900/40 hover:bg-blue-900/60 p-6 rounded-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h3 className="text-xl font-semibold text-white mb-6">Create New Agent</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-gray-400 text-sm mb-2">Agent Name</label>
                <input
                  type="text"
                  value={newAgent.name}
                  onChange={(e) => setNewAgent({ ...newAgent, name: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  placeholder="Enter agent name..."
                />
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Description</label>
                <textarea
                  value={newAgent.description}
                  onChange={(e) => setNewAgent({ ...newAgent, description: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 h-20"
                  placeholder="Describe what this agent will do..."
                />
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Agent Type</label>
                <select
                  value={newAgent.type}
                  onChange={(e) => setNewAgent({ ...newAgent, type: e.target.value as any })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="custom">Custom</option>
                  <option value="research">Research</option>
                  <option value="analysis">Analysis</option>
                  <option value="communication">Communication</option>
                  <option value="automation">Automation</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Prompt Template</label>
                <textarea
                  value={newAgent.prompt_template}
                  onChange={(e) => setNewAgent({ ...newAgent, prompt_template: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 h-32"
                  placeholder="Enter the system prompt for this agent..."
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCreateAgent}
                disabled={createAgentMutation.isPending || !newAgent.name || !newAgent.description}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {createAgentMutation.isPending ? 'Creating...' : 'Create Agent'}
              </button>
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 bg-blue-700 hover:bg-blue-800 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentBuilder;
