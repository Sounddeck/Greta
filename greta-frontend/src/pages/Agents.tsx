import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Bot,
  Plus,
  Play,
  Pause,
  Settings,
  Trash2,
  Activity,
  Zap,
  Brain,
  Search
} from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error';
  description: string;
  lastActive: string;
  tasksCompleted: number;
  capabilities: string[];
  performance: number;
}

// Mock API functions
const fetchAgents = async (): Promise<Agent[]> => {
  await new Promise(resolve => setTimeout(resolve, 1000));
  return [
    {
      id: '1',
      name: 'Research Assistant',
      type: 'research',
      status: 'active',
      description: 'Specialized in web research and data analysis',
      lastActive: '2 minutes ago',
      tasksCompleted: 47,
      capabilities: ['Web Search', 'Data Analysis', 'Report Generation'],
      performance: 94
    },
    {
      id: '2',
      name: 'Content Creator',
      type: 'content',
      status: 'active',
      description: 'Creates engaging content for various platforms',
      lastActive: '15 minutes ago',
      tasksCompleted: 23,
      capabilities: ['Writing', 'SEO Optimization', 'Social Media'],
      performance: 87
    },
    {
      id: '3',
      name: 'Code Assistant',
      type: 'development',
      status: 'inactive',
      description: 'Helps with coding tasks and debugging',
      lastActive: '2 hours ago',
      tasksCompleted: 156,
      capabilities: ['Code Generation', 'Debugging', 'Documentation'],
      performance: 96
    },
    {
      id: '4',
      name: 'Data Analyst',
      type: 'analytics',
      status: 'active',
      description: 'Analyzes data patterns and generates insights',
      lastActive: '5 minutes ago',
      tasksCompleted: 89,
      capabilities: ['Statistical Analysis', 'Visualization', 'Reporting'],
      performance: 91
    }
  ];
};

const toggleAgent = async (agentId: string, action: 'start' | 'stop') => {
  await new Promise(resolve => setTimeout(resolve, 500));
  return { success: true, agentId, action };
};

const AgentCard: React.FC<{ agent: Agent }> = ({ agent }) => {
  const queryClient = useQueryClient();
  
  const toggleMutation = useMutation({
    mutationFn: ({ agentId, action }: { agentId: string; action: 'start' | 'stop' }) =>
      toggleAgent(agentId, action),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agents'] });
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-400/20';
      case 'inactive': return 'text-blue-400 bg-blue-400/20';
      case 'error': return 'text-red-400 bg-red-400/20';
      default: return 'text-blue-400 bg-blue-400/20';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'research': return Search;
      case 'content': return Brain;
      case 'development': return Zap;
      case 'analytics': return Activity;
      default: return Bot;
    }
  };

  const TypeIcon = getTypeIcon(agent.type);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      whileHover={{ scale: 1.02 }}
      className="card"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg">
            <TypeIcon className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
            <p className="text-sm text-gray-400">{agent.type}</p>
          </div>
        </div>
        
        <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
          {agent.status}
        </div>
      </div>

      <p className="text-gray-300 text-sm mb-4">{agent.description}</p>

      {/* Capabilities */}
      <div className="mb-4">
        <p className="text-xs font-medium text-gray-400 mb-2">Capabilities</p>
        <div className="flex flex-wrap gap-1">
          {agent.capabilities.map((capability, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-blue-500/15 text-xs text-gray-300 rounded-md"
            >
              {capability}
            </span>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <p className="text-lg font-bold text-white">{agent.tasksCompleted}</p>
          <p className="text-xs text-gray-400">Tasks</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-bold text-white">{agent.performance}%</p>
          <p className="text-xs text-gray-400">Performance</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-white">{agent.lastActive}</p>
          <p className="text-xs text-gray-400">Last Active</p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => toggleMutation.mutate({
              agentId: agent.id,
              action: agent.status === 'active' ? 'stop' : 'start'
            })}
            disabled={toggleMutation.isPending}
            className={`p-2 rounded-lg transition-colors ${
              agent.status === 'active'
                ? 'bg-red-600/20 text-red-400 hover:bg-red-600/30'
                : 'bg-green-600/20 text-green-400 hover:bg-green-600/30'
            }`}
          >
            {agent.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          
          <button className="p-2 bg-blue-500/15 text-gray-400 hover:text-white hover:bg-blue-500/20 rounded-lg transition-colors">
            <Settings className="w-4 h-4" />
          </button>
        </div>
        
        <button className="p-2 bg-red-600/20 text-red-400 hover:bg-red-600/30 rounded-lg transition-colors">
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  );
};

export const Agents: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');

  const { data: agents, isLoading, error } = useQuery({
    queryKey: ['agents'],
    queryFn: fetchAgents,
    refetchInterval: 30000
  });

  const filteredAgents = agents?.filter(agent => {
    const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         agent.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || agent.status === statusFilter;
    const matchesType = typeFilter === 'all' || agent.type === typeFilter;
    
    return matchesSearch && matchesStatus && matchesType;
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center text-red-400 p-8">
        <p>Error loading agents</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">🤖 Agent Management</h1>
          <p className="text-gray-400">Manage and monitor your AI agents</p>
        </div>
        
        <button className="btn-primary flex items-center space-x-2">
          <Plus className="w-5 h-5" />
          <span>Create Agent</span>
        </button>
      </div>

      {/* Filters */}
      <div className="glass-strong p-4 rounded-xl mb-6">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search agents..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input-field pl-10 w-full"
              />
            </div>
          </div>
          
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="input-field"
          >
            <option value="all">All Status</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="error">Error</option>
          </select>
          
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="input-field"
          >
            <option value="all">All Types</option>
            <option value="research">Research</option>
            <option value="content">Content</option>
            <option value="development">Development</option>
            <option value="analytics">Analytics</option>
          </select>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="card text-center">
          <div className="text-2xl font-bold text-white mb-1">
            {agents?.length || 0}
          </div>
          <div className="text-sm text-gray-400">Total Agents</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-green-400 mb-1">
            {agents?.filter(a => a.status === 'active').length || 0}
          </div>
          <div className="text-sm text-gray-400">Active</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-blue-400 mb-1">
            {agents?.reduce((sum, agent) => sum + agent.tasksCompleted, 0) || 0}
          </div>
          <div className="text-sm text-gray-400">Tasks Completed</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-purple-400 mb-1">
            {agents ? Math.round(agents.reduce((sum, agent) => sum + agent.performance, 0) / agents.length) : 0}%
          </div>
          <div className="text-sm text-gray-400">Avg Performance</div>
        </div>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredAgents?.map((agent) => (
          <AgentCard key={agent.id} agent={agent} />
        ))}
      </div>

      {filteredAgents?.length === 0 && (
        <div className="text-center py-12">
          <Bot className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-400">No agents found matching your criteria</p>
        </div>
      )}
    </div>
  );
};
