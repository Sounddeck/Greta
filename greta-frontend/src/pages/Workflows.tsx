import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import {
  Workflow,
  Plus,
  Play,
  Pause,
  Edit,
  Trash2,
  Clock,
  CheckCircle,
  AlertCircle,
  Zap
} from 'lucide-react';

interface WorkflowItem {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'running' | 'error';
  lastRun: string;
  nextRun?: string;
  executions: number;
  successRate: number;
  steps: number;
  trigger: string;
}

// Mock API function
const fetchWorkflows = async (): Promise<WorkflowItem[]> => {
  await new Promise(resolve => setTimeout(resolve, 1000));
  return [
    {
      id: '1',
      name: 'Daily Report Generation',
      description: 'Automatically generates daily performance reports',
      status: 'active',
      lastRun: '2 hours ago',
      nextRun: 'Tomorrow at 9:00 AM',
      executions: 45,
      successRate: 98,
      steps: 5,
      trigger: 'Schedule'
    },
    {
      id: '2',
      name: 'Content Optimization',
      description: 'Optimizes content for SEO and engagement',
      status: 'running',
      lastRun: '5 minutes ago',
      executions: 23,
      successRate: 87,
      steps: 8,
      trigger: 'Manual'
    },
    {
      id: '3',
      name: 'Data Backup',
      description: 'Backs up important data to secure storage',
      status: 'inactive',
      lastRun: '1 week ago',
      executions: 156,
      successRate: 100,
      steps: 3,
      trigger: 'Schedule'
    },
    {
      id: '4',
      name: 'Email Processing',
      description: 'Processes and categorizes incoming emails',
      status: 'error',
      lastRun: '3 hours ago',
      executions: 89,
      successRate: 94,
      steps: 6,
      trigger: 'Event'
    }
  ];
};

const WorkflowCard: React.FC<{ workflow: WorkflowItem }> = ({ workflow }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-400/20';
      case 'running': return 'text-blue-400 bg-blue-400/20';
      case 'inactive': return 'text-blue-400 bg-blue-400/20';
      case 'error': return 'text-red-400 bg-red-400/20';
      default: return 'text-blue-400 bg-blue-400/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return CheckCircle;
      case 'running': return Zap;
      case 'inactive': return Pause;
      case 'error': return AlertCircle;
      default: return Clock;
    }
  };

  const StatusIcon = getStatusIcon(workflow.status);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      className="card"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg">
            <Workflow className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">{workflow.name}</h3>
            <p className="text-sm text-gray-400">{workflow.trigger} trigger</p>
          </div>
        </div>
        
        <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(workflow.status)}`}>
          <StatusIcon className="w-3 h-3" />
          <span>{workflow.status}</span>
        </div>
      </div>

      <p className="text-gray-300 text-sm mb-4">{workflow.description}</p>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-xs text-gray-400 mb-1">Executions</p>
          <p className="text-lg font-bold text-white">{workflow.executions}</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 mb-1">Success Rate</p>
          <p className="text-lg font-bold text-green-400">{workflow.successRate}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 mb-1">Steps</p>
          <p className="text-lg font-bold text-blue-400">{workflow.steps}</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 mb-1">Last Run</p>
          <p className="text-sm text-white">{workflow.lastRun}</p>
        </div>
      </div>

      {workflow.nextRun && (
        <div className="mb-4 p-2 bg-blue-600/10 border border-blue-600/20 rounded-lg">
          <p className="text-xs text-blue-400">Next run: {workflow.nextRun}</p>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <div className="flex items-center space-x-2">
          <button className="p-2 bg-green-600/20 text-green-400 hover:bg-green-600/30 rounded-lg transition-colors">
            <Play className="w-4 h-4" />
          </button>
          
          <button className="p-2 bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 rounded-lg transition-colors">
            <Edit className="w-4 h-4" />
          </button>
        </div>
        
        <button className="p-2 bg-red-600/20 text-red-400 hover:bg-red-600/30 rounded-lg transition-colors">
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  );
};

export const Workflows: React.FC = () => {
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const { data: workflows, isLoading, error } = useQuery({
    queryKey: ['workflows'],
    queryFn: fetchWorkflows,
    refetchInterval: 30000
  });

  const filteredWorkflows = workflows?.filter(workflow => 
    statusFilter === 'all' || workflow.status === statusFilter
  );

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
        <p>Error loading workflows</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">⚡ Workflow Automation</h1>
          <p className="text-gray-400">Create and manage automated workflows</p>
        </div>
        
        <button className="btn-primary flex items-center space-x-2">
          <Plus className="w-5 h-5" />
          <span>Create Workflow</span>
        </button>
      </div>

      {/* Filter */}
      <div className="glass-strong p-4 rounded-xl mb-6">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-300">Filter by status:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="input-field"
          >
            <option value="all">All Status</option>
            <option value="active">Active</option>
            <option value="running">Running</option>
            <option value="inactive">Inactive</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="card text-center">
          <div className="text-2xl font-bold text-white mb-1">
            {workflows?.length || 0}
          </div>
          <div className="text-sm text-gray-400">Total Workflows</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-green-400 mb-1">
            {workflows?.filter(w => w.status === 'active').length || 0}
          </div>
          <div className="text-sm text-gray-400">Active</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-blue-400 mb-1">
            {workflows?.filter(w => w.status === 'running').length || 0}
          </div>
          <div className="text-sm text-gray-400">Running</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-purple-400 mb-1">
            {workflows ? Math.round(workflows.reduce((sum, w) => sum + w.successRate, 0) / workflows.length) : 0}%
          </div>
          <div className="text-sm text-gray-400">Avg Success Rate</div>
        </div>
      </div>

      {/* Workflows Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredWorkflows?.map((workflow) => (
          <WorkflowCard key={workflow.id} workflow={workflow} />
        ))}
      </div>

      {filteredWorkflows?.length === 0 && (
        <div className="text-center py-12">
          <Workflow className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-400">No workflows found matching your criteria</p>
        </div>
      )}
    </div>
  );
};
