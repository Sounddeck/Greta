import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Edit, Trash2, Play, Pause, Settings } from 'lucide-react';
import { toast } from 'sonner';
import { agentBuilderAPI } from '../../lib/api';

interface Agent {
  id: string;
  name: string;
  description: string;
  type: string;
  status: 'active' | 'inactive' | 'error';
  created_at: string;
  updated_at: string;
  config: Record<string, any>;
  metrics?: {
    total_executions: number;
    success_rate: number;
    avg_response_time: number;
  };
}

export default function AgentDetail() {
  const { agentId } = useParams<{ agentId: string }>();
  const navigate = useNavigate();
  const [agent, setAgent] = useState<Agent | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (agentId) {
      loadAgent();
    }
  }, [agentId]);

  const loadAgent = async () => {
    try {
      setLoading(true);
      const response = await agentBuilderAPI.getAgent(agentId!);
      setAgent(response.data);
    } catch (error: any) {
      console.error('Failed to load agent:', error);
      toast.error(error.response?.data?.detail || 'Failed to load agent');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this agent?')) return;

    try {
      await agentBuilderAPI.deleteAgent(agentId!);
      toast.success('Agent deleted successfully');
      navigate('/agent-builder/agents');
    } catch (error: any) {
      console.error('Failed to delete agent:', error);
      toast.error(error.response?.data?.detail || 'Failed to delete agent');
    }
  };

  const handleToggleStatus = async () => {
    if (!agent) return;

    try {
      const newStatus = agent.status === 'active' ? 'inactive' : 'active';
      await agentBuilderAPI.updateAgent(agentId!, { status: newStatus });
      setAgent({ ...agent, status: newStatus });
      toast.success(`Agent ${newStatus === 'active' ? 'activated' : 'deactivated'}`);
    } catch (error: any) {
      console.error('Failed to update agent status:', error);
      toast.error(error.response?.data?.detail || 'Failed to update agent status');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading agent...</p>
        </div>
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400">Agent not found</p>
          <button
            onClick={() => navigate('/agent-builder/agents')}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Back to Agents
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto bg-gray-50 dark:bg-gray-900">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={() => navigate('/agent-builder/agents')}
            className="flex items-center text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Agents
          </button>

          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{agent.name}</h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">{agent.description}</p>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={handleToggleStatus}
                className={`px-4 py-2 rounded flex items-center ${
                  agent.status === 'active'
                    ? 'bg-yellow-500 hover:bg-yellow-600 text-white'
                    : 'bg-green-500 hover:bg-green-600 text-white'
                }`}
              >
                {agent.status === 'active' ? (
                  <>
                    <Pause className="w-4 h-4 mr-2" />
                    Deactivate
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Activate
                  </>
                )}
              </button>

              <button
                onClick={() => navigate(`/agent-builder/agents/${agentId}/edit`)}
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded flex items-center"
              >
                <Edit className="w-4 h-4 mr-2" />
                Edit
              </button>

              <button
                onClick={handleDelete}
                className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded flex items-center"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Delete
              </button>
            </div>
          </div>
        </div>

        {/* Status Badge */}
        <div className="mb-6">
          <span
            className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              agent.status === 'active'
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                : agent.status === 'error'
                ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
            }`}
          >
            {agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}
          </span>
        </div>

        {/* Metrics */}
        {agent.metrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Executions</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {agent.metrics.total_executions}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
              <p className="text-sm text-gray-600 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {(agent.metrics.success_rate * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
              <p className="text-sm text-gray-600 dark:text-gray-400">Avg Response Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {agent.metrics.avg_response_time.toFixed(0)}ms
              </p>
            </div>
          </div>
        )}

        {/* Configuration */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Configuration
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Type
              </label>
              <p className="text-gray-900 dark:text-white">{agent.type}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Created
              </label>
              <p className="text-gray-900 dark:text-white">
                {new Date(agent.created_at).toLocaleString()}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Last Updated
              </label>
              <p className="text-gray-900 dark:text-white">
                {new Date(agent.updated_at).toLocaleString()}
              </p>
            </div>

            {Object.keys(agent.config).length > 0 && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Additional Configuration
                </label>
                <pre className="bg-gray-100 dark:bg-gray-700 p-4 rounded text-sm overflow-x-auto">
                  {JSON.stringify(agent.config, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

