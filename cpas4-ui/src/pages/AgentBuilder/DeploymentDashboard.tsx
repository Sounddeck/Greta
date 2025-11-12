import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Rocket, CheckCircle, XCircle, Clock, TrendingUp, Activity } from 'lucide-react';
import { toast } from 'sonner';
import { agentBuilderAPI } from '../../lib/api';

interface Deployment {
  id: string;
  agent_id: string;
  agent_name: string;
  environment: 'development' | 'staging' | 'production';
  status: 'pending' | 'deploying' | 'deployed' | 'failed';
  version: string;
  deployed_at?: string;
  deployed_by?: string;
  health_status?: 'healthy' | 'degraded' | 'unhealthy';
}

export default function DeploymentDashboard() {
  const navigate = useNavigate();
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'development' | 'staging' | 'production'>('all');

  useEffect(() => {
    loadDeployments();
  }, []);

  const loadDeployments = async () => {
    try {
      setLoading(true);
      const response = await agentBuilderAPI.listDeployments();
      setDeployments(response.data.deployments || []);
    } catch (error: any) {
      console.error('Failed to load deployments:', error);
      toast.error(error.response?.data?.detail || 'Failed to load deployments');
      // Set empty array on error
      setDeployments([]);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: Deployment['status']) => {
    switch (status) {
      case 'deployed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'deploying':
        return <Activity className="w-5 h-5 text-blue-500 animate-pulse" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return null;
    }
  };

  const getHealthIcon = (health?: Deployment['health_status']) => {
    if (!health) return null;
    
    switch (health) {
      case 'healthy':
        return <div className="w-3 h-3 rounded-full bg-green-500"></div>;
      case 'degraded':
        return <div className="w-3 h-3 rounded-full bg-yellow-500"></div>;
      case 'unhealthy':
        return <div className="w-3 h-3 rounded-full bg-red-500"></div>;
      default:
        return null;
    }
  };

  const filteredDeployments = filter === 'all'
    ? deployments
    : deployments.filter(d => d.environment === filter);

  const stats = {
    total: deployments.length,
    deployed: deployments.filter(d => d.status === 'deployed').length,
    failed: deployments.filter(d => d.status === 'failed').length,
    pending: deployments.filter(d => d.status === 'pending' || d.status === 'deploying').length,
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading deployments...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
            <Rocket className="w-8 h-8 mr-3" />
            Deployment Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Monitor and manage agent deployments across environments
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Deployments</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.total}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Deployed</p>
                <p className="text-2xl font-bold text-green-600">{stats.deployed}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Failed</p>
                <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
              </div>
              <XCircle className="w-8 h-8 text-red-500" />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Pending</p>
                <p className="text-2xl font-bold text-yellow-600">{stats.pending}</p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="mb-6 flex space-x-2">
          {(['all', 'development', 'staging', 'production'] as const).map((env) => (
            <button
              key={env}
              onClick={() => setFilter(env)}
              className={`px-4 py-2 rounded ${
                filter === env
                  ? 'bg-blue-500 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {env.charAt(0).toUpperCase() + env.slice(1)}
            </button>
          ))}
        </div>

        {/* Deployments List */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          {filteredDeployments.length === 0 ? (
            <div className="p-8 text-center text-gray-600 dark:text-gray-400">
              <Rocket className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No deployments found</p>
              <button
                onClick={() => navigate('/agent-builder/agents')}
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Deploy an Agent
              </button>
            </div>
          ) : (
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Agent
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Environment
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Version
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Health
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Deployed
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {filteredDeployments.map((deployment) => (
                  <tr
                    key={deployment.id}
                    className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                    onClick={() => navigate(`/agent-builder/agents/${deployment.agent_id}`)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {deployment.agent_name}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          deployment.environment === 'production'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                            : deployment.environment === 'staging'
                            ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                            : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                        }`}
                      >
                        {deployment.environment}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {deployment.version}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {getStatusIcon(deployment.status)}
                        <span className="ml-2 text-sm text-gray-900 dark:text-white">
                          {deployment.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {getHealthIcon(deployment.health_status)}
                        <span className="ml-2 text-sm text-gray-900 dark:text-white">
                          {deployment.health_status || 'N/A'}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {deployment.deployed_at
                        ? new Date(deployment.deployed_at).toLocaleString()
                        : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

