import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Brain, 
  Activity, 
  Users, 
  Settings, 
  BarChart3, 
  Zap, 
  CheckCircle, 
  AlertTriangle, 
  Clock, 
  TrendingUp,
  Database,
  Network,
  Shield,
  Target,
  Globe,
  MessageSquare,
  FileText,
  Download,
  Upload,
  RefreshCw,
  Play,
  Pause,
  Square,
  Plus,
  Edit,
  Trash2,
  Copy,
  Save,
  Code,
  Terminal,
  Server,
  Monitor,
  Cpu,
  HardDrive,
  Wifi,
  Eye,
  Search,
  Filter,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  X,
  ChevronRight,
  ChevronDown,
  Package,
  Rocket,
  Layers,
  GitBranch,
  History,
  Star,
  Award,
  Wrench
} from 'lucide-react';
import { apiService } from '../services/api';

interface Agent {
  id: string;
  name: string;
  type: 'osint' | 'research' | 'analysis' | 'communication' | 'automation' | 'monitoring';
  status: 'active' | 'idle' | 'busy' | 'error' | 'offline' | 'deploying' | 'building';
  version: string;
  environment: 'development' | 'staging' | 'production';
  created_at: string;
  last_deployed: string;
  performance_score: number;
  resource_usage: {
    cpu: number;
    memory: number;
    network: number;
  };
  build_info: {
    last_build: string;
    build_status: 'success' | 'failed' | 'building';
    build_time: number;
    tests_passed: number;
    tests_total: number;
  };
  deployment_info: {
    replicas: number;
    health_checks: boolean;
    auto_scaling: boolean;
    rollback_available: boolean;
  };
}

interface DeploymentPipeline {
  id: string;
  name: string;
  agent_id: string;
  stages: Array<{
    name: string;
    status: 'pending' | 'running' | 'success' | 'failed';
    duration?: number;
    logs?: string[];
  }>;
  status: 'pending' | 'running' | 'success' | 'failed';
  created_at: string;
  completed_at?: string;
}

interface SystemMetrics {
  total_agents: number;
  active_deployments: number;
  successful_builds: number;
  failed_builds: number;
  avg_build_time: number;
  system_health: number;
  resource_utilization: {
    cpu: number;
    memory: number;
    storage: number;
    network: number;
  };
}

export const PAIHubManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'agents' | 'deployments' | 'builds' | 'monitoring' | 'settings'>('overview');
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [showCreateAgent, setShowCreateAgent] = useState(false);
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterEnvironment, setFilterEnvironment] = useState<string>('all');

  const queryClient = useQueryClient();

  // Mock data for development
  const mockAgents: Agent[] = [
    {
      id: 'agent-1',
      name: 'OSINT Lead Hunter',
      type: 'osint',
      status: 'active',
      version: 'v2.1.3',
      environment: 'production',
      created_at: '2024-01-15T10:30:00Z',
      last_deployed: '2024-01-20T14:22:00Z',
      performance_score: 94,
      resource_usage: { cpu: 45, memory: 62, network: 23 },
      build_info: {
        last_build: '2024-01-20T14:15:00Z',
        build_status: 'success',
        build_time: 180,
        tests_passed: 47,
        tests_total: 50
      },
      deployment_info: {
        replicas: 3,
        health_checks: true,
        auto_scaling: true,
        rollback_available: true
      }
    },
    {
      id: 'agent-2',
      name: 'Market Research Analyst',
      type: 'research',
      status: 'building',
      version: 'v1.8.2',
      environment: 'staging',
      created_at: '2024-01-10T09:15:00Z',
      last_deployed: '2024-01-19T11:45:00Z',
      performance_score: 87,
      resource_usage: { cpu: 78, memory: 55, network: 34 },
      build_info: {
        last_build: '2024-01-20T15:30:00Z',
        build_status: 'building',
        build_time: 0,
        tests_passed: 0,
        tests_total: 42
      },
      deployment_info: {
        replicas: 2,
        health_checks: true,
        auto_scaling: false,
        rollback_available: true
      }
    },
    {
      id: 'agent-3',
      name: 'Communication Coordinator',
      type: 'communication',
      status: 'idle',
      version: 'v3.0.1',
      environment: 'production',
      created_at: '2024-01-08T16:20:00Z',
      last_deployed: '2024-01-18T13:10:00Z',
      performance_score: 91,
      resource_usage: { cpu: 12, memory: 28, network: 15 },
      build_info: {
        last_build: '2024-01-18T13:05:00Z',
        build_status: 'success',
        build_time: 145,
        tests_passed: 38,
        tests_total: 38
      },
      deployment_info: {
        replicas: 1,
        health_checks: true,
        auto_scaling: false,
        rollback_available: false
      }
    },
    {
      id: 'agent-4',
      name: 'Predictive Analytics Engine',
      type: 'analysis',
      status: 'error',
      version: 'v2.3.1',
      environment: 'production',
      created_at: '2024-01-12T11:45:00Z',
      last_deployed: '2024-01-19T16:30:00Z',
      performance_score: 76,
      resource_usage: { cpu: 89, memory: 92, network: 67 },
      build_info: {
        last_build: '2024-01-20T10:20:00Z',
        build_status: 'failed',
        build_time: 0,
        tests_passed: 23,
        tests_total: 35
      },
      deployment_info: {
        replicas: 2,
        health_checks: false,
        auto_scaling: true,
        rollback_available: true
      }
    }
  ];

  const mockPipelines: DeploymentPipeline[] = [
    {
      id: 'pipeline-1',
      name: 'OSINT Lead Hunter - Production Deploy',
      agent_id: 'agent-1',
      stages: [
        { name: 'Build', status: 'success', duration: 120 },
        { name: 'Test', status: 'success', duration: 85 },
        { name: 'Security Scan', status: 'success', duration: 45 },
        { name: 'Deploy', status: 'success', duration: 30 }
      ],
      status: 'success',
      created_at: '2024-01-20T14:15:00Z',
      completed_at: '2024-01-20T14:22:00Z'
    },
    {
      id: 'pipeline-2',
      name: 'Market Research Analyst - Staging Deploy',
      agent_id: 'agent-2',
      stages: [
        { name: 'Build', status: 'running', duration: 0 },
        { name: 'Test', status: 'pending' },
        { name: 'Security Scan', status: 'pending' },
        { name: 'Deploy', status: 'pending' }
      ],
      status: 'running',
      created_at: '2024-01-20T15:30:00Z'
    }
  ];

  const mockMetrics: SystemMetrics = {
    total_agents: 4,
    active_deployments: 2,
    successful_builds: 23,
    failed_builds: 2,
    avg_build_time: 165,
    system_health: 87,
    resource_utilization: {
      cpu: 56,
      memory: 59,
      storage: 34,
      network: 35
    }
  };

  const { data: agents = mockAgents, isLoading: agentsLoading } = useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: async (): Promise<Agent[]> => {
      try {
        const response = await apiService.request<Agent[]>('/api/v1/agents', { method: 'GET' });
        return response.data || mockAgents;
      } catch (error) {
        return mockAgents;
      }
    },
    refetchInterval: 10000
  });

  const { data: pipelines = mockPipelines, isLoading: pipelinesLoading } = useQuery<DeploymentPipeline[]>({
    queryKey: ['deployment-pipelines'],
    queryFn: async (): Promise<DeploymentPipeline[]> => {
      try {
        const response = await apiService.request<DeploymentPipeline[]>('/api/v1/deployments/pipelines', { method: 'GET' });
        return response.data || mockPipelines;
      } catch (error) {
        return mockPipelines;
      }
    },
    refetchInterval: 5000
  });

  const { data: metrics = mockMetrics, isLoading: metricsLoading } = useQuery<SystemMetrics>({
    queryKey: ['system-metrics'],
    queryFn: async (): Promise<SystemMetrics> => {
      try {
        const response = await apiService.request<SystemMetrics>('/api/v1/system/metrics', { method: 'GET' });
        return response.data || mockMetrics;
      } catch (error) {
        return mockMetrics;
      }
    },
    refetchInterval: 30000
  });

  const buildAgentMutation = useMutation({
    mutationFn: async (agentId: string) => {
      const response = await apiService.request(`/api/v1/agents/${agentId}/build`, { method: 'POST' });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agents'] });
      queryClient.invalidateQueries({ queryKey: ['deployment-pipelines'] });
    }
  });

  const deployAgentMutation = useMutation({
    mutationFn: async ({ agentId, environment }: { agentId: string; environment: string }) => {
      const response = await apiService.request(`/api/v1/agents/${agentId}/deploy`, { method: 'POST', body: JSON.stringify({ environment }), headers: { 'Content-Type': 'application/json' } });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agents'] });
      queryClient.invalidateQueries({ queryKey: ['deployment-pipelines'] });
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'building': return 'text-yellow-400';
      case 'deploying': return 'text-blue-400';
      case 'idle': return 'text-gray-400';
      case 'error': return 'text-red-400';
      case 'offline': return 'text-gray-600';
      default: return 'text-gray-400';
    }
  };

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500/20';
      case 'building': return 'bg-yellow-500/20';
      case 'deploying': return 'bg-blue-500/20';
      case 'idle': return 'bg-gray-500/20';
      case 'error': return 'bg-red-500/20';
      case 'offline': return 'bg-blue-700 hover:bg-blue-800/20';
      default: return 'bg-gray-500/20';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'osint': return <Shield className="w-4 h-4" />;
      case 'research': return <Search className="w-4 h-4" />;
      case 'analysis': return <BarChart3 className="w-4 h-4" />;
      case 'communication': return <MessageSquare className="w-4 h-4" />;
      case 'automation': return <Zap className="w-4 h-4" />;
      case 'monitoring': return <Monitor className="w-4 h-4" />;
      default: return <Brain className="w-4 h-4" />;
    }
  };

  const filteredAgents = agents.filter(agent => {
    const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         agent.type.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || agent.status === filterStatus;
    const matchesEnvironment = filterEnvironment === 'all' || agent.environment === filterEnvironment;
    return matchesSearch && matchesStatus && matchesEnvironment;
  });

  const handleBuildAgent = (agentId: string) => {
    buildAgentMutation.mutate(agentId);
  };

  const handleDeployAgent = (agentId: string, environment: string) => {
    deployAgentMutation.mutate({ agentId, environment });
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Rocket className="w-8 h-8 mr-3 text-blue-400" />
            PAI Hub Management
          </h1>
          <p className="text-gray-400 mt-2">
            Build, deploy, and manage your Personal AI agent ecosystem
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowCreateAgent(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Create Agent</span>
          </button>
          <div className="flex items-center space-x-2 px-3 py-1 rounded-full glass">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-green-400">System Healthy</span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-white/5 p-1 rounded-lg">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'agents', label: 'Agents', icon: Users },
          { id: 'deployments', label: 'Deployments', icon: Rocket },
          { id: 'builds', label: 'Builds', icon: Package },
          { id: 'monitoring', label: 'Monitoring', icon: Monitor },
          { id: 'settings', label: 'Settings', icon: Settings }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* System Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Agents</p>
                  <p className="text-2xl font-bold text-white">{metrics.total_agents}</p>
                </div>
                <Users className="w-8 h-8 text-blue-400" />
              </div>
              <p className="text-green-400 text-sm mt-2">+2 this week</p>
            </div>

            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Active Deployments</p>
                  <p className="text-2xl font-bold text-white">{metrics.active_deployments}</p>
                </div>
                <Rocket className="w-8 h-8 text-green-400" />
              </div>
              <p className="text-blue-400 text-sm mt-2">Running smoothly</p>
            </div>

            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Build Success Rate</p>
                  <p className="text-2xl font-bold text-white">
                    {Math.round((metrics.successful_builds / (metrics.successful_builds + metrics.failed_builds)) * 100)}%
                  </p>
                </div>
                <CheckCircle className="w-8 h-8 text-green-400" />
              </div>
              <p className="text-green-400 text-sm mt-2">{metrics.successful_builds}/{metrics.successful_builds + metrics.failed_builds} builds</p>
            </div>

            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Avg Build Time</p>
                  <p className="text-2xl font-bold text-white">{metrics.avg_build_time}s</p>
                </div>
                <Clock className="w-8 h-8 text-yellow-400" />
              </div>
              <p className="text-yellow-400 text-sm mt-2">-15s from last week</p>
            </div>
          </div>

          {/* Resource Utilization */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Cpu className="w-5 h-5 mr-2 text-blue-400" />
                Resource Utilization
              </h3>
              <div className="space-y-4">
                {Object.entries(metrics.resource_utilization).map(([resource, usage]) => (
                  <div key={resource}>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400 capitalize">{resource}</span>
                      <span className="text-white">{usage}%</span>
                    </div>
                    <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          usage > 80 ? 'bg-gradient-to-r from-red-500 to-red-700' :
                          usage > 60 ? 'bg-gradient-to-r from-yellow-500 to-orange-500' :
                          'bg-gradient-to-r from-green-500 to-blue-500'
                        }`}
                        style={{ width: `${usage}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-green-400" />
                Recent Deployments
              </h3>
              <div className="space-y-3">
                {pipelines.slice(0, 5).map((pipeline) => (
                  <div key={pipeline.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-2 h-2 rounded-full ${
                        pipeline.status === 'success' ? 'bg-green-400' :
                        pipeline.status === 'running' ? 'bg-yellow-400 animate-pulse' :
                        pipeline.status === 'failed' ? 'bg-red-400' : 'bg-gray-400'
                      }`}></div>
                      <div>
                        <p className="text-sm text-white">{pipeline.name}</p>
                        <p className="text-xs text-gray-400">
                          {new Date(pipeline.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      pipeline.status === 'success' ? 'bg-green-500/20 text-green-400' :
                      pipeline.status === 'running' ? 'bg-yellow-500/20 text-yellow-400' :
                      pipeline.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      {pipeline.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'agents' && (
        <div className="space-y-6">
          {/* Filters and Search */}
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search agents..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
              </div>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="idle">Idle</option>
                <option value="building">Building</option>
                <option value="deploying">Deploying</option>
                <option value="error">Error</option>
                <option value="offline">Offline</option>
              </select>
              <select
                value={filterEnvironment}
                onChange={(e) => setFilterEnvironment(e.target.value)}
                className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Environments</option>
                <option value="development">Development</option>
                <option value="staging">Staging</option>
                <option value="production">Production</option>
              </select>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => queryClient.invalidateQueries({ queryKey: ['agents'] })}
                className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Refresh</span>
              </button>
            </div>
          </div>

          {/* Agents Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredAgents.map((agent) => (
              <div key={agent.id} className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${getStatusBg(agent.status)}`}>
                      {getTypeIcon(agent.type)}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
                      <p className="text-sm text-gray-400">{agent.type} • {agent.version}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`text-xs px-2 py-1 rounded-full ${getStatusBg(agent.status)} ${getStatusColor(agent.status)}`}>
                      {agent.status}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      agent.environment === 'production' ? 'bg-green-500/20 text-green-400' :
                      agent.environment === 'staging' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-blue-500/20 text-blue-400'
                    }`}>
                      {agent.environment}
                    </span>
                  </div>
                </div>

                <div className="space-y-3 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Performance</span>
                    <span className="text-white">{agent.performance_score}%</span>
                  </div>
                  <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                      style={{ width: `${agent.performance_score}%` }}
                    ></div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="text-center">
                      <p className="text-gray-400">CPU</p>
                      <p className="text-white">{agent.resource_usage.cpu}%</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-400">Memory</p>
                      <p className="text-white">{agent.resource_usage.memory}%</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-400">Network</p>
                      <p className="text-white">{agent.resource_usage.network}%</p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleBuildAgent(agent.id)}
                      disabled={agent.status === 'building' || buildAgentMutation.isPending}
                      className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-700 hover:bg-blue-800 disabled:cursor-not-allowed rounded text-xs transition-colors"
                    >
                      <Package className="w-3 h-3" />
                      <span>Build</span>
                    </button>
                    <button
                      onClick={() => {
                        setSelectedAgent(agent.id);
                        setShowDeployModal(true);
                      }}
                      disabled={agent.status === 'deploying' || deployAgentMutation.isPending}
                      className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 disabled:bg-blue-700 hover:bg-blue-800 disabled:cursor-not-allowed rounded text-xs transition-colors"
                    >
                      <Rocket className="w-3 h-3" />
                      <span>Deploy</span>
                    </button>
                  </div>
                  <div className="flex items-center space-x-1">
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Eye className="w-4 h-4 text-gray-400" />
                    </button>
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Edit className="w-4 h-4 text-gray-400" />
                    </button>
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Settings className="w-4 h-4 text-gray-400" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'deployments' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">Deployment Pipelines</h2>
            <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
              <Plus className="w-4 h-4" />
              <span>New Pipeline</span>
            </button>
          </div>

          <div className="space-y-4">
            {pipelines.map((pipeline) => (
              <div key={pipeline.id} className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{pipeline.name}</h3>
                    <p className="text-sm text-gray-400">
                      Started {new Date(pipeline.created_at).toLocaleString()}
                      {pipeline.completed_at && ` • Completed ${new Date(pipeline.completed_at).toLocaleString()}`}
                    </p>
                  </div>
                  <span className={`text-xs px-3 py-1 rounded-full ${
                    pipeline.status === 'success' ? 'bg-green-500/20 text-green-400' :
                    pipeline.status === 'running' ? 'bg-yellow-500/20 text-yellow-400' :
                    pipeline.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {pipeline.status}
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {pipeline.stages.map((stage, index) => (
                    <div key={index} className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        stage.status === 'success' ? 'bg-green-500/20 text-green-400' :
                        stage.status === 'running' ? 'bg-yellow-500/20 text-yellow-400' :
                        stage.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {stage.status === 'success' ? <CheckCircle className="w-4 h-4" /> :
                         stage.status === 'running' ? <Clock className="w-4 h-4 animate-spin" /> :
                         stage.status === 'failed' ? <AlertTriangle className="w-4 h-4" /> :
                         <Clock className="w-4 h-4" />}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-white">{stage.name}</p>
                        {stage.duration && (
                          <p className="text-xs text-gray-400">{stage.duration}s</p>
                        )}
                      </div>
                      {index < pipeline.stages.length - 1 && (
                        <ArrowRight className="w-4 h-4 text-gray-600" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'builds' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">Build History</h2>
            <div className="flex items-center space-x-2">
              <button className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors">
                <Filter className="w-4 h-4" />
                <span>Filter</span>
              </button>
              <button className="flex items-center space-x-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
                <RefreshCw className="w-4 h-4" />
                <span>Refresh</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {agents.map((agent) => (
              <div key={agent.id} className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${getStatusBg(agent.build_info.build_status)}`}>
                      {getTypeIcon(agent.type)}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
                      <p className="text-sm text-gray-400">Version {agent.version}</p>
                    </div>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    agent.build_info.build_status === 'success' ? 'bg-green-500/20 text-green-400' :
                    agent.build_info.build_status === 'building' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {agent.build_info.build_status}
                  </span>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Last Build</span>
                    <span className="text-white">
                      {new Date(agent.build_info.last_build).toLocaleString()}
                    </span>
                  </div>

                  {agent.build_info.build_time > 0 && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Build Time</span>
                      <span className="text-white">{agent.build_info.build_time}s</span>
                    </div>
                  )}

                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Tests</span>
                    <span className={`${
                      agent.build_info.tests_passed === agent.build_info.tests_total
                        ? 'text-green-400'
                        : 'text-yellow-400'
                    }`}>
                      {agent.build_info.tests_passed}/{agent.build_info.tests_total}
                    </span>
                  </div>

                  <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        agent.build_info.tests_passed === agent.build_info.tests_total
                          ? 'bg-gradient-to-r from-green-500 to-green-700'
                          : 'bg-gradient-to-r from-yellow-500 to-orange-500'
                      }`}
                      style={{
                        width: `${(agent.build_info.tests_passed / agent.build_info.tests_total) * 100}%`
                      }}
                    ></div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600 mt-4">
                  <button
                    onClick={() => handleBuildAgent(agent.id)}
                    disabled={agent.build_info.build_status === 'building'}
                    className="flex items-center space-x-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-700 hover:bg-blue-800 disabled:cursor-not-allowed rounded text-sm transition-colors"
                  >
                    <Package className="w-4 h-4" />
                    <span>Rebuild</span>
                  </button>
                  <div className="flex items-center space-x-1">
                    <button className="p-1 hover:bg-white/10 rounded">
                      <FileText className="w-4 h-4 text-gray-400" />
                    </button>
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Download className="w-4 h-4 text-gray-400" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'monitoring' && (
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-white">System Monitoring</h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-green-400" />
                Real-time Metrics
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">System Health</span>
                  <span className="text-green-400">{metrics.system_health}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Active Agents</span>
                  <span className="text-blue-400">{agents.filter(a => a.status === 'active').length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Running Deployments</span>
                  <span className="text-yellow-400">{pipelines.filter(p => p.status === 'running').length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Error Rate</span>
                  <span className="text-red-400">0.2%</span>
                </div>
              </div>
            </div>

            <div className="glass p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-yellow-400" />
                Alerts & Issues
              </h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  <div>
                    <p className="text-sm text-white">High Memory Usage</p>
                    <p className="text-xs text-gray-400">Predictive Analytics Engine</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <div>
                    <p className="text-sm text-white">Build Failed</p>
                    <p className="text-xs text-gray-400">Predictive Analytics Engine v2.3.1</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'settings' && (
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-white">System Settings</h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-white mb-4">Build Configuration</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Default Build Timeout</label>
                  <input
                    type="number"
                    defaultValue="300"
                    className="w-full px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Auto-deploy on Success</label>
                  <input type="checkbox" className="rounded" />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Parallel Builds</label>
                  <input type="checkbox" defaultChecked className="rounded" />
                </div>
              </div>
            </div>

            <div className="glass p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-white mb-4">Deployment Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Default Environment</label>
                  <select className="w-full px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500">
                    <option value="staging">Staging</option>
                    <option value="production">Production</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Health Check Timeout</label>
                  <input
                    type="number"
                    defaultValue="60"
                    className="w-full px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Auto-rollback on Failure</label>
                  <input type="checkbox" defaultChecked className="rounded" />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PAIHubManagement;
