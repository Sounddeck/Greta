import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Brain, 
  Command, 
  Activity, 
  Users, 
  Settings, 
  Play, 
  Pause, 
  Square, 
  BarChart3, 
  Zap, 
  Target, 
  Globe, 
  Database, 
  MessageSquare, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  TrendingUp,
  Layers,
  Network,
  Eye,
  Shield,
  Cpu,
  HardDrive,
  Wifi,
  Server,
  Monitor,
  Terminal,
  Code,
  FileText,
  Search,
  Filter,
  Download,
  Upload,
  RefreshCw,
  Plus,
  Minus,
  Edit,
  Trash2,
  Save,
  X,
  ChevronRight,
  ChevronDown,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  Rocket
} from 'lucide-react';
import { apiService } from '../services/api';

interface SystemAgent {
  id: string;
  name: string;
  type: 'osint' | 'research' | 'analysis' | 'communication' | 'automation' | 'monitoring';
  status: 'active' | 'idle' | 'busy' | 'error' | 'offline';
  current_task?: string;
  performance_score: number;
  tasks_completed: number;
  uptime: number;
  last_activity: string;
  capabilities: string[];
  resource_usage: {
    cpu: number;
    memory: number;
    network: number;
  };
}

interface SystemTask {
  id: string;
  title: string;
  description: string;
  type: 'osint' | 'research' | 'analysis' | 'automation' | 'monitoring';
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'queued' | 'assigned' | 'in_progress' | 'completed' | 'failed';
  assigned_agent?: string;
  created_at: string;
  estimated_duration: number;
  progress: number;
  dependencies: string[];
}

interface SystemMetrics {
  total_agents: number;
  active_agents: number;
  total_tasks: number;
  completed_tasks: number;
  system_load: number;
  memory_usage: number;
  network_activity: number;
  error_rate: number;
  uptime: number;
  performance_score: number;
}

interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  steps: Array<{
    id: string;
    name: string;
    agent_type: string;
    parameters: any;
    dependencies: string[];
  }>;
  estimated_duration: number;
  success_rate: number;
}

export const MasterAgentEnhanced: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'command' | 'agents' | 'tasks' | 'workflows' | 'monitoring' | 'analytics'>('command');
  const [commandInput, setCommandInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<Array<{
    id: string;
    command: string;
    response: string;
    timestamp: string;
    status: 'success' | 'error' | 'pending';
  }>>([]);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [showCreateWorkflow, setShowCreateWorkflow] = useState(false);
  const [autoMode, setAutoMode] = useState(false);
  const [showCreateAgent, setShowCreateAgent] = useState(false);
  const [showAgentDetails, setShowAgentDetails] = useState(false);
  const [agentScaling, setAgentScaling] = useState<{[key: string]: number}>({});
  const [systemMode, setSystemMode] = useState<'manual' | 'auto' | 'hybrid'>('manual');
  const [selectedEnvironment, setSelectedEnvironment] = useState<'development' | 'staging' | 'production'>('production');

  const queryClient = useQueryClient();

  const { data: agents, isLoading: agentsLoading } = useQuery<SystemAgent[]>({
    queryKey: ['system-agents'],
    queryFn: async () => {
      try {
        const response = await apiService.request<SystemAgent[]>('/api/v1/master-agent/agents');
        return response.data || getMockAgents();
      } catch (error) {
        return getMockAgents();
      }
    },
    refetchInterval: 5000
  });

  const { data: tasks, isLoading: tasksLoading } = useQuery<SystemTask[]>({
    queryKey: ['system-tasks'],
    queryFn: async () => {
      try {
        const response = await apiService.request<SystemTask[]>('/api/v1/master-agent/tasks');
        return response.data || getMockTasks();
      } catch (error) {
        return getMockTasks();
      }
    },
    refetchInterval: 10000
  });

  const { data: metrics, isLoading: metricsLoading } = useQuery<SystemMetrics>({
    queryKey: ['system-metrics'],
    queryFn: async () => {
      try {
        const response = await apiService.request<SystemMetrics>('/api/v1/master-agent/metrics');
        return response.data || getMockMetrics();
      } catch (error) {
        return getMockMetrics();
      }
    },
    refetchInterval: 2000
  });

  const { data: workflows, isLoading: workflowsLoading } = useQuery<WorkflowTemplate[]>({
    queryKey: ['workflow-templates'],
    queryFn: async () => {
      try {
        const response = await apiService.request<WorkflowTemplate[]>('/api/v1/master-agent/workflows');
        return response.data || getMockWorkflows();
      } catch (error) {
        return getMockWorkflows();
      }
    }
  });

  const executeCommandMutation = useMutation({
    mutationFn: async (command: string) => {
      const response = await apiService.request('/api/v1/master-agent/command', {
        method: 'POST',
        body: JSON.stringify({ command })
      });
      return response.data;
    },
    onSuccess: (data, command) => {
      const newEntry = {
        id: Date.now().toString(),
        command,
        response: (data as any)?.response || 'Command executed successfully',
        timestamp: new Date().toISOString(),
        status: 'success' as const
      };
      setCommandHistory(prev => [newEntry, ...prev.slice(0, 49)]);
      setCommandInput('');
      queryClient.invalidateQueries({ queryKey: ['system-agents'] });
      queryClient.invalidateQueries({ queryKey: ['system-tasks'] });
    },
    onError: (error, command) => {
      const newEntry = {
        id: Date.now().toString(),
        command,
        response: `Error: ${error instanceof Error ? error.message : 'Command failed'}`,
        timestamp: new Date().toISOString(),
        status: 'error' as const
      };
      setCommandHistory(prev => [newEntry, ...prev.slice(0, 49)]);
    }
  });

  const createAgentMutation = useMutation({
    mutationFn: async (agentData: {
      name: string;
      type: string;
      description: string;
      environment: string;
      config: any;
    }) => {
      const response = await apiService.request('/api/v1/master-agent/agents', {
        method: 'POST',
        body: JSON.stringify(agentData)
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-agents'] });
      setShowCreateAgent(false);
    }
  });

  const scaleAgentMutation = useMutation({
    mutationFn: async ({ agentId, replicas }: { agentId: string; replicas: number }) => {
      const response = await apiService.request(`/api/v1/master-agent/agents/${agentId}/scale`, {
        method: 'POST',
        body: JSON.stringify({ replicas })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-agents'] });
    }
  });

  const deployAgentMutation = useMutation({
    mutationFn: async ({ agentId, environment }: { agentId: string; environment: string }) => {
      const response = await apiService.request(`/api/v1/master-agent/agents/${agentId}/deploy`, {
        method: 'POST',
        body: JSON.stringify({ environment })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-agents'] });
    }
  });

  const stopAgentMutation = useMutation({
    mutationFn: async (agentId: string) => {
      const response = await apiService.request(`/api/v1/master-agent/agents/${agentId}/stop`, {
        method: 'POST'
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-agents'] });
    }
  });

  const startAgentMutation = useMutation({
    mutationFn: async (agentId: string) => {
      const response = await apiService.request(`/api/v1/master-agent/agents/${agentId}/start`, {
        method: 'POST'
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-agents'] });
    }
  });

  const getMockAgents = (): SystemAgent[] => [
    {
      id: 'agent-osint-1',
      name: 'OSINT Lead Hunter',
      type: 'osint',
      status: 'active',
      current_task: 'Scanning LinkedIn for tech startup CTOs',
      performance_score: 94,
      tasks_completed: 247,
      uptime: 99.7,
      last_activity: new Date(Date.now() - 300000).toISOString(),
      capabilities: ['social_media_analysis', 'company_research', 'contact_discovery'],
      resource_usage: { cpu: 45, memory: 67, network: 23 }
    },
    {
      id: 'agent-research-1',
      name: 'Market Research Analyst',
      type: 'research',
      status: 'busy',
      current_task: 'Analyzing fintech market trends Q4 2024',
      performance_score: 87,
      tasks_completed: 156,
      uptime: 98.2,
      last_activity: new Date(Date.now() - 120000).toISOString(),
      capabilities: ['market_analysis', 'trend_identification', 'competitive_intelligence'],
      resource_usage: { cpu: 72, memory: 84, network: 56 }
    },
    {
      id: 'agent-comm-1',
      name: 'Communication Coordinator',
      type: 'communication',
      status: 'idle',
      performance_score: 91,
      tasks_completed: 89,
      uptime: 99.1,
      last_activity: new Date(Date.now() - 600000).toISOString(),
      capabilities: ['email_automation', 'social_media_posting', 'content_generation'],
      resource_usage: { cpu: 12, memory: 34, network: 8 }
    },
    {
      id: 'agent-monitor-1',
      name: 'System Monitor',
      type: 'monitoring',
      status: 'active',
      current_task: 'Real-time system health monitoring',
      performance_score: 99,
      tasks_completed: 1247,
      uptime: 99.9,
      last_activity: new Date(Date.now() - 30000).toISOString(),
      capabilities: ['system_monitoring', 'performance_analysis', 'alert_management'],
      resource_usage: { cpu: 15, memory: 28, network: 45 }
    }
  ];

  const getMockTasks = (): SystemTask[] => [
    {
      id: 'task-1',
      title: 'Healthcare IT Decision Maker Research',
      description: 'Identify and profile IT decision makers at mid-size healthcare organizations',
      type: 'osint',
      priority: 'high',
      status: 'in_progress',
      assigned_agent: 'agent-osint-1',
      created_at: new Date(Date.now() - 3600000).toISOString(),
      estimated_duration: 120,
      progress: 65,
      dependencies: []
    },
    {
      id: 'task-2',
      title: 'Competitor Analysis - Q4 Report',
      description: 'Comprehensive analysis of top 5 competitors including funding, personnel, and product updates',
      type: 'research',
      priority: 'medium',
      status: 'assigned',
      assigned_agent: 'agent-research-1',
      created_at: new Date(Date.now() - 1800000).toISOString(),
      estimated_duration: 240,
      progress: 0,
      dependencies: ['task-1']
    },
    {
      id: 'task-3',
      title: 'Lead Nurturing Campaign',
      description: 'Automated email sequence for qualified leads from OSINT research',
      type: 'automation',
      priority: 'medium',
      status: 'queued',
      created_at: new Date(Date.now() - 900000).toISOString(),
      estimated_duration: 60,
      progress: 0,
      dependencies: ['task-1']
    }
  ];

  const getMockMetrics = (): SystemMetrics => ({
    total_agents: 4,
    active_agents: 3,
    total_tasks: 12,
    completed_tasks: 8,
    system_load: 42,
    memory_usage: 67,
    network_activity: 34,
    error_rate: 2.1,
    uptime: 99.7,
    performance_score: 91
  });

  const getMockWorkflows = (): WorkflowTemplate[] => [
    {
      id: 'workflow-1',
      name: 'Complete Lead Generation Pipeline',
      description: 'End-to-end lead generation from OSINT research to qualified outreach',
      steps: [
        { id: 'step-1', name: 'OSINT Research', agent_type: 'osint', parameters: {}, dependencies: [] },
        { id: 'step-2', name: 'Lead Qualification', agent_type: 'analysis', parameters: {}, dependencies: ['step-1'] },
        { id: 'step-3', name: 'Outreach Campaign', agent_type: 'communication', parameters: {}, dependencies: ['step-2'] }
      ],
      estimated_duration: 180,
      success_rate: 87.3
    }
  ];

  const getAgentStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Activity className="w-4 h-4 text-green-400 animate-pulse" />;
      case 'busy': return <Zap className="w-4 h-4 text-yellow-400" />;
      case 'idle': return <Clock className="w-4 h-4 text-gray-400" />;
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case 'offline': return <X className="w-4 h-4 text-red-500" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getAgentTypeIcon = (type: string) => {
    switch (type) {
      case 'osint': return <Eye className="w-5 h-5" />;
      case 'research': return <Search className="w-5 h-5" />;
      case 'analysis': return <BarChart3 className="w-5 h-5" />;
      case 'communication': return <MessageSquare className="w-5 h-5" />;
      case 'automation': return <Zap className="w-5 h-5" />;
      case 'monitoring': return <Monitor className="w-5 h-5" />;
      default: return <Brain className="w-5 h-5" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'text-red-400 bg-red-400/20';
      case 'high': return 'text-orange-400 bg-orange-400/20';
      case 'medium': return 'text-yellow-400 bg-yellow-400/20';
      case 'low': return 'text-green-400 bg-green-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const handleCommand = () => {
    if (commandInput.trim()) {
      executeCommandMutation.mutate(commandInput.trim());
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleCommand();
    }
  };

  if (agentsLoading || tasksLoading || metricsLoading) {
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
            <Command className="w-8 h-8 mr-3 text-blue-400" />
            Master Agent Control Center
          </h1>
          <p className="text-gray-400 mt-2">
            Central command and orchestration for the entire GRETA PAI system
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center space-x-2">
            <span className="text-gray-400 text-sm">Auto Mode</span>
            <button
              onClick={() => setAutoMode(!autoMode)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                autoMode ? 'bg-blue-600' : 'bg-blue-700 hover:bg-blue-800'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  autoMode ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <div className="flex items-center space-x-2 text-sm">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-green-400">System Online</span>
          </div>
        </div>
      </div>

      {/* System Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        <div className="glass p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Agents</p>
              <p className="text-2xl font-bold text-white">{metrics?.active_agents || 0}</p>
            </div>
            <Users className="w-6 h-6 text-blue-400" />
          </div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Running Tasks</p>
              <p className="text-2xl font-bold text-white">
                {(tasks || []).filter(t => t.status === 'in_progress').length}
              </p>
            </div>
            <Activity className="w-6 h-6 text-green-400" />
          </div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">System Load</p>
              <p className="text-2xl font-bold text-white">{metrics?.system_load || 0}%</p>
            </div>
            <Cpu className="w-6 h-6 text-yellow-400" />
          </div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Memory Usage</p>
              <p className="text-2xl font-bold text-white">{metrics?.memory_usage || 0}%</p>
            </div>
            <HardDrive className="w-6 h-6 text-purple-400" />
          </div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Performance</p>
              <p className="text-2xl font-bold text-white">{metrics?.performance_score || 0}%</p>
            </div>
            <TrendingUp className="w-6 h-6 text-green-400" />
          </div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Uptime</p>
              <p className="text-2xl font-bold text-white">{metrics?.uptime || 0}%</p>
            </div>
            <Server className="w-6 h-6 text-blue-400" />
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-white/5 p-1 rounded-lg">
        {[
          { id: 'command', label: 'Command Center', icon: Terminal },
          { id: 'agents', label: 'Agent Fleet', icon: Users },
          { id: 'tasks', label: 'Task Queue', icon: Target },
          { id: 'workflows', label: 'Workflows', icon: Network },
          { id: 'monitoring', label: 'Monitoring', icon: Monitor },
          { id: 'analytics', label: 'Analytics', icon: BarChart3 }
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

      {/* Command Center Tab */}
      {activeTab === 'command' && (
        <div className="space-y-6">
          {/* Command Input */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Terminal className="w-5 h-5 mr-2 text-blue-400" />
              System Command Interface
            </h3>

            <div className="space-y-4">
              <div className="flex space-x-4">
                <div className="flex-1">
                  <textarea
                    value={commandInput}
                    onChange={(e) => setCommandInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="w-full h-24 px-4 py-3 bg-black/50 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 font-mono text-sm resize-none"
                    placeholder="Enter system commands... (e.g., 'Start OSINT program for fintech startups', 'Deploy lead generation workflow', 'Scale up research agents')"
                  />
                </div>
                <div className="flex flex-col space-y-2">
                  <button
                    onClick={handleCommand}
                    disabled={executeCommandMutation.isPending || !commandInput.trim()}
                    className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                  >
                    <Play className="w-4 h-4" />
                    <span>Execute</span>
                  </button>
                  <button
                    onClick={() => setCommandInput('')}
                    className="px-6 py-3 bg-blue-700 hover:bg-blue-800 text-white rounded-lg transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Quick Commands */}
              <div className="flex flex-wrap gap-2">
                {[
                  'Status report',
                  'Start OSINT scan',
                  'Deploy lead workflow',
                  'Scale agents',
                  'System health check',
                  'Emergency stop'
                ].map((cmd) => (
                  <button
                    key={cmd}
                    onClick={() => setCommandInput(cmd)}
                    className="px-3 py-1 bg-white/10 hover:bg-white/20 text-gray-300 text-sm rounded-lg transition-colors"
                  >
                    {cmd}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Command History */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <FileText className="w-5 h-5 mr-2 text-green-400" />
              Command History
            </h3>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {commandHistory.length === 0 ? (
                <p className="text-gray-400 text-center py-8">No commands executed yet</p>
              ) : (
                commandHistory.map((entry) => (
                  <div key={entry.id} className="bg-white/5 p-4 rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {entry.status === 'success' ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : entry.status === 'error' ? (
                          <AlertTriangle className="w-4 h-4 text-red-400" />
                        ) : (
                          <Clock className="w-4 h-4 text-yellow-400 animate-pulse" />
                        )}
                        <span className="text-white font-mono text-sm">$ {entry.command}</span>
                      </div>
                      <span className="text-gray-400 text-xs">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className={`text-sm font-mono ml-6 ${
                      entry.status === 'error' ? 'text-red-300' : 'text-gray-300'
                    }`}>
                      {entry.response}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Agent Fleet Tab */}
      {activeTab === 'agents' && (
        <div className="space-y-6">
          <div className="glass p-6 rounded-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Agent Fleet Management</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowCreateAgent(true)}
                  className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
                >
                  <Plus className="w-4 h-4 inline mr-1" />
                  Create Agent
                </button>
                <select
                  value={selectedEnvironment}
                  onChange={(e) => setSelectedEnvironment(e.target.value as any)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="development">Development</option>
                  <option value="staging">Staging</option>
                  <option value="production">Production</option>
                </select>
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['system-agents'] })}
                  className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4 inline mr-1" />
                  Refresh
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {(agents || []).map((agent) => (
                <div key={agent.id} className="bg-white/5 p-6 rounded-xl hover:bg-white/10 transition-all">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        agent.type === 'osint' ? 'text-blue-400 bg-blue-400/20' :
                        agent.type === 'research' ? 'text-green-400 bg-green-400/20' :
                        agent.type === 'communication' ? 'text-purple-400 bg-purple-400/20' :
                        agent.type === 'monitoring' ? 'text-yellow-400 bg-yellow-400/20' :
                        'text-gray-400 bg-gray-400/20'
                      }`}>
                        {getAgentTypeIcon(agent.type)}
                      </div>
                      <div>
                        <h4 className="text-white font-semibold">{agent.name}</h4>
                        <p className="text-gray-400 text-sm capitalize">{agent.type} Agent</p>
                      </div>
                    </div>
                    {getAgentStatusIcon(agent.status)}
                  </div>

                  {agent.current_task && (
                    <div className="mb-4 p-3 bg-white/5 rounded-lg">
                      <p className="text-gray-400 text-xs mb-1">Current Task</p>
                      <p className="text-white text-sm">{agent.current_task}</p>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-gray-400 text-xs">Performance</p>
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${agent.performance_score}%` }}
                          ></div>
                        </div>
                        <span className="text-white text-sm">{agent.performance_score}%</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs">Uptime</p>
                      <p className="text-white text-sm">{agent.uptime}%</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
                    <div className="bg-white/5 p-2 rounded">
                      <p className="text-gray-400">CPU</p>
                      <p className="text-white">{agent.resource_usage.cpu}%</p>
                    </div>
                    <div className="bg-white/5 p-2 rounded">
                      <p className="text-gray-400">Memory</p>
                      <p className="text-white">{agent.resource_usage.memory}%</p>
                    </div>
                    <div className="bg-white/5 p-2 rounded">
                      <p className="text-gray-400">Network</p>
                      <p className="text-white">{agent.resource_usage.network}%</p>
                    </div>
                  </div>

                  {/* Agent Management Controls */}
                  <div className="space-y-3">
                    <div className="flex gap-2">
                      {agent.status === 'idle' || agent.status === 'offline' ? (
                        <button
                          onClick={() => startAgentMutation.mutate(agent.id)}
                          disabled={startAgentMutation.isPending}
                          className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-blue-700 hover:bg-blue-800 text-white text-sm rounded-lg transition-colors"
                        >
                          <Play className="w-4 h-4 inline mr-1" />
                          Start
                        </button>
                      ) : (
                        <button
                          onClick={() => stopAgentMutation.mutate(agent.id)}
                          disabled={stopAgentMutation.isPending}
                          className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-blue-700 hover:bg-blue-800 text-white text-sm rounded-lg transition-colors"
                        >
                          <Square className="w-4 h-4 inline mr-1" />
                          Stop
                        </button>
                      )}
                      <button
                        onClick={() => deployAgentMutation.mutate({ agentId: agent.id, environment: selectedEnvironment })}
                        disabled={deployAgentMutation.isPending}
                        className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-700 hover:bg-blue-800 text-white text-sm rounded-lg transition-colors"
                      >
                        <Rocket className="w-4 h-4 inline mr-1" />
                        Deploy
                      </button>
                      <button
                        onClick={() => {
                          setSelectedAgent(agent.id);
                          setShowAgentDetails(true);
                        }}
                        className="px-3 py-2 bg-blue-700 hover:bg-blue-800 text-white text-sm rounded-lg transition-colors"
                      >
                        <Settings className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Scaling Controls */}
                    <div className="flex items-center space-x-2 text-sm">
                      <span className="text-gray-400">Replicas:</span>
                      <div className="flex items-center space-x-1">
                        <button
                          onClick={() => {
                            const currentReplicas = agentScaling[agent.id] || 1;
                            if (currentReplicas > 1) {
                              const newReplicas = currentReplicas - 1;
                              setAgentScaling(prev => ({ ...prev, [agent.id]: newReplicas }));
                              scaleAgentMutation.mutate({ agentId: agent.id, replicas: newReplicas });
                            }
                          }}
                          disabled={scaleAgentMutation.isPending || (agentScaling[agent.id] || 1) <= 1}
                          className="w-6 h-6 bg-blue-700 hover:bg-blue-800 disabled:bg-slate-800 hover:bg-slate-900 text-white rounded text-xs flex items-center justify-center"
                        >
                          -
                        </button>
                        <span className="w-8 text-center text-white">{agentScaling[agent.id] || 1}</span>
                        <button
                          onClick={() => {
                            const currentReplicas = agentScaling[agent.id] || 1;
                            const newReplicas = currentReplicas + 1;
                            setAgentScaling(prev => ({ ...prev, [agent.id]: newReplicas }));
                            scaleAgentMutation.mutate({ agentId: agent.id, replicas: newReplicas });
                          }}
                          disabled={scaleAgentMutation.isPending || (agentScaling[agent.id] || 1) >= 10}
                          className="w-6 h-6 bg-blue-700 hover:bg-blue-800 disabled:bg-slate-800 hover:bg-slate-900 text-white rounded text-xs flex items-center justify-center"
                        >
                          +
                        </button>
                      </div>
                      <span className="text-gray-400 text-xs">instances</span>
                    </div>

                    {/* Agent Metrics */}
                    <div className="flex justify-between text-xs text-gray-400">
                      <span>Tasks: {agent.tasks_completed || 0}</span>
                      <span>Last: {agent.last_activity ? new Date(agent.last_activity).toLocaleTimeString() : 'Never'}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Task Queue Tab */}
      {activeTab === 'tasks' && (
        <div className="space-y-6">
          <div className="glass p-6 rounded-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Task Queue Management</h3>
              <div className="flex gap-2">
                <button className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors">
                  <Plus className="w-4 h-4 inline mr-1" />
                  New Task
                </button>
                <button className="px-3 py-2 bg-blue-700 hover:bg-blue-800 text-white text-sm rounded-lg transition-colors">
                  <Filter className="w-4 h-4 inline mr-1" />
                  Filter
                </button>
              </div>
            </div>

            <div className="space-y-4">
              {(tasks || []).map((task) => (
                <div key={task.id} className="bg-white/5 p-6 rounded-xl hover:bg-white/10 transition-all">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="text-white font-semibold">{task.title}</h4>
                        <span className={`px-2 py-1 text-xs rounded-full ${getPriorityColor(task.priority)}`}>
                          {task.priority}
                        </span>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          task.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                          task.status === 'in_progress' ? 'bg-blue-500/20 text-blue-400' :
                          task.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {task.status.replace('_', ' ')}
                        </span>
                      </div>
                      <p className="text-gray-300 text-sm mb-3">{task.description}</p>

                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-gray-400">Type</p>
                          <p className="text-white capitalize">{task.type}</p>
                        </div>
                        <div>
                          <p className="text-gray-400">Assigned Agent</p>
                          <p className="text-white">
                            {task.assigned_agent ?
                              (agents || []).find(a => a.id === task.assigned_agent)?.name || 'Unknown' :
                              'Unassigned'
                            }
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-400">Duration</p>
                          <p className="text-white">{task.estimated_duration}m</p>
                        </div>
                        <div>
                          <p className="text-gray-400">Created</p>
                          <p className="text-white">{new Date(task.created_at).toLocaleDateString()}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {task.status === 'in_progress' && (
                    <div className="mb-4">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">Progress</span>
                        <span className="text-white">{task.progress}%</span>
                      </div>
                      <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${task.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2">
                    {task.status === 'queued' && (
                      <button className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors">
                        <Play className="w-4 h-4 inline mr-1" />
                        Start
                      </button>
                    )}
                    {task.status === 'in_progress' && (
                      <button className="px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded-lg transition-colors">
                        <Pause className="w-4 h-4 inline mr-1" />
                        Pause
                      </button>
                    )}
                    <button className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors">
                      <Edit className="w-4 h-4 inline mr-1" />
                      Edit
                    </button>
                    <button className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors">
                      <Trash2 className="w-4 h-4 inline mr-1" />
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
