import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Server, 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  Settings, 
  Search,
  Globe,
  Database,
  Code,
  Brain,
  Zap,
  Shield,
  Eye,
  TrendingUp,
  FileText,
  Image,
  MessageSquare,
  Mail,
  Calendar,
  DollarSign,
  BarChart3,
  Users,
  Wrench
} from 'lucide-react';
import { apiService } from '../services/api';

interface MCPServer {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'error' | 'connecting';
  capabilities: string[];
  provider: string;
  endpoint: string;
  last_used?: string;
  usage_count: number;
  error_count: number;
  category: 'search' | 'communication' | 'analytics' | 'development' | 'business' | 'ai' | 'security';
}

interface MCPMetrics {
  total_servers: number;
  active_servers: number;
  total_requests: number;
  success_rate: number;
  servers: MCPServer[];
}

export const MCPHub: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const queryClient = useQueryClient();

  const { data: mcpMetrics, isLoading } = useQuery<MCPMetrics>({
    queryKey: ['mcp-metrics'],
    queryFn: async () => {
      try {
        const response = await apiService.request<MCPMetrics>('/api/v1/mcp/status');
        return response.data || getMockMCPData();
      } catch (error) {
        return getMockMCPData();
      }
    },
    refetchInterval: 10000
  });

  const connectServerMutation = useMutation({
    mutationFn: async (serverId: string) => {
      const response = await apiService.request(`/api/v1/mcp/servers/${serverId}/connect`, {
        method: 'POST'
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-metrics'] });
    }
  });

  const getMockMCPData = (): MCPMetrics => ({
    total_servers: 15,
    active_servers: 12,
    total_requests: 2847,
    success_rate: 94.2,
    servers: [
      {
        id: 'exa-search',
        name: 'Exa Search',
        description: 'Advanced AI-powered web search and content discovery',
        status: 'active',
        capabilities: ['web_search', 'content_analysis', 'semantic_search'],
        provider: 'exa',
        endpoint: '/tools/exa_search',
        last_used: new Date(Date.now() - 300000).toISOString(),
        usage_count: 456,
        error_count: 2,
        category: 'search'
      },
      {
        id: 'claude-taskmaster',
        name: 'Claude Taskmaster',
        description: 'Advanced task orchestration and workflow management',
        status: 'active',
        capabilities: ['task_management', 'workflow_orchestration', 'ai_reasoning'],
        provider: 'anthropic',
        endpoint: '/tools/claude_taskmaster',
        last_used: new Date(Date.now() - 180000).toISOString(),
        usage_count: 234,
        error_count: 0,
        category: 'ai'
      },
      {
        id: 'context7',
        name: 'Context7',
        description: 'Advanced context management and retrieval system',
        status: 'active',
        capabilities: ['context_retrieval', 'semantic_search', 'knowledge_graph'],
        provider: 'context7',
        endpoint: '/tools/context7',
        last_used: new Date(Date.now() - 600000).toISOString(),
        usage_count: 189,
        error_count: 1,
        category: 'ai'
      },
      {
        id: 'magic-ui',
        name: 'Magic UI',
        description: 'Dynamic UI generation and interface creation',
        status: 'active',
        capabilities: ['ui_generation', 'component_creation', 'design_automation'],
        provider: 'magic',
        endpoint: '/tools/magic_ui',
        last_used: new Date(Date.now() - 900000).toISOString(),
        usage_count: 67,
        error_count: 0,
        category: 'development'
      },
      {
        id: 'jina-web-search',
        name: 'Jina Web Search',
        description: 'Parallel web search with content reading capabilities',
        status: 'active',
        capabilities: ['web_search', 'content_reading', 'url_extraction'],
        provider: 'jina',
        endpoint: '/tools/web_search',
        last_used: new Date(Date.now() - 1200000).toISOString(),
        usage_count: 342,
        error_count: 5,
        category: 'search'
      },
      {
        id: 'arxiv-search',
        name: 'ArXiv Academic Search',
        description: 'Search academic papers and research publications',
        status: 'active',
        capabilities: ['academic_search', 'paper_download', 'citation_extraction'],
        provider: 'arxiv',
        endpoint: '/tools/arxiv_search',
        last_used: new Date(Date.now() - 1800000).toISOString(),
        usage_count: 123,
        error_count: 1,
        category: 'search'
      },
      {
        id: 'google-workspace',
        name: 'Google Workspace',
        description: 'Gmail, Calendar, Drive, and Docs integration',
        status: 'inactive',
        capabilities: ['email_management', 'calendar_sync', 'document_creation'],
        provider: 'google',
        endpoint: '/tools/google_workspace',
        usage_count: 0,
        error_count: 0,
        category: 'business'
      },
      {
        id: 'slack-integration',
        name: 'Slack Integration',
        description: 'Team communication and workflow automation',
        status: 'active',
        capabilities: ['messaging', 'channel_management', 'workflow_automation'],
        provider: 'slack',
        endpoint: '/tools/slack',
        last_used: new Date(Date.now() - 2400000).toISOString(),
        usage_count: 89,
        error_count: 2,
        category: 'communication'
      },
      {
        id: 'github-integration',
        name: 'GitHub Integration',
        description: 'Code repository management and CI/CD automation',
        status: 'active',
        capabilities: ['repo_management', 'code_analysis', 'ci_cd_automation'],
        provider: 'github',
        endpoint: '/tools/github',
        last_used: new Date(Date.now() - 3600000).toISOString(),
        usage_count: 156,
        error_count: 3,
        category: 'development'
      },
      {
        id: 'financial-data',
        name: 'Financial Data API',
        description: 'Real-time financial data and market analysis',
        status: 'active',
        capabilities: ['market_data', 'financial_analysis', 'portfolio_tracking'],
        provider: 'financial',
        endpoint: '/tools/financial_data',
        last_used: new Date(Date.now() - 7200000).toISOString(),
        usage_count: 78,
        error_count: 1,
        category: 'business'
      },
      {
        id: 'security-scanner',
        name: 'Security Scanner',
        description: 'Vulnerability assessment and security monitoring',
        status: 'active',
        capabilities: ['vulnerability_scan', 'security_monitoring', 'threat_detection'],
        provider: 'security',
        endpoint: '/tools/security_scanner',
        last_used: new Date(Date.now() - 10800000).toISOString(),
        usage_count: 45,
        error_count: 0,
        category: 'security'
      },
      {
        id: 'analytics-engine',
        name: 'Analytics Engine',
        description: 'Advanced data analytics and visualization',
        status: 'active',
        capabilities: ['data_analysis', 'visualization', 'predictive_modeling'],
        provider: 'analytics',
        endpoint: '/tools/analytics',
        last_used: new Date(Date.now() - 14400000).toISOString(),
        usage_count: 234,
        error_count: 4,
        category: 'analytics'
      }
    ]
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'inactive': return <AlertCircle className="w-5 h-5 text-gray-400" />;
      case 'error': return <AlertCircle className="w-5 h-5 text-red-400" />;
      case 'connecting': return <Activity className="w-5 h-5 text-blue-400 animate-spin" />;
      default: return <AlertCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'search': return <Search className="w-5 h-5" />;
      case 'communication': return <MessageSquare className="w-5 h-5" />;
      case 'analytics': return <BarChart3 className="w-5 h-5" />;
      case 'development': return <Code className="w-5 h-5" />;
      case 'business': return <DollarSign className="w-5 h-5" />;
      case 'ai': return <Brain className="w-5 h-5" />;
      case 'security': return <Shield className="w-5 h-5" />;
      default: return <Server className="w-5 h-5" />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'search': return 'text-blue-400 bg-blue-400/20';
      case 'communication': return 'text-green-400 bg-green-400/20';
      case 'analytics': return 'text-purple-400 bg-purple-400/20';
      case 'development': return 'text-orange-400 bg-orange-400/20';
      case 'business': return 'text-yellow-400 bg-yellow-400/20';
      case 'ai': return 'text-cyan-400 bg-cyan-400/20';
      case 'security': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const filteredServers = mcpMetrics?.servers.filter(server => {
    const matchesCategory = selectedCategory === 'all' || server.category === selectedCategory;
    const matchesSearch = server.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         server.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  }) || [];

  const categories = ['all', 'search', 'ai', 'development', 'business', 'communication', 'analytics', 'security'];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Server className="w-8 h-8 mr-3 text-blue-400" />
            MCP Hub
          </h1>
          <p className="text-gray-400 mt-2">
            Model Context Protocol servers and tool integrations
          </p>
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Servers</p>
              <p className="text-2xl font-bold text-white">{mcpMetrics?.total_servers || 0}</p>
            </div>
            <Server className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Servers</p>
              <p className="text-2xl font-bold text-white">{mcpMetrics?.active_servers || 0}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Requests</p>
              <p className="text-2xl font-bold text-white">{mcpMetrics?.total_requests?.toLocaleString() || 0}</p>
            </div>
            <Activity className="w-8 h-8 text-purple-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Success Rate</p>
              <p className="text-2xl font-bold text-white">{mcpMetrics?.success_rate || 0}%</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-400" />
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search MCP servers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
        <div className="flex gap-2 overflow-x-auto">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                selectedCategory === category
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              {category.charAt(0).toUpperCase() + category.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* MCP Servers Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredServers.map((server) => (
          <div key={server.id} className="glass p-6 rounded-xl hover:bg-white/15 transition-all duration-200">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${getCategoryColor(server.category)}`}>
                  {getCategoryIcon(server.category)}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">{server.name}</h3>
                  <p className="text-sm text-gray-400">{server.provider}</p>
                </div>
              </div>
              {getStatusIcon(server.status)}
            </div>

            <p className="text-gray-300 text-sm mb-4">{server.description}</p>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Usage Count</span>
                <span className="text-white">{server.usage_count.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Error Count</span>
                <span className={`${server.error_count > 0 ? 'text-red-400' : 'text-green-400'}`}>
                  {server.error_count}
                </span>
              </div>
              {server.last_used && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Last Used</span>
                  <span className="text-white">
                    {new Date(server.last_used).toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>

            <div className="mt-4 pt-4 border-t border-white/10">
              <div className="flex flex-wrap gap-2 mb-3">
                {server.capabilities.slice(0, 3).map((capability, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded-full"
                  >
                    {capability.replace('_', ' ')}
                  </span>
                ))}
                {server.capabilities.length > 3 && (
                  <span className="px-2 py-1 bg-gray-500/20 text-gray-300 text-xs rounded-full">
                    +{server.capabilities.length - 3} more
                  </span>
                )}
              </div>

              <div className="flex gap-2">
                {server.status === 'inactive' ? (
                  <button
                    onClick={() => connectServerMutation.mutate(server.id)}
                    disabled={connectServerMutation.isPending}
                    className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors disabled:opacity-50"
                  >
                    {connectServerMutation.isPending ? 'Connecting...' : 'Connect'}
                  </button>
                ) : (
                  <button className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors">
                    Configure
                  </button>
                )}
                <button className="px-3 py-2 bg-white/10 hover:bg-white/20 text-white text-sm rounded-lg transition-colors">
                  <Settings className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredServers.length === 0 && (
        <div className="text-center py-12">
          <Server className="w-16 h-16 mx-auto mb-4 text-gray-400 opacity-50" />
          <p className="text-gray-400 text-lg">No MCP servers found</p>
          <p className="text-gray-500 text-sm">Try adjusting your search or category filter</p>
        </div>
      )}
    </div>
  );
};

export default MCPHub;
