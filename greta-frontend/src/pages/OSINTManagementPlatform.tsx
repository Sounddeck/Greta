import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Search,
  Eye,
  Globe,
  Shield,
  Database,
  Users,
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  MapPin,
  Filter,
  Download,
  RefreshCw,
  Play,
  Pause,
  Settings,
  Target,
  Building,
  User,
  Mail,
  Phone,
  ExternalLink,
  FileText,
  BarChart3,
  Zap,
  Plus,
  Trash2,
  Edit,
  Save,
  X,
  ChevronRight,
  ChevronDown,
  Activity,
  Layers,
  Network,
  Brain,
  MessageSquare,
  Calendar,
  DollarSign,
  Package,
  Workflow,
  TestTube,
  LineChart,
  PieChart,
  Star,
  Award,
  Wrench,
  Monitor,
  Terminal,
  Copy,
  Rocket,
  HardDrive,
  Wifi,
  Server,
  Cpu,
  Minus
} from 'lucide-react';
import { apiService } from '../services/api';

// OSINT Management Platform Interfaces
interface OSINTProgram {
  id: string;
  name: string;
  description: string;
  type: 'person_intel' | 'company_intel' | 'social_media' | 'technical_intel' | 'market_intel' | 'geographic_intel' | 'threat_intel' | 'custom';
  status: 'draft' | 'active' | 'completed' | 'paused' | 'error';
  priority: 'low' | 'medium' | 'high' | 'critical';
  targets: string[];
  data_sources: string[];
  automation_level: 'manual' | 'semi_automated' | 'fully_automated';
  schedule: {
    enabled: boolean;
    frequency: 'once' | 'daily' | 'weekly' | 'monthly';
    next_run: string;
  };
  results: {
    total_findings: number;
    high_confidence: number;
    medium_confidence: number;
    low_confidence: number;
    last_updated: string;
  };
  performance: {
    success_rate: number;
    avg_execution_time: number;
    data_quality_score: number;
    cost_per_execution: number;
  };
  created_at: string;
  updated_at: string;
  created_by: string;
}

interface OSINTTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  type: 'person_intel' | 'company_intel' | 'social_media' | 'technical_intel' | 'market_intel' | 'geographic_intel' | 'threat_intel' | 'custom';
  complexity: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  estimated_time: number;
  data_sources: string[];
  tools_required: string[];
  steps: {
    id: string;
    name: string;
    description: string;
    type: 'search' | 'analysis' | 'verification' | 'documentation';
    automated: boolean;
    estimated_time: number;
  }[];
  outputs: string[];
  use_cases: string[];
  tags: string[];
  author: string;
  downloads: number;
  rating: number;
  reviews: number;
  last_updated: string;
}

interface OSINTResult {
  id: string;
  program_id: string;
  target: string;
  findings: {
    id: string;
    type: 'personal_info' | 'contact_info' | 'social_media' | 'professional_info' | 'location_data' | 'technical_data' | 'financial_data' | 'other';
    source: string;
    data: any;
    confidence_score: number;
    verification_status: 'verified' | 'unverified' | 'disputed' | 'false';
    timestamp: string;
    relevance_score: number;
  }[];
  metadata: {
    execution_time: number;
    sources_queried: number;
    data_points_collected: number;
    verification_checks: number;
    cost: number;
  };
  status: 'processing' | 'completed' | 'failed' | 'partial';
  created_at: string;
  expires_at: string;
}

interface OSINTIntegration {
  id: string;
  name: string;
  description: string;
  type: 'crm' | 'threat_intel' | 'case_management' | 'reporting' | 'notification' | 'storage' | 'api';
  status: 'active' | 'inactive' | 'error';
  configuration: any;
  usage_count: number;
  last_used: string;
}

// Mock Data
const mockPrograms: OSINTProgram[] = [
  {
    id: 'program-1',
    name: 'Executive Background Check',
    description: 'Comprehensive background investigation for C-level executives including professional history, social media presence, and public records',
    type: 'person_intel',
    status: 'active',
    priority: 'high',
    targets: ['John Smith', 'Jane Doe', 'Michael Johnson'],
    data_sources: ['LinkedIn', 'Twitter', 'Public Records', 'News Articles', 'Company Filings'],
    automation_level: 'semi_automated',
    schedule: {
      enabled: true,
      frequency: 'weekly',
      next_run: '2024-01-27T09:00:00Z'
    },
    results: {
      total_findings: 247,
      high_confidence: 89,
      medium_confidence: 124,
      low_confidence: 34,
      last_updated: '2024-01-20T15:30:00Z'
    },
    performance: {
      success_rate: 94.2,
      avg_execution_time: 1847,
      data_quality_score: 87.5,
      cost_per_execution: 12.50
    },
    created_at: '2024-01-10T09:00:00Z',
    updated_at: '2024-01-20T14:00:00Z',
    created_by: 'john.doe'
  }
];

const OSINTManagementPlatform: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'programs' | 'templates' | 'results' | 'integrations' | 'analytics' | 'settings'>('programs');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');
  const [selectedProgram, setSelectedProgram] = useState<string | null>(null);
  const [showCreateProgram, setShowCreateProgram] = useState(false);

  const queryClient = useQueryClient();

  // Queries
  const { data: programs = mockPrograms, isLoading: programsLoading } = useQuery<OSINTProgram[]>({
    queryKey: ['osint-programs'],
    queryFn: async (): Promise<OSINTProgram[]> => {
      try {
        const response = await apiService.request('/api/v1/osint/programs');
        return response.data as OSINTProgram[];
      } catch (error) {
        return mockPrograms;
      }
    }
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-red-600 rounded-xl">
              <Eye className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">OSINT Management Platform</h1>
              <p className="text-gray-400">Program templates, deployment automation, result management, and performance optimization</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowCreateProgram(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>New Program</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
              <Package className="w-4 h-4" />
              <span>Templates</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors">
              <Workflow className="w-4 h-4" />
              <span>Automation</span>
            </button>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-400">OSINT Platform Active</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-white/5 p-1 rounded-lg">
          {[
            { id: 'programs', label: 'Programs', icon: Eye },
            { id: 'templates', label: 'Templates', icon: Package },
            { id: 'results', label: 'Results', icon: Database },
            { id: 'integrations', label: 'Integrations', icon: Workflow },
            { id: 'analytics', label: 'Analytics', icon: BarChart3 },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-red-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/10'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Programs Tab */}
        {activeTab === 'programs' && (
          <div className="space-y-6">
            {/* Filters and Search */}
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search programs..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-red-500"
                  />
                </div>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-red-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="completed">Completed</option>
                  <option value="paused">Paused</option>
                  <option value="draft">Draft</option>
                  <option value="error">Error</option>
                </select>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-red-500"
                >
                  <option value="all">All Types</option>
                  <option value="person_intel">Person Intel</option>
                  <option value="company_intel">Company Intel</option>
                  <option value="social_media">Social Media</option>
                  <option value="technical_intel">Technical Intel</option>
                  <option value="market_intel">Market Intel</option>
                  <option value="geographic_intel">Geographic Intel</option>
                  <option value="threat_intel">Threat Intel</option>
                  <option value="custom">Custom</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['osint-programs'] })}
                  className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Programs Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {programs.map((program) => (
                <div key={program.id} className="glass p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        program.status === 'active' ? 'bg-green-500/20 text-green-400' :
                        program.status === 'completed' ? 'bg-blue-500/20 text-blue-400' :
                        program.status === 'paused' ? 'bg-yellow-500/20 text-yellow-400' :
                        program.status === 'error' ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        <Eye className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">{program.name}</h3>
                        <p className="text-sm text-gray-400">{program.type} • {program.priority} priority</p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      program.status === 'active' ? 'bg-green-500/20 text-green-400' :
                      program.status === 'completed' ? 'bg-blue-500/20 text-blue-400' :
                      program.status === 'paused' ? 'bg-yellow-500/20 text-yellow-400' :
                      program.status === 'error' ? 'bg-red-500/20 text-red-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      {program.status}
                    </span>
                  </div>

                  <p className="text-gray-300 text-sm mb-4 line-clamp-2">{program.description}</p>

                  {/* Program Stats */}
                  <div className="space-y-3 mb-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Total Findings</p>
                        <p className="text-white font-semibold">{program.results.total_findings}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Success Rate</p>
                        <p className="text-white font-semibold">{program.performance.success_rate}%</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Targets</p>
                        <p className="text-white font-semibold">{program.targets.length}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Data Sources</p>
                        <p className="text-white font-semibold">{program.data_sources.length}</p>
                      </div>
                    </div>
                    <div className="text-sm">
                      <p className="text-gray-400">Automation Level</p>
                      <p className="text-white font-semibold capitalize">{program.automation_level.replace('_', ' ')}</p>
                    </div>
                  </div>

                  {/* High Confidence Findings */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400">High Confidence Findings</span>
                      <span className="text-white">{program.results.high_confidence}/{program.results.total_findings}</span>
                    </div>
                    <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                        style={{ width: `${(program.results.high_confidence / program.results.total_findings) * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                    <div className="flex items-center space-x-2">
                      {program.status === 'paused' || program.status === 'draft' ? (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                          <Play className="w-3 h-3" />
                          <span>Start</span>
                        </button>
                      ) : program.status === 'active' ? (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs transition-colors">
                          <Pause className="w-3 h-3" />
                          <span>Pause</span>
                        </button>
                      ) : null}
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <Database className="w-3 h-3" />
                        <span>Results</span>
                      </button>
                      <button className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-xs transition-colors">
                        <Download className="w-3 h-3" />
                        <span>Export</span>
                      </button>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => setSelectedProgram(program.id)}
                        className="p-1 hover:bg-white/10 rounded"
                      >
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

        {/* Templates Tab */}
        {activeTab === 'templates' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">OSINT Program Templates</h2>
              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search templates..."
                    className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-red-500"
                  />
                </div>
                <select className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-red-500">
                  <option value="all">All Categories</option>
                  <option value="person_intel">Person Intel</option>
                  <option value="company_intel">Company Intel</option>
                  <option value="social_media">Social Media</option>
                  <option value="technical_intel">Technical Intel</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
                      <User className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Executive Background Check</h3>
                      <p className="text-sm text-gray-400">Advanced • 2-4 hours</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400 fill-current" />
                    <span className="text-sm text-white">4.8</span>
                  </div>
                </div>

                <p className="text-gray-300 text-sm mb-4">Comprehensive background investigation for executives including professional history, social media, and public records.</p>

                <div className="flex flex-wrap gap-1 mb-4">
                  <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded">Person Intel</span>
                  <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">Professional</span>
                  <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded">Advanced</span>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                  <div>
                    <p className="text-gray-400">Downloads</p>
                    <p className="text-white">1.2K</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Data Sources</p>
                    <p className="text-white">12</p>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <div className="flex items-center space-x-2">
                    <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                      <Download className="w-3 h-3" />
                      <span>Use Template</span>
                    </button>
                    <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                      <Eye className="w-3 h-3" />
                      <span>Preview</span>
                    </button>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-green-400 font-semibold">Free</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">OSINT Analytics & Performance</h2>
              <div className="flex items-center space-x-2">
                <select className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-red-500">
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                  <option value="90d">Last 90 Days</option>
                </select>
                <button className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors">
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Active Programs</p>
                    <p className="text-2xl font-bold text-white">12</p>
                  </div>
                  <Eye className="w-8 h-8 text-red-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+3 this week</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Total Findings</p>
                    <p className="text-2xl font-bold text-white">2,847</p>
                  </div>
                  <Database className="w-8 h-8 text-blue-400" />
                </div>
                <p className="text-blue-400 text-sm mt-2">847 this week</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Avg Success Rate</p>
                    <p className="text-2xl font-bold text-white">94.2%</p>
                  </div>
                  <Target className="w-8 h-8 text-green-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+2.1% improvement</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Cost Efficiency</p>
                    <p className="text-2xl font-bold text-white">$8.50</p>
                  </div>
                  <DollarSign className="w-8 h-8 text-yellow-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">Per finding</p>
              </div>
            </div>

            {/* Performance Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <LineChart className="w-5 h-5 mr-2 text-blue-400" />
                  Program Performance Trends
                </h3>
                <div className="h-64 flex items-center justify-center text-gray-400">
                  <p>Performance trends chart visualization would go here</p>
                </div>
              </div>

              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <PieChart className="w-5 h-5 mr-2 text-green-400" />
                  Finding Types Distribution
                </h3>
                <div className="h-64 flex items-center justify-center text-gray-400">
                  <p>Finding types distribution chart would go here</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default OSINTManagementPlatform;
