import React, { useState, useEffect } from 'react';
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
  DollarSign
} from 'lucide-react';
import { apiService } from '../services/api';

interface OSINTProgram {
  id: string;
  name: string;
  description: string;
  category: 'social_media' | 'company_intel' | 'people_intel' | 'technical_intel' | 'market_intel' | 'geographic_intel';
  status: 'idle' | 'running' | 'completed' | 'error';
  progress: number;
  created_at: string;
  last_run?: string;
  results_count: number;
  configuration: {
    target: string;
    depth: 'surface' | 'deep' | 'comprehensive';
    sources: string[];
    filters: any;
  };
}

interface OSINTLead {
  id: string;
  name: string;
  company?: string;
  title?: string;
  email?: string;
  phone?: string;
  linkedin?: string;
  twitter?: string;
  location?: string;
  confidence_score: number;
  source_programs: string[];
  last_updated: string;
  status: 'new' | 'qualified' | 'contacted' | 'converted' | 'rejected';
  notes: string;
}

interface OSINTContext {
  id: string;
  title: string;
  description: string;
  active_programs: string[];
  total_leads: number;
  qualified_leads: number;
  conversion_rate: number;
  created_at: string;
  last_activity: string;
}

export const OSINTAgent: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'programs' | 'leads' | 'context' | 'analytics'>('dashboard');
  const [selectedProgram, setSelectedProgram] = useState<string>('');
  const [showCreateProgram, setShowCreateProgram] = useState(false);
  const [showCreateContext, setShowCreateContext] = useState(false);
  const [contextWindow, setContextWindow] = useState<string>('');
  const [newProgram, setNewProgram] = useState({
    name: '',
    description: '',
    category: 'company_intel' as const,
    target: '',
    depth: 'surface' as const,
    sources: [] as string[]
  });

  const queryClient = useQueryClient();

  const { data: programs, isLoading: programsLoading } = useQuery<OSINTProgram[]>({
    queryKey: ['osint-programs'],
    queryFn: async () => {
      try {
        const response = await apiService.request<OSINTProgram[]>('/api/v1/osint/programs');
        return response.data || getMockPrograms();
      } catch (error) {
        return getMockPrograms();
      }
    },
    refetchInterval: 10000
  });

  const { data: leads, isLoading: leadsLoading } = useQuery<OSINTLead[]>({
    queryKey: ['osint-leads'],
    queryFn: async () => {
      try {
        const response = await apiService.request<OSINTLead[]>('/api/v1/osint/leads');
        return response.data || getMockLeads();
      } catch (error) {
        return getMockLeads();
      }
    },
    refetchInterval: 30000
  });

  const { data: contexts, isLoading: contextsLoading } = useQuery<OSINTContext[]>({
    queryKey: ['osint-contexts'],
    queryFn: async () => {
      try {
        const response = await apiService.request<OSINTContext[]>('/api/v1/osint/contexts');
        return response.data || getMockContexts();
      } catch (error) {
        return getMockContexts();
      }
    }
  });

  const createProgramMutation = useMutation({
    mutationFn: async (programData: any) => {
      const response = await apiService.request('/api/v1/osint/programs', {
        method: 'POST',
        body: JSON.stringify(programData)
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['osint-programs'] });
      setShowCreateProgram(false);
      setNewProgram({
        name: '',
        description: '',
        category: 'company_intel',
        target: '',
        depth: 'surface',
        sources: []
      });
    }
  });

  const runProgramMutation = useMutation({
    mutationFn: async (programId: string) => {
      const response = await apiService.request(`/api/v1/osint/programs/${programId}/run`, {
        method: 'POST'
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['osint-programs'] });
      queryClient.invalidateQueries({ queryKey: ['osint-leads'] });
    }
  });

  const getMockPrograms = (): OSINTProgram[] => [
    {
      id: 'prog-1',
      name: 'Tech Startup Lead Generation',
      description: 'Identify decision makers at Series A-C tech startups in SF Bay Area',
      category: 'company_intel',
      status: 'completed',
      progress: 100,
      created_at: new Date(Date.now() - 86400000).toISOString(),
      last_run: new Date(Date.now() - 3600000).toISOString(),
      results_count: 247,
      configuration: {
        target: 'Tech startups Series A-C, San Francisco Bay Area',
        depth: 'deep',
        sources: ['linkedin', 'crunchbase', 'company_websites', 'news_sources'],
        filters: {
          company_size: '10-500',
          funding_stage: ['Series A', 'Series B', 'Series C'],
          location: 'San Francisco Bay Area',
          industry: 'Technology'
        }
      }
    },
    {
      id: 'prog-2',
      name: 'Healthcare Decision Makers',
      description: 'Find CTOs and IT Directors at mid-size healthcare organizations',
      category: 'people_intel',
      status: 'running',
      progress: 65,
      created_at: new Date(Date.now() - 172800000).toISOString(),
      last_run: new Date(Date.now() - 1800000).toISOString(),
      results_count: 89,
      configuration: {
        target: 'Healthcare IT decision makers, 100-1000 employees',
        depth: 'comprehensive',
        sources: ['linkedin', 'company_directories', 'conference_speakers', 'industry_publications'],
        filters: {
          titles: ['CTO', 'IT Director', 'Chief Information Officer', 'VP Technology'],
          company_size: '100-1000',
          industry: 'Healthcare'
        }
      }
    },
    {
      id: 'prog-3',
      name: 'Competitor Analysis - FinTech',
      description: 'Monitor competitor activities, funding, and key personnel changes',
      category: 'market_intel',
      status: 'idle',
      progress: 0,
      created_at: new Date(Date.now() - 259200000).toISOString(),
      results_count: 0,
      configuration: {
        target: 'FinTech competitors: Stripe, Square, Plaid, Adyen',
        depth: 'surface',
        sources: ['news_monitoring', 'social_media', 'job_postings', 'patent_filings'],
        filters: {
          companies: ['Stripe', 'Square', 'Plaid', 'Adyen'],
          monitoring_types: ['funding', 'personnel', 'product_launches', 'partnerships']
        }
      }
    }
  ];

  const getMockLeads = (): OSINTLead[] => [
    {
      id: 'lead-1',
      name: 'Sarah Chen',
      company: 'TechFlow AI',
      title: 'CTO',
      email: 'sarah.chen@techflow.ai',
      phone: '+1-415-555-0123',
      linkedin: 'https://linkedin.com/in/sarahchen-cto',
      twitter: '@sarahchen_tech',
      location: 'San Francisco, CA',
      confidence_score: 94,
      source_programs: ['prog-1'],
      last_updated: new Date(Date.now() - 3600000).toISOString(),
      status: 'qualified',
      notes: 'Recently raised Series B, actively hiring engineering team. Strong AI/ML background.'
    },
    {
      id: 'lead-2',
      name: 'Michael Rodriguez',
      company: 'HealthTech Solutions',
      title: 'VP of Technology',
      email: 'm.rodriguez@healthtech.com',
      linkedin: 'https://linkedin.com/in/mrodriguez-healthtech',
      location: 'Austin, TX',
      confidence_score: 87,
      source_programs: ['prog-2'],
      last_updated: new Date(Date.now() - 7200000).toISOString(),
      status: 'new',
      notes: 'Leading digital transformation initiative. Company expanding IT infrastructure.'
    },
    {
      id: 'lead-3',
      name: 'Jennifer Kim',
      company: 'MedConnect',
      title: 'Chief Information Officer',
      email: 'j.kim@medconnect.org',
      phone: '+1-512-555-0456',
      linkedin: 'https://linkedin.com/in/jenniferkim-cio',
      location: 'Dallas, TX',
      confidence_score: 91,
      source_programs: ['prog-2'],
      last_updated: new Date(Date.now() - 10800000).toISOString(),
      status: 'contacted',
      notes: 'Responded to initial outreach. Interested in cloud migration solutions.'
    }
  ];

  const getMockContexts = (): OSINTContext[] => [
    {
      id: 'ctx-1',
      title: 'Q4 2024 Lead Generation Campaign',
      description: 'Comprehensive lead generation for enterprise software sales targeting healthcare and fintech sectors',
      active_programs: ['prog-1', 'prog-2'],
      total_leads: 336,
      qualified_leads: 89,
      conversion_rate: 26.5,
      created_at: new Date(Date.now() - 2592000000).toISOString(),
      last_activity: new Date(Date.now() - 3600000).toISOString()
    }
  ];

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'social_media': return <MessageSquare className="w-5 h-5" />;
      case 'company_intel': return <Building className="w-5 h-5" />;
      case 'people_intel': return <Users className="w-5 h-5" />;
      case 'technical_intel': return <Database className="w-5 h-5" />;
      case 'market_intel': return <TrendingUp className="w-5 h-5" />;
      case 'geographic_intel': return <MapPin className="w-5 h-5" />;
      default: return <Eye className="w-5 h-5" />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'social_media': return 'text-blue-400 bg-blue-400/20';
      case 'company_intel': return 'text-green-400 bg-green-400/20';
      case 'people_intel': return 'text-purple-400 bg-purple-400/20';
      case 'technical_intel': return 'text-orange-400 bg-orange-400/20';
      case 'market_intel': return 'text-yellow-400 bg-yellow-400/20';
      case 'geographic_intel': return 'text-red-400 bg-red-400/20';
      default: return 'text-blue-400 bg-blue-400/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Activity className="w-4 h-4 text-blue-400 animate-pulse" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const handleCreateProgram = () => {
    if (newProgram.name && newProgram.target) {
      createProgramMutation.mutate(newProgram);
    }
  };

  const handleRunProgram = (programId: string) => {
    runProgramMutation.mutate(programId);
  };

  if (programsLoading || leadsLoading || contextsLoading) {
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
            <Brain className="w-8 h-8 mr-3 text-blue-400" />
            OSINT Agent
          </h1>
          <p className="text-gray-400 mt-2">
            Advanced Open Source Intelligence for Lead Generation & Business Intelligence
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowCreateContext(true)}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Context</span>
          </button>
          <button
            onClick={() => setShowCreateProgram(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Program</span>
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-blue-500/10 p-1 rounded-lg">
        {[
          { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
          { id: 'programs', label: 'Programs', icon: Zap },
          { id: 'leads', label: 'Leads', icon: Target },
          { id: 'context', label: 'Context Window', icon: Layers },
          { id: 'analytics', label: 'Analytics', icon: TrendingUp }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-blue-500/15'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-6">
          {/* Overview Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Active Programs</p>
                  <p className="text-2xl font-bold text-white">
                    {(programs || []).filter(p => p.status === 'running').length}
                  </p>
                </div>
                <Activity className="w-8 h-8 text-blue-400" />
              </div>
            </div>
            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Leads</p>
                  <p className="text-2xl font-bold text-white">{(leads || []).length}</p>
                </div>
                <Target className="w-8 h-8 text-green-400" />
              </div>
            </div>
            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Qualified Leads</p>
                  <p className="text-2xl font-bold text-white">
                    {(leads || []).filter(l => l.status === 'qualified').length}
                  </p>
                </div>
                <CheckCircle className="w-8 h-8 text-purple-400" />
              </div>
            </div>
            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Conversion Rate</p>
                  <p className="text-2xl font-bold text-white">
                    {leads && leads.length > 0
                      ? Math.round((leads.filter(l => l.status === 'qualified').length / leads.length) * 100)
                      : 0}%
                  </p>
                </div>
                <TrendingUp className="w-8 h-8 text-yellow-400" />
              </div>
            </div>
          </div>

          {/* Recent Programs */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Zap className="w-5 h-5 mr-2 text-blue-400" />
              Recent Programs
            </h3>
            <div className="space-y-4">
              {(programs || []).slice(0, 3).map((program) => (
                <div key={program.id} className="flex items-center justify-between p-4 bg-blue-500/10 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className={`p-2 rounded-lg ${getCategoryColor(program.category)}`}>
                      {getCategoryIcon(program.category)}
                    </div>
                    <div>
                      <h4 className="text-white font-medium">{program.name}</h4>
                      <p className="text-gray-400 text-sm">{program.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(program.status)}
                    <span className="text-gray-400 text-sm">{program.results_count} results</span>
                    {program.status === 'idle' && (
                      <button
                        onClick={() => handleRunProgram(program.id)}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
                      >
                        Run
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Leads */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-green-400" />
              Top Leads
            </h3>
            <div className="space-y-4">
              {(leads || []).slice(0, 5).map((lead) => (
                <div key={lead.id} className="flex items-center justify-between p-4 bg-blue-500/10 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
                      <User className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h4 className="text-white font-medium">{lead.name}</h4>
                      <p className="text-gray-400 text-sm">{lead.title} at {lead.company}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="text-white text-sm">{lead.confidence_score}% confidence</p>
                      <p className="text-gray-400 text-xs">{lead.status}</p>
                    </div>
                    <div className="flex space-x-2">
                      {lead.email && (
                        <a href={`mailto:${lead.email}`} className="text-blue-400 hover:text-blue-300">
                          <Mail className="w-4 h-4" />
                        </a>
                      )}
                      {lead.linkedin && (
                        <a href={lead.linkedin} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">
                          <ExternalLink className="w-4 h-4" />
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Programs Tab */}
      {activeTab === 'programs' && (
        <div className="space-y-6">
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4">OSINT Programs</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {(programs || []).map((program) => (
                <div key={program.id} className="bg-blue-500/10 p-6 rounded-xl hover:bg-blue-500/15 transition-all">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-2 rounded-lg ${getCategoryColor(program.category)}`}>
                      {getCategoryIcon(program.category)}
                    </div>
                    {getStatusIcon(program.status)}
                  </div>

                  <h4 className="text-white font-semibold mb-2">{program.name}</h4>
                  <p className="text-gray-300 text-sm mb-4">{program.description}</p>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Target</span>
                      <span className="text-white text-right">{program.configuration.target.substring(0, 30)}...</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Depth</span>
                      <span className="text-white capitalize">{program.configuration.depth}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Results</span>
                      <span className="text-white">{program.results_count}</span>
                    </div>
                  </div>

                  {program.status === 'running' && (
                    <div className="mb-4">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">Progress</span>
                        <span className="text-white">{program.progress}%</span>
                      </div>
                      <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${program.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2">
                    {program.status === 'idle' && (
                      <button
                        onClick={() => handleRunProgram(program.id)}
                        className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
                      >
                        <Play className="w-4 h-4 inline mr-1" />
                        Run
                      </button>
                    )}
                    {program.status === 'running' && (
                      <button className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors">
                        <Pause className="w-4 h-4 inline mr-1" />
                        Stop
                      </button>
                    )}
                    <button className="px-3 py-2 bg-blue-700 hover:bg-blue-800 text-white text-sm rounded-lg transition-colors">
                      <Settings className="w-4 h-4" />
                    </button>
                    <button className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors">
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Leads Tab */}
      {activeTab === 'leads' && (
        <div className="space-y-6">
          <div className="glass p-6 rounded-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Lead Database</h3>
              <div className="flex gap-2">
                <button className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors">
                  <Filter className="w-4 h-4 inline mr-1" />
                  Filter
                </button>
                <button className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors">
                  <Download className="w-4 h-4 inline mr-1" />
                  Export
                </button>
              </div>
            </div>

            <div className="space-y-4">
              {(leads || []).map((lead) => (
                <div key={lead.id} className="bg-blue-500/10 p-6 rounded-xl hover:bg-blue-500/15 transition-all">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4">
                      <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                        <User className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <h4 className="text-white font-semibold text-lg">{lead.name}</h4>
                        <p className="text-gray-300">{lead.title} at {lead.company}</p>
                        <p className="text-gray-400 text-sm">{lead.location}</p>

                        <div className="flex items-center space-x-4 mt-3">
                          {lead.email && (
                            <div className="flex items-center space-x-1">
                              <Mail className="w-4 h-4 text-gray-400" />
                              <span className="text-gray-300 text-sm">{lead.email}</span>
                            </div>
                          )}
                          {lead.phone && (
                            <div className="flex items-center space-x-1">
                              <Phone className="w-4 h-4 text-gray-400" />
                              <span className="text-gray-300 text-sm">{lead.phone}</span>
                            </div>
                          )}
                        </div>

                        {lead.notes && (
                          <p className="text-gray-400 text-sm mt-2 italic">{lead.notes}</p>
                        )}
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="text-white font-medium">{lead.confidence_score}%</span>
                        <div className="w-16 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${lead.confidence_score}%` }}
                          ></div>
                        </div>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        lead.status === 'qualified' ? 'bg-green-500/20 text-green-400' :
                        lead.status === 'contacted' ? 'bg-blue-500/20 text-blue-400' :
                        lead.status === 'converted' ? 'bg-purple-500/20 text-purple-400' :
                        'bg-blue-600/20 text-gray-400'
                      }`}>
                        {lead.status}
                      </span>

                      <div className="flex space-x-2 mt-3">
                        {lead.linkedin && (
                          <a href={lead.linkedin} target="_blank" rel="noopener noreferrer"
                             className="text-blue-400 hover:text-blue-300">
                            <ExternalLink className="w-4 h-4" />
                          </a>
                        )}
                        <button className="text-gray-400 hover:text-white">
                          <Edit className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Context Window Tab */}
      {activeTab === 'context' && (
        <div className="space-y-6">
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Layers className="w-5 h-5 mr-2 text-purple-400" />
              Context Window Management
            </h3>

            {/* Active Contexts */}
            <div className="space-y-4 mb-6">
              {(contexts || []).map((context) => (
                <div key={context.id} className="bg-blue-500/10 p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h4 className="text-white font-semibold text-lg">{context.title}</h4>
                      <p className="text-gray-300">{context.description}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-white font-medium">{context.conversion_rate}% conversion</p>
                      <p className="text-gray-400 text-sm">{context.qualified_leads}/{context.total_leads} qualified</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div className="bg-blue-500/10 p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">Active Programs</p>
                      <p className="text-white text-xl font-bold">{context.active_programs.length}</p>
                    </div>
                    <div className="bg-blue-500/10 p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">Total Leads</p>
                      <p className="text-white text-xl font-bold">{context.total_leads}</p>
                    </div>
                    <div className="bg-blue-500/10 p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">Last Activity</p>
                      <p className="text-white text-sm">{new Date(context.last_activity).toLocaleDateString()}</p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors">
                      <Edit className="w-4 h-4 inline mr-1" />
                      Edit Context
                    </button>
                    <button className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors">
                      <Download className="w-4 h-4 inline mr-1" />
                      Export Results
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Context Input */}
            <div className="bg-blue-500/10 p-6 rounded-xl">
              <h4 className="text-white font-medium mb-4">Context Window Input</h4>
              <textarea
                value={contextWindow}
                onChange={(e) => setContextWindow(e.target.value)}
                className="w-full h-32 px-4 py-3 bg-blue-500/15 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 resize-none"
                placeholder="Enter context for OSINT operations... (e.g., 'Find decision makers at Series A startups in fintech space, focusing on companies with 10-100 employees in San Francisco Bay Area')"
              />
              <div className="flex justify-between items-center mt-4">
                <p className="text-gray-400 text-sm">{contextWindow.length} characters</p>
                <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
                  <Brain className="w-4 h-4 inline mr-1" />
                  Process Context
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analytics Tab */}
      {activeTab === 'analytics' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4">Performance Metrics</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Total Programs Run</span>
                  <span className="text-white font-medium">{(programs || []).length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Success Rate</span>
                  <span className="text-green-400 font-medium">87.3%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Avg. Leads per Program</span>
                  <span className="text-white font-medium">
                    {programs && programs.length > 0
                      ? Math.round(programs.reduce((sum, p) => sum + p.results_count, 0) / programs.length)
                      : 0}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Lead Quality Score</span>
                  <span className="text-blue-400 font-medium">
                    {leads && leads.length > 0
                      ? Math.round(leads.reduce((sum, l) => sum + l.confidence_score, 0) / leads.length)
                      : 0}%
                  </span>
                </div>
              </div>
            </div>

            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4">Category Distribution</h3>
              <div className="space-y-3">
                {['company_intel', 'people_intel', 'market_intel', 'social_media', 'technical_intel'].map((category) => {
                  const count = (programs || []).filter(p => p.category === category).length;
                  const percentage = programs && programs.length > 0 ? (count / programs.length) * 100 : 0;
                  return (
                    <div key={category} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400 capitalize">{category.replace('_', ' ')}</span>
                        <span className="text-white">{count} ({Math.round(percentage)}%)</span>
                      </div>
                      <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create Program Modal */}
      {showCreateProgram && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-blue-900/40 hover:bg-blue-900/60 p-6 rounded-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Create OSINT Program</h3>
              <button
                onClick={() => setShowCreateProgram(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-gray-400 text-sm mb-2">Program Name</label>
                <input
                  type="text"
                  value={newProgram.name}
                  onChange={(e) => setNewProgram({ ...newProgram, name: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  placeholder="e.g., Tech Startup Lead Generation"
                />
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Description</label>
                <textarea
                  value={newProgram.description}
                  onChange={(e) => setNewProgram({ ...newProgram, description: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 h-20"
                  placeholder="Describe the purpose and scope of this OSINT program..."
                />
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Category</label>
                <select
                  value={newProgram.category}
                  onChange={(e) => setNewProgram({ ...newProgram, category: e.target.value as any })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="company_intel">Company Intelligence</option>
                  <option value="people_intel">People Intelligence</option>
                  <option value="market_intel">Market Intelligence</option>
                  <option value="social_media">Social Media Intelligence</option>
                  <option value="technical_intel">Technical Intelligence</option>
                  <option value="geographic_intel">Geographic Intelligence</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Target</label>
                <textarea
                  value={newProgram.target}
                  onChange={(e) => setNewProgram({ ...newProgram, target: e.target.value })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 h-24"
                  placeholder="Define your target (companies, people, technologies, etc.)..."
                />
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Investigation Depth</label>
                <select
                  value={newProgram.depth}
                  onChange={(e) => setNewProgram({ ...newProgram, depth: e.target.value as any })}
                  className="w-full px-3 py-2 bg-blue-500/15 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="surface">Surface - Basic information gathering</option>
                  <option value="deep">Deep - Comprehensive analysis</option>
                  <option value="comprehensive">Comprehensive - Full intelligence profile</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Data Sources</label>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    'LinkedIn', 'Twitter', 'Facebook', 'Company Websites',
                    'News Sources', 'Crunchbase', 'SEC Filings', 'Patent Databases',
                    'Job Postings', 'Conference Speakers', 'Industry Publications', 'Government Records'
                  ].map((source) => (
                    <label key={source} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={newProgram.sources.includes(source.toLowerCase().replace(' ', '_'))}
                        onChange={(e) => {
                          const sourceId = source.toLowerCase().replace(' ', '_');
                          if (e.target.checked) {
                            setNewProgram({
                              ...newProgram,
                              sources: [...newProgram.sources, sourceId]
                            });
                          } else {
                            setNewProgram({
                              ...newProgram,
                              sources: newProgram.sources.filter(s => s !== sourceId)
                            });
                          }
                        }}
                        className="rounded border-gray-600 bg-blue-800 hover:bg-blue-900 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-gray-300 text-sm">{source}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCreateProgram}
                disabled={createProgramMutation.isPending || !newProgram.name || !newProgram.target}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {createProgramMutation.isPending ? 'Creating...' : 'Create Program'}
              </button>
              <button
                onClick={() => setShowCreateProgram(false)}
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
