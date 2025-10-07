import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Search, 
  Eye, 
  Globe, 
  Shield, 
  Target,
  Clock,
  CheckCircle,
  AlertTriangle,
  Database,
  Activity,
  TrendingUp,
  Users,
  MapPin,
  Link,
  FileText,
  Image,
  Video,
  Download
} from 'lucide-react';
import { apiService } from '../services/api';

interface OSINTSource {
  id: string;
  name: string;
  type: 'web' | 'social' | 'news' | 'academic' | 'government' | 'commercial';
  status: 'active' | 'inactive' | 'error';
  last_updated: string;
  reliability_score: number;
}

interface OSINTQuery {
  id: string;
  query: string;
  type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  results_count: number;
  confidence: number;
}

interface OSINTResult {
  id: string;
  source: string;
  title: string;
  content: string;
  url?: string;
  timestamp: string;
  relevance_score: number;
  content_type: 'text' | 'image' | 'video' | 'document';
  metadata: Record<string, any>;
}

interface OSINTMetrics {
  total_queries: number;
  active_sources: number;
  success_rate: number;
  avg_response_time: number;
  sources: OSINTSource[];
  recent_queries: OSINTQuery[];
}

export const OSINT: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState('comprehensive');
  const [selectedQuery, setSelectedQuery] = useState<OSINTQuery | null>(null);
  const [results, setResults] = useState<OSINTResult[]>([]);

  const queryClient = useQueryClient();

  const { data: metrics, isLoading } = useQuery({
    queryKey: ['osint-metrics'],
    queryFn: async () => {
      try {
        const response = await apiService.request('/api/v1/osint/metrics');
        return response.data as OSINTMetrics;
      } catch (error) {
        // Mock data for development
        return {
          total_queries: 847,
          active_sources: 23,
          success_rate: 91.5,
          avg_response_time: 4.2,
          sources: [
            {
              id: 'web-crawler',
              name: 'Web Crawler',
              type: 'web',
              status: 'active',
              last_updated: new Date(Date.now() - 300000).toISOString(),
              reliability_score: 94
            },
            {
              id: 'news-aggregator',
              name: 'News Aggregator',
              type: 'news',
              status: 'active',
              last_updated: new Date(Date.now() - 600000).toISOString(),
              reliability_score: 89
            },
            {
              id: 'social-monitor',
              name: 'Social Media Monitor',
              type: 'social',
              status: 'active',
              last_updated: new Date(Date.now() - 180000).toISOString(),
              reliability_score: 76
            },
            {
              id: 'academic-db',
              name: 'Academic Database',
              type: 'academic',
              status: 'active',
              last_updated: new Date(Date.now() - 900000).toISOString(),
              reliability_score: 97
            }
          ],
          recent_queries: [
            {
              id: 'query-001',
              query: 'cybersecurity trends 2024',
              type: 'comprehensive',
              status: 'completed',
              created_at: new Date(Date.now() - 1800000).toISOString(),
              results_count: 156,
              confidence: 92
            },
            {
              id: 'query-002',
              query: 'AI regulation updates',
              type: 'news',
              status: 'completed',
              created_at: new Date(Date.now() - 3600000).toISOString(),
              results_count: 89,
              confidence: 87
            },
            {
              id: 'query-003',
              query: 'emerging tech startups',
              type: 'business',
              status: 'processing',
              created_at: new Date(Date.now() - 300000).toISOString(),
              results_count: 0,
              confidence: 0
            }
          ]
        } as OSINTMetrics;
      }
    },
    refetchInterval: 10000
  });

  const searchMutation = useMutation({
    mutationFn: async ({ query, type }: { query: string; type: string }) => {
      const response = await apiService.request('/api/v1/osint/search', {
        method: 'POST',
        body: JSON.stringify({ query, search_type: type })
      });
      return response.data;
    },
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['osint-metrics'] });
      if (data.results) {
        setResults(data.results);
      }
    }
  });

  const getSourceTypeColor = (type: string) => {
    switch (type) {
      case 'web': return 'text-blue-400 bg-blue-400/20';
      case 'social': return 'text-blue-400 bg-blue-400/20';
      case 'news': return 'text-green-400 bg-green-400/20';
      case 'academic': return 'text-yellow-400 bg-yellow-400/20';
      case 'government': return 'text-red-400 bg-red-400/20';
      case 'commercial': return 'text-orange-400 bg-orange-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'completed': return 'text-green-400';
      case 'processing': return 'text-blue-400';
      case 'pending': return 'text-yellow-400';
      case 'inactive': return 'text-gray-400';
      case 'error': return 'text-red-400';
      case 'failed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getContentTypeIcon = (type: string) => {
    switch (type) {
      case 'text': return <FileText className="w-4 h-4" />;
      case 'image': return <Image className="w-4 h-4" />;
      case 'video': return <Video className="w-4 h-4" />;
      case 'document': return <Download className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  const handleSearch = () => {
    if (searchQuery.trim()) {
      searchMutation.mutate({ query: searchQuery, type: searchType });
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Eye className="w-8 h-8 mr-3 text-green-400" />
            OSINT Intelligence Hub
          </h1>
          <p className="text-gray-400 mt-2">
            Open Source Intelligence gathering and analysis platform
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="text-right">
            <p className="text-sm text-gray-400">Success Rate</p>
            <p className="text-xl font-bold text-green-400">{metrics?.success_rate || 0}%</p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Queries</p>
              <p className="text-2xl font-bold text-white">{metrics?.total_queries || 0}</p>
            </div>
            <Search className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Sources</p>
              <p className="text-2xl font-bold text-white">{metrics?.active_sources || 0}</p>
            </div>
            <Database className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Success Rate</p>
              <p className="text-2xl font-bold text-white">{metrics?.success_rate || 0}%</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Response</p>
              <p className="text-2xl font-bold text-white">{metrics?.avg_response_time || 0}s</p>
            </div>
            <Clock className="w-8 h-8 text-purple-400" />
          </div>
        </div>
      </div>

      {/* Search Interface */}
      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Search className="w-5 h-5 mr-2 text-blue-400" />
          Intelligence Search
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          <div className="lg:col-span-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Enter search query for intelligence gathering..."
              className="w-full px-4 py-3 bg-white/10 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
          </div>
          <div>
            <select
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
              className="w-full px-4 py-3 bg-white/10 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
            >
              <option value="comprehensive">Comprehensive Search</option>
              <option value="news">News & Media</option>
              <option value="social">Social Media</option>
              <option value="academic">Academic Sources</option>
              <option value="business">Business Intelligence</option>
              <option value="technical">Technical Analysis</option>
            </select>
          </div>
          <div>
            <button
              onClick={handleSearch}
              disabled={!searchQuery.trim() || searchMutation.isPending}
              className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {searchMutation.isPending ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>Search</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sources Status */}
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Database className="w-5 h-5 mr-2 text-green-400" />
            Intelligence Sources
          </h3>
          <div className="space-y-4">
            {metrics?.sources.map((source) => (
              <div key={source.id} className="p-4 bg-white/5 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white">{source.name}</h4>
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(source.status)}`}></div>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className={`px-2 py-1 text-xs rounded-full ${getSourceTypeColor(source.type)}`}>
                    {source.type}
                  </span>
                  <span className="text-sm text-gray-400">
                    {source.reliability_score}% reliable
                  </span>
                </div>
                <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-1">
                  <div 
                    className="bg-gradient-to-r from-green-500 to-blue-500 h-1 rounded-full"
                    style={{ width: `${source.reliability_score}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Updated: {new Date(source.last_updated).toLocaleTimeString()}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Queries */}
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-400" />
            Recent Queries
          </h3>
          <div className="space-y-4">
            {metrics?.recent_queries.map((query) => (
              <div 
                key={query.id}
                className={`p-4 bg-white/5 rounded-lg cursor-pointer transition-all hover:bg-white/10 ${
                  selectedQuery?.id === query.id ? 'ring-2 ring-purple-500' : ''
                }`}
                onClick={() => setSelectedQuery(query)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-white text-sm">{query.query}</h4>
                  <span className={`text-xs ${getStatusColor(query.status)}`}>
                    {query.status}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>{query.type}</span>
                  <span>{query.results_count} results</span>
                </div>
                <div className="flex items-center justify-between text-xs text-gray-500 mt-1">
                  <span>{new Date(query.created_at).toLocaleString()}</span>
                  {query.confidence > 0 && (
                    <span>{query.confidence}% confidence</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Search Results */}
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-yellow-400" />
            Search Results
          </h3>
          {results.length > 0 ? (
            <div className="space-y-4">
              {results.slice(0, 5).map((result, index) => (
                <div key={result.id || index} className="p-4 bg-white/5 rounded-lg">
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-medium text-white text-sm">{result.title}</h4>
                    <div className="flex items-center space-x-1">
                      {getContentTypeIcon(result.content_type)}
                      <span className="text-xs text-gray-400">{result.relevance_score}%</span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mb-2 line-clamp-2">{result.content}</p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{result.source}</span>
                    <span>{new Date(result.timestamp).toLocaleDateString()}</span>
                  </div>
                  {result.url && (
                    <a 
                      href={result.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-flex items-center space-x-1 text-xs text-blue-400 hover:text-blue-300 mt-2"
                    >
                      <Link className="w-3 h-3" />
                      <span>View Source</span>
                    </a>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No search results yet</p>
              <p className="text-sm">Perform a search to see intelligence data</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OSINT;
