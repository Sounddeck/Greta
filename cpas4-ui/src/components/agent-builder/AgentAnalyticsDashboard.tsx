/**
 * Agent Performance Analytics Dashboard
 * 
 * GitHub Issue #26: Agent Performance Analytics Dashboard
 * 
 * Features:
 * - Real-time performance metrics display
 * - Success/failure rate visualizations
 * - Average execution time charts
 * - Token usage statistics
 * - Historical trend analysis with interactive charts
 * - Filtering by date range and agent
 */

import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Select } from '../ui/select';
import { toast } from 'sonner';
import api from '../../lib/api';

// Types
interface AgentMetrics {
  agent_id: string;
  agent_name?: string;
  total_executions: number;
  successful_executions: number;
  failed_executions: number;
  total_duration_seconds: number;
  average_duration_seconds: number;
  min_duration_seconds: number;
  max_duration_seconds: number;
  success_rate: number;
  error_patterns?: Record<string, number>;
}

interface TrendData {
  date: string;
  executions: number;
  success_rate: number;
  average_duration: number;
}

interface TokenUsage {
  total_tokens: number;
  total_executions: number;
  average_tokens_per_execution: number;
  daily_usage: Array<{ date: string; tokens: number }>;
}

interface DateRange {
  label: string;
  days?: number;
  start?: string;
  end?: string;
}

const DATE_RANGES: DateRange[] = [
  { label: 'Last 24 Hours', days: 1 },
  { label: 'Last 7 Days', days: 7 },
  { label: 'Last 30 Days', days: 30 },
  { label: 'Last 90 Days', days: 90 },
];

const COLORS = {
  success: '#10b981',
  failure: '#ef4444',
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  warning: '#f59e0b',
};

export const AgentAnalyticsDashboard: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState<string>('all');
  const [selectedRange, setSelectedRange] = useState<DateRange>(DATE_RANGES[1]); // Default: Last 7 Days
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  
  // Data states
  const [agentMetrics, setAgentMetrics] = useState<AgentMetrics | null>(null);
  const [allAgentsMetrics, setAllAgentsMetrics] = useState<AgentMetrics[]>([]);
  const [trends, setTrends] = useState<TrendData[]>([]);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage | null>(null);

  // Fetch agent summary metrics
  const fetchAgentSummary = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      
      if (selectedRange.start) {
        params.append('start_date', selectedRange.start);
      }
      if (selectedRange.end) {
        params.append('end_date', selectedRange.end);
      }
      
      const response = await api.get(`/api/v1/analytics/agents/summary?${params.toString()}`);
      setAllAgentsMetrics(response.data.agents || []);
    } catch (error) {
      console.error('Error fetching agent summary:', error);
      toast.error('Failed to fetch agent summary metrics');
    } finally {
      setLoading(false);
    }
  };

  // Fetch specific agent metrics
  const fetchAgentMetrics = async (agentId: string) => {
    try {
      setLoading(true);
      const response = await api.get(`/api/v1/analytics/agents/${agentId}/metrics`);
      setAgentMetrics(response.data);
    } catch (error) {
      console.error('Error fetching agent metrics:', error);
      toast.error('Failed to fetch agent metrics');
    } finally {
      setLoading(false);
    }
  };

  // Fetch performance trends
  const fetchTrends = async (agentId: string) => {
    try {
      const days = selectedRange.days || 7;
      const response = await api.get(`/api/v1/analytics/agents/${agentId}/trends?days=${days}`);
      setTrends(response.data.trends || []);
    } catch (error) {
      console.error('Error fetching trends:', error);
      toast.error('Failed to fetch performance trends');
    }
  };

  // Fetch token usage
  const fetchTokenUsage = async () => {
    try {
      const params = new URLSearchParams();
      
      if (selectedAgent !== 'all') {
        params.append('agent_id', selectedAgent);
      }
      if (selectedRange.start) {
        params.append('start_date', selectedRange.start);
      }
      if (selectedRange.end) {
        params.append('end_date', selectedRange.end);
      }
      
      const response = await api.get(`/api/v1/analytics/tokens/usage?${params.toString()}`);
      setTokenUsage(response.data);
    } catch (error) {
      console.error('Error fetching token usage:', error);
      toast.error('Failed to fetch token usage');
    }
  };

  // Load data
  const loadData = async () => {
    if (selectedAgent === 'all') {
      await fetchAgentSummary();
      await fetchTokenUsage();
    } else {
      await Promise.all([
        fetchAgentMetrics(selectedAgent),
        fetchTrends(selectedAgent),
        fetchTokenUsage(),
      ]);
    }
  };

  // Initial load and refresh
  useEffect(() => {
    loadData();
  }, [selectedAgent, selectedRange]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      loadData();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [autoRefresh, selectedAgent, selectedRange]);

  // Calculate success/failure pie chart data
  const getSuccessFailureData = (metrics: AgentMetrics) => [
    { name: 'Success', value: metrics.successful_executions, color: COLORS.success },
    { name: 'Failed', value: metrics.failed_executions, color: COLORS.failure },
  ];

  // Render metrics cards
  const renderMetricsCards = (metrics: AgentMetrics) => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <Card className="p-4">
        <div className="text-sm text-gray-500">Total Executions</div>
        <div className="text-2xl font-bold">{metrics.total_executions}</div>
      </Card>
      
      <Card className="p-4">
        <div className="text-sm text-gray-500">Success Rate</div>
        <div className="text-2xl font-bold" style={{ color: COLORS.success }}>
          {metrics.success_rate.toFixed(1)}%
        </div>
      </Card>
      
      <Card className="p-4">
        <div className="text-sm text-gray-500">Avg Duration</div>
        <div className="text-2xl font-bold">
          {metrics.average_duration_seconds.toFixed(2)}s
        </div>
      </Card>
      
      <Card className="p-4">
        <div className="text-sm text-gray-500">Total Duration</div>
        <div className="text-2xl font-bold">
          {(metrics.total_duration_seconds / 60).toFixed(1)}m
        </div>
      </Card>
    </div>
  );

  // Render token usage card
  const renderTokenUsageCard = () => {
    if (!tokenUsage) return null;
    
    return (
      <Card className="p-4 mb-6">
        <h3 className="text-lg font-semibold mb-4">Token Usage</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <div className="text-sm text-gray-500">Total Tokens</div>
            <div className="text-xl font-bold">{tokenUsage.total_tokens.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Avg per Execution</div>
            <div className="text-xl font-bold">{tokenUsage.average_tokens_per_execution.toFixed(0)}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Executions</div>
            <div className="text-xl font-bold">{tokenUsage.total_executions}</div>
          </div>
        </div>
        
        {tokenUsage.daily_usage.length > 0 && (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={tokenUsage.daily_usage}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="tokens" fill={COLORS.primary} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Card>
    );
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Agent Performance Analytics</h1>

        <div className="flex gap-4">
          <Button
            variant={autoRefresh ? 'default' : 'outline'}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? '🔄 Auto-Refresh ON' : '⏸️ Auto-Refresh OFF'}
          </Button>

          <Button onClick={loadData} disabled={loading}>
            {loading ? 'Loading...' : '🔄 Refresh'}
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4 mb-6">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-2">Agent</label>
          <select
            className="w-full p-2 border rounded"
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
          >
            <option value="all">All Agents</option>
            {allAgentsMetrics.map((agent) => (
              <option key={agent.agent_id} value={agent.agent_id}>
                {agent.agent_name || agent.agent_id}
              </option>
            ))}
          </select>
        </div>

        <div className="flex-1">
          <label className="block text-sm font-medium mb-2">Date Range</label>
          <select
            className="w-full p-2 border rounded"
            value={selectedRange.label}
            onChange={(e) => {
              const range = DATE_RANGES.find(r => r.label === e.target.value);
              if (range) setSelectedRange(range);
            }}
          >
            {DATE_RANGES.map((range) => (
              <option key={range.label} value={range.label}>
                {range.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          <p className="mt-2 text-gray-600">Loading analytics...</p>
        </div>
      )}

      {!loading && selectedAgent === 'all' && (
        <>
          {/* Summary metrics */}
          <Card className="p-4 mb-6">
            <h3 className="text-lg font-semibold mb-4">All Agents Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-gray-500">Total Agents</div>
                <div className="text-2xl font-bold">{allAgentsMetrics.length}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Total Executions</div>
                <div className="text-2xl font-bold">
                  {allAgentsMetrics.reduce((sum, a) => sum + a.total_executions, 0)}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Avg Success Rate</div>
                <div className="text-2xl font-bold" style={{ color: COLORS.success }}>
                  {allAgentsMetrics.length > 0
                    ? (allAgentsMetrics.reduce((sum, a) => sum + a.success_rate, 0) / allAgentsMetrics.length).toFixed(1)
                    : 0}%
                </div>
              </div>
            </div>
          </Card>

          {/* Token usage */}
          {renderTokenUsageCard()}

          {/* Agent comparison */}
          <Card className="p-4 mb-6">
            <h3 className="text-lg font-semibold mb-4">Agent Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={allAgentsMetrics.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="agent_name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="total_executions" fill={COLORS.primary} name="Executions" />
                <Bar dataKey="success_rate" fill={COLORS.success} name="Success Rate %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {!loading && selectedAgent !== 'all' && agentMetrics && (
        <>
          {/* Metrics cards */}
          {renderMetricsCards(agentMetrics)}

          {/* Token usage */}
          {renderTokenUsageCard()}

          {/* Success/Failure pie chart */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <Card className="p-4">
              <h3 className="text-lg font-semibold mb-4">Success vs Failure</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={getSuccessFailureData(agentMetrics)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {getSuccessFailureData(agentMetrics).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Error patterns */}
            {agentMetrics.error_patterns && Object.keys(agentMetrics.error_patterns).length > 0 && (
              <Card className="p-4">
                <h3 className="text-lg font-semibold mb-4">Error Patterns</h3>
                <div className="space-y-2">
                  {Object.entries(agentMetrics.error_patterns)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 5)
                    .map(([error, count]) => (
                      <div key={error} className="flex justify-between items-center">
                        <span className="text-sm truncate flex-1">{error}</span>
                        <span className="text-sm font-semibold ml-2">{count}</span>
                      </div>
                    ))}
                </div>
              </Card>
            )}
          </div>

          {/* Performance trends */}
          {trends.length > 0 && (
            <Card className="p-4 mb-6">
              <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="executions"
                    stroke={COLORS.primary}
                    name="Executions"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="success_rate"
                    stroke={COLORS.success}
                    name="Success Rate %"
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="average_duration"
                    stroke={COLORS.secondary}
                    name="Avg Duration (s)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </>
      )}
    </div>
  );
};

export default AgentAnalyticsDashboard;

