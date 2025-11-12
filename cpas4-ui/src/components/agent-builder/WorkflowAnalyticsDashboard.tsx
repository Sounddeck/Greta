/**
 * Workflow Analytics Dashboard Component
 * Display workflow performance metrics and analytics
 */
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, TrendingUp, Clock, CheckCircle, XCircle, Users, Zap } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface WorkflowAnalytics {
  total_executions: number;
  completed: number;
  failed: number;
  running: number;
  success_rate: number;
  avg_execution_time: number;
  min_execution_time: number;
  max_execution_time: number;
  by_workflow_type: Record<string, {
    total: number;
    completed: number;
    failed: number;
    avg_execution_time: number;
  }>;
  filter: string;
}

const WORKFLOW_TYPE_COLORS: Record<string, string> = {
  sequential: '#3b82f6',
  parallel: '#10b981',
  hierarchical: '#f59e0b',
  consensus: '#8b5cf6',
  pipeline: '#ec4899',
  autonomous: '#06b6d4',
  round_robin: '#f97316',
  selector: '#14b8a6'
};

const WORKFLOW_TYPE_LABELS: Record<string, string> = {
  sequential: 'Sequential',
  parallel: 'Parallel',
  hierarchical: 'Hierarchical',
  consensus: 'Consensus',
  pipeline: 'Pipeline',
  autonomous: 'Autonomous',
  round_robin: 'Round Robin',
  selector: 'Selector'
};

export function WorkflowAnalyticsDashboard() {
  const [analytics, setAnalytics] = useState<WorkflowAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    loadAnalytics();
  }, [filter]);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const params = filter !== 'all' ? { workflow_type: filter } : {};
      const response = await axios.get(`${API_BASE_URL}/api/v1/agent-builder/workflows/analytics`, { params });
      setAnalytics(response.data);
    } catch (error) {
      console.error('Failed to load workflow analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading analytics...</div>
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">No analytics data available</div>
      </div>
    );
  }

  // Prepare data for charts
  const workflowTypeData = Object.entries(analytics.by_workflow_type).map(([type, data]) => ({
    name: WORKFLOW_TYPE_LABELS[type] || type,
    total: data.total,
    completed: data.completed,
    failed: data.failed,
    avgTime: data.avg_execution_time
  }));

  const statusData = [
    { name: 'Completed', value: analytics.completed, color: '#10b981' },
    { name: 'Failed', value: analytics.failed, color: '#ef4444' },
    { name: 'Running', value: analytics.running, color: '#3b82f6' }
  ];

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Workflow Analytics</h2>
          <p className="text-muted-foreground mt-1">
            Performance metrics and insights for team workflows
          </p>
        </div>

        {/* Filter */}
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
        >
          <option value="all">All Workflows</option>
          <option value="sequential">Sequential</option>
          <option value="parallel">Parallel</option>
          <option value="hierarchical">Hierarchical</option>
          <option value="consensus">Consensus</option>
          <option value="pipeline">Pipeline</option>
          <option value="autonomous">Autonomous</option>
          <option value="round_robin">Round Robin</option>
          <option value="selector">Selector</option>
        </select>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Executions</p>
              <p className="text-2xl font-bold mt-1">{analytics.total_executions}</p>
            </div>
            <Activity className="h-8 w-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Success Rate</p>
              <p className="text-2xl font-bold mt-1">{analytics.success_rate}%</p>
            </div>
            <TrendingUp className="h-8 w-8 text-green-500" />
          </div>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Execution Time</p>
              <p className="text-2xl font-bold mt-1">{analytics.avg_execution_time.toFixed(2)}s</p>
            </div>
            <Clock className="h-8 w-8 text-yellow-500" />
          </div>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Completed</p>
              <p className="text-2xl font-bold mt-1">{analytics.completed}</p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-500" />
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Workflow Type Distribution */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Executions by Workflow Type</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={workflowTypeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Bar dataKey="completed" fill="#10b981" name="Completed" />
              <Bar dataKey="failed" fill="#ef4444" name="Failed" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Status Distribution */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Execution Status Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={statusData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {statusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Average Execution Time by Type */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Avg Execution Time by Type</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={workflowTypeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Bar dataKey="avgTime" fill="#3b82f6" name="Avg Time (s)" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Performance Metrics */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Min Execution Time</span>
              <span className="font-semibold">{analytics.min_execution_time.toFixed(2)}s</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Max Execution Time</span>
              <span className="font-semibold">{analytics.max_execution_time.toFixed(2)}s</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Avg Execution Time</span>
              <span className="font-semibold">{analytics.avg_execution_time.toFixed(2)}s</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Total Completed</span>
              <span className="font-semibold text-green-500">{analytics.completed}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Total Failed</span>
              <span className="font-semibold text-red-500">{analytics.failed}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Currently Running</span>
              <span className="font-semibold text-blue-500">{analytics.running}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Workflow Type Details Table */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Workflow Type Details</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Workflow Type</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Total</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Completed</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Failed</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Success Rate</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Avg Time</th>
              </tr>
            </thead>
            <tbody>
              {workflowTypeData.map((row) => {
                const successRate = row.total > 0 ? (row.completed / row.total * 100).toFixed(1) : '0.0';
                return (
                  <tr key={row.name} className="border-b border-gray-700/50">
                    <td className="py-3 px-4 font-medium">{row.name}</td>
                    <td className="text-right py-3 px-4">{row.total}</td>
                    <td className="text-right py-3 px-4 text-green-500">{row.completed}</td>
                    <td className="text-right py-3 px-4 text-red-500">{row.failed}</td>
                    <td className="text-right py-3 px-4">{successRate}%</td>
                    <td className="text-right py-3 px-4">{row.avgTime.toFixed(2)}s</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

