import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Brain, 
  Activity, 
  Zap, 
  TrendingUp, 
  Users, 
  MessageSquare,
  Settings,
  AlertCircle,
  CheckCircle,
  Clock,
  BarChart3,
  Cpu,
  Database
} from 'lucide-react';
import { apiService } from '../services/api';

interface PAIStatus {
  intelligence_level: number;
  active_agents: number;
  reasoning_efficiency: number;
  memory_utilization: number;
  learning_rate: number;
  system_health: string;
  capabilities: string[];
  recent_activities: Array<{
    timestamp: string;
    type: string;
    description: string;
    status: string;
  }>;
}

export const PAIHub: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<string>('overview');

  const { data: paiStatus, isLoading } = useQuery({
    queryKey: ['pai-status'],
    queryFn: async () => {
      try {
        const response = await apiService.getSystemStatus();
        return response.data as PAIStatus;
      } catch (error) {
        // Mock data for development
        return {
          intelligence_level: 87,
          active_agents: 12,
          reasoning_efficiency: 94,
          memory_utilization: 68,
          learning_rate: 76,
          system_health: 'optimal',
          capabilities: [
            'Hierarchical Reasoning',
            'Memory Orchestration',
            'Multi-Agent Coordination',
            'Predictive Analytics',
            'Natural Language Processing',
            'Voice Synthesis',
            'Learning Adaptation'
          ],
          recent_activities: [
            {
              timestamp: new Date(Date.now() - 300000).toISOString(),
              type: 'reasoning',
              description: 'Completed strategic analysis for business planning',
              status: 'success'
            },
            {
              timestamp: new Date(Date.now() - 600000).toISOString(),
              type: 'learning',
              description: 'Updated user preference model',
              status: 'success'
            },
            {
              timestamp: new Date(Date.now() - 900000).toISOString(),
              type: 'memory',
              description: 'Consolidated conversation memories',
              status: 'success'
            }
          ]
        } as PAIStatus;
      }
    },
    refetchInterval: 30000
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'optimal': return 'text-green-400';
      case 'good': return 'text-blue-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'reasoning': return <Brain className="w-4 h-4" />;
      case 'learning': return <TrendingUp className="w-4 h-4" />;
      case 'memory': return <Database className="w-4 h-4" />;
      case 'agent': return <Users className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
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
            <Brain className="w-8 h-8 mr-3 text-blue-400" />
            PAI Intelligence Hub
          </h1>
          <p className="text-gray-400 mt-2">
            Central command for Personal AI orchestration and monitoring
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full glass ${getStatusColor(paiStatus?.system_health || 'unknown')}`}>
            <div className="w-2 h-2 bg-current rounded-full animate-pulse"></div>
            <span className="text-sm font-medium capitalize">{paiStatus?.system_health || 'Unknown'}</span>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Intelligence Level</p>
              <p className="text-2xl font-bold text-white">{paiStatus?.intelligence_level || 0}%</p>
            </div>
            <Brain className="w-8 h-8 text-blue-400" />
          </div>
          <div className="mt-4 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-blue-500 to-blue-700 h-2 rounded-full transition-all duration-500"
              style={{ width: `${paiStatus?.intelligence_level || 0}%` }}
            ></div>
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Agents</p>
              <p className="text-2xl font-bold text-white">{paiStatus?.active_agents || 0}</p>
            </div>
            <Users className="w-8 h-8 text-blue-400" />
          </div>
          <p className="text-green-400 text-sm mt-2">+2 from last hour</p>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Reasoning Efficiency</p>
              <p className="text-2xl font-bold text-white">{paiStatus?.reasoning_efficiency || 0}%</p>
            </div>
            <Zap className="w-8 h-8 text-yellow-400" />
          </div>
          <div className="mt-4 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-yellow-500 to-orange-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${paiStatus?.reasoning_efficiency || 0}%` }}
            ></div>
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Learning Rate</p>
              <p className="text-2xl font-bold text-white">{paiStatus?.learning_rate || 0}%</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-400" />
          </div>
          <p className="text-blue-400 text-sm mt-2">Adaptive learning active</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Capabilities */}
        <div className="lg:col-span-2 glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Cpu className="w-5 h-5 mr-2 text-blue-400" />
            Active Capabilities
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(paiStatus?.capabilities || []).map((capability, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 bg-white/5 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-gray-300">{capability}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Activities */}
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-400" />
            Recent Activities
          </h3>
          <div className="space-y-4">
            {(paiStatus?.recent_activities || []).map((activity, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-white/5 rounded-lg">
                <div className="text-gray-400 mt-1">
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-300">{activity.description}</p>
                  <div className="flex items-center space-x-2 mt-1">
                    <Clock className="w-3 h-3 text-gray-500" />
                    <span className="text-xs text-gray-500">
                      {new Date(activity.timestamp).toLocaleTimeString()}
                    </span>
                    <CheckCircle className="w-3 h-3 text-green-400" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Memory and Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Database className="w-5 h-5 mr-2 text-green-400" />
            Memory Utilization
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Working Memory</span>
                <span className="text-white">{paiStatus?.memory_utilization || 0}%</span>
              </div>
              <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                  style={{ width: `${paiStatus?.memory_utilization || 0}%` }}
                ></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Long-term Storage</span>
                <span className="text-white">45%</span>
              </div>
              <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                <div className="bg-gradient-to-r from-blue-500 to-blue-700 h-2 rounded-full w-[45%]"></div>
              </div>
            </div>
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-yellow-400" />
            Performance Metrics
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Response Time</span>
              <span className="text-green-400">1.2s avg</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Accuracy Rate</span>
              <span className="text-green-400">96.8%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Uptime</span>
              <span className="text-green-400">99.9%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Tasks Completed</span>
              <span className="text-blue-400">1,247 today</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PAIHub;
