import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Zap, 
  Brain, 
  TrendingUp, 
  Clock, 
  Target,
  CheckCircle,
  AlertCircle,
  BarChart3,
  Activity,
  Layers,
  ArrowRight,
  Lightbulb,
  Cpu,
  Database
} from 'lucide-react';
import { apiService } from '../services/api';

interface ReasoningLevel {
  name: string;
  description: string;
  speed: string;
  accuracy: number;
  use_cases: string[];
  active: boolean;
}

interface ReasoningTask {
  id: string;
  type: string;
  description: string;
  level: 'fast' | 'medium' | 'deep';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  start_time: string;
  completion_time?: string;
  result?: any;
  confidence: number;
}

interface ReasoningMetrics {
  total_tasks: number;
  success_rate: number;
  avg_response_time: number;
  efficiency_score: number;
  reasoning_levels: ReasoningLevel[];
  recent_tasks: ReasoningTask[];
}

export const Reasoning: React.FC = () => {
  const [selectedLevel, setSelectedLevel] = useState<string>('medium');
  const [testPrompt, setTestPrompt] = useState('');

  const { data: metrics, isLoading } = useQuery({
    queryKey: ['reasoning-metrics'],
    queryFn: async () => {
      try {
        const response = await apiService.request('/api/v1/reasoning/metrics');
        return response.data as ReasoningMetrics;
      } catch (error) {
        // Mock data for development
        return {
          total_tasks: 1247,
          success_rate: 94.2,
          avg_response_time: 2.3,
          efficiency_score: 87,
          reasoning_levels: [
            {
              name: 'Fast Reasoning',
              description: 'Quick responses for simple queries and immediate decisions',
              speed: '< 1s',
              accuracy: 89,
              use_cases: ['Simple Q&A', 'Basic Classification', 'Quick Decisions'],
              active: true
            },
            {
              name: 'Medium Reasoning',
              description: 'Balanced approach for most complex reasoning tasks',
              speed: '1-5s',
              accuracy: 94,
              use_cases: ['Analysis', 'Problem Solving', 'Strategic Planning'],
              active: true
            },
            {
              name: 'Deep Reasoning',
              description: 'Comprehensive analysis for complex, multi-step problems',
              speed: '5-30s',
              accuracy: 97,
              use_cases: ['Research', 'Complex Analysis', 'Critical Decisions'],
              active: true
            }
          ],
          recent_tasks: [
            {
              id: 'task-001',
              type: 'Strategic Analysis',
              description: 'Analyze market trends for Q4 planning',
              level: 'deep',
              status: 'completed',
              start_time: new Date(Date.now() - 300000).toISOString(),
              completion_time: new Date(Date.now() - 120000).toISOString(),
              confidence: 96
            },
            {
              id: 'task-002',
              type: 'Problem Solving',
              description: 'Optimize workflow efficiency',
              level: 'medium',
              status: 'completed',
              start_time: new Date(Date.now() - 600000).toISOString(),
              completion_time: new Date(Date.now() - 480000).toISOString(),
              confidence: 91
            },
            {
              id: 'task-003',
              type: 'Quick Decision',
              description: 'Prioritize incoming requests',
              level: 'fast',
              status: 'processing',
              start_time: new Date(Date.now() - 30000).toISOString(),
              confidence: 0
            }
          ]
        } as ReasoningMetrics;
      }
    },
    refetchInterval: 5000
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400';
      case 'processing': return 'text-blue-400';
      case 'pending': return 'text-yellow-400';
      case 'failed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'processing': return <Activity className="w-4 h-4 animate-spin" />;
      case 'pending': return <Clock className="w-4 h-4" />;
      case 'failed': return <AlertCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'fast': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'deep': return 'bg-red-500/20 text-red-400 border-red-500/30';
      default: return 'bg-blue-600/20 text-gray-400 border-gray-500/30';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Zap className="w-8 h-8 mr-3 text-yellow-400" />
            Hierarchical Reasoning Engine
          </h1>
          <p className="text-gray-400 mt-2">
            Multi-level reasoning system for optimal decision making and analysis
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="text-right">
            <p className="text-sm text-gray-400">Efficiency Score</p>
            <p className="text-xl font-bold text-yellow-400">{metrics?.efficiency_score || 0}%</p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Tasks</p>
              <p className="text-2xl font-bold text-white">{metrics?.total_tasks || 0}</p>
            </div>
            <Target className="w-8 h-8 text-blue-400" />
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

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Levels</p>
              <p className="text-2xl font-bold text-white">
                {metrics?.reasoning_levels.filter(l => l.active).length || 0}
              </p>
            </div>
            <Layers className="w-8 h-8 text-orange-400" />
          </div>
        </div>
      </div>

      {/* Reasoning Levels */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {metrics?.reasoning_levels.map((level, index) => (
          <div 
            key={level.name}
            className={`glass p-6 rounded-xl cursor-pointer transition-all hover:scale-105 ${
              selectedLevel === level.name.toLowerCase().split(' ')[0] ? 'ring-2 ring-purple-500' : ''
            }`}
            onClick={() => setSelectedLevel(level.name.toLowerCase().split(' ')[0])}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">{level.name}</h3>
              <div className={`px-3 py-1 rounded-full text-sm border ${getLevelColor(level.name.toLowerCase().split(' ')[0])}`}>
                {level.speed}
              </div>
            </div>
            
            <p className="text-gray-400 text-sm mb-4">{level.description}</p>
            
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Accuracy</span>
                  <span className="text-white">{level.accuracy}%</span>
                </div>
                <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                    style={{ width: `${level.accuracy}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <p className="text-sm text-gray-400 mb-2">Use Cases</p>
                <div className="space-y-1">
                  {level.use_cases.map((useCase, idx) => (
                    <div key={idx} className="flex items-center space-x-2">
                      <ArrowRight className="w-3 h-3 text-gray-500" />
                      <span className="text-xs text-gray-300">{useCase}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Tasks and Test Interface */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Tasks */}
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-400" />
            Recent Reasoning Tasks
          </h3>
          <div className="space-y-4">
            {metrics?.recent_tasks.map((task) => (
              <div key={task.id} className="p-4 bg-blue-500/10 rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h4 className="font-medium text-white">{task.type}</h4>
                      <span className={`px-2 py-1 text-xs rounded-full border ${getLevelColor(task.level)}`}>
                        {task.level}
                      </span>
                      <div className={`flex items-center space-x-1 ${getStatusColor(task.status)}`}>
                        {getStatusIcon(task.status)}
                        <span className="text-xs capitalize">{task.status}</span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-400 mb-2">{task.description}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>Started: {new Date(task.start_time).toLocaleTimeString()}</span>
                      {task.completion_time && (
                        <span>Completed: {new Date(task.completion_time).toLocaleTimeString()}</span>
                      )}
                      {task.confidence > 0 && (
                        <span>Confidence: {task.confidence}%</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Test Interface */}
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Lightbulb className="w-5 h-5 mr-2 text-yellow-400" />
            Test Reasoning Engine
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Reasoning Level</label>
              <select
                value={selectedLevel}
                onChange={(e) => setSelectedLevel(e.target.value)}
                className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
              >
                <option value="fast">Fast Reasoning (&lt; 1s)</option>
                <option value="medium">Medium Reasoning (1-5s)</option>
                <option value="deep">Deep Reasoning (5-30s)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-2">Test Prompt</label>
              <textarea
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
                placeholder="Enter a reasoning task to test the engine..."
                className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none resize-none"
                rows={4}
              />
            </div>
            
            <button
              disabled={!testPrompt.trim()}
              className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              <Brain className="w-4 h-4" />
              <span>Execute Reasoning Task</span>
            </button>
            
            <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Cpu className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-blue-400">System Status</span>
              </div>
              <div className="space-y-2 text-sm text-gray-300">
                <div className="flex justify-between">
                  <span>Processing Queue:</span>
                  <span className="text-blue-400">2 tasks</span>
                </div>
                <div className="flex justify-between">
                  <span>Memory Usage:</span>
                  <span className="text-green-400">67%</span>
                </div>
                <div className="flex justify-between">
                  <span>Response Time:</span>
                  <span className="text-yellow-400">1.2s avg</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Reasoning;
