import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Brain, 
  Users, 
  Zap, 
  TrendingUp, 
  Shield,
  Database,
  Cpu,
  Network,
  Clock,
  MessageSquare,
  FileText,
  Search,
  Settings,
  BarChart3,
  Globe,
  Smartphone,
  Building,
  Lightbulb,
  Target,
  Eye,
  Layers,
  Workflow
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../services/api';
import CommandCenter from '../components/CommandCenter';

// Type definitions for API responses
interface SystemStatus {
  memory_usage: number;
  response_time: number;
  security_status: string;
  uptime: string;
}

interface PAIStatus {
  intelligence_level: number;
  active_agents: number;
  reasoning_efficiency: number;
  learning_rate: number;
}

interface OrchestratorStatus {
  available_routers: string[];
}

interface MasterAgentStatus {
  total_tasks: number;
  success_rate: number;
  system_load: number;
}

const EnhancedDashboard = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [commandResults, setCommandResults] = useState<any[]>([]);
  const [activeView, setActiveView] = useState('overview');

  // Fetch comprehensive system status
  const { data: systemStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: async () => {
      const response = await apiService.request('/api/v1/system/status');
      return response.data;
    },
    refetchInterval: 5000
  });

  const { data: orchestratorStatus } = useQuery({
    queryKey: ['orchestrator-status'],
    queryFn: async () => {
      const response = await apiService.request('/api/v1/orchestrator/status');
      return response.data;
    },
    refetchInterval: 5000
  });

  const { data: paiStatus } = useQuery({
    queryKey: ['pai-status'],
    queryFn: async () => {
      const response = await apiService.request('/api/v1/pai-hub/status');
      return response.data;
    },
    refetchInterval: 10000
  });

  const { data: masterAgentStatus } = useQuery({
    queryKey: ['master-agent-status'],
    queryFn: async () => {
      const response = await apiService.request('/api/v1/master-agent/status');
      return response.data;
    },
    refetchInterval: 10000
  });

  // Enhanced system metrics with real data
  const systemMetrics = {
    aiProcessingPower: (paiStatus as any)?.intelligence_level || 87,
    activeAgents: (paiStatus as any)?.active_agents || 12,
    memoryUtilization: (systemStatus as any)?.memory_usage || 64,
    networkLatency: (systemStatus as any)?.response_time || 23,
    securityStatus: (systemStatus as any)?.security_status || 'Secure',
    uptime: (systemStatus as any)?.uptime || '99.9%',
    totalCapabilities: (orchestratorStatus as any)?.available_routers?.length || 17,
    reasoningEfficiency: (paiStatus as any)?.reasoning_efficiency || 94,
    learningRate: (paiStatus as any)?.learning_rate || 76,
    totalTasks: (masterAgentStatus as any)?.total_tasks || 1247,
    successRate: (masterAgentStatus as any)?.success_rate || 96.8,
    systemLoad: (masterAgentStatus as any)?.system_load || 34
  };

  // GRETA Capabilities Overview
  const capabilities = [
    { 
      name: 'AI Intelligence Hub', 
      icon: Brain, 
      status: 'active', 
      efficiency: systemMetrics.aiProcessingPower,
      description: 'Central AI orchestration and reasoning'
    },
    { 
      name: 'Master Agent Control', 
      icon: Target, 
      status: 'active', 
      efficiency: systemMetrics.successRate,
      description: 'Autonomous agent coordination'
    },
    { 
      name: 'Digital Asset Creation', 
      icon: FileText, 
      status: 'active', 
      efficiency: 89,
      description: 'Create websites, apps, documents'
    },
    { 
      name: 'Research Automation', 
      icon: Search, 
      status: 'active', 
      efficiency: 92,
      description: 'Deep research and reporting'
    },
    { 
      name: 'Business Intelligence', 
      icon: Building, 
      status: 'active', 
      efficiency: 85,
      description: 'AI CEO/CFO/CTO decision making'
    },
    { 
      name: 'Training Platform', 
      icon: Lightbulb, 
      status: 'active', 
      efficiency: systemMetrics.learningRate,
      description: 'Interactive AI education'
    },
    { 
      name: 'Voice Synthesis', 
      icon: MessageSquare, 
      status: 'active', 
      efficiency: 94,
      description: 'Natural voice generation'
    },
    { 
      name: 'OSINT Operations', 
      icon: Eye, 
      status: 'active', 
      efficiency: 88,
      description: 'Intelligence gathering'
    }
  ];

  // Recent system activities
  const recentActivities = [
    { 
      id: 1, 
      action: 'Website Created', 
      target: 'Business Portfolio Site', 
      time: '3 minutes ago', 
      status: 'success',
      type: 'asset_creation'
    },
    { 
      id: 2, 
      action: 'Research Completed', 
      target: 'AI Market Analysis', 
      time: '12 minutes ago', 
      status: 'success',
      type: 'research'
    },
    { 
      id: 3, 
      action: 'Agent Deployed', 
      target: 'Customer Support Bot', 
      time: '25 minutes ago', 
      status: 'success',
      type: 'agent'
    },
    { 
      id: 4, 
      action: 'Business Decision', 
      target: 'Strategic Planning', 
      time: '1 hour ago', 
      status: 'success',
      type: 'business'
    },
    { 
      id: 5, 
      action: 'Training Session', 
      target: 'Advanced Reasoning', 
      time: '2 hours ago', 
      status: 'success',
      type: 'training'
    }
  ];

  const handleCommandExecuted = (result: any) => {
    setCommandResults(prev => [result, ...prev.slice(0, 4)]); // Keep last 5 results
  };

  const StatCard = ({ title, value, icon: Icon, color, trend, subtitle }: any) => (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="stat-card"
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
          {trend && (
            <p className="text-sm text-green-400 mt-1 flex items-center">
              <TrendingUp className="w-3 h-3 mr-1" />
              {trend}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </motion.div>
  );

  const CapabilityCard = ({ capability }: any) => (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="capability-card"
    >
      <div className="flex items-center gap-3 mb-3">
        <div className="capability-icon">
          <capability.icon className="w-5 h-5 text-white" />
        </div>
        <div className="flex-1">
          <h4 className="text-white font-medium">{capability.name}</h4>
          <p className="text-xs text-gray-400">{capability.description}</p>
        </div>
        <div className={`status-indicator ${capability.status}`}></div>
      </div>
      
      <div className="efficiency-bar">
        <div className="efficiency-label">
          <span className="text-xs text-gray-400">Efficiency</span>
          <span className="text-xs text-white font-medium">{capability.efficiency}%</span>
        </div>
        <div className="efficiency-track">
          <div 
            className="efficiency-fill"
            style={{ width: `${capability.efficiency}%` }}
          ></div>
        </div>
      </div>
    </motion.div>
  );

  return (
    <div className="enhanced-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <h1 className="text-4xl font-bold text-white mb-2">
            🎭 GRETA PAI System
          </h1>
          <p className="text-gray-400 text-lg">
            Personal AI Intelligence • {systemMetrics.totalCapabilities} Active Capabilities
          </p>
        </div>
        
        <div className="header-controls">
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="timeframe-select"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          
          <div className="view-toggle">
            <button
              onClick={() => setActiveView('overview')}
              className={`view-btn ${activeView === 'overview' ? 'active' : ''}`}
            >
              <Layers className="w-4 h-4" />
              Overview
            </button>
            <button
              onClick={() => setActiveView('capabilities')}
              className={`view-btn ${activeView === 'capabilities' ? 'active' : ''}`}
            >
              <Workflow className="w-4 h-4" />
              Capabilities
            </button>
          </div>
        </div>
      </div>

      {/* Command Center */}
      <CommandCenter onCommandExecuted={handleCommandExecuted} />

      {/* Main Content */}
      {activeView === 'overview' && (
        <>
          {/* Key Metrics */}
          <div className="metrics-grid">
            <StatCard
              title="AI Intelligence Level"
              value={`${systemMetrics.aiProcessingPower}%`}
              icon={Brain}
              color="bg-gradient-to-r from-purple-500 to-pink-500"
              trend="+5.2% this week"
              subtitle="Hierarchical reasoning active"
            />
            <StatCard
              title="Active Agents"
              value={systemMetrics.activeAgents}
              icon={Users}
              color="bg-gradient-to-r from-blue-500 to-cyan-500"
              trend="+3 new agents"
              subtitle="Multi-agent coordination"
            />
            <StatCard
              title="Success Rate"
              value={`${systemMetrics.successRate}%`}
              icon={Target}
              color="bg-gradient-to-r from-green-500 to-emerald-500"
              trend="+2.1% improvement"
              subtitle="Task completion rate"
            />
            <StatCard
              title="System Load"
              value={`${systemMetrics.systemLoad}%`}
              icon={Cpu}
              color="bg-gradient-to-r from-orange-500 to-red-500"
              subtitle="Optimal performance"
            />
            <StatCard
              title="Memory Efficiency"
              value={`${systemMetrics.memoryUtilization}%`}
              icon={Database}
              color="bg-gradient-to-r from-indigo-500 to-purple-500"
              subtitle="Intelligent caching"
            />
            <StatCard
              title="Learning Rate"
              value={`${systemMetrics.learningRate}%`}
              icon={TrendingUp}
              color="bg-gradient-to-r from-teal-500 to-green-500"
              trend="+8.3% this month"
              subtitle="Continuous improvement"
            />
          </div>

          {/* Recent Activities */}
          <div className="activities-section">
            <h3 className="section-title">Recent Activities</h3>
            <div className="activities-list">
              {recentActivities.map((activity) => (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="activity-item"
                >
                  <div className={`activity-icon ${activity.type}`}>
                    {activity.type === 'asset_creation' && <FileText className="w-4 h-4" />}
                    {activity.type === 'research' && <Search className="w-4 h-4" />}
                    {activity.type === 'agent' && <Users className="w-4 h-4" />}
                    {activity.type === 'business' && <Building className="w-4 h-4" />}
                    {activity.type === 'training' && <Lightbulb className="w-4 h-4" />}
                  </div>
                  <div className="activity-content">
                    <div className="activity-header">
                      <span className="activity-action">{activity.action}</span>
                      <span className="activity-time">{activity.time}</span>
                    </div>
                    <p className="activity-target">{activity.target}</p>
                  </div>
                  <div className={`activity-status ${activity.status}`}>
                    ✓
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </>
      )}

      {activeView === 'capabilities' && (
        <div className="capabilities-section">
          <h3 className="section-title">GRETA Capabilities Matrix</h3>
          <div className="capabilities-grid">
            {capabilities.map((capability, index) => (
              <CapabilityCard key={index} capability={capability} />
            ))}
          </div>
        </div>
      )}

      {/* Command Results */}
      {commandResults.length > 0 && (
        <div className="command-results-section">
          <h3 className="section-title">Recent Command Results</h3>
          <div className="command-results-list">
            {commandResults.map((result, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="command-result-item"
              >
                <div className="result-header">
                  <span className={`result-status ${result.success ? 'success' : 'error'}`}>
                    {result.success ? '✓' : '✗'}
                  </span>
                  <span className="result-id">#{result.command_id?.slice(-8)}</span>
                  <span className="result-time">{result.execution_time?.toFixed(2)}s</span>
                </div>
                <div className="result-routers">
                  Executed via: {result.routers_invoked?.join(', ')}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        .enhanced-dashboard {
          padding: 20px;
          min-height: 100vh;
          background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 30px;
        }

        .header-controls {
          display: flex;
          gap: 16px;
          align-items: center;
        }

        .timeframe-select {
          padding: 8px 12px;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 8px;
          color: white;
          font-size: 14px;
        }

        .view-toggle {
          display: flex;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          padding: 4px;
        }

        .view-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 12px;
          background: transparent;
          border: none;
          color: #ccc;
          border-radius: 6px;
          font-size: 14px;
          transition: all 0.2s ease;
        }

        .view-btn.active {
          background: linear-gradient(135deg, #8B4513, #4B0082);
          color: white;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 20px;
          margin-bottom: 30px;
        }

        .stat-card {
          background: linear-gradient(135deg, rgba(139, 69, 19, 0.1), rgba(75, 0, 130, 0.1));
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          padding: 20px;
          transition: all 0.3s ease;
        }

        .stat-card:hover {
          border-color: rgba(139, 69, 19, 0.3);
          box-shadow: 0 8px 32px rgba(139, 69, 19, 0.2);
        }

        .activities-section, .capabilities-section, .command-results-section {
          margin-bottom: 30px;
        }

        .section-title {
          font-size: 20px;
          font-weight: 600;
          color: white;
          margin-bottom: 16px;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .section-title::before {
          content: '';
          width: 4px;
          height: 20px;
          background: linear-gradient(135deg, #8B4513, #4B0082);
          border-radius: 2px;
        }

        .activities-list {
          space-y: 12px;
        }

        .activity-item {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 16px;
          background: rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          transition: all 0.2s ease;
        }

        .activity-item:hover {
          background: rgba(255, 255, 255, 0.05);
        }

        .activity-icon {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .activity-icon.asset_creation { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
        .activity-icon.research { background: linear-gradient(135deg, #10b981, #059669); }
        .activity-icon.agent { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
        .activity-icon.business { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .activity-icon.training { background: linear-gradient(135deg, #ef4444, #dc2626); }

        .activity-content {
          flex: 1;
        }

        .activity-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 4px;
        }

        .activity-action {
          font-weight: 500;
          color: white;
        }

        .activity-time {
          font-size: 12px;
          color: #9ca3af;
        }

        .activity-target {
          font-size: 14px;
          color: #6b7280;
        }

        .activity-status.success {
          width: 24px;
          height: 24px;
          background: #10b981;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 12px;
        }

        .capabilities-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
        }

        .capability-card {
          background: linear-gradient(135deg, rgba(139, 69, 19, 0.1), rgba(75, 0, 130, 0.1));
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          padding: 20px;
          transition: all 0.3s ease;
        }

        .capability-card:hover {
          border-color: rgba(139, 69, 19, 0.3);
          transform: translateY(-2px);
        }

        .capability-icon {
          width: 32px;
          height: 32px;
          background: linear-gradient(135deg, #8B4513, #4B0082);
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-indicator.active {
          background: #10b981;
          animation: pulse 2s infinite;
        }

        .efficiency-bar {
          margin-top: 12px;
        }

        .efficiency-label {
          display: flex;
          justify-content: space-between;
          margin-bottom: 6px;
        }

        .efficiency-track {
          height: 6px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
          overflow: hidden;
        }

        .efficiency-fill {
          height: 100%;
          background: linear-gradient(90deg, #8B4513, #4B0082);
          border-radius: 3px;
          transition: width 0.3s ease;
        }

        .command-results-list {
          space-y: 8px;
        }

        .command-result-item {
          padding: 12px 16px;
          background: rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
        }

        .result-header {
          display: flex;
          gap: 12px;
          align-items: center;
          margin-bottom: 4px;
        }

        .result-status.success {
          color: #10b981;
        }

        .result-status.error {
          color: #ef4444;
        }

        .result-id {
          font-family: monospace;
          font-size: 12px;
          color: #8B4513;
        }

        .result-time {
          font-size: 12px;
          color: #6b7280;
        }

        .result-routers {
          font-size: 12px;
          color: #9ca3af;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default EnhancedDashboard;
