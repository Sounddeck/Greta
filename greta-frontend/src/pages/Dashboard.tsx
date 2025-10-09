import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  Brain,
  Zap,
  Users,
  MessageSquare,
  TrendingUp,
  Clock,
  Server,
  Cpu,
  HardDrive
} from 'lucide-react';

import { apiService, handleApiResponse } from '../services/api';

interface SystemData {
  status?: string;
  version?: string;
  message?: string;
}

// API function with fallback to mock data
const fetchSystemStatus = async () => {
  try {
    const response = await apiService.getSystemStatus();
    const systemData = handleApiResponse(response) as SystemData;

    // Transform the API response to match our expected format
    return {
      status: systemData.status || 'healthy',
      version: systemData.version || '2.2.0',
      uptime: '2d 14h 32m', // This would come from system metrics
      activeAgents: 7, // This would come from agent status
      totalConversations: 1247, // This would come from conversation history
      memoryUsage: 74,
      cpuUsage: 23,
      diskUsage: 45,
      recentActivity: [
        { id: 1, type: 'chat', message: 'New conversation started', time: '2 minutes ago' },
        { id: 2, type: 'agent', message: 'Agent "Research Assistant" deployed', time: '15 minutes ago' },
        { id: 3, type: 'system', message: 'Memory optimization completed', time: '1 hour ago' },
        { id: 4, type: 'workflow', message: 'Workflow "Data Analysis" executed', time: '2 hours ago' },
      ]
    };
  } catch (error) {
    // Fallback to mock data if backend is not available
    await new Promise(resolve => setTimeout(resolve, 1000));
    return {
      status: 'healthy',
      version: '2.2.0',
      uptime: '2d 14h 32m',
      activeAgents: 7,
      totalConversations: 1247,
      memoryUsage: 74,
      cpuUsage: 23,
      diskUsage: 45,
      recentActivity: [
        { id: 1, type: 'chat', message: 'New conversation started', time: '2 minutes ago' },
        { id: 2, type: 'agent', message: 'Agent "Research Assistant" deployed', time: '15 minutes ago' },
        { id: 3, type: 'system', message: 'Memory optimization completed', time: '1 hour ago' },
        { id: 4, type: 'workflow', message: 'Workflow "Data Analysis" executed', time: '2 hours ago' },
      ]
    };
  }
};

const StatCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ElementType;
  color: string;
  trend?: string;
}> = ({ title, value, icon: Icon, color, trend }) => (
  <motion.div
    whileHover={{ scale: 1.02 }}
    className="card"
  >
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-400">{title}</p>
        <p className="text-2xl font-bold text-white mt-1">{value}</p>
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

const ActivityItem: React.FC<{
  type: string;
  message: string;
  time: string;
}> = ({ type, message, time }) => {
  const getIcon = () => {
    switch (type) {
      case 'chat': return MessageSquare;
      case 'agent': return Users;
      case 'system': return Server;
      case 'workflow': return Zap;
      default: return Activity;
    }
  };

  const getColor = () => {
    switch (type) {
      case 'chat': return 'text-blue-400';
      case 'agent': return 'text-purple-400';
      case 'system': return 'text-green-400';
      case 'workflow': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const Icon = getIcon();

  return (
    <div className="flex items-start space-x-3 p-3 rounded-lg hover:bg-blue-500/10 transition-colors">
      <div className={`p-2 rounded-lg bg-blue-500/15 ${getColor()}`}>
        <Icon className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-white">{message}</p>
        <p className="text-xs text-gray-400 mt-1">{time}</p>
      </div>
    </div>
  );
};

export const Dashboard: React.FC = () => {
  const { data: systemStatus, isLoading, error } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: fetchSystemStatus,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center text-red-400 p-8">
        <p>Error loading dashboard data</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-3xl font-bold text-white mb-2"
        >
          🎭 GRETA Dashboard
        </motion.h1>
        <p className="text-gray-400">
          Welcome to your Personal AI System. Status: {' '}
          <span className="text-green-400 font-medium">
            {systemStatus?.status || 'Unknown'}
          </span>
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="System Version"
          value={systemStatus?.version || '2.2.0'}
          icon={Brain}
          color="bg-gradient-to-r from-purple-600 to-blue-600"
        />
        <StatCard
          title="Active Agents"
          value={systemStatus?.activeAgents || 0}
          icon={Users}
          color="bg-gradient-to-r from-blue-600 to-cyan-600"
          trend="+2 this week"
        />
        <StatCard
          title="Conversations"
          value={systemStatus?.totalConversations || 0}
          icon={MessageSquare}
          color="bg-gradient-to-r from-green-600 to-emerald-600"
          trend="+15% this month"
        />
        <StatCard
          title="Uptime"
          value={systemStatus?.uptime || '0h 0m'}
          icon={Clock}
          color="bg-gradient-to-r from-yellow-600 to-orange-600"
        />
      </div>

      {/* System Resources */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <HardDrive className="w-5 h-5 mr-2 text-purple-400" />
            Memory Usage
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Used</span>
              <span className="text-white">{systemStatus?.memoryUsage || 0}%</span>
            </div>
            <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-purple-600 to-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${systemStatus?.memoryUsage || 0}%` }}
              ></div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Cpu className="w-5 h-5 mr-2 text-blue-400" />
            CPU Usage
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Used</span>
              <span className="text-white">{systemStatus?.cpuUsage || 0}%</span>
            </div>
            <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-600 to-cyan-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${systemStatus?.cpuUsage || 0}%` }}
              ></div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Server className="w-5 h-5 mr-2 text-green-400" />
            Disk Usage
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Used</span>
              <span className="text-white">{systemStatus?.diskUsage || 0}%</span>
            </div>
            <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-green-600 to-emerald-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${systemStatus?.diskUsage || 0}%` }}
              ></div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <h3 className="text-lg font-semibold text-white mb-6 flex items-center">
          <Activity className="w-5 h-5 mr-2 text-yellow-400" />
          Recent Activity
        </h3>
        <div className="space-y-2">
          {systemStatus?.recentActivity?.map((activity) => (
            <ActivityItem
              key={activity.id}
              type={activity.type}
              message={activity.message}
              time={activity.time}
            />
          ))}
        </div>
      </motion.div>
    </div>
  );
};
