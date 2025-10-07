import React from 'react';
import { Activity, Cpu, HardDrive, Wifi } from 'lucide-react';

export const SystemMonitor: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Activity className="w-8 h-8 mr-3 text-red-400" />
            System Monitor
          </h1>
          <p className="text-gray-400 mt-2">
            Real-time system performance and health monitoring
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">CPU Usage</p>
              <p className="text-2xl font-bold text-white">34%</p>
            </div>
            <Cpu className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Memory</p>
              <p className="text-2xl font-bold text-white">67%</p>
            </div>
            <HardDrive className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Network</p>
              <p className="text-2xl font-bold text-white">12MB/s</p>
            </div>
            <Wifi className="w-8 h-8 text-purple-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Uptime</p>
              <p className="text-2xl font-bold text-white">99.9%</p>
            </div>
            <Activity className="w-8 h-8 text-red-400" />
          </div>
        </div>
      </div>

      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4">System Health</h3>
        <div className="text-center text-gray-400 py-8">
          <Activity className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">System Monitoring Dashboard</p>
          <p className="text-sm">Real-time performance metrics and alerts</p>
        </div>
      </div>
    </div>
  );
};

export default SystemMonitor;
