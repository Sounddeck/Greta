import React from 'react';
import { Zap, Activity, Settings, Play } from 'lucide-react';

export const Autonomous: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Zap className="w-8 h-8 mr-3 text-yellow-400" />
            Autonomous Operations
          </h1>
          <p className="text-gray-400 mt-2">
            Self-directed AI workflows and automated decision making
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Workflows</p>
              <p className="text-2xl font-bold text-white">12</p>
            </div>
            <Activity className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Automation Rate</p>
              <p className="text-2xl font-bold text-white">87%</p>
            </div>
            <Settings className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Tasks Completed</p>
              <p className="text-2xl font-bold text-white">1,247</p>
            </div>
            <Play className="w-8 h-8 text-purple-400" />
          </div>
        </div>
      </div>

      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4">Autonomous Systems</h3>
        <div className="text-center text-gray-400 py-8">
          <Zap className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Autonomous Operations Center</p>
          <p className="text-sm">Self-directed AI workflows coming soon</p>
        </div>
      </div>
    </div>
  );
};

export default Autonomous;
