import React from 'react';
import { Shield, Users, BarChart3, Settings } from 'lucide-react';

export const Enterprise: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Shield className="w-8 h-8 mr-3 text-gold-400" />
            Enterprise Features
          </h1>
          <p className="text-gray-400 mt-2">
            Commercial-grade AI capabilities and business intelligence
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Enterprise Users</p>
              <p className="text-2xl font-bold text-white">247</p>
            </div>
            <Users className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Business ROI</p>
              <p className="text-2xl font-bold text-white">340%</p>
            </div>
            <BarChart3 className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Security Score</p>
              <p className="text-2xl font-bold text-white">98%</p>
            </div>
            <Shield className="w-8 h-8 text-red-400" />
          </div>
        </div>
      </div>

      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4">Enterprise Dashboard</h3>
        <div className="text-center text-gray-400 py-8">
          <Shield className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Enterprise AI Platform</p>
          <p className="text-sm">Commercial-grade features and business intelligence</p>
        </div>
      </div>
    </div>
  );
};

export default Enterprise;
