import React from 'react';
import { BookOpen, Brain, Search, Database } from 'lucide-react';

export const Knowledge: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <BookOpen className="w-8 h-8 mr-3 text-green-400" />
            Knowledge Base
          </h1>
          <p className="text-gray-400 mt-2">
            Centralized knowledge management and intelligent search
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Knowledge Items</p>
              <p className="text-2xl font-bold text-white">2,847</p>
            </div>
            <Database className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Search Accuracy</p>
              <p className="text-2xl font-bold text-white">96%</p>
            </div>
            <Search className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Learning Rate</p>
              <p className="text-2xl font-bold text-white">78%</p>
            </div>
            <Brain className="w-8 h-8 text-purple-400" />
          </div>
        </div>
      </div>

      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4">Knowledge Management</h3>
        <div className="text-center text-gray-400 py-8">
          <BookOpen className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Knowledge Base System</p>
          <p className="text-sm">Intelligent knowledge organization and retrieval</p>
        </div>
      </div>
    </div>
  );
};

export default Knowledge;
