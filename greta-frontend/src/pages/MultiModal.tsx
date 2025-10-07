import React from 'react';
import { Layers, Image, Video, Mic } from 'lucide-react';

export const MultiModal: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Layers className="w-8 h-8 mr-3 text-orange-400" />
            Multi-Modal Processing
          </h1>
          <p className="text-gray-400 mt-2">
            Process and analyze text, voice, images, and video content
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Supported Formats</p>
              <p className="text-2xl font-bold text-white">12</p>
            </div>
            <Layers className="w-8 h-8 text-orange-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Processing Speed</p>
              <p className="text-2xl font-bold text-white">3.2s</p>
            </div>
            <Video className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Accuracy Rate</p>
              <p className="text-2xl font-bold text-white">91%</p>
            </div>
            <Image className="w-8 h-8 text-green-400" />
          </div>
        </div>
      </div>

      <div className="glass p-6 rounded-xl">
        <h3 className="text-xl font-semibold text-white mb-4">Multi-Modal Capabilities</h3>
        <div className="text-center text-gray-400 py-8">
          <Layers className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Multi-Modal Processing Engine</p>
          <p className="text-sm">Advanced content processing across all media types</p>
        </div>
      </div>
    </div>
  );
};

export default MultiModal;
