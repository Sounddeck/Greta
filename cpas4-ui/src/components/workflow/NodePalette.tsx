/**
 * Node Palette Component
 * 
 * Draggable palette of node types for the workflow canvas
 */

import React from 'react';
import { Play, Zap, GitBranch, Bot } from 'lucide-react';

interface NodeTypeItem {
  type: string;
  label: string;
  icon: React.ReactNode;
  color: string;
  description: string;
}

const nodeTypes: NodeTypeItem[] = [
  {
    type: 'trigger',
    label: 'Trigger',
    icon: <Play className="h-4 w-4" />,
    color: 'bg-green-500',
    description: 'Start workflow',
  },
  {
    type: 'action',
    label: 'Action',
    icon: <Zap className="h-4 w-4" />,
    color: 'bg-blue-500',
    description: 'Perform task',
  },
  {
    type: 'condition',
    label: 'Condition',
    icon: <GitBranch className="h-4 w-4" />,
    color: 'bg-amber-500',
    description: 'Branch logic',
  },
  {
    type: 'agent',
    label: 'AI Agent',
    icon: <Bot className="h-4 w-4" />,
    color: 'bg-purple-500',
    description: 'AI processing',
  },
];

export default function NodePalette() {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 w-64">
      <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
        Node Palette
      </h3>
      <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">
        Drag nodes onto the canvas
      </p>

      <div className="space-y-2">
        {nodeTypes.map((node) => (
          <div
            key={node.type}
            draggable
            onDragStart={(e) => onDragStart(e, node.type)}
            className="flex items-center gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-750 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-move transition-colors"
          >
            <div className={`flex items-center justify-center w-8 h-8 rounded-full ${node.color} text-white flex-shrink-0`}>
              {node.icon}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-900 dark:text-white">
                {node.label}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                {node.description}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-500 text-center">
          Click and drag to add nodes
        </p>
      </div>
    </div>
  );
}

