/**
 * Action Node Component
 * 
 * Represents action/task nodes in the workflow
 */

import React from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { Zap, Send, Database, FileText, Code, Globe } from 'lucide-react';

export interface ActionNodeData {
  label: string;
  actionType?: 'api' | 'email' | 'database' | 'file' | 'code' | 'webhook';
  config?: any;
  status?: 'idle' | 'running' | 'completed' | 'failed';
}

export default function ActionNode({ data, selected }: NodeProps<ActionNodeData>) {
  const getActionIcon = () => {
    switch (data.actionType) {
      case 'email':
        return <Send className="h-4 w-4" />;
      case 'database':
        return <Database className="h-4 w-4" />;
      case 'file':
        return <FileText className="h-4 w-4" />;
      case 'code':
        return <Code className="h-4 w-4" />;
      case 'webhook':
        return <Globe className="h-4 w-4" />;
      default:
        return <Zap className="h-4 w-4" />;
    }
  };

  const getStatusColor = () => {
    switch (data.status) {
      case 'running':
        return 'border-yellow-500 dark:border-yellow-600';
      case 'completed':
        return 'border-green-500 dark:border-green-600';
      case 'failed':
        return 'border-red-500 dark:border-red-600';
      default:
        return 'border-blue-500 dark:border-blue-600';
    }
  };

  const getStatusBadge = () => {
    if (!data.status || data.status === 'idle') return null;

    const statusConfig = {
      running: { color: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400', label: 'Running' },
      completed: { color: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400', label: 'Completed' },
      failed: { color: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400', label: 'Failed' },
    };

    const config = statusConfig[data.status];
    if (!config) return null;

    return (
      <div className={`mt-2 inline-flex items-center px-2 py-1 rounded text-xs font-medium ${config.color}`}>
        {config.label}
      </div>
    );
  };

  return (
    <div
      className={`
        px-4 py-3 rounded-lg border-2 shadow-lg min-w-[180px]
        bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20
        ${getStatusColor()}
        ${selected ? 'ring-2 ring-blue-500 ring-offset-2' : ''}
        ${data.status === 'running' ? 'animate-pulse' : ''}
        transition-all duration-200
      `}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-blue-500 border-2 border-white dark:border-gray-800"
      />

      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-500 text-white">
          {getActionIcon()}
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-blue-700 dark:text-blue-400 uppercase tracking-wide">
            Action
          </div>
          <div className="text-sm font-semibold text-gray-900 dark:text-white">
            {data.label}
          </div>
        </div>
      </div>

      {/* Action type badge */}
      {data.actionType && (
        <div className="mt-2 inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
          {data.actionType}
        </div>
      )}

      {/* Status badge */}
      {getStatusBadge()}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-blue-500 border-2 border-white dark:border-gray-800"
      />
    </div>
  );
}

