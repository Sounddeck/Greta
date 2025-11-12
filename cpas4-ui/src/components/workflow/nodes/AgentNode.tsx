/**
 * Agent Node Component
 * 
 * Represents AI agent nodes in the workflow
 */

import React from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { Bot, Brain, MessageSquare, Search } from 'lucide-react';

export interface AgentNodeData {
  label: string;
  agentType?: 'assistant' | 'researcher' | 'analyzer' | 'custom';
  model?: string;
  config?: any;
  status?: 'idle' | 'thinking' | 'completed' | 'failed';
}

export default function AgentNode({ data, selected }: NodeProps<AgentNodeData>) {
  const getAgentIcon = () => {
    switch (data.agentType) {
      case 'researcher':
        return <Search className="h-4 w-4" />;
      case 'analyzer':
        return <Brain className="h-4 w-4" />;
      case 'assistant':
        return <MessageSquare className="h-4 w-4" />;
      default:
        return <Bot className="h-4 w-4" />;
    }
  };

  const getStatusColor = () => {
    switch (data.status) {
      case 'thinking':
        return 'border-yellow-500 dark:border-yellow-600';
      case 'completed':
        return 'border-green-500 dark:border-green-600';
      case 'failed':
        return 'border-red-500 dark:border-red-600';
      default:
        return 'border-purple-500 dark:border-purple-600';
    }
  };

  const getStatusBadge = () => {
    if (!data.status || data.status === 'idle') return null;

    const statusConfig = {
      thinking: { color: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400', label: 'Thinking...' },
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
        bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20
        ${getStatusColor()}
        ${selected ? 'ring-2 ring-purple-500 ring-offset-2' : ''}
        ${data.status === 'thinking' ? 'animate-pulse' : ''}
        transition-all duration-200
      `}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-purple-500 border-2 border-white dark:border-gray-800"
      />

      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-500 text-white">
          {getAgentIcon()}
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-purple-700 dark:text-purple-400 uppercase tracking-wide">
            AI Agent
          </div>
          <div className="text-sm font-semibold text-gray-900 dark:text-white">
            {data.label}
          </div>
        </div>
      </div>

      {/* Agent type badge */}
      {data.agentType && (
        <div className="mt-2 inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400">
          {data.agentType}
        </div>
      )}

      {/* Model info */}
      {data.model && (
        <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
          Model: {data.model}
        </div>
      )}

      {/* Status badge */}
      {getStatusBadge()}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-purple-500 border-2 border-white dark:border-gray-800"
      />
    </div>
  );
}

