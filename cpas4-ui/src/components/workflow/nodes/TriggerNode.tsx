/**
 * Trigger Node Component
 * 
 * Represents workflow trigger nodes (start points)
 */

import React from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { Play, Clock, Webhook, Mail } from 'lucide-react';

export interface TriggerNodeData {
  label: string;
  triggerType?: 'manual' | 'scheduled' | 'webhook' | 'email';
  config?: any;
}

export default function TriggerNode({ data, selected }: NodeProps<TriggerNodeData>) {
  const getTriggerIcon = () => {
    switch (data.triggerType) {
      case 'scheduled':
        return <Clock className="h-4 w-4" />;
      case 'webhook':
        return <Webhook className="h-4 w-4" />;
      case 'email':
        return <Mail className="h-4 w-4" />;
      default:
        return <Play className="h-4 w-4" />;
    }
  };

  return (
    <div
      className={`
        px-4 py-3 rounded-lg border-2 shadow-lg min-w-[180px]
        bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20
        border-green-500 dark:border-green-600
        ${selected ? 'ring-2 ring-green-500 ring-offset-2' : ''}
        transition-all duration-200
      `}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-green-500 text-white">
          {getTriggerIcon()}
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-green-700 dark:text-green-400 uppercase tracking-wide">
            Trigger
          </div>
          <div className="text-sm font-semibold text-gray-900 dark:text-white">
            {data.label}
          </div>
        </div>
      </div>

      {/* Trigger type badge */}
      {data.triggerType && (
        <div className="mt-2 inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
          {data.triggerType}
        </div>
      )}

      {/* Output handle (only) - triggers don't have inputs */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-green-500 border-2 border-white dark:border-gray-800"
      />
    </div>
  );
}

