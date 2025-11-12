/**
 * Condition Node Component
 * 
 * Represents conditional logic nodes (if/else branching)
 */

import React from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { GitBranch } from 'lucide-react';

export interface ConditionNodeData {
  label: string;
  condition?: string;
  operator?: 'equals' | 'contains' | 'greater' | 'less' | 'exists';
  config?: any;
}

export default function ConditionNode({ data, selected }: NodeProps<ConditionNodeData>) {
  return (
    <div
      className={`
        px-4 py-3 rounded-lg border-2 shadow-lg min-w-[180px]
        bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20
        border-amber-500 dark:border-amber-600
        ${selected ? 'ring-2 ring-amber-500 ring-offset-2' : ''}
        transition-all duration-200
      `}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-amber-500 border-2 border-white dark:border-gray-800"
      />

      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-amber-500 text-white">
          <GitBranch className="h-4 w-4" />
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-amber-700 dark:text-amber-400 uppercase tracking-wide">
            Condition
          </div>
          <div className="text-sm font-semibold text-gray-900 dark:text-white">
            {data.label}
          </div>
        </div>
      </div>

      {/* Condition info */}
      {data.condition && (
        <div className="mt-2 text-xs text-gray-600 dark:text-gray-400 font-mono bg-white dark:bg-gray-800 px-2 py-1 rounded">
          {data.condition}
        </div>
      )}

      {/* Operator badge */}
      {data.operator && (
        <div className="mt-2 inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400">
          {data.operator}
        </div>
      )}

      {/* Output handles - True (left) and False (right) */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="true"
        className="w-3 h-3 bg-green-500 border-2 border-white dark:border-gray-800"
        style={{ left: '30%' }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        className="w-3 h-3 bg-red-500 border-2 border-white dark:border-gray-800"
        style={{ left: '70%' }}
      />

      {/* Labels for true/false outputs */}
      <div className="flex justify-between mt-6 text-xs font-medium">
        <span className="text-green-600 dark:text-green-400">True</span>
        <span className="text-red-600 dark:text-red-400">False</span>
      </div>
    </div>
  );
}

