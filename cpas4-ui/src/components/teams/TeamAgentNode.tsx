/**
 * Team Agent Node Component
 * 
 * Custom React Flow node for displaying agents in team canvas
 */

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';

interface TeamAgentNodeData {
  agentId: string;
  role: string;
  order: number;
  description?: string;
  status?: 'idle' | 'executing' | 'completed' | 'error';
}

function TeamAgentNode({ data, selected }: NodeProps<TeamAgentNodeData>) {
  const { role, order, description, status = 'idle' } = data;

  // Status colors
  const statusColors = {
    idle: 'bg-gray-100 border-gray-300',
    executing: 'bg-blue-50 border-blue-400 shadow-lg',
    completed: 'bg-green-50 border-green-400',
    error: 'bg-red-50 border-red-400',
  };

  // Status icons
  const statusIcons = {
    idle: '⚪',
    executing: '🔵',
    completed: '✅',
    error: '❌',
  };

  return (
    <div
      className={`
        min-w-[220px] rounded-lg border-2 p-3 transition-all
        ${statusColors[status]}
        ${selected ? 'ring-2 ring-blue-500 shadow-xl' : 'shadow-md'}
      `}
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-blue-500"
      />

      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{getAgentIcon(role)}</span>
          <div>
            <div className="font-semibold text-sm">{role}</div>
            <Badge variant="outline" className="text-xs mt-1">
              Order: {order}
            </Badge>
          </div>
        </div>
        <span className="text-lg">{statusIcons[status]}</span>
      </div>

      {/* Description */}
      {description && (
        <div className="text-xs text-gray-600 mt-2 border-t pt-2">
          {description}
        </div>
      )}

      {/* Status Indicator */}
      {status === 'executing' && (
        <div className="mt-2 flex items-center gap-2 text-xs text-blue-600">
          <div className="animate-spin">⚙️</div>
          <span>Executing...</span>
        </div>
      )}

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-blue-500"
      />
    </div>
  );
}

// Helper function to get agent icon based on role
function getAgentIcon(role: string): string {
  const roleLower = role.toLowerCase();
  
  if (roleLower.includes('research') || roleLower.includes('investigat')) return '🔍';
  if (roleLower.includes('analyst') || roleLower.includes('analyz')) return '📈';
  if (roleLower.includes('writer') || roleLower.includes('content')) return '✍️';
  if (roleLower.includes('review') || roleLower.includes('validator')) return '👁️';
  if (roleLower.includes('manager') || roleLower.includes('coordinator')) return '💼';
  if (roleLower.includes('engineer') || roleLower.includes('developer')) return '🔧';
  if (roleLower.includes('designer') || roleLower.includes('architect')) return '🎨';
  if (roleLower.includes('tester') || roleLower.includes('qa')) return '🧪';
  if (roleLower.includes('devops') || roleLower.includes('deploy')) return '🚀';
  if (roleLower.includes('support') || roleLower.includes('help')) return '🎧';
  if (roleLower.includes('sales') || roleLower.includes('lead')) return '💰';
  if (roleLower.includes('marketing') || roleLower.includes('seo')) return '📢';
  if (roleLower.includes('social')) return '📱';
  if (roleLower.includes('strategist') || roleLower.includes('planner')) return '🎯';
  
  return '🤖'; // Default icon
}

export default memo(TeamAgentNode);

