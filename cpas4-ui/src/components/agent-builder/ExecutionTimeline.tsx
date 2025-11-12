/**
 * Execution Timeline Component
 * Visual timeline showing agent execution order and progress
 */
import React from 'react';
import { CheckCircle, Clock, Loader2, XCircle, Users } from 'lucide-react';
import type { TeamMessage } from '../../types/team';

interface ExecutionTimelineProps {
  messages: TeamMessage[];
  currentAgent: string | null;
  totalAgents: number;
  isExecuting: boolean;
}

export function ExecutionTimeline({ messages, currentAgent, totalAgents, isExecuting }: ExecutionTimelineProps) {
  // Group messages by agent
  const agentTurns = messages.map((msg, index) => ({
    turnNumber: msg.turn_number,
    agentName: msg.agent_name,
    role: msg.role,
    timestamp: msg.timestamp,
    tokensUsed: msg.metadata?.tokens_used || 0,
    status: 'completed' as const
  }));

  // Add current agent if executing
  if (isExecuting && currentAgent) {
    agentTurns.push({
      turnNumber: messages.length + 1,
      agentName: currentAgent,
      role: 'Current',
      timestamp: new Date(),
      tokensUsed: 0,
      status: 'running' as const
    });
  }

  // Add pending agents
  const remainingAgents = Math.max(0, totalAgents - agentTurns.length);
  for (let i = 0; i < remainingAgents; i++) {
    agentTurns.push({
      turnNumber: agentTurns.length + 1,
      agentName: 'Pending',
      role: 'Pending',
      timestamp: new Date(),
      tokensUsed: 0,
      status: 'pending' as const
    });
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'running':
        return <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-400" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'border-green-500/30 bg-green-900/20';
      case 'running':
        return 'border-blue-500/50 bg-blue-900/30 ring-2 ring-blue-500/20';
      case 'failed':
        return 'border-red-500/30 bg-red-900/20';
      default:
        return 'border-gray-700 bg-gray-800/50';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Execution Timeline</h3>
        <div className="text-sm text-gray-400">
          {messages.length} of {totalAgents} agents completed
        </div>
      </div>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-700" />

        {/* Timeline items */}
        <div className="space-y-4">
          {agentTurns.map((turn, index) => (
            <div key={index} className="relative flex items-start gap-4">
              {/* Timeline dot */}
              <div className={`relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-2 ${getStatusColor(turn.status)}`}>
                {getStatusIcon(turn.status)}
              </div>

              {/* Content */}
              <div className={`flex-1 border rounded-lg p-4 ${getStatusColor(turn.status)}`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Users className="h-4 w-4 text-gray-400" />
                      <span className="font-semibold text-white">{turn.agentName}</span>
                      <span className="text-sm text-gray-400">({turn.role})</span>
                    </div>
                    
                    <div className="flex items-center gap-4 text-sm text-gray-400">
                      <span>Turn {turn.turnNumber}</span>
                      {turn.tokensUsed > 0 && (
                        <span>{turn.tokensUsed} tokens</span>
                      )}
                      {turn.status === 'running' && (
                        <span className="text-blue-400 font-medium">In Progress...</span>
                      )}
                    </div>
                  </div>

                  {turn.status === 'completed' && (
                    <div className="text-xs text-gray-500">
                      {new Date(turn.timestamp).toLocaleTimeString()}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      {messages.length > 0 && (
        <div className="mt-6 grid grid-cols-3 gap-4">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">Total Turns</div>
            <div className="text-2xl font-bold text-white">{messages.length}</div>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">Total Tokens</div>
            <div className="text-2xl font-bold text-white">
              {messages.reduce((sum, msg) => sum + (msg.metadata?.tokens_used || 0), 0)}
            </div>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">Progress</div>
            <div className="text-2xl font-bold text-white">
              {Math.round((messages.length / totalAgents) * 100)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

