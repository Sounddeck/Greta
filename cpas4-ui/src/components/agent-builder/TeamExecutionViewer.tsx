/**
 * Team Execution Viewer Component
 * Display real-time team execution progress and conversation flow
 */
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Play, Pause, Square, ArrowLeft, Clock, MessageSquare, Users, CheckCircle, XCircle, AlertCircle, Network } from 'lucide-react';
import { toast } from 'sonner';
import { useTeamStore } from '../../stores/teamStore';
import { ControlTransitionGraph } from './ControlTransitionGraph';
import type { TeamExecutionResult, TeamMessage } from '../../types/team';

const STATUS_COLORS = {
  running: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  completed: 'bg-green-500/20 text-green-300 border-green-500/30',
  failed: 'bg-red-500/20 text-red-300 border-red-500/30',
  pending: 'bg-gray-500/20 text-gray-300 border-gray-500/30'
};

const STATUS_ICONS = {
  running: Play,
  completed: CheckCircle,
  failed: XCircle,
  pending: Clock
};

type ViewMode = 'messages' | 'graph';

export function TeamExecutionViewer() {
  const { teamId } = useParams<{ teamId: string }>();
  const navigate = useNavigate();
  const { currentTeam, executions, executeTeam, fetchTeam, fetchExecutions, isLoading } = useTeamStore();

  const [task, setTask] = useState('');
  const [context, setContext] = useState('{}');
  const [currentExecution, setCurrentExecution] = useState<TeamExecutionResult | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('messages');

  // Fetch team on mount
  useEffect(() => {
    if (teamId) {
      fetchTeam(teamId);
      fetchExecutions(teamId);
    }
  }, [teamId, fetchTeam, fetchExecutions]);

  useEffect(() => {
    // Find the latest execution for this team
    if (executions.length > 0) {
      const latestExecution = executions.find(e => e.team_id === teamId);
      if (latestExecution) {
        setCurrentExecution(latestExecution);
      }
    }
  }, [executions, teamId]);

  const handleExecute = async () => {
    if (!teamId || !task.trim()) {
      toast.error('Please enter a task');
      return;
    }

    setIsExecuting(true);
    try {
      let contextObj = {};
      if (context.trim()) {
        try {
          contextObj = JSON.parse(context);
        } catch (e) {
          toast.error('Invalid JSON in context field');
          setIsExecuting(false);
          return;
        }
      }

      const result = await executeTeam(teamId, task, contextObj);
      setCurrentExecution(result);
      toast.success('Team execution started!');
    } catch (error) {
      // Error already handled in store
    } finally {
      setIsExecuting(false);
    }
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getStatusIcon = (status: string) => {
    const Icon = STATUS_ICONS[status as keyof typeof STATUS_ICONS] || Clock;
    return <Icon className="h-4 w-4" />;
  };

  const getStatusColor = (status: string) => {
    return STATUS_COLORS[status as keyof typeof STATUS_COLORS] || STATUS_COLORS.pending;
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/teams')}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <ArrowLeft className="h-5 w-5 text-gray-400" />
          </button>
          <div>
            <h2 className="text-2xl font-bold text-white">Team Execution</h2>
            <p className="text-gray-400 mt-1">
              {currentTeam?.name || 'Loading...'}
            </p>
          </div>
        </div>
      </div>

      {/* Execution Form */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Execute Team</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Task *
            </label>
            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="Enter the task for the team to accomplish..."
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
              rows={3}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Context (JSON, optional)
            </label>
            <textarea
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder='{"key": "value"}'
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 font-mono text-sm"
              rows={3}
            />
          </div>

          <button
            onClick={handleExecute}
            disabled={isExecuting || !task.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
          >
            <Play className="h-4 w-4" />
            {isExecuting ? 'Executing...' : 'Execute Team'}
          </button>
        </div>
      </div>

      {/* Execution Results */}
      {currentExecution && (
        <div className="space-y-4">
          {/* Execution Status */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Execution Status</h3>
              <span className={`px-3 py-1 rounded-lg text-sm font-medium border flex items-center gap-2 ${getStatusColor(currentExecution.status)}`}>
                {getStatusIcon(currentExecution.status)}
                {currentExecution.status.toUpperCase()}
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-400">Total Turns</p>
                <p className="text-2xl font-bold text-white">{currentExecution.metadata.total_turns}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Total Tokens</p>
                <p className="text-2xl font-bold text-white">{currentExecution.metadata.total_tokens}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Duration</p>
                <p className="text-2xl font-bold text-white">
                  {formatDuration(currentExecution.metadata.duration_ms)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Termination</p>
                <p className="text-sm font-medium text-white">
                  {currentExecution.metadata.termination_reason}
                </p>
              </div>
            </div>
          </div>

          {/* View Tabs */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
            {/* Tab Headers */}
            <div className="flex border-b border-gray-700">
              <button
                onClick={() => setViewMode('messages')}
                className={`flex items-center gap-2 px-6 py-3 font-medium transition-colors ${
                  viewMode === 'messages'
                    ? 'bg-gray-900 text-white border-b-2 border-blue-500'
                    : 'text-gray-400 hover:text-white hover:bg-gray-750'
                }`}
              >
                <MessageSquare className="h-4 w-4" />
                Messages
              </button>
              <button
                onClick={() => setViewMode('graph')}
                className={`flex items-center gap-2 px-6 py-3 font-medium transition-colors ${
                  viewMode === 'graph'
                    ? 'bg-gray-900 text-white border-b-2 border-blue-500'
                    : 'text-gray-400 hover:text-white hover:bg-gray-750'
                }`}
              >
                <Network className="h-4 w-4" />
                Flow Graph
              </button>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              {viewMode === 'messages' ? (
                // Messages View
                currentExecution.messages.length === 0 ? (
                  <div className="text-center py-8 text-gray-400">
                    <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No messages yet</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {currentExecution.messages.map((message, index) => (
                      <div
                        key={index}
                        className="bg-gray-900 border border-gray-700 rounded-lg p-4"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Users className="h-4 w-4 text-blue-400" />
                            <span className="font-semibold text-white">{message.role}</span>
                            <span className="text-sm text-gray-400">({message.agent_name})</span>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-400">
                            <span>Turn {message.turn_number}</span>
                            <span>{message.metadata?.tokens_used || 0} tokens</span>
                          </div>
                        </div>
                        <div className="text-gray-300 whitespace-pre-wrap">
                          {message.content}
                        </div>
                      </div>
                    ))}
                  </div>
                )
              ) : (
                // Graph View
                currentTeam && currentExecution.messages.length > 0 ? (
                  <div className="h-[600px]">
                    <ControlTransitionGraph
                      team={currentTeam}
                      messages={currentExecution.messages}
                      currentTurn={currentExecution.metadata.total_turns}
                    />
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-400">
                    <Network className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No execution data to visualize</p>
                  </div>
                )
              )}
            </div>
          </div>

          {/* Final Output */}
          {currentExecution.final_output && (
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Final Output</h3>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-gray-300 whitespace-pre-wrap">
                {currentExecution.final_output}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

