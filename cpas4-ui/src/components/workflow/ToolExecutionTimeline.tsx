/**
 * Tool Execution Timeline Component
 * 
 * Visual timeline of tool executions within a workflow.
 */

import React, { useState, useEffect } from 'react';
import type { ToolExecutionEvent } from '../../types/workflow';
import { getToolExecutionEvents } from '../../api/workflowApi';

interface ToolExecutionTimelineProps {
  workflowId: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const ToolExecutionTimeline: React.FC<ToolExecutionTimelineProps> = ({
  workflowId,
  autoRefresh = true,
  refreshInterval = 2000,
}) => {
  const [events, setEvents] = useState<ToolExecutionEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const data = await getToolExecutionEvents(workflowId);
        setEvents(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load events');
      } finally {
        setLoading(false);
      }
    };

    fetchEvents();

    if (autoRefresh) {
      const interval = setInterval(fetchEvents, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [workflowId, autoRefresh, refreshInterval]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800 text-sm">{error}</p>
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center">
        <p className="text-gray-600">No tool executions yet</p>
      </div>
    );
  }

  const getEventIcon = (status: string) => {
    switch (status) {
      case 'started':
        return '🔄';
      case 'completed':
        return '✅';
      case 'failed':
        return '❌';
      default:
        return '⚪';
    }
  };

  const getEventColor = (status: string) => {
    switch (status) {
      case 'started':
        return 'border-blue-400 bg-blue-50';
      case 'completed':
        return 'border-green-400 bg-green-50';
      case 'failed':
        return 'border-red-400 bg-red-50';
      default:
        return 'border-gray-400 bg-gray-50';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-6">
        Tool Execution Timeline ({events.length} events)
      </h3>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-300"></div>

        {/* Events */}
        <div className="space-y-6">
          {events.map((event, index) => (
            <div key={event.event_id} className="relative flex items-start space-x-4">
              {/* Timeline dot */}
              <div className={`relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-2 ${getEventColor(event.status)}`}>
                <span className="text-xl">{getEventIcon(event.status)}</span>
              </div>

              {/* Event content */}
              <div className="flex-1 bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900">{event.tool_name}</h4>
                  <span className="text-xs text-gray-500">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>

                <div className="flex items-center space-x-4 text-sm text-gray-600">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    event.status === 'completed' ? 'bg-green-100 text-green-800' :
                    event.status === 'failed' ? 'bg-red-100 text-red-800' :
                    'bg-blue-100 text-blue-800'
                  }`}>
                    {event.status.toUpperCase()}
                  </span>
                  {event.duration !== undefined && (
                    <span className="text-gray-500">
                      Duration: {event.duration.toFixed(2)}s
                    </span>
                  )}
                </div>

                {event.error && (
                  <div className="mt-3 text-sm text-red-600 bg-red-50 p-2 rounded">
                    <span className="font-medium">Error:</span> {event.error}
                  </div>
                )}

                {event.result && event.status === 'completed' && (
                  <div className="mt-3 text-sm text-gray-700 bg-gray-50 p-2 rounded">
                    <span className="font-medium">Result:</span>{' '}
                    {typeof event.result === 'string' 
                      ? event.result.substring(0, 100) + (event.result.length > 100 ? '...' : '')
                      : JSON.stringify(event.result).substring(0, 100) + '...'
                    }
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {events.filter(e => e.status === 'started').length}
            </div>
            <div className="text-sm text-gray-600">Running</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">
              {events.filter(e => e.status === 'completed').length}
            </div>
            <div className="text-sm text-gray-600">Completed</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-600">
              {events.filter(e => e.status === 'failed').length}
            </div>
            <div className="text-sm text-gray-600">Failed</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ToolExecutionTimeline;

