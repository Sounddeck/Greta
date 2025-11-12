/**
 * Workflow Visualization Component
 * 
 * Real-time visualization of workflow execution with subtasks,
 * tool selections, and pattern matches.
 */

import React, { useState, useEffect } from 'react';
import type { WorkflowStatus } from '../../types/workflow';
import { getWorkflowStatus, getActiveWorkflows } from '../../api/workflowApi';

interface WorkflowVisualizationProps {
  workflowId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const WorkflowVisualization: React.FC<WorkflowVisualizationProps> = ({
  workflowId,
  autoRefresh = true,
  refreshInterval = 2000,
}) => {
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!workflowId) {
      setLoading(false);
      return;
    }

    const fetchWorkflow = async () => {
      try {
        const data = await getWorkflowStatus(workflowId);
        setWorkflow(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load workflow');
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflow();

    if (autoRefresh) {
      const interval = setInterval(fetchWorkflow, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [workflowId, autoRefresh, refreshInterval]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">{error}</p>
      </div>
    );
  }

  if (!workflow) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
        <p className="text-gray-600">No workflow selected</p>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'executing':
      case 'analyzing':
      case 'planning':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'cancelled':
        return 'bg-gray-100 text-gray-800 border-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="space-y-6">
      {/* Workflow Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-gray-900">Workflow Execution</h2>
          <span className={`px-4 py-2 rounded-full text-sm font-medium border ${getStatusColor(workflow.status)}`}>
            {workflow.status.toUpperCase()}
          </span>
        </div>
        <p className="text-gray-700 mb-2">{workflow.task_description}</p>
        <div className="text-sm text-gray-500">
          <span>ID: {workflow.workflow_id}</span>
          <span className="mx-2">•</span>
          <span>Created: {new Date(workflow.created_at).toLocaleString()}</span>
        </div>
      </div>

      {/* Subtasks */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Subtasks ({workflow.subtasks.length})</h3>
        <div className="space-y-3">
          {workflow.subtasks.map((subtask, index) => (
            <div
              key={subtask.subtask_id}
              className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <span className="text-sm font-medium text-gray-500">#{index + 1}</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(subtask.status)}`}>
                    {subtask.status}
                  </span>
                </div>
                {subtask.execution_time && (
                  <span className="text-sm text-gray-500">{subtask.execution_time.toFixed(2)}s</span>
                )}
              </div>
              <p className="text-gray-700 mb-2">{subtask.description}</p>
              {subtask.assigned_tool && (
                <div className="text-sm text-gray-600">
                  <span className="font-medium">Tool:</span> {subtask.assigned_tool}
                </div>
              )}
              {subtask.error && (
                <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
                  {subtask.error}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Selected Tools */}
      {workflow.selected_tools.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Selected Tools ({workflow.selected_tools.length})</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {workflow.selected_tools.map((tool) => (
              <div key={tool.tool_id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{tool.tool_name}</span>
                  <span className="text-sm text-blue-600">{(tool.relevance_score * 100).toFixed(0)}%</span>
                </div>
                <p className="text-sm text-gray-600 mb-2">{tool.reason}</p>
                <span className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                  {tool.category}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Matched Patterns */}
      {workflow.matched_patterns.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Matched Patterns ({workflow.matched_patterns.length})</h3>
          <div className="space-y-3">
            {workflow.matched_patterns.map((pattern) => (
              <div key={pattern.pattern_id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{pattern.pattern_name}</span>
                  <span className="text-sm text-green-600">{(pattern.similarity_score * 100).toFixed(0)}% match</span>
                </div>
                <p className="text-sm text-gray-600 mb-2">{pattern.description}</p>
                <div className="text-sm text-gray-700 bg-gray-50 p-2 rounded">
                  <span className="font-medium">Solution:</span> {pattern.solution_summary}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* RAG Context */}
      {workflow.rag_context && workflow.rag_context.results.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            RAG Context ({workflow.rag_context.total_results} results)
          </h3>
          <div className="space-y-3">
            {workflow.rag_context.results.slice(0, 3).map((result, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Result #{index + 1}</span>
                  <span className="text-sm text-purple-600">Score: {result.score.toFixed(3)}</span>
                </div>
                <p className="text-sm text-gray-600">{result.content.substring(0, 200)}...</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Message */}
      {workflow.error_message && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error</h3>
          <p className="text-red-700">{workflow.error_message}</p>
        </div>
      )}
    </div>
  );
};

export default WorkflowVisualization;

