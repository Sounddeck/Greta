/**
 * Workflow Dashboard Page
 * 
 * Main dashboard for workflow visualization, pattern analysis,
 * and tool execution monitoring.
 */

import React, { useState, useEffect } from 'react';
import { WorkflowVisualization } from '../components/workflow/WorkflowVisualization';
import { ToolExecutionTimeline } from '../components/workflow/ToolExecutionTimeline';
import { getActiveWorkflows, getWorkflowMetrics, executeWorkflow } from '../api/workflowApi';
import type { WorkflowStatus, WorkflowMetrics } from '../types/workflow';

export const WorkflowDashboard: React.FC = () => {
  const [activeWorkflows, setActiveWorkflows] = useState<WorkflowStatus[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<WorkflowMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [taskInput, setTaskInput] = useState('');
  const [executing, setExecuting] = useState(false);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [workflows, metricsData] = await Promise.all([
        getActiveWorkflows(),
        getWorkflowMetrics(),
      ]);
      setActiveWorkflows(workflows);
      setMetrics(metricsData);
      
      // Auto-select first workflow if none selected
      if (!selectedWorkflowId && workflows.length > 0) {
        setSelectedWorkflowId(workflows[0].workflow_id);
      }
    } catch (error) {
      console.error('Failed to load workflow data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteWorkflow = async () => {
    if (!taskInput.trim()) return;

    setExecuting(true);
    try {
      const workflow = await executeWorkflow(taskInput, {}, 'user');
      setSelectedWorkflowId(workflow.workflow_id);
      setTaskInput('');
      await loadData();
    } catch (error) {
      console.error('Failed to execute workflow:', error);
      alert('Failed to execute workflow. Please try again.');
    } finally {
      setExecuting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Workflow Dashboard</h1>
          <p className="text-gray-600">Monitor and visualize workflow executions in real-time</p>
        </div>

        {/* Metrics */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="text-sm text-gray-600 mb-1">Total Workflows</div>
              <div className="text-2xl font-bold text-gray-900">{metrics.total_workflows}</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="text-sm text-gray-600 mb-1">Active</div>
              <div className="text-2xl font-bold text-blue-600">{metrics.active_workflows}</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="text-sm text-gray-600 mb-1">Completed</div>
              <div className="text-2xl font-bold text-green-600">{metrics.completed_workflows}</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="text-sm text-gray-600 mb-1">Failed</div>
              <div className="text-2xl font-bold text-red-600">{metrics.failed_workflows}</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="text-sm text-gray-600 mb-1">Avg Time</div>
              <div className="text-2xl font-bold text-purple-600">{metrics.average_execution_time.toFixed(1)}s</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <div className="text-sm text-gray-600 mb-1">Success Rate</div>
              <div className="text-2xl font-bold text-green-600">{(metrics.success_rate * 100).toFixed(0)}%</div>
            </div>
          </div>
        )}

        {/* Execute Workflow */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Execute New Workflow</h2>
          <div className="flex space-x-4">
            <input
              type="text"
              value={taskInput}
              onChange={(e) => setTaskInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleExecuteWorkflow()}
              placeholder="Describe the task you want to execute..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={executing}
            />
            <button
              onClick={handleExecuteWorkflow}
              disabled={executing || !taskInput.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {executing ? 'Executing...' : 'Execute'}
            </button>
          </div>
        </div>

        {/* Active Workflows List */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Active Workflows ({activeWorkflows.length})</h2>
          {activeWorkflows.length === 0 ? (
            <p className="text-gray-600 text-center py-8">No active workflows</p>
          ) : (
            <div className="space-y-2">
              {activeWorkflows.map((workflow) => (
                <button
                  key={workflow.workflow_id}
                  onClick={() => setSelectedWorkflowId(workflow.workflow_id)}
                  className={`w-full text-left p-4 rounded-lg border transition-colors ${
                    selectedWorkflowId === workflow.workflow_id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="font-medium text-gray-900">{workflow.task_description}</div>
                      <div className="text-sm text-gray-500 mt-1">
                        {workflow.workflow_id} • {new Date(workflow.created_at).toLocaleString()}
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      workflow.status === 'completed' ? 'bg-green-100 text-green-800' :
                      workflow.status === 'failed' ? 'bg-red-100 text-red-800' :
                      workflow.status === 'executing' ? 'bg-blue-100 text-blue-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {workflow.status.toUpperCase()}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Workflow Visualization */}
        {selectedWorkflowId && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <WorkflowVisualization workflowId={selectedWorkflowId} />
            </div>
            <div>
              <ToolExecutionTimeline workflowId={selectedWorkflowId} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WorkflowDashboard;

