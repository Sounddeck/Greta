/**
 * Workflow API Client
 * 
 * API client for workflow orchestration, pattern visualization,
 * and tool execution timeline.
 */

import type {
  WorkflowStatus,
  WorkflowMetrics,
  ToolExecutionEvent,
  PatternVisualization,
  WorkflowDebugInfo
} from '../types/workflow';

// Use the same base URL as other API clients for consistency
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace('/api/v1', '') || 'http://localhost:8000';

/**
 * Execute a workflow
 */
export async function executeWorkflow(
  taskDescription: string,
  context: Record<string, any> = {},
  userId: string = 'user'
): Promise<WorkflowStatus> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/execute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      task_description: taskDescription,
      context,
      user_id: userId,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to execute workflow: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get workflow status
 */
export async function getWorkflowStatus(workflowId: string): Promise<WorkflowStatus> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}`);

  if (!response.ok) {
    throw new Error(`Failed to get workflow status: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get workflow metrics
 */
export async function getWorkflowMetrics(): Promise<WorkflowMetrics> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/metrics`);

  if (!response.ok) {
    throw new Error(`Failed to get workflow metrics: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get active workflows
 */
export async function getActiveWorkflows(): Promise<WorkflowStatus[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/active`);

  if (!response.ok) {
    throw new Error(`Failed to get active workflows: ${response.statusText}`);
  }

  const data = await response.json();
  return data.workflows || [];
}

/**
 * Get tool execution events for a workflow
 */
export async function getToolExecutionEvents(workflowId: string): Promise<ToolExecutionEvent[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/events`);

  if (!response.ok) {
    throw new Error(`Failed to get tool execution events: ${response.statusText}`);
  }

  const data = await response.json();
  return data.events || [];
}

/**
 * Get pattern visualizations
 */
export async function getPatternVisualizations(): Promise<PatternVisualization[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/patterns/visualizations`);

  if (!response.ok) {
    throw new Error(`Failed to get pattern visualizations: ${response.statusText}`);
  }

  const data = await response.json();
  return data.patterns || [];
}

/**
 * Get workflow debug info
 */
export async function getWorkflowDebugInfo(workflowId: string): Promise<WorkflowDebugInfo> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/debug`);

  if (!response.ok) {
    throw new Error(`Failed to get workflow debug info: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Cancel a workflow
 */
export async function cancelWorkflow(workflowId: string): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/cancel`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel workflow: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Retry a failed workflow
 */
export async function retryWorkflow(workflowId: string): Promise<WorkflowStatus> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/retry`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Failed to retry workflow: ${response.statusText}`);
  }

  return response.json();
}

