/**
 * Workflow Visualization Types
 * 
 * Type definitions for workflow orchestration, pattern visualization,
 * and tool execution timeline components.
 */

export interface WorkflowStatus {
  workflow_id: string;
  status: 'pending' | 'analyzing' | 'planning' | 'executing' | 'completed' | 'failed' | 'cancelled';
  task_description: string;
  context: Record<string, any>;
  user_id: string;
  created_at: string;
  updated_at: string;
  subtasks: Subtask[];
  selected_tools: ToolSelection[];
  matched_patterns: PatternMatch[];
  rag_context: RAGContext | null;
  execution_results: ExecutionResult[];
  error_message?: string;
}

export interface Subtask {
  subtask_id: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  assigned_tool?: string;
  result?: any;
  error?: string;
  started_at?: string;
  completed_at?: string;
  execution_time?: number;
}

export interface ToolSelection {
  tool_id: string;
  tool_name: string;
  relevance_score: number;
  reason: string;
  category: string;
}

export interface PatternMatch {
  pattern_id: string;
  pattern_name: string;
  similarity_score: number;
  pattern_type: string;
  description: string;
  solution_summary: string;
}

export interface RAGContext {
  query: string;
  results: RAGResult[];
  total_results: number;
}

export interface RAGResult {
  content: string;
  metadata: Record<string, any>;
  score: number;
}

export interface ExecutionResult {
  subtask_id: string;
  tool_id: string;
  success: boolean;
  result: any;
  execution_time: number;
  timestamp: string;
  error?: string;
}

export interface WorkflowMetrics {
  total_workflows: number;
  active_workflows: number;
  completed_workflows: number;
  failed_workflows: number;
  average_execution_time: number;
  success_rate: number;
}

export interface ToolExecutionEvent {
  event_id: string;
  workflow_id: string;
  tool_id: string;
  tool_name: string;
  status: 'started' | 'completed' | 'failed';
  timestamp: string;
  duration?: number;
  result?: any;
  error?: string;
}

export interface PatternVisualization {
  pattern_id: string;
  pattern_name: string;
  pattern_type: string;
  usage_count: number;
  success_rate: number;
  average_execution_time: number;
  last_used: string;
  steps: PatternStep[];
}

export interface PatternStep {
  step_number: number;
  description: string;
  tool_id?: string;
  expected_output: string;
}

export interface WorkflowDebugInfo {
  workflow_id: string;
  breakpoints: Breakpoint[];
  variables: Record<string, any>;
  call_stack: CallStackFrame[];
  logs: LogEntry[];
}

export interface Breakpoint {
  id: string;
  subtask_id: string;
  enabled: boolean;
  condition?: string;
}

export interface CallStackFrame {
  frame_id: string;
  subtask_id: string;
  tool_id: string;
  timestamp: string;
}

export interface LogEntry {
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  context?: Record<string, any>;
}

