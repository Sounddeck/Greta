/**
 * Team Types for CPAS5 Agent Builder
 * Multi-agent collaboration patterns similar to AutoGen Studio
 */

/**
 * Workflow pattern types supported by teams
 */
export type WorkflowPattern = 
  | 'sequential'        // Agents execute in predefined order
  | 'autonomous'        // LLM decides next speaker (GroupChat)
  | 'round_robin'       // Agents take turns in rotation
  | 'selector';         // Model-based speaker selection

/**
 * Termination condition types
 */
export type TerminationType = 
  | 'max_turns'              // Maximum number of conversation turns
  | 'keyword'                // Specific keyword detected in message
  | 'max_consecutive'        // Max consecutive auto-replies from same agent
  | 'function';              // Custom termination function

/**
 * Team status
 */
export type TeamStatus = 'active' | 'archived';

/**
 * Execution status
 */
export type ExecutionStatus = 'running' | 'completed' | 'failed' | 'terminated';

/**
 * Team agent role assignment
 */
export interface TeamAgent {
  role: string;              // e.g., "researcher", "writer", "reviewer"
  agent_id: string;          // Reference to CustomAgent
  order?: number;            // Execution order (for sequential pattern)
  description?: string;      // Role description
}

/**
 * Termination condition configuration
 */
export interface TerminationCondition {
  type: TerminationType;
  value: string | number;    // Depends on type: number for max_turns, string for keyword
  description?: string;
}

/**
 * Team configuration for multi-agent collaboration
 */
export interface Team {
  team_id: string;
  name: string;
  description: string;
  agents: TeamAgent[];
  workflow_pattern: WorkflowPattern;
  termination: TerminationCondition;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
  created_by?: string;
  status: TeamStatus;
}

/**
 * Team creation payload
 */
export interface TeamCreate {
  name: string;
  description: string;
  agents: TeamAgent[];
  workflow_pattern: WorkflowPattern;
  termination: TerminationCondition;
  metadata?: Record<string, any>;
}

/**
 * Team update payload
 */
export interface TeamUpdate {
  name?: string;
  description?: string;
  agents?: TeamAgent[];
  workflow_pattern?: WorkflowPattern;
  termination?: TerminationCondition;
  metadata?: Record<string, any>;
  status?: TeamStatus;
}

/**
 * Team execution request
 */
export interface TeamExecutionRequest {
  team_id: string;
  task: string;              // The task/prompt for the team
  context?: Record<string, any>;
  max_iterations?: number;   // Override termination max_turns
}

/**
 * Message in team conversation
 */
export interface TeamMessage {
  message_id: string;
  agent_id: string;
  agent_role: string;
  content: string;
  timestamp: string;
  metadata?: {
    tokens?: number;
    model?: string;
    tool_calls?: any[];
  };
}

/**
 * Team execution result
 */
export interface TeamExecutionResult {
  execution_id: string;
  team_id: string;
  status: ExecutionStatus;
  messages: TeamMessage[];
  final_output?: string;
  metadata: {
    total_turns: number;
    total_tokens?: number;
    duration_ms: number;
    termination_reason: string;
  };
  created_at: string;
  completed_at?: string;
}

/**
 * Team validation result
 */
export interface TeamValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Team list response
 */
export interface TeamListResponse {
  teams: Team[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Execution list response
 */
export interface ExecutionListResponse {
  executions: TeamExecutionResult[];
  total: number;
  limit: number;
}

/**
 * Workflow pattern display labels
 */
export const WORKFLOW_PATTERN_LABELS: Record<WorkflowPattern, string> = {
  sequential: 'Sequential Chat',
  autonomous: 'Autonomous Chat',
  round_robin: 'Round Robin',
  selector: 'Selector'
};

/**
 * Workflow pattern descriptions
 */
export const WORKFLOW_PATTERN_DESCRIPTIONS: Record<WorkflowPattern, string> = {
  sequential: 'Agents execute in the order you define',
  autonomous: 'An LLM decides which agent speaks next based on context',
  round_robin: 'Agents take turns speaking in rotation',
  selector: 'A model intelligently selects the next speaker'
};

/**
 * Termination type display labels
 */
export const TERMINATION_TYPE_LABELS: Record<TerminationType, string> = {
  max_turns: 'Max Turns',
  keyword: 'Keyword Detection',
  max_consecutive: 'Max Consecutive Auto-Reply',
  function: 'Custom Function'
};

/**
 * Termination type descriptions
 */
export const TERMINATION_TYPE_DESCRIPTIONS: Record<TerminationType, string> = {
  max_turns: 'Stop after a maximum number of conversation turns',
  keyword: 'Stop when a specific keyword is detected in a message',
  max_consecutive: 'Stop after maximum consecutive auto-replies from the same agent',
  function: 'Use a custom termination function'
};

