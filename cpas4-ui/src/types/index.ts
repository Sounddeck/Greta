// Agent Types
export interface Agent {
  id: string;
  name: string;
  type: AgentType;
  description?: string;
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export type AgentType =
  | 'research'
  | 'coding'
  | 'data_analysis'
  | 'content_creation'
  | 'task_automation'
  | 'customer_support'
  | 'sales'
  | 'marketing'
  | 'finance'
  | 'general';

// Task Types
export interface Task {
  id: string;
  agent_id: string;
  prompt: string;
  status: TaskStatus;
  result?: any;
  error?: string;
  created_at: string;
  completed_at?: string;
  execution_time?: number;
}

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

// Message Types (for chat interface)
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

// Prompt Enhancement Types
export interface PromptEnhancement {
  id: string;
  original_prompt: string;
  enhanced_prompt: string;
  quality_metrics: {
    clarity: number;
    specificity: number;
    structure: number;
  };
  engine: 'dspy' | 'template' | 'custom';
  created_at: string;
}

// Report Types
export interface Report {
  id: string;
  type: ReportType;
  title: string;
  format: 'html' | 'json' | 'pdf' | 'word';
  status: 'pending' | 'generating' | 'completed' | 'failed';
  file_path?: string;
  created_at: string;
  completed_at?: string;
}

export type ReportType =
  | 'agent_performance'
  | 'task_summary'
  | 'knowledge_analytics'
  | 'scraping_summary'
  | 'prompt_enhancement_analysis'
  | 'system_health';

// Knowledge Types
export interface Document {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  collection?: string;
  uploaded_at: string;
  metadata?: Record<string, any>;
}

// Scraping Types
export interface ScrapingJob {
  id: string;
  url: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: string;
  created_at: string;
  completed_at?: string;
}

// UI State Types
export interface UIPreferences {
  theme: 'light' | 'dark';
  backgroundImage?: string;
  backgroundOpacity: number;
  leftSidebarCollapsed: boolean;
  rightSidebarCollapsed: boolean;
}

