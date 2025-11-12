/**
 * Business Management Types
 * Type definitions for multi-business management module
 */

export type BusinessStatus = 'active' | 'paused' | 'archived';

export type DecisionAuthority = 'advisory_only' | 'low' | 'medium' | 'high';

export type DecisionStatus = 'pending' | 'approved' | 'rejected' | 'implemented' | 'cancelled';

export interface AgentConfig {
  name: string;
  role: string;
  capabilities: string[];
  tools: string[];
  personality: string;
  decision_authority: DecisionAuthority;
  requires_approval_for: string[];
  agent_id?: string;
}

export interface GoogleWorkspaceConfig {
  domain: string;
  admin_email: string;
  credentials_encrypted?: string;
  accounts: Record<string, string>;
  shared_drives: string[];
  calendars: string[];
  setup_completed: boolean;
}

export interface GoogleNotebookLMConfig {
  notebook_id?: string;
  api_key_encrypted?: string;
  enabled: boolean;
}

export interface BusinessInfrastructure {
  google_workspace?: GoogleWorkspaceConfig;
  google_notebooklm?: GoogleNotebookLMConfig;
  crm_id?: string;
  ats_id?: string;
  database_name?: string;
  storage_path?: string;
}

export interface ExecutiveTeam {
  business_advisor?: AgentConfig;
  ceo?: AgentConfig;
  cfo?: AgentConfig;
  coo?: AgentConfig;
  cto?: AgentConfig;
  accountant?: AgentConfig;
  finance?: AgentConfig;
}

export interface ApprovalWorkflow {
  autonomous_mode: boolean;
  approval_required_for: string[];
  approval_threshold_amount?: number;
  notification_emails: string[];
}

export interface Business {
  _id?: string;
  id?: string;
  name: string;
  description: string;
  industry: string;
  owner_id: string;
  workspace_path: string;
  isolated_data: boolean;
  executive_team: ExecutiveTeam;
  infrastructure: BusinessInfrastructure;
  approval_workflow: ApprovalWorkflow;
  created_at: string;
  updated_at: string;
  status: BusinessStatus;
  tags: string[];
}

export interface BusinessCreateRequest {
  name: string;
  description: string;
  industry: string;
  owner_id?: string;
  tags?: string[];
}

export interface BusinessUpdateRequest {
  name?: string;
  description?: string;
  industry?: string;
  tags?: string[];
  status?: BusinessStatus;
}

