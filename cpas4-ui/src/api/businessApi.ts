/**
 * Business Management API Client
 * Handles all API calls for multi-business management with AI executive teams
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// ============================================================================
// Types
// ============================================================================

export const BusinessStatusValues = {
  ACTIVE: 'active',
  INACTIVE: 'inactive',
  ARCHIVED: 'archived',
} as const;

export type BusinessStatus = typeof BusinessStatusValues[keyof typeof BusinessStatusValues];

export const DecisionAuthorityValues = {
  ADVISORY_ONLY: 'advisory_only',
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
} as const;

export type DecisionAuthority = typeof DecisionAuthorityValues[keyof typeof DecisionAuthorityValues];

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

export interface ExecutiveTeam {
  business_advisor?: AgentConfig;
  ceo?: AgentConfig;
  cfo?: AgentConfig;
  coo?: AgentConfig;
  cto?: AgentConfig;
  accountant?: AgentConfig;
  finance?: AgentConfig;
}

export interface BusinessInfrastructure {
  google_workspace?: {
    domain: string;
    admin_email: string;
    org_unit: string;
  };
  notebooklm?: {
    project_id: string;
    notebook_id: string;
  };
  crm_integration?: {
    enabled: boolean;
    sync_interval: number;
  };
  ats_integration?: {
    enabled: boolean;
    sync_interval: number;
  };
}

export interface ApprovalWorkflow {
  enabled: boolean;
  approval_threshold: number;
  auto_approve_below: number;
  require_approval_for: string[];
}

export interface Business {
  _id?: string;
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

export interface BusinessCreate {
  name: string;
  description: string;
  industry: string;
  tags?: string[];
}

export interface BusinessUpdate {
  name?: string;
  description?: string;
  industry?: string;
  status?: BusinessStatus;
  tags?: string[];
  infrastructure?: BusinessInfrastructure;
  approval_workflow?: ApprovalWorkflow;
}

export interface BusinessStats {
  total_decisions: number;
  pending_approvals: number;
  approved_decisions: number;
  rejected_decisions: number;
  total_revenue: number;
  total_expenses: number;
  active_projects: number;
  team_size: number;
}

export interface ExecutiveDecision {
  _id?: string;
  business_id: string;
  agent_role: string;
  decision_type: string;
  description: string;
  amount?: number;
  requires_approval: boolean;
  approved?: boolean;
  approved_by?: string;
  approved_at?: string;
  created_at: string;
  metadata?: Record<string, any>;
}

export interface DecisionCreate {
  agent_role: string;
  decision_type: string;
  description: string;
  amount?: number;
  metadata?: Record<string, any>;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Create a new business
 */
export const createBusiness = async (data: BusinessCreate): Promise<Business> => {
  const response = await axios.post(`${API_BASE_URL}/businesses`, data);
  return response.data;
};

/**
 * Get a business by ID
 */
export const getBusiness = async (businessId: string): Promise<Business> => {
  const response = await axios.get(`${API_BASE_URL}/businesses/${businessId}`);
  return response.data;
};

/**
 * List all businesses with optional filtering
 */
export const listBusinesses = async (params?: {
  status?: BusinessStatus;
  skip?: number;
  limit?: number;
}): Promise<{ businesses: Business[]; total: number }> => {
  const response = await axios.get(`${API_BASE_URL}/businesses`, { params });
  return response.data;
};

/**
 * Update a business
 */
export const updateBusiness = async (
  businessId: string,
  data: BusinessUpdate
): Promise<Business> => {
  const response = await axios.put(`${API_BASE_URL}/businesses/${businessId}`, data);
  return response.data;
};

/**
 * Delete (archive) a business
 */
export const deleteBusiness = async (businessId: string): Promise<{ message: string }> => {
  const response = await axios.delete(`${API_BASE_URL}/businesses/${businessId}`);
  return response.data;
};

/**
 * Get business statistics
 */
export const getBusinessStats = async (businessId: string): Promise<BusinessStats> => {
  const response = await axios.get(`${API_BASE_URL}/businesses/${businessId}/stats`);
  return response.data;
};

/**
 * Get pending decisions for a business
 */
export const getPendingDecisions = async (businessId: string): Promise<ExecutiveDecision[]> => {
  const response = await axios.get(`${API_BASE_URL}/businesses/${businessId}/decisions/pending`);
  return response.data;
};

/**
 * Create a new executive decision
 */
export const createDecision = async (
  businessId: string,
  data: DecisionCreate
): Promise<ExecutiveDecision> => {
  const response = await axios.post(`${API_BASE_URL}/businesses/${businessId}/decisions`, data);
  return response.data;
};

/**
 * Approve or reject a decision
 */
export const approveDecision = async (
  businessId: string,
  decisionId: string,
  approved: boolean
): Promise<ExecutiveDecision> => {
  const response = await axios.post(
    `${API_BASE_URL}/businesses/${businessId}/decisions/${decisionId}/approve`,
    { approved }
  );
  return response.data;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get executive team member count
 */
export const getExecutiveCount = (team: ExecutiveTeam): number => {
  return Object.values(team).filter((exec) => exec !== undefined).length;
};

/**
 * Get executive team roles
 */
export const getExecutiveRoles = (team: ExecutiveTeam): string[] => {
  return Object.entries(team)
    .filter(([_, exec]) => exec !== undefined)
    .map(([role, _]) => role);
};

/**
 * Format decision amount
 */
export const formatDecisionAmount = (amount?: number): string => {
  if (!amount) return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
};

/**
 * Get decision authority label
 */
export const getDecisionAuthorityLabel = (authority: DecisionAuthority): string => {
  const labels: Record<DecisionAuthority, string> = {
    [DecisionAuthorityValues.ADVISORY_ONLY]: 'Advisory Only',
    [DecisionAuthorityValues.LOW]: 'Low Authority',
    [DecisionAuthorityValues.MEDIUM]: 'Medium Authority',
    [DecisionAuthorityValues.HIGH]: 'High Authority',
  };
  return labels[authority];
};

/**
 * Get business status color
 */
export const getBusinessStatusColor = (status: BusinessStatus): string => {
  const colors: Record<BusinessStatus, string> = {
    [BusinessStatusValues.ACTIVE]: 'text-green-600 bg-green-100',
    [BusinessStatusValues.INACTIVE]: 'text-yellow-600 bg-yellow-100',
    [BusinessStatusValues.ARCHIVED]: 'text-gray-600 bg-gray-100',
  };
  return colors[status];
};

