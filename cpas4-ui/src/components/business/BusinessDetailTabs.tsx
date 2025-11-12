/**
 * Business Detail Tabs
 * Tab components for business detail page
 */

import React from 'react';
import {
  Users,
  Briefcase,
  Shield,
  Zap,
  Mail,
  Calendar,
  Database,
  FolderOpen,
  CheckCircle,
  XCircle,
  AlertCircle,
  TrendingUp,
} from 'lucide-react';
import type { Business, BusinessStats, AgentConfig } from '../../api/businessApi';

// Overview Tab
interface OverviewTabProps {
  business: Business;
  stats: BusinessStats | null;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({ business, stats }) => {
  return (
    <div className="space-y-6">
      {/* Business Information */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Business Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Owner ID</p>
            <p className="text-gray-900 font-medium">{business.owner_id}</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Workspace Path</p>
            <p className="text-gray-900 font-medium">{business.workspace_path}</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Created At</p>
            <p className="text-gray-900 font-medium">
              {new Date(business.created_at).toLocaleDateString()}
            </p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Last Updated</p>
            <p className="text-gray-900 font-medium">
              {new Date(business.updated_at).toLocaleDateString()}
            </p>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      {stats && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-green-700">Total Revenue</p>
                <TrendingUp className="w-5 h-5 text-green-600" />
              </div>
              <p className="text-2xl font-bold text-green-900">
                ${stats.total_revenue.toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-blue-700">Active Clients</p>
                <Users className="w-5 h-5 text-blue-600" />
              </div>
              <p className="text-2xl font-bold text-blue-900">{stats.active_clients}</p>
            </div>
            <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-orange-700">Pending Approvals</p>
                <AlertCircle className="w-5 h-5 text-orange-600" />
              </div>
              <p className="text-2xl font-bold text-orange-900">{stats.pending_approvals}</p>
            </div>
          </div>
        </div>
      )}

      {/* Data Isolation */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Isolation</h3>
        <div className="p-4 bg-gray-50 rounded-lg flex items-center gap-3">
          {business.isolated_data ? (
            <>
              <CheckCircle className="w-6 h-6 text-green-600" />
              <div>
                <p className="font-medium text-gray-900">Data Isolation Enabled</p>
                <p className="text-sm text-gray-600">
                  This business has its own isolated data workspace
                </p>
              </div>
            </>
          ) : (
            <>
              <XCircle className="w-6 h-6 text-red-600" />
              <div>
                <p className="font-medium text-gray-900">Data Isolation Disabled</p>
                <p className="text-sm text-gray-600">
                  This business shares data with other businesses
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// Executive Team Tab
interface ExecutiveTeamTabProps {
  business: Business;
}

export const ExecutiveTeamTab: React.FC<ExecutiveTeamTabProps> = ({ business }) => {
  const executives = [
    { key: 'business_advisor', config: business.executive_team.business_advisor, icon: Shield },
    { key: 'ceo', config: business.executive_team.ceo, icon: Briefcase },
    { key: 'cfo', config: business.executive_team.cfo, icon: TrendingUp },
    { key: 'coo', config: business.executive_team.coo, icon: Zap },
    { key: 'cto', config: business.executive_team.cto, icon: Database },
    { key: 'accountant', config: business.executive_team.accountant, icon: Users },
    { key: 'finance', config: business.executive_team.finance, icon: TrendingUp },
  ].filter((exec) => exec.config !== null);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Executive Team ({executives.length} members)
        </h3>
        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          Add Executive
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {executives.map(({ key, config, icon: Icon }) => (
          <ExecutiveCard key={key} config={config!} icon={Icon} />
        ))}
      </div>

      {executives.length === 0 && (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No executives configured</p>
          <button className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Add First Executive
          </button>
        </div>
      )}
    </div>
  );
};

// Executive Card Component
interface ExecutiveCardProps {
  config: AgentConfig;
  icon: React.ElementType;
}

const ExecutiveCard: React.FC<ExecutiveCardProps> = ({ config, icon: Icon }) => {
  const getAuthorityColor = (authority: string) => {
    switch (authority) {
      case 'advisory_only':
        return 'bg-gray-100 text-gray-700';
      case 'low':
        return 'bg-blue-100 text-blue-700';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700';
      case 'high':
        return 'bg-red-100 text-red-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start gap-3 mb-3">
        <div className="p-2 bg-blue-100 rounded-lg">
          <Icon className="w-5 h-5 text-blue-600" />
        </div>
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900">{config.name}</h4>
          <p className="text-sm text-gray-600">{config.role}</p>
        </div>
        <span className={`px-2 py-1 rounded text-xs ${getAuthorityColor(config.decision_authority)}`}>
          {config.decision_authority.replace('_', ' ')}
        </span>
      </div>

      <p className="text-sm text-gray-600 mb-3">{config.personality}</p>

      <div className="space-y-2">
        <div>
          <p className="text-xs font-medium text-gray-700 mb-1">Capabilities</p>
          <div className="flex flex-wrap gap-1">
            {config.capabilities.slice(0, 3).map((cap) => (
              <span key={cap} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                {cap.replace('_', ' ')}
              </span>
            ))}
            {config.capabilities.length > 3 && (
              <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                +{config.capabilities.length - 3} more
              </span>
            )}
          </div>
        </div>

        {config.requires_approval_for.length > 0 && (
          <div>
            <p className="text-xs font-medium text-gray-700 mb-1">Requires Approval For</p>
            <div className="flex flex-wrap gap-1">
              {config.requires_approval_for.map((item) => (
                <span key={item} className="px-2 py-1 bg-orange-100 text-orange-700 rounded text-xs">
                  {item.replace('_', ' ')}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Infrastructure Tab
interface InfrastructureTabProps {
  business: Business;
}

export const InfrastructureTab: React.FC<InfrastructureTabProps> = ({ business }) => {
  const { infrastructure } = business;

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Infrastructure Configuration</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfrastructureItem
          icon={Mail}
          label="Google Workspace"
          value={infrastructure.google_workspace || 'Not configured'}
          configured={!!infrastructure.google_workspace}
        />
        <InfrastructureItem
          icon={Database}
          label="Google NotebookLM"
          value={infrastructure.google_notebooklm || 'Not configured'}
          configured={!!infrastructure.google_notebooklm}
        />
        <InfrastructureItem
          icon={Users}
          label="CRM ID"
          value={infrastructure.crm_id || 'Not configured'}
          configured={!!infrastructure.crm_id}
        />
        <InfrastructureItem
          icon={Briefcase}
          label="ATS ID"
          value={infrastructure.ats_id || 'Not configured'}
          configured={!!infrastructure.ats_id}
        />
        <InfrastructureItem
          icon={Database}
          label="Database Name"
          value={infrastructure.database_name || 'Not configured'}
          configured={!!infrastructure.database_name}
        />
        <InfrastructureItem
          icon={FolderOpen}
          label="Storage Path"
          value={infrastructure.storage_path || 'Not configured'}
          configured={!!infrastructure.storage_path}
        />
      </div>

      <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
        Configure Infrastructure
      </button>
    </div>
  );
};

interface InfrastructureItemProps {
  icon: React.ElementType;
  label: string;
  value: string;
  configured: boolean;
}

const InfrastructureItem: React.FC<InfrastructureItemProps> = ({ icon: Icon, label, value, configured }) => {
  return (
    <div className="p-4 bg-gray-50 rounded-lg">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-gray-600" />
        <p className="text-sm font-medium text-gray-700">{label}</p>
        {configured ? (
          <CheckCircle className="w-4 h-4 text-green-600 ml-auto" />
        ) : (
          <XCircle className="w-4 h-4 text-gray-400 ml-auto" />
        )}
      </div>
      <p className="text-gray-900">{value}</p>
    </div>
  );
};

// Approval Workflow Tab
interface ApprovalWorkflowTabProps {
  business: Business;
}

export const ApprovalWorkflowTab: React.FC<ApprovalWorkflowTabProps> = ({ business }) => {
  const { approval_workflow } = business;

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Approval Workflow Configuration</h3>

      <div className="p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="font-medium text-gray-900">Autonomous Mode</p>
            <p className="text-sm text-gray-600">
              Allow AI executives to make decisions without human approval
            </p>
          </div>
          <div className={`px-3 py-1 rounded-full text-sm ${
            approval_workflow.autonomous_mode
              ? 'bg-green-100 text-green-700'
              : 'bg-gray-100 text-gray-700'
          }`}>
            {approval_workflow.autonomous_mode ? 'Enabled' : 'Disabled'}
          </div>
        </div>

        {approval_workflow.approval_threshold_amount && (
          <div className="mb-4">
            <p className="text-sm font-medium text-gray-700 mb-1">Approval Threshold</p>
            <p className="text-gray-900">
              ${approval_workflow.approval_threshold_amount.toLocaleString()}
            </p>
          </div>
        )}

        {approval_workflow.approval_required_for.length > 0 && (
          <div className="mb-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Approval Required For</p>
            <div className="flex flex-wrap gap-2">
              {approval_workflow.approval_required_for.map((item) => (
                <span key={item} className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm">
                  {item.replace('_', ' ')}
                </span>
              ))}
            </div>
          </div>
        )}

        {approval_workflow.notification_emails.length > 0 && (
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Notification Emails</p>
            <div className="space-y-1">
              {approval_workflow.notification_emails.map((email) => (
                <div key={email} className="flex items-center gap-2 text-sm text-gray-600">
                  <Mail className="w-4 h-4" />
                  {email}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
        Configure Workflow
      </button>
    </div>
  );
};

