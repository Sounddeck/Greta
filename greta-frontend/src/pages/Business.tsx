import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Building,
  Users,
  DollarSign,
  TrendingUp,
  Calendar,
  Mail,
  FileText,
  BarChart3,
  Settings,
  Plus,
  Search,
  Filter,
  Download,
  Upload,
  CheckCircle,
  AlertCircle,
  Clock,
  Phone,
  Globe,
  Shield,
  Zap,
  Activity
} from 'lucide-react';
import { apiService } from '../services/api';

interface BusinessMetrics {
  total_contacts: number;
  active_deals: number;
  monthly_revenue: number;
  conversion_rate: number;
  google_workspace_connected: boolean;
  recent_activities: Array<{
    id: string;
    type: 'crm' | 'ats' | 'calendar' | 'email';
    description: string;
    timestamp: string;
    status: 'success' | 'pending' | 'error';
  }>;
}

interface GoogleWorkspaceStatus {
  connected: boolean;
  email: string;
  services: {
    gmail: boolean;
    calendar: boolean;
    drive: boolean;
    docs: boolean;
    sheets: boolean;
    meet: boolean;
  };
  last_sync: string;
}

export const Business: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'crm' | 'ats' | 'workspace'>('overview');
  const [showConnectModal, setShowConnectModal] = useState(false);
  const queryClient = useQueryClient();

  const { data: businessMetrics, isLoading } = useQuery<BusinessMetrics>({
    queryKey: ['business-metrics'],
    queryFn: async () => {
      try {
        const response = await apiService.request<BusinessMetrics>('/api/v1/business/dashboard');
        return response.data || getMockBusinessData();
      } catch (error) {
        return getMockBusinessData();
      }
    },
    refetchInterval: 30000
  });

  const { data: workspaceStatus } = useQuery<GoogleWorkspaceStatus>({
    queryKey: ['google-workspace-status'],
    queryFn: async () => {
      try {
        const response = await apiService.request<GoogleWorkspaceStatus>('/api/v1/business/google-workspace/status');
        return response.data || getMockWorkspaceData();
      } catch (error) {
        return getMockWorkspaceData();
      }
    },
    refetchInterval: 60000
  });

  const connectWorkspaceMutation = useMutation({
    mutationFn: async (authCode: string) => {
      const response = await apiService.request('/api/v1/business/google-workspace/connect', {
        method: 'POST',
        body: JSON.stringify({ auth_code: authCode })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['google-workspace-status'] });
      queryClient.invalidateQueries({ queryKey: ['business-metrics'] });
      setShowConnectModal(false);
    }
  });

  const getMockBusinessData = (): BusinessMetrics => ({
    total_contacts: 1247,
    active_deals: 23,
    monthly_revenue: 145000,
    conversion_rate: 18.5,
    google_workspace_connected: false,
    recent_activities: [
      {
        id: '1',
        type: 'crm',
        description: 'New lead added: TechCorp Solutions',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        status: 'success'
      },
      {
        id: '2',
        type: 'ats',
        description: 'Interview scheduled for Senior Developer position',
        timestamp: new Date(Date.now() - 600000).toISOString(),
        status: 'success'
      },
      {
        id: '3',
        type: 'email',
        description: 'Follow-up email sent to 5 prospects',
        timestamp: new Date(Date.now() - 900000).toISOString(),
        status: 'success'
      },
      {
        id: '4',
        type: 'calendar',
        description: 'Team meeting scheduled for tomorrow',
        timestamp: new Date(Date.now() - 1200000).toISOString(),
        status: 'pending'
      }
    ]
  });

  const getMockWorkspaceData = (): GoogleWorkspaceStatus => ({
    connected: false,
    email: '',
    services: {
      gmail: false,
      calendar: false,
      drive: false,
      docs: false,
      sheets: false,
      meet: false
    },
    last_sync: ''
  });

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'crm': return <Users className="w-4 h-4" />;
      case 'ats': return <Building className="w-4 h-4" />;
      case 'email': return <Mail className="w-4 h-4" />;
      case 'calendar': return <Calendar className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-400';
      case 'pending': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="w-4 h-4" />;
      case 'pending': return <Clock className="w-4 h-4" />;
      case 'error': return <AlertCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Building className="w-8 h-8 mr-3 text-blue-400" />
            Business Hub
          </h1>
          <p className="text-gray-400 mt-2">
            CRM, ATS, and Google Workspace integration
          </p>
        </div>
        <button
          onClick={() => setShowConnectModal(true)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>Connect Service</span>
        </button>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Contacts</p>
              <p className="text-2xl font-bold text-white">{businessMetrics?.total_contacts?.toLocaleString() || 0}</p>
            </div>
            <Users className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Deals</p>
              <p className="text-2xl font-bold text-white">{businessMetrics?.active_deals || 0}</p>
            </div>
            <DollarSign className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Monthly Revenue</p>
              <p className="text-2xl font-bold text-white">
                ${(businessMetrics?.monthly_revenue || 0).toLocaleString()}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Conversion Rate</p>
              <p className="text-2xl font-bold text-white">{businessMetrics?.conversion_rate || 0}%</p>
            </div>
            <BarChart3 className="w-8 h-8 text-orange-400" />
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-blue-500/15 rounded-lg p-1">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'crm', label: 'CRM', icon: Users },
          { id: 'ats', label: 'ATS', icon: Building },
          { id: 'workspace', label: 'Google Workspace', icon: Globe }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:text-white hover:bg-blue-500/15'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Activities */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Zap className="w-5 h-5 mr-2 text-blue-400" />
              Recent Activities
            </h3>
            <div className="space-y-4">
              {(businessMetrics?.recent_activities || []).map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3 p-3 bg-blue-500/10 rounded-lg">
                  <div className="text-gray-400 mt-1">
                    {getActivityIcon(activity.type)}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-300">{activity.description}</p>
                    <div className="flex items-center space-x-2 mt-1">
                      <Clock className="w-3 h-3 text-gray-500" />
                      <span className="text-xs text-gray-500">
                        {new Date(activity.timestamp).toLocaleTimeString()}
                      </span>
                      <div className={`flex items-center space-x-1 ${getStatusColor(activity.status)}`}>
                        {getStatusIcon(activity.status)}
                        <span className="text-xs">{activity.status}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Zap className="w-5 h-5 mr-2 text-yellow-400" />
              Quick Actions
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <button className="p-4 bg-blue-600/20 hover:bg-blue-600/30 rounded-lg transition-colors text-left">
                <Users className="w-6 h-6 text-blue-400 mb-2" />
                <p className="text-white font-medium">Add Contact</p>
                <p className="text-gray-400 text-sm">Create new CRM contact</p>
              </button>
              <button className="p-4 bg-green-600/20 hover:bg-green-600/30 rounded-lg transition-colors text-left">
                <DollarSign className="w-6 h-6 text-green-400 mb-2" />
                <p className="text-white font-medium">New Deal</p>
                <p className="text-gray-400 text-sm">Start sales opportunity</p>
              </button>
              <button className="p-4 bg-purple-600/20 hover:bg-purple-600/30 rounded-lg transition-colors text-left">
                <Building className="w-6 h-6 text-purple-400 mb-2" />
                <p className="text-white font-medium">Post Job</p>
                <p className="text-gray-400 text-sm">Create job posting</p>
              </button>
              <button className="p-4 bg-orange-600/20 hover:bg-orange-600/30 rounded-lg transition-colors text-left">
                <Calendar className="w-6 h-6 text-orange-400 mb-2" />
                <p className="text-white font-medium">Schedule Meeting</p>
                <p className="text-gray-400 text-sm">Book calendar event</p>
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'workspace' && (
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <Globe className="w-5 h-5 mr-2 text-blue-400" />
            Google Workspace Integration
          </h3>

          {workspaceStatus?.connected ? (
            <div className="space-y-6">
              <div className="flex items-center justify-between p-4 bg-green-600/20 rounded-lg">
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-6 h-6 text-green-400" />
                  <div>
                    <p className="text-white font-medium">Connected to Google Workspace</p>
                    <p className="text-gray-400 text-sm">{workspaceStatus.email}</p>
                  </div>
                </div>
                <button className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors">
                  Disconnect
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(workspaceStatus.services).map(([service, enabled]) => (
                  <div key={service} className={`p-4 rounded-lg border ${
                    enabled ? 'bg-green-600/20 border-green-600/30' : 'bg-blue-700 hover:bg-blue-800/20 border-gray-600/30'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="text-white capitalize">{service}</span>
                      {enabled ? (
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-gray-400" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Globe className="w-16 h-16 mx-auto mb-4 text-gray-400 opacity-50" />
              <p className="text-gray-400 text-lg mb-2">Google Workspace Not Connected</p>
              <p className="text-gray-500 text-sm mb-6">
                Connect your Google Workspace to sync emails, calendar, and documents
              </p>
              <button
                onClick={() => setShowConnectModal(true)}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Connect Google Workspace
              </button>
            </div>
          )}
        </div>
      )}

      {/* Connect Modal */}
      {showConnectModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-blue-900/40 hover:bg-blue-900/60 p-6 rounded-xl max-w-md w-full mx-4">
            <h3 className="text-xl font-semibold text-white mb-4">Connect Google Workspace</h3>
            <p className="text-gray-400 mb-6">
              You'll be redirected to Google to authorize GRETA to access your workspace.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  // In a real implementation, this would redirect to Google OAuth
                  window.open('https://accounts.google.com/oauth/authorize?client_id=your-client-id', '_blank');
                }}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Authorize
              </button>
              <button
                onClick={() => setShowConnectModal(false)}
                className="px-4 py-2 bg-blue-700 hover:bg-blue-800 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Business;
