/**
 * Business Detail Page
 * Detailed view of a single business with executive team management
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Building2,
  ArrowLeft,
  Users,
  Settings,
  TrendingUp,
  DollarSign,
  AlertCircle,
  CheckCircle,
  Edit,
  Trash2,
  UserPlus,
  Briefcase,
  Target,
  Activity,
} from 'lucide-react';
import {
  getBusiness,
  getBusinessStats,
  updateBusiness,
  deleteBusiness,
  getBusinessStatusColor,
  getExecutiveCount,
} from '../api/businessApi';
import type {
  Business,
  BusinessStats,
  BusinessUpdate,
} from '../api/businessApi';
import {
  OverviewTab,
  ExecutiveTeamTab,
  InfrastructureTab,
  ApprovalWorkflowTab,
} from '../components/business/BusinessDetailTabs';

const BusinessDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [business, setBusiness] = useState<Business | null>(null);
  const [stats, setStats] = useState<BusinessStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'team' | 'infrastructure' | 'approvals'>('overview');

  useEffect(() => {
    if (id) {
      loadBusiness();
    }
  }, [id]);

  const loadBusiness = async () => {
    if (!id) return;
    
    try {
      setLoading(true);
      const [businessData, statsData] = await Promise.all([
        getBusiness(id),
        getBusinessStats(id),
      ]);
      setBusiness(businessData);
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load business:', error);
      alert('Failed to load business details');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!id || !business) return;
    
    if (confirm(`Are you sure you want to archive "${business.name}"?`)) {
      try {
        await deleteBusiness(id);
        navigate('/businesses');
      } catch (error) {
        alert('Failed to archive business');
      }
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading business details...</p>
        </div>
      </div>
    );
  }

  if (!business) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-600 mx-auto mb-4" />
          <p className="text-gray-600">Business not found</p>
          <button
            onClick={() => navigate('/businesses')}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Back to Businesses
          </button>
        </div>
      </div>
    );
  }

  const executiveCount = getExecutiveCount(business.executive_team);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate('/businesses')}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Businesses
        </button>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <Building2 className="w-8 h-8 text-blue-600" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{business.name}</h1>
                <p className="text-gray-600 mt-1">{business.description}</p>
                <div className="flex items-center gap-4 mt-3">
                  <span className="text-sm text-gray-500">
                    <strong>Industry:</strong> {business.industry}
                  </span>
                  <span className={`px-3 py-1 rounded-full text-sm ${getBusinessStatusColor(business.status)}`}>
                    {business.status}
                  </span>
                </div>
                {business.tags.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-3">
                    {business.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <div className="flex gap-2">
              <button className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
                <Edit className="w-5 h-5 text-gray-600" />
              </button>
              <button
                onClick={handleDelete}
                className="p-2 hover:bg-red-50 rounded-lg transition-colors"
              >
                <Trash2 className="w-5 h-5 text-red-600" />
              </button>
            </div>
          </div>

          {/* Stats Cards */}
          {stats && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Total Revenue</p>
                    <p className="text-xl font-bold text-gray-900">
                      ${stats.total_revenue.toLocaleString()}
                    </p>
                  </div>
                  <DollarSign className="w-8 h-8 text-green-600" />
                </div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Active Clients</p>
                    <p className="text-xl font-bold text-gray-900">{stats.active_clients}</p>
                  </div>
                  <Users className="w-8 h-8 text-blue-600" />
                </div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Pending Approvals</p>
                    <p className="text-xl font-bold text-gray-900">{stats.pending_approvals}</p>
                  </div>
                  <AlertCircle className="w-8 h-8 text-orange-600" />
                </div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Executive Team</p>
                    <p className="text-xl font-bold text-gray-900">{executiveCount}</p>
                  </div>
                  <Briefcase className="w-8 h-8 text-purple-600" />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
        <div className="border-b border-gray-200">
          <nav className="flex gap-8 px-6">
            <button
              onClick={() => setActiveTab('overview')}
              className={`py-4 border-b-2 font-medium transition-colors ${
                activeTab === 'overview'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Overview
              </div>
            </button>
            <button
              onClick={() => setActiveTab('team')}
              className={`py-4 border-b-2 font-medium transition-colors ${
                activeTab === 'team'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4" />
                Executive Team
              </div>
            </button>
            <button
              onClick={() => setActiveTab('infrastructure')}
              className={`py-4 border-b-2 font-medium transition-colors ${
                activeTab === 'infrastructure'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Infrastructure
              </div>
            </button>
            <button
              onClick={() => setActiveTab('approvals')}
              className={`py-4 border-b-2 font-medium transition-colors ${
                activeTab === 'approvals'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                Approval Workflow
              </div>
            </button>
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'overview' && <OverviewTab business={business} stats={stats} />}
          {activeTab === 'team' && <ExecutiveTeamTab business={business} />}
          {activeTab === 'infrastructure' && <InfrastructureTab business={business} />}
          {activeTab === 'approvals' && <ApprovalWorkflowTab business={business} />}
        </div>
      </div>
    </div>
  );
};

export default BusinessDetail;

