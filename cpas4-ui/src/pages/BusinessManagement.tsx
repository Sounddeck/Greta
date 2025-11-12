/**
 * Business Management Dashboard
 * Main page for managing multiple autonomous AI-powered businesses
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Building2,
  Plus,
  Search,
  Filter,
  TrendingUp,
  Users,
  DollarSign,
  AlertCircle,
  CheckCircle,
  XCircle,
  MoreVertical,
} from 'lucide-react';
import {
  listBusinesses,
  getBusinessStats,
  deleteBusiness,
  getBusinessStatusColor,
  getExecutiveCount,
  BusinessStatusValues,
} from '../api/businessApi';
import type {
  Business,
  BusinessStats,
} from '../api/businessApi';
import CreateBusinessModal from '../components/business/CreateBusinessModal';

const BusinessManagement: React.FC = () => {
  const navigate = useNavigate();
  const [businesses, setBusinesses] = useState<Business[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<BusinessStatus | 'all'>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedBusiness, setSelectedBusiness] = useState<Business | null>(null);
  const [businessStats, setBusinessStats] = useState<Record<string, BusinessStats>>({});

  // Load businesses
  useEffect(() => {
    loadBusinesses();
  }, [statusFilter]);

  const loadBusinesses = async () => {
    try {
      setLoading(true);
      const params = statusFilter !== 'all' ? { status: statusFilter } : {};
      const data = await listBusinesses(params);
      setBusinesses(data.businesses);

      // Load stats for each business
      const stats: Record<string, BusinessStats> = {};
      for (const business of data.businesses) {
        if (business._id) {
          try {
            stats[business._id] = await getBusinessStats(business._id);
          } catch (error) {
            console.error(`Failed to load stats for business ${business._id}:`, error);
          }
        }
      }
      setBusinessStats(stats);
    } catch (error) {
      console.error('Failed to load businesses:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteBusiness = async (businessId: string) => {
    if (!confirm('Are you sure you want to archive this business?')) return;

    try {
      await deleteBusiness(businessId);
      loadBusinesses();
    } catch (error) {
      console.error('Failed to delete business:', error);
      alert('Failed to archive business');
    }
  };

  // Filter businesses by search query
  const filteredBusinesses = businesses.filter((business) => {
    const matchesSearch =
      business.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      business.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      business.industry.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  // Calculate summary stats
  const summaryStats = {
    total: businesses.length,
    active: businesses.filter((b) => b.status === BusinessStatusValues.ACTIVE).length,
    inactive: businesses.filter((b) => b.status === BusinessStatusValues.INACTIVE).length,
    totalRevenue: Object.values(businessStats).reduce((sum, stats) => sum + stats.total_revenue, 0),
    pendingApprovals: Object.values(businessStats).reduce(
      (sum, stats) => sum + stats.pending_approvals,
      0
    ),
  };

  return (
    <div className="h-full overflow-auto bg-background">
      <div className="p-6">
        {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
              <Building2 className="w-8 h-8 text-primary" />
              Business Management
            </h1>
            <p className="text-muted-foreground mt-1">
              Manage your autonomous AI-powered businesses and executive teams
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Plus className="w-5 h-5" />
            Create Business
          </button>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="bg-card p-4 rounded-lg shadow-sm border border-border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Businesses</p>
                <p className="text-2xl font-bold text-foreground">{summaryStats.total}</p>
              </div>
              <Building2 className="w-8 h-8 text-primary" />
            </div>
          </div>

          <div className="bg-card p-4 rounded-lg shadow-sm border border-border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active</p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">{summaryStats.active}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
            </div>
          </div>

          <div className="bg-card p-4 rounded-lg shadow-sm border border-border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Inactive</p>
                <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{summaryStats.inactive}</p>
              </div>
              <XCircle className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
            </div>
          </div>

          <div className="bg-card p-4 rounded-lg shadow-sm border border-border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Revenue</p>
                <p className="text-2xl font-bold text-foreground">
                  ${(summaryStats.totalRevenue / 1000).toFixed(1)}K
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-green-600 dark:text-green-400" />
            </div>
          </div>

          <div className="bg-card p-4 rounded-lg shadow-sm border border-border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Pending Approvals</p>
                <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">{summaryStats.pendingApprovals}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-orange-600 dark:text-orange-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-card p-4 rounded-lg shadow-sm border border-border mb-6">
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search businesses..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-background border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-foreground"
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-muted-foreground" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as BusinessStatus | 'all')}
              className="px-4 py-2 bg-background border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-foreground"
            >
              <option value="all">All Status</option>
              <option value={BusinessStatusValues.ACTIVE}>Active</option>
              <option value={BusinessStatusValues.INACTIVE}>Inactive</option>
              <option value={BusinessStatusValues.ARCHIVED}>Archived</option>
            </select>
          </div>
        </div>
      </div>

      {/* Business List */}
      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
          <p className="mt-4 text-muted-foreground">Loading businesses...</p>
        </div>
      ) : filteredBusinesses.length === 0 ? (
        <div className="text-center py-12 bg-card rounded-lg shadow-sm border border-border">
          <Building2 className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-foreground mb-2">No businesses found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery
              ? 'Try adjusting your search criteria'
              : 'Get started by creating your first business'}
          </p>
          {!searchQuery && (
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            >
              Create Business
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredBusinesses.map((business) => (
            <BusinessCard
              key={business._id}
              business={business}
              stats={business._id ? businessStats[business._id] : undefined}
              onDelete={() => business._id && handleDeleteBusiness(business._id)}
              onSelect={() => business._id && navigate(`/businesses/${business._id}`)}
            />
          ))}
        </div>
      )}



      {/* Create Business Modal */}
      <CreateBusinessModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={loadBusinesses}
      />
      </div>
    </div>
  );
};

// Business Card Component
interface BusinessCardProps {
  business: Business;
  stats?: BusinessStats;
  onDelete: () => void;
  onSelect: () => void;
}

const BusinessCard: React.FC<BusinessCardProps> = ({ business, stats, onDelete, onSelect }) => {
  const executiveCount = getExecutiveCount(business.executive_team);

  return (
    <div
      className="bg-card rounded-lg shadow-sm border border-border p-6 hover:shadow-md transition-shadow cursor-pointer"
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-foreground mb-1">{business.name}</h3>
          <p className="text-sm text-muted-foreground">{business.industry}</p>
        </div>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getBusinessStatusColor(business.status)}`}>
          {business.status}
        </span>
      </div>

      <p className="text-sm text-foreground mb-4 line-clamp-2">{business.description}</p>

      {stats && (
        <div className="grid grid-cols-2 gap-4 mb-4 pb-4 border-b border-gray-200">
          <div>
            <p className="text-xs text-gray-600">Revenue</p>
            <p className="text-sm font-semibold text-gray-900">${(stats.total_revenue / 1000).toFixed(1)}K</p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Pending Approvals</p>
            <p className="text-sm font-semibold text-orange-600">{stats.pending_approvals}</p>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-gray-600">
          <Users className="w-4 h-4" />
          <span>{executiveCount} Executives</span>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="text-red-600 hover:text-red-700"
        >
          Archive
        </button>
      </div>
    </div>
  );
};

export default BusinessManagement;

