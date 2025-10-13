/**
 * Lead Generation Dashboard
 * 
 * Main dashboard for managing lead generation campaigns and leads.
 */

import React, { useEffect, useState } from 'react';
import {
  Plus,
  Search,
  Filter,
  TrendingUp,
  Users,
  Target,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { useLeadGenerationStore } from '../stores/leadGenerationStore';
import { toast } from 'sonner';
import CampaignStats from '../components/leads/CampaignStats';
import LeadList from '../components/leads/LeadList';
import CreateCampaignModal from '../components/leads/CreateCampaignModal';
import CreateLeadModal from '../components/leads/CreateLeadModal';
import LeadDetailModal from '../components/leads/LeadDetailModal';

export default function LeadGenerationDashboard() {
  const {
    campaigns,
    selectedCampaign,
    leads,
    campaignStats,
    loading,
    error,
    fetchCampaigns,
    selectCampaign,
    clearError,
  } = useLeadGenerationStore();

  const [showCreateCampaign, setShowCreateCampaign] = useState(false);
  const [showCreateLead, setShowCreateLead] = useState(false);
  const [selectedLeadId, setSelectedLeadId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // TODO: Get actual user ID from auth context
  const userId = 'test_user';

  // Fetch campaigns on mount
  useEffect(() => {
    fetchCampaigns(userId);
  }, [fetchCampaigns, userId]);

  // Auto-select first campaign
  useEffect(() => {
    if (campaigns.length > 0 && !selectedCampaign) {
      selectCampaign(campaigns[0].campaign_id);
    }
  }, [campaigns, selectedCampaign, selectCampaign]);

  // Filter leads
  const filteredLeads = leads.filter((lead) => {
    const matchesSearch = lead.company_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      lead.industry.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || lead.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Handle campaign selection
  const handleCampaignSelect = (campaignId: string) => {
    selectCampaign(campaignId);
  };

  // Handle lead click
  const handleLeadClick = (leadId: string) => {
    setSelectedLeadId(leadId);
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              Lead Generation
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Automated lead research and qualification
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setShowCreateLead(true)}
              disabled={!selectedCampaign}
              className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-750 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Add Lead</span>
            </button>

            <button
              onClick={() => setShowCreateCampaign(true)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>New Campaign</span>
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>
            <button
              onClick={clearError}
              className="text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
            >
              ×
            </button>
          </div>
        )}

        {/* Campaign Selector */}
        {campaigns.length > 0 && (
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Active Campaign
            </label>
            <select
              value={selectedCampaign || ''}
              onChange={(e) => handleCampaignSelect(e.target.value)}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {campaigns.map((campaign) => (
                <option key={campaign.campaign_id} value={campaign.campaign_id}>
                  {campaign.config.campaign_name} ({campaign.leads_generated} leads)
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Campaign Stats */}
        {selectedCampaign && campaignStats && (
          <CampaignStats stats={campaignStats} />
        )}

        {/* Search and Filters */}
        {selectedCampaign && (
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <div className="flex gap-4">
              {/* Search */}
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search leads..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Status Filter */}
              <div className="w-48">
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="all">All Status</option>
                  <option value="new">New</option>
                  <option value="researching">Researching</option>
                  <option value="qualified">Qualified</option>
                  <option value="contacted">Contacted</option>
                  <option value="converted">Converted</option>
                  <option value="disqualified">Disqualified</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Lead List */}
        {selectedCampaign ? (
          loading && leads.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : filteredLeads.length === 0 ? (
            <div className="text-center py-12 bg-white dark:bg-gray-800 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700">
              <Target className="w-12 h-12 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                No leads found
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                {searchQuery || statusFilter !== 'all'
                  ? 'Try adjusting your search or filters'
                  : 'Add your first lead to get started'}
              </p>
              {!searchQuery && statusFilter === 'all' && (
                <button
                  onClick={() => setShowCreateLead(true)}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add Your First Lead</span>
                </button>
              )}
            </div>
          ) : (
            <LeadList
              leads={filteredLeads}
              onLeadClick={handleLeadClick}
            />
          )
        ) : (
          <div className="text-center py-12 bg-white dark:bg-gray-800 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700">
            <TrendingUp className="w-12 h-12 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              No campaigns yet
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Create your first lead generation campaign to get started
            </p>
            <button
              onClick={() => setShowCreateCampaign(true)}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Create Your First Campaign</span>
            </button>
          </div>
        )}
      </div>

      {/* Modals */}
      {showCreateCampaign && (
        <CreateCampaignModal
          userId={userId}
          onClose={() => setShowCreateCampaign(false)}
        />
      )}

      {showCreateLead && selectedCampaign && (
        <CreateLeadModal
          campaignId={selectedCampaign}
          userId={userId}
          onClose={() => setShowCreateLead(false)}
        />
      )}

      {selectedLeadId && (
        <LeadDetailModal
          leadId={selectedLeadId}
          onClose={() => setSelectedLeadId(null)}
        />
      )}
    </div>
  );
}

