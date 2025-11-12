/**
 * Lead Generation Store
 * 
 * Zustand store for managing lead generation campaigns, leads, and research.
 */

import { create } from 'zustand';
import { api } from '../lib/api';

// ============================================================================
// Types
// ============================================================================

export type LeadStatus = 'new' | 'researching' | 'qualified' | 'contacted' | 'converted' | 'disqualified';
export type ResearchDepth = 'shallow' | 'medium' | 'deep';
export type MonitoringFrequency = 'daily' | 'weekly' | 'monthly';
export type CompanySize = '1-10' | '11-50' | '51-200' | '201-1000' | '1000+';
export type LeadClassification = 'unclassified' | 'candidate' | 'client' | 'both';

export interface LeadCampaign {
  campaign_id: string;
  user_id: string;
  config: {
    campaign_name: string;
    target_industry: string;
    target_location: string;
    company_size: CompanySize;
    keywords: string[];
    research_depth: ResearchDepth;
    monitoring_frequency: MonitoringFrequency;
    max_leads: number;
    auto_research: boolean;
  };
  status: string;
  leads_generated: number;
  leads_qualified: number;
  created_at: string;
  updated_at: string;
}

export interface KeyPerson {
  name: string;
  title: string;
  linkedin?: string;
  email?: string;
  phone?: string;
  bio?: string;
}

export interface Opportunity {
  opportunity_type: string;
  description: string;
  confidence: number;
  source: string;
  detected_at: string;
}

export interface Lead {
  lead_id: string;
  campaign_id: string;
  user_id: string;
  company_name: string;
  industry: string;
  location: string;
  company_size?: string;
  website?: string;
  lead_score: number;
  status: LeadStatus;
  research_history: string[];
  key_people: KeyPerson[];
  recent_news: any[];
  opportunities: Opportunity[];
  monitoring_active: boolean;
  next_research_date?: string;
  classification?: LeadClassification;
  ats_candidate_id?: string;
  crm_contact_id?: string;
  created_at: string;
  updated_at: string;
}

export interface ClassifyLeadResponse {
  lead_id: string;
  classification: LeadClassification;
  ats_candidate_id?: string;
  crm_contact_id?: string;
  message: string;
}

export interface CampaignStats {
  campaign_id: string;
  total_leads: number;
  qualified_leads: number;
  contacted_leads: number;
  converted_leads: number;
  avg_lead_score: number;
  top_opportunities: Opportunity[];
}

export interface LeadScoreData {
  lead_id: string;
  score: number;
  factors: {
    company_size: number;
    industry: number;
    engagement: number;
    budget: number;
    decision_maker: number;
    geographic: number;
  };
  confidence: number;
  recommendations: string[];
  calculated_at: string;
}

// ============================================================================
// Store Interface
// ============================================================================

interface LeadGenerationState {
  // State
  campaigns: LeadCampaign[];
  selectedCampaign: string | null;
  leads: Lead[];
  selectedLead: Lead | null;
  campaignStats: CampaignStats | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchCampaigns: (userId: string) => Promise<void>;
  createCampaign: (userId: string, config: LeadCampaign['config']) => Promise<string>;
  selectCampaign: (campaignId: string) => void;
  fetchCampaignStats: (campaignId: string) => Promise<void>;
  
  fetchLeads: (campaignId?: string, userId?: string, status?: LeadStatus) => Promise<void>;
  createLead: (campaignId: string, userId: string, leadData: any) => Promise<string>;
  selectLead: (leadId: string) => Promise<void>;
  updateLeadStatus: (leadId: string, status: LeadStatus) => Promise<void>;
  
  researchLead: (leadId: string, depth: ResearchDepth) => Promise<void>;
  startMonitoring: (leadId: string, frequency: MonitoringFrequency) => Promise<void>;
  stopMonitoring: (leadId: string) => Promise<void>;

  calculateLeadScore: (leadId: string, engagementData?: { email_opens?: number; website_visits?: number; content_downloads?: number }) => Promise<LeadScoreData>;

  classifyLead: (leadId: string, classification: LeadClassification) => Promise<ClassifyLeadResponse>;
  fetchCandidates: () => Promise<void>;
  fetchClients: () => Promise<void>;

  clearError: () => void;
  reset: () => void;
}

// ============================================================================
// Store Implementation
// ============================================================================

export const useLeadGenerationStore = create<LeadGenerationState>((set, get) => ({
  // Initial State
  campaigns: [],
  selectedCampaign: null,
  leads: [],
  selectedLead: null,
  campaignStats: null,
  loading: false,
  error: null,

  // Fetch Campaigns
  fetchCampaigns: async (userId: string) => {
    set({ loading: true, error: null });
    try {
      const response = await api.get(`/api/v1/leads/campaigns?user_id=${userId}`);
      set({ campaigns: response.data, loading: false });
    } catch (error: any) {
      // Handle 404 gracefully - might be no campaigns yet or backend not running
      if (error.response?.status === 404) {
        console.warn('⚠️ No campaigns found or backend endpoint not available');
        set({
          campaigns: [],
          error: null, // Don't show error for empty state
          loading: false
        });
      } else {
        set({
          error: error.response?.data?.detail || 'Failed to fetch campaigns',
          loading: false,
        });
      }
    }
  },

  // Create Campaign
  createCampaign: async (userId: string, config: LeadCampaign['config']) => {
    set({ loading: true, error: null });
    try {
      const response = await api.post('/api/v1/leads/campaigns', {
        user_id: userId,
        config,
      });
      
      // Refresh campaigns
      await get().fetchCampaigns(userId);
      
      set({ loading: false });
      return response.data.campaign_id;
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to create campaign',
        loading: false,
      });
      throw error;
    }
  },

  // Select Campaign
  selectCampaign: (campaignId: string) => {
    set({ selectedCampaign: campaignId });
    // Fetch leads for this campaign
    get().fetchLeads(campaignId);
    // Fetch campaign stats
    get().fetchCampaignStats(campaignId);
  },

  // Fetch Campaign Stats
  fetchCampaignStats: async (campaignId: string) => {
    try {
      const response = await api.get(`/api/v1/leads/campaigns/${campaignId}/stats`);
      set({ campaignStats: response.data });
    } catch (error: any) {
      // Handle 404 gracefully - campaign might not have stats yet
      if (error.response?.status === 404) {
        console.warn('⚠️ Campaign stats not found - campaign might be new');
        set({
          campaignStats: {
            total_leads: 0,
            qualified_leads: 0,
            contacted_leads: 0,
            converted_leads: 0,
            avg_lead_score: 0,
            total_opportunities: 0,
            total_opportunity_value: 0,
          }
        });
      } else {
        console.error('Failed to fetch campaign stats:', error);
      }
    }
  },

  // Fetch Leads
  fetchLeads: async (campaignId?: string, userId?: string, status?: LeadStatus) => {
    set({ loading: true, error: null });
    try {
      const params = new URLSearchParams();
      if (campaignId) params.append('campaign_id', campaignId);
      if (userId) params.append('user_id', userId);
      if (status) params.append('status', status);

      const response = await api.get(`/api/v1/leads/?${params.toString()}`);
      set({ leads: response.data.leads, loading: false });
    } catch (error: any) {
      // Handle 404 gracefully - might be no leads yet
      if (error.response?.status === 404) {
        console.warn('⚠️ No leads found for this campaign');
        set({
          leads: [],
          error: null, // Don't show error for empty state
          loading: false
        });
      } else {
        set({
          error: error.response?.data?.detail || 'Failed to fetch leads',
          loading: false,
        });
      }
    }
  },

  // Create Lead
  createLead: async (campaignId: string, userId: string, leadData: any) => {
    set({ loading: true, error: null });
    try {
      const params = new URLSearchParams({
        campaign_id: campaignId,
        user_id: userId,
        company_name: leadData.company_name,
        industry: leadData.industry,
        location: leadData.location,
      });
      
      if (leadData.company_size) params.append('company_size', leadData.company_size);
      if (leadData.website) params.append('website', leadData.website);
      
      const response = await api.post(`/api/v1/leads/?${params.toString()}`);
      
      // Refresh leads
      await get().fetchLeads(campaignId);
      
      set({ loading: false });
      return response.data.lead_id;
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to create lead',
        loading: false,
      });
      throw error;
    }
  },

  // Select Lead
  selectLead: async (leadId: string) => {
    set({ loading: true, error: null });
    try {
      const response = await api.get(`/api/v1/leads/${leadId}`);
      set({ selectedLead: response.data, loading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to fetch lead',
        loading: false,
      });
    }
  },

  // Update Lead Status
  updateLeadStatus: async (leadId: string, status: LeadStatus) => {
    set({ loading: true, error: null });
    try {
      await api.patch(`/api/v1/leads/${leadId}/status?status=${status}`);
      
      // Refresh leads
      const { selectedCampaign } = get();
      if (selectedCampaign) {
        await get().fetchLeads(selectedCampaign);
      }
      
      // Refresh selected lead if it's the one being updated
      if (get().selectedLead?.lead_id === leadId) {
        await get().selectLead(leadId);
      }
      
      set({ loading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to update lead status',
        loading: false,
      });
      throw error;
    }
  },

  // Research Lead
  researchLead: async (leadId: string, depth: ResearchDepth) => {
    set({ loading: true, error: null });
    try {
      await api.post(`/api/v1/leads/${leadId}/research`, {
        lead_id: leadId,
        research_depth: depth,
      });
      
      // Refresh lead
      await get().selectLead(leadId);
      
      // Refresh leads list
      const { selectedCampaign } = get();
      if (selectedCampaign) {
        await get().fetchLeads(selectedCampaign);
      }
      
      set({ loading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to research lead',
        loading: false,
      });
      throw error;
    }
  },

  // Start Monitoring
  startMonitoring: async (leadId: string, frequency: MonitoringFrequency) => {
    set({ loading: true, error: null });
    try {
      await api.post(`/api/v1/leads/${leadId}/monitoring/start?frequency=${frequency}`);
      
      // Refresh lead
      await get().selectLead(leadId);
      
      set({ loading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to start monitoring',
        loading: false,
      });
      throw error;
    }
  },

  // Stop Monitoring
  stopMonitoring: async (leadId: string) => {
    set({ loading: true, error: null });
    try {
      await api.post(`/api/v1/leads/${leadId}/monitoring/stop`);

      // Refresh lead
      await get().selectLead(leadId);

      set({ loading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to stop monitoring',
        loading: false,
      });
      throw error;
    }
  },

  // Calculate Lead Score
  calculateLeadScore: async (
    leadId: string,
    engagementData?: { email_opens?: number; website_visits?: number; content_downloads?: number }
  ) => {
    try {
      const params = new URLSearchParams();
      if (engagementData?.email_opens !== undefined) {
        params.append('engagement_opens', engagementData.email_opens.toString());
      }
      if (engagementData?.website_visits !== undefined) {
        params.append('engagement_visits', engagementData.website_visits.toString());
      }
      if (engagementData?.content_downloads !== undefined) {
        params.append('engagement_downloads', engagementData.content_downloads.toString());
      }

      const response = await api.post(`/api/v1/leads/${leadId}/score?${params.toString()}`);

      // Refresh lead to get updated score
      await get().selectLead(leadId);

      return response.data;
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to calculate lead score',
      });
      throw error;
    }
  },

  // Classify Lead
  classifyLead: async (leadId: string, classification: LeadClassification) => {
    set({ loading: true, error: null });
    try {
      const response = await api.post(`/api/v1/leads/${leadId}/classify`, {
        classification,
      });

      // Refresh lead to get updated classification
      await get().selectLead(leadId);

      // Refresh leads list
      const { selectedCampaign } = get();
      if (selectedCampaign) {
        await get().fetchLeads(selectedCampaign);
      }

      set({ loading: false });
      return response.data;
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Failed to classify lead',
        loading: false,
      });
      throw error;
    }
  },

  // Fetch Candidates (from ATS)
  fetchCandidates: async () => {
    set({ loading: true, error: null });
    try {
      const response = await api.get('/api/v1/ats/candidates');
      // Store candidates in a separate state if needed
      set({ loading: false });
    } catch (error: any) {
      if (error.response?.status === 404) {
        console.warn('⚠️ No candidates found');
        set({ loading: false });
      } else {
        set({
          error: error.response?.data?.detail || 'Failed to fetch candidates',
          loading: false,
        });
      }
    }
  },

  // Fetch Clients (from CRM)
  fetchClients: async () => {
    set({ loading: true, error: null });
    try {
      const response = await api.get('/api/v1/crm/contacts');
      // Store clients in a separate state if needed
      set({ loading: false });
    } catch (error: any) {
      if (error.response?.status === 404) {
        console.warn('⚠️ No clients found');
        set({ loading: false });
      } else {
        set({
          error: error.response?.data?.detail || 'Failed to fetch clients',
          loading: false,
        });
      }
    }
  },

  // Clear Error
  clearError: () => set({ error: null }),

  // Reset
  reset: () => set({
    campaigns: [],
    selectedCampaign: null,
    leads: [],
    selectedLead: null,
    campaignStats: null,
    loading: false,
    error: null,
  }),
}));

