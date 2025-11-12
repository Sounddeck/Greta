import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from 'sonner';
import { api } from '../lib/api';

// Types
export interface Contact {
  contact_id: string;
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
  title?: string;
  company_id?: string;
  status: 'lead' | 'prospect' | 'customer' | 'inactive' | 'lost';
  lead_score: number;
  tags: string[];
  last_contacted?: string;
  created_at: string;
  updated_at: string;
  custom_fields?: Record<string, any>;
}

export interface Company {
  company_id: string;
  name: string;
  industry?: string;
  size?: string;
  website?: string;
  phone?: string;
  address?: string;
  city?: string;
  state?: string;
  country?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  custom_fields?: Record<string, any>;
}

export interface Deal {
  deal_id: string;
  title: string;
  contact_id: string;
  company_id?: string;
  value: number;
  currency: string;
  stage: 'lead' | 'qualified' | 'proposal' | 'negotiation' | 'closed_won' | 'closed_lost';
  probability: number;
  expected_close_date?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  custom_fields?: Record<string, any>;
}

export interface Activity {
  activity_id: string;
  type: 'call' | 'email' | 'meeting' | 'note' | 'task' | 'demo';
  subject: string;
  description?: string;
  contact_id?: string;
  company_id?: string;
  deal_id?: string;
  scheduled_at?: string;
  completed_at?: string;
  created_at: string;
  created_by: string;
}

export interface CRMStats {
  total_contacts: number;
  total_companies: number;
  total_deals: number;
  total_deal_value: number;
  deals_by_stage: Record<string, number>;
  contacts_by_status: Record<string, number>;
  recent_activities: number;
}

interface CRMStore {
  // State
  contacts: Contact[];
  companies: Company[];
  deals: Deal[];
  activities: Activity[];
  stats: CRMStats | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchContacts: (filters?: any) => Promise<void>;
  fetchCompanies: (filters?: any) => Promise<void>;
  fetchDeals: (filters?: any) => Promise<void>;
  fetchActivities: (filters?: any) => Promise<void>;
  fetchStats: () => Promise<void>;
  
  createContact: (data: any) => Promise<Contact | null>;
  updateContact: (id: string, data: any) => Promise<Contact | null>;
  deleteContact: (id: string) => Promise<boolean>;
  
  createCompany: (data: any) => Promise<Company | null>;
  updateCompany: (id: string, data: any) => Promise<Company | null>;
  deleteCompany: (id: string) => Promise<boolean>;
  
  createDeal: (data: any) => Promise<Deal | null>;
  updateDeal: (id: string, data: any) => Promise<Deal | null>;
  deleteDeal: (id: string) => Promise<boolean>;
  
  createActivity: (data: any) => Promise<Activity | null>;
}

const extractErrorMessage = (error: any, defaultMessage: string): string => {
  if (error?.response?.data?.detail) return error.response.data.detail;
  if (error?.response?.data?.message) return error.response.data.message;
  if (error?.response?.data?.error) return error.response.data.error;
  if (error?.message) return error.message;
  return defaultMessage;
};

export const useCRMStore = create<CRMStore>()(
  persist(
    (set, get) => ({
      // Initial state
      contacts: [],
      companies: [],
      deals: [],
      activities: [],
      stats: null,
      isLoading: false,
      error: null,

      // Fetch contacts
      fetchContacts: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/crm/contacts', { params: filters });
          set({ contacts: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch contacts');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch companies
      fetchCompanies: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/crm/companies', { params: filters });
          set({ companies: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch companies');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch deals
      fetchDeals: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/crm/deals', { params: filters });
          set({ deals: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch deals');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch activities
      fetchActivities: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/crm/activities', { params: filters });
          set({ activities: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch activities');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch stats
      fetchStats: async () => {
        try {
          const response = await api.get('/crm/stats');
          set({ stats: response.data });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch stats');
          toast.error(errorMsg);
        }
      },

      // Create contact
      createContact: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/crm/contacts', data);
          const newContact = response.data;
          set((state) => ({
            contacts: [newContact, ...state.contacts],
            isLoading: false,
          }));
          toast.success('Contact created successfully');
          return newContact;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to create contact');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Update contact
      updateContact: async (id, data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.patch(`/crm/contacts/${id}`, data);
          const updatedContact = response.data;
          set((state) => ({
            contacts: state.contacts.map((c) =>
              c.contact_id === id ? updatedContact : c
            ),
            isLoading: false,
          }));
          toast.success('Contact updated successfully');
          return updatedContact;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to update contact');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Delete contact
      deleteContact: async (id) => {
        set({ isLoading: true, error: null });
        try {
          await api.delete(`/crm/contacts/${id}`);
          set((state) => ({
            contacts: state.contacts.filter((c) => c.contact_id !== id),
            isLoading: false,
          }));
          toast.success('Contact deleted successfully');
          return true;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to delete contact');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return false;
        }
      },

      // Create company
      createCompany: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/crm/companies', data);
          const newCompany = response.data;
          set((state) => ({
            companies: [newCompany, ...state.companies],
            isLoading: false,
          }));
          toast.success('Company created successfully');
          return newCompany;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to create company');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Update company
      updateCompany: async (id, data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.patch(`/crm/companies/${id}`, data);
          const updatedCompany = response.data;
          set((state) => ({
            companies: state.companies.map((c) =>
              c.company_id === id ? updatedCompany : c
            ),
            isLoading: false,
          }));
          toast.success('Company updated successfully');
          return updatedCompany;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to update company');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Delete company
      deleteCompany: async (id) => {
        set({ isLoading: true, error: null });
        try {
          await api.delete(`/crm/companies/${id}`);
          set((state) => ({
            companies: state.companies.filter((c) => c.company_id !== id),
            isLoading: false,
          }));
          toast.success('Company deleted successfully');
          return true;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to delete company');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return false;
        }
      },

      // Create deal
      createDeal: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/crm/deals', data);
          const newDeal = response.data;
          set((state) => ({
            deals: [newDeal, ...state.deals],
            isLoading: false,
          }));
          toast.success('Deal created successfully');
          return newDeal;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to create deal');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Update deal
      updateDeal: async (id, data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.patch(`/crm/deals/${id}`, data);
          const updatedDeal = response.data;
          set((state) => ({
            deals: state.deals.map((d) => (d.deal_id === id ? updatedDeal : d)),
            isLoading: false,
          }));
          toast.success('Deal updated successfully');
          return updatedDeal;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to update deal');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Delete deal
      deleteDeal: async (id) => {
        set({ isLoading: true, error: null });
        try {
          await api.delete(`/crm/deals/${id}`);
          set((state) => ({
            deals: state.deals.filter((d) => d.deal_id !== id),
            isLoading: false,
          }));
          toast.success('Deal deleted successfully');
          return true;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to delete deal');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return false;
        }
      },

      // Create activity
      createActivity: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/crm/activities', data);
          const newActivity = response.data;
          set((state) => ({
            activities: [newActivity, ...state.activities],
            isLoading: false,
          }));
          toast.success('Activity created successfully');
          return newActivity;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to create activity');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // CSV Import
      importContactsCSV: async (file: File) => {
        set({ isLoading: true, error: null });
        try {
          const formData = new FormData();
          formData.append('file', file);

          const response = await api.post('/crm/contacts/import-csv', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });

          const result = response.data;

          if (result.success) {
            toast.success(result.message);
            // Refresh contacts list
            get().fetchContacts();
          } else {
            toast.warning(result.message);
            if (result.errors.length > 0) {
              console.error('Import errors:', result.errors);
            }
          }

          set({ isLoading: false });
          return result;
        } catch (error) {
          const errorMsg = extractErrorMessage(error, 'Failed to import contacts');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      importCompaniesCSV: async (file: File) => {
        set({ isLoading: true, error: null });
        try {
          const formData = new FormData();
          formData.append('file', file);

          const response = await api.post('/crm/companies/import-csv', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });

          const result = response.data;

          if (result.success) {
            toast.success(result.message);
            // Refresh companies list
            get().fetchCompanies();
          } else {
            toast.warning(result.message);
            if (result.errors.length > 0) {
              console.error('Import errors:', result.errors);
            }
          }

          set({ isLoading: false });
          return result;
        } catch (error) {
          const errorMsg = extractErrorMessage(error, 'Failed to import companies');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },
    }),
    {
      name: 'crm-storage',
      partialize: (state) => ({
        contacts: state.contacts,
        companies: state.companies,
        deals: state.deals,
      }),
    }
  )
);

