import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { promptTemplateAPI } from '../lib/api';

export interface PromptTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  template: string;
  variables: string[];
  tags: string[];
  version: string;
  is_active: boolean;
  is_system: boolean;
  usage_count: number;
  rating: number;
  created_by: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
}

interface PromptTemplateStore {
  templates: PromptTemplate[];
  selectedTemplate: PromptTemplate | null;
  loading: boolean;
  error: string | null;
  searchQuery: string;
  selectedCategory: string | null;
  currentPage: number;
  pageSize: number;
  totalTemplates: number;

  // Actions
  fetchTemplates: (filters?: any) => Promise<void>;
  searchTemplates: (query: string, category?: string) => Promise<void>;
  selectTemplate: (template: PromptTemplate | null) => void;
  renderTemplate: (templateId: string, variables: Record<string, string>) => Promise<string>;
  rateTemplate: (templateId: string, rating: number, feedback?: string) => Promise<void>;
  createTemplate: (data: any) => Promise<void>;
  updateTemplate: (id: string, data: any) => Promise<void>;
  deleteTemplate: (id: string) => Promise<void>;
  setSearchQuery: (query: string) => void;
  setSelectedCategory: (category: string | null) => void;
  setPage: (page: number) => void;
  clearError: () => void;
  initializeTemplates: () => Promise<void>;
}

export const usePromptTemplateStore = create<PromptTemplateStore>()(
  persist(
    (set, get) => ({
      templates: [],
      selectedTemplate: null,
      loading: false,
      error: null,
      searchQuery: '',
      selectedCategory: null,
      currentPage: 1,
      pageSize: 20,
      totalTemplates: 0,

      fetchTemplates: async (filters = {}) => {
        set({ loading: true, error: null });
        try {
          const params = {
            page: get().currentPage,
            page_size: get().pageSize,
            category: get().selectedCategory,
            ...filters,
          };

          const response = await promptTemplateAPI.list(params);
          
          if (response.data.success) {
            set({
              templates: response.data.templates,
              totalTemplates: response.data.total,
              loading: false,
            });
          } else {
            set({ error: response.data.message, loading: false });
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to fetch templates', loading: false });
        }
      },

      searchTemplates: async (query: string, category?: string) => {
        set({ loading: true, error: null, searchQuery: query });
        try {
          const response = await promptTemplateAPI.search(query, category || get().selectedCategory || undefined);
          
          if (response.data.success) {
            set({
              templates: response.data.templates,
              totalTemplates: response.data.total,
              loading: false,
            });
          } else {
            set({ error: response.data.message, loading: false });
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to search templates', loading: false });
        }
      },

      selectTemplate: (template: PromptTemplate | null) => {
        set({ selectedTemplate: template });
      },

      renderTemplate: async (templateId: string, variables: Record<string, string>): Promise<string> => {
        set({ loading: true, error: null });
        try {
          const response = await promptTemplateAPI.render(templateId, variables);
          
          if (response.data.success) {
            set({ loading: false });
            return response.data.rendered_prompt;
          } else {
            set({ error: response.data.message, loading: false });
            throw new Error(response.data.message);
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to render template', loading: false });
          throw error;
        }
      },

      rateTemplate: async (templateId: string, rating: number, feedback?: string) => {
        set({ loading: true, error: null });
        try {
          const response = await promptTemplateAPI.rate(templateId, rating, feedback);
          
          if (response.data.success) {
            // Update the template in the list
            const templates = get().templates.map(t =>
              t.id === templateId ? { ...t, rating: response.data.new_rating } : t
            );
            set({ templates, loading: false });
          } else {
            set({ error: response.data.message, loading: false });
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to rate template', loading: false });
        }
      },

      createTemplate: async (data: any) => {
        set({ loading: true, error: null });
        try {
          const response = await promptTemplateAPI.create(data);
          
          if (response.data.success) {
            // Refresh templates
            await get().fetchTemplates();
            set({ loading: false });
          } else {
            set({ error: response.data.message, loading: false });
            throw new Error(response.data.message);
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to create template', loading: false });
          throw error;
        }
      },

      updateTemplate: async (id: string, data: any) => {
        set({ loading: true, error: null });
        try {
          const response = await promptTemplateAPI.update(id, data);
          
          if (response.data.success) {
            // Update the template in the list
            const templates = get().templates.map(t =>
              t.id === id ? response.data.template : t
            );
            set({ templates, loading: false });
          } else {
            set({ error: response.data.message, loading: false });
            throw new Error(response.data.message);
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to update template', loading: false });
          throw error;
        }
      },

      deleteTemplate: async (id: string) => {
        set({ loading: true, error: null });
        try {
          const response = await promptTemplateAPI.delete(id);
          
          if (response.data.success) {
            // Remove the template from the list
            const templates = get().templates.filter(t => t.id !== id);
            set({ templates, totalTemplates: get().totalTemplates - 1, loading: false });
          } else {
            set({ error: response.data.message, loading: false });
            throw new Error(response.data.message);
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to delete template', loading: false });
          throw error;
        }
      },

      setSearchQuery: (query: string) => {
        set({ searchQuery: query });
      },

      setSelectedCategory: (category: string | null) => {
        set({ selectedCategory: category, currentPage: 1 });
      },

      setPage: (page: number) => {
        set({ currentPage: page });
      },

      clearError: () => {
        set({ error: null });
      },

      initializeTemplates: async () => {
        set({ loading: true, error: null });
        try {
          const response = await promptTemplateAPI.initialize();
          
          if (response.data.success) {
            // Fetch templates after initialization
            await get().fetchTemplates();
            set({ loading: false });
          } else {
            set({ error: response.data.message, loading: false });
          }
        } catch (error: any) {
          set({ error: error.message || 'Failed to initialize templates', loading: false });
        }
      },
    }),
    {
      name: 'prompt-template-storage',
      partialize: (state) => ({
        selectedCategory: state.selectedCategory,
        pageSize: state.pageSize,
      }),
    }
  )
);

