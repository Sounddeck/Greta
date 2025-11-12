import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from 'sonner';
import { promptAPI } from '../lib/api';

export interface PromptEnhancement {
  id: string;
  original_prompt: string;
  enhanced_prompt: string;
  quality_metrics: {
    clarity: number;
    specificity: number;
    structure: number;
  };
  engine: 'dspy' | 'template' | 'custom';
  created_at: string;
  feedback?: {
    rating: number;
    helpful: boolean;
    comments?: string;
    improvements?: string;
  };
}

export interface EnhancementSettings {
  engine: 'dspy' | 'template' | 'custom';
  autoEnhance: boolean;
  qualityThresholds: {
    clarity: number;
    specificity: number;
    structure: number;
  };
  advancedParams: {
    temperature: number;
    maxLength: number;
    tone: 'professional' | 'casual' | 'technical' | 'creative' | 'formal';
    domainOptimization: boolean;
  };
}

export interface EnhancementStatistics {
  total_enhancements: number;
  average_quality: {
    clarity: number;
    specificity: number;
    structure: number;
  };
  engine_usage: {
    dspy: number;
    template: number;
    custom: number;
  };
  feedback_summary: {
    average_rating: number;
    helpful_percentage: number;
  };
}

interface PromptEnhancementState {
  enhancements: PromptEnhancement[];
  currentEnhancement: PromptEnhancement | null;
  settings: EnhancementSettings;
  statistics: EnhancementStatistics | null;
  isLoading: boolean;
  error: string | null;
  
  // Search and filter state
  searchQuery: string;
  filterEngine: 'all' | 'dspy' | 'template' | 'custom';
  filterDateRange: { start: Date | null; end: Date | null };
  currentPage: number;
  pageSize: number;
  
  // Actions
  enhancePrompt: (prompt: string, agentType?: string) => Promise<PromptEnhancement | null>;
  enhancePromptAsync: (prompt: string, agentType?: string) => Promise<string | null>;
  getEnhancement: (id: string) => Promise<PromptEnhancement | null>;
  submitFeedback: (id: string, feedback: PromptEnhancement['feedback']) => Promise<void>;
  fetchStatistics: () => Promise<void>;
  fetchEnhancements: () => Promise<void>;
  fetchEnhancements: () => Promise<void>;

  updateSettings: (settings: Partial<EnhancementSettings>) => void;
  setSearchQuery: (query: string) => void;
  setFilterEngine: (engine: 'all' | 'dspy' | 'template' | 'custom') => void;
  setFilterDateRange: (range: { start: Date | null; end: Date | null }) => void;
  setPage: (page: number) => void;
  exportHistory: (format: 'csv' | 'json') => void;
}

const defaultSettings: EnhancementSettings = {
  engine: 'dspy',
  autoEnhance: false,
  qualityThresholds: {
    clarity: 70,
    specificity: 70,
    structure: 70,
  },
  advancedParams: {
    temperature: 0.7,
    maxLength: 500,
    tone: 'professional',
    domainOptimization: true,
  },
};

export const usePromptEnhancementStore = create<PromptEnhancementState>()(
  persist(
    (set, get) => ({
      // Initial state
      enhancements: [],
      currentEnhancement: null,
      settings: defaultSettings,
      statistics: null,
      isLoading: false,
      error: null,
      searchQuery: '',
      filterEngine: 'all',
      filterDateRange: { start: null, end: null },
      currentPage: 1,
      pageSize: 10,

      // Enhance prompt (synchronous)
      enhancePrompt: async (prompt: string, agentType?: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await promptAPI.enhance({
            prompt,
            agent_type: agentType || 'general',
            engine: get().settings.engine,
            ...get().settings.advancedParams,
          });

          const enhancement: PromptEnhancement = response.data;
          
          set((state) => ({
            enhancements: [enhancement, ...state.enhancements],
            currentEnhancement: enhancement,
            isLoading: false,
          }));

          toast.success('Prompt enhanced successfully!');
          return enhancement;
        } catch (error: any) {
          const errorMsg = error.response?.data?.detail || error.message || 'Failed to enhance prompt';
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Enhance prompt (asynchronous)
      enhancePromptAsync: async (prompt: string, agentType?: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await promptAPI.enhanceAsync({
            prompt,
            agent_type: agentType || 'general',
            engine: get().settings.engine,
            ...get().settings.advancedParams,
          });

          const enhancementId = response.data.id;
          set({ isLoading: false });
          toast.success('Enhancement started! Check back soon.');
          return enhancementId;
        } catch (error: any) {
          const errorMsg = error.response?.data?.detail || error.message || 'Failed to start enhancement';
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Get enhancement by ID
      getEnhancement: async (id: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await promptAPI.get(id);
          const enhancement: PromptEnhancement = response.data;
          
          set((state) => {
            const existingIndex = state.enhancements.findIndex((e) => e.id === id);
            const updatedEnhancements = existingIndex >= 0
              ? state.enhancements.map((e, i) => (i === existingIndex ? enhancement : e))
              : [enhancement, ...state.enhancements];
            
            return {
              enhancements: updatedEnhancements,
              currentEnhancement: enhancement,
              isLoading: false,
            };
          });

          return enhancement;
        } catch (error: any) {
          const errorMsg = error.response?.data?.detail || error.message || 'Failed to fetch enhancement';
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Submit feedback
      submitFeedback: async (id: string, feedback: PromptEnhancement['feedback']) => {
        try {
          await promptAPI.feedback(id, feedback);
          
          set((state) => ({
            enhancements: state.enhancements.map((e) =>
              e.id === id ? { ...e, feedback } : e
            ),
          }));

          toast.success('Feedback submitted successfully!');
        } catch (error: any) {
          const errorMsg = error.response?.data?.detail || error.message || 'Failed to submit feedback';
          toast.error(errorMsg);
        }
      },

      // Fetch statistics
      fetchStatistics: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await promptAPI.statistics();
          set({ statistics: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = error.response?.data?.detail || error.message || 'Failed to fetch statistics';
          set({ error: errorMsg, isLoading: false });
          console.error('Failed to fetch statistics:', errorMsg);
        }
      },

      // Fetch all enhancements (mock for now)
      fetchEnhancements: async () => {
        set({ isLoading: true, error: null });
        try {
          // TODO: Implement backend endpoint to fetch all enhancements
          // For now, we're using the persisted state
          set({ isLoading: false });
        } catch (error: any) {
          const errorMsg = error.response?.data?.detail || error.message || 'Failed to fetch enhancements';
          set({ error: errorMsg, isLoading: false });
        }
      },

      // Update settings
      updateSettings: (newSettings: Partial<EnhancementSettings>) => {
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        }));
        toast.success('Settings updated successfully!');
      },

      // Set search query
      setSearchQuery: (query: string) => {
        set({ searchQuery: query, currentPage: 1 });
      },

      // Set filter engine
      setFilterEngine: (engine: 'all' | 'dspy' | 'template' | 'custom') => {
        set({ filterEngine: engine, currentPage: 1 });
      },

      // Set filter date range
      setFilterDateRange: (range: { start: Date | null; end: Date | null }) => {
        set({ filterDateRange: range, currentPage: 1 });
      },

      // Set page
      setPage: (page: number) => {
        set({ currentPage: page });
      },

      // Export history
      exportHistory: (format: 'csv' | 'json') => {
        const { enhancements } = get();
        
        if (format === 'json') {
          const dataStr = JSON.stringify(enhancements, null, 2);
          const dataBlob = new Blob([dataStr], { type: 'application/json' });
          const url = URL.createObjectURL(dataBlob);
          const link = document.createElement('a');
          link.href = url;
          link.download = `prompt-enhancements-${new Date().toISOString()}.json`;
          link.click();
          URL.revokeObjectURL(url);
        } else if (format === 'csv') {
          const headers = ['ID', 'Date', 'Original Prompt', 'Enhanced Prompt', 'Clarity', 'Specificity', 'Structure', 'Engine'];
          const rows = enhancements.map((e) => [
            e.id,
            e.created_at,
            `"${e.original_prompt.replace(/"/g, '""')}"`,
            `"${e.enhanced_prompt.replace(/"/g, '""')}"`,
            e.quality_metrics.clarity,
            e.quality_metrics.specificity,
            e.quality_metrics.structure,
            e.engine,
          ]);
          
          const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
          const dataBlob = new Blob([csvContent], { type: 'text/csv' });
          const url = URL.createObjectURL(dataBlob);
          const link = document.createElement('a');
          link.href = url;
          link.download = `prompt-enhancements-${new Date().toISOString()}.csv`;
          link.click();
          URL.revokeObjectURL(url);
        }
        
        toast.success(`Exported ${enhancements.length} enhancements as ${format.toUpperCase()}`);
      },
    }),
    {
      name: 'cpas5-prompt-enhancement',
      partialize: (state) => ({
        enhancements: state.enhancements,
        settings: state.settings,
      }),
    }
  )
);
