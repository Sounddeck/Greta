/**
 * Hook Store - Zustand State Management
 * Manages hooks, plugins, event types, and hook execution
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { hookApi } from '../lib/api';
import { toast } from 'sonner';

export interface Hook {
  id: string;
  name: string;
  eventType: string;
  pluginId: string;
  config?: Record<string, any>;
  isEnabled: boolean;
  priority: number;
  createdAt: string;
  updatedAt: string;
}

export interface Plugin {
  id: string;
  name: string;
  description?: string;
  version: string;
  isInstalled: boolean;
  isEnabled: boolean;
}

export interface EventType {
  id: string;
  name: string;
  description?: string;
  category: string;
}

export interface HookState {
  // State
  hooks: Hook[];
  plugins: Plugin[];
  eventTypes: EventType[];
  selectedHook: Hook | null;
  isLoading: boolean;
  isExecuting: boolean;
  error: string | null;

  // Actions
  fetchHooks: () => Promise<void>;
  fetchPlugins: () => Promise<void>;
  fetchEventTypes: () => Promise<void>;
  registerHook: (hook: Omit<Hook, 'id' | 'createdAt' | 'updatedAt'>) => Promise<void>;
  updateHook: (id: string, updates: Partial<Hook>) => Promise<void>;
  deleteHook: (id: string) => Promise<void>;
  testHook: (id: string, payload?: Record<string, any>) => Promise<any>;
  setSelectedHook: (hook: Hook | null) => void;
  clearError: () => void;
  reset: () => void;
}

const initialState = {
  hooks: [],
  plugins: [],
  eventTypes: [],
  selectedHook: null,
  isLoading: false,
  isExecuting: false,
  error: null,
};

export const useHookStore = create<HookState>()(
  persist(
    (set, get) => ({
      ...initialState,

      fetchHooks: async () => {
        set({ isLoading: true, error: null });
        try {
          const hooks = await hookApi.getHooks();
          set({ hooks, isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to fetch hooks';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      fetchPlugins: async () => {
        set({ isLoading: true, error: null });
        try {
          const plugins = await hookApi.getPlugins();
          set({ plugins, isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to fetch plugins';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      fetchEventTypes: async () => {
        set({ isLoading: true, error: null });
        try {
          const eventTypes = await hookApi.getEventTypes();
          set({ eventTypes, isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to fetch event types';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      registerHook: async (hookData) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
          const tempId = `temp-${Date.now()}`;
          const optimisticHook: Hook = {
            ...hookData,
            id: tempId,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          };
          set((state) => ({ hooks: [...state.hooks, optimisticHook] }));

          // API call
          const newHook = await hookApi.registerHook(hookData);
          
          // Replace optimistic update with real data
          set((state) => ({
            hooks: state.hooks.map((h) => (h.id === tempId ? newHook : h)),
            isLoading: false,
          }));
          
          toast.success('Hook registered successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set((state) => ({
            hooks: state.hooks.filter((h) => !h.id.startsWith('temp-')),
            error: error.response?.data?.message || 'Failed to register hook',
            isLoading: false,
          }));
          toast.error(error.response?.data?.message || 'Failed to register hook');
        }
      },

      updateHook: async (id, updates) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
          const previousHooks = get().hooks;
          set((state) => ({
            hooks: state.hooks.map((h) =>
              h.id === id ? { ...h, ...updates, updatedAt: new Date().toISOString() } : h
            ),
          }));

          // API call
          const updatedHook = await hookApi.updateHook(id, updates);
          
          // Update with real data
          set((state) => ({
            hooks: state.hooks.map((h) => (h.id === id ? updatedHook : h)),
            isLoading: false,
          }));
          
          toast.success('Hook updated successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set({ hooks: previousHooks, error: error.response?.data?.message || 'Failed to update hook', isLoading: false });
          toast.error(error.response?.data?.message || 'Failed to update hook');
        }
      },

      deleteHook: async (id) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
          const previousHooks = get().hooks;
          set((state) => ({
            hooks: state.hooks.filter((h) => h.id !== id),
          }));

          // API call
          await hookApi.deleteHook(id);
          
          set({ isLoading: false });
          toast.success('Hook deleted successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set({ hooks: previousHooks, error: error.response?.data?.message || 'Failed to delete hook', isLoading: false });
          toast.error(error.response?.data?.message || 'Failed to delete hook');
        }
      },

      testHook: async (id, payload) => {
        set({ isExecuting: true, error: null });
        try {
          const result = await hookApi.testHook(id, payload);
          set({ isExecuting: false });
          toast.success('Hook test completed successfully');
          return result;
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Hook test failed';
          set({ error: errorMessage, isExecuting: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      setSelectedHook: (hook) => {
        set({ selectedHook: hook });
      },

      clearError: () => set({ error: null }),

      reset: () => set(initialState),
    }),
    {
      name: 'hook-storage',
      partialize: (state) => ({
        hooks: state.hooks,
        plugins: state.plugins,
        eventTypes: state.eventTypes,
        selectedHook: state.selectedHook,
      }),
    }
  )
);
