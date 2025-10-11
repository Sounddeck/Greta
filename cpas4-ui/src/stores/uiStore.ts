import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { UIPreferences, AgentType } from '../types';

interface UIStore extends UIPreferences {
  currentAgentType: AgentType | null;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleTheme: () => void;
  setBackgroundImage: (image: string | undefined) => void;
  setBackgroundOpacity: (opacity: number) => void;
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setLeftSidebarCollapsed: (collapsed: boolean) => void;
  setRightSidebarCollapsed: (collapsed: boolean) => void;
  setCurrentAgentType: (agentType: AgentType | null) => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      // Default values
      theme: 'light',
      backgroundImage: undefined,
      backgroundOpacity: 0.3,
      leftSidebarCollapsed: false,
      rightSidebarCollapsed: false,
      currentAgentType: null,

      // Actions
      setTheme: (theme) => {
        set({ theme });
        // Apply theme to document
        if (theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      },

      toggleTheme: () => {
        set((state) => {
          const newTheme = state.theme === 'light' ? 'dark' : 'light';
          // Apply theme to document
          if (newTheme === 'dark') {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
          return { theme: newTheme };
        });
      },

      setBackgroundImage: (image) => set({ backgroundImage: image }),
      setBackgroundOpacity: (opacity) => set({ backgroundOpacity: opacity }),
      toggleLeftSidebar: () => set((state) => ({ leftSidebarCollapsed: !state.leftSidebarCollapsed })),
      toggleRightSidebar: () => set((state) => ({ rightSidebarCollapsed: !state.rightSidebarCollapsed })),
      setLeftSidebarCollapsed: (collapsed) => set({ leftSidebarCollapsed: collapsed }),
      setRightSidebarCollapsed: (collapsed) => set({ rightSidebarCollapsed: collapsed }),
      setCurrentAgentType: (agentType) => set({ currentAgentType: agentType }),
    }),
    {
      name: 'cpas4-ui-preferences',
    }
  )
);

