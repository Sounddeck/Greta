import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Message, AgentType } from '../types';

// Per-agent message history
type AgentMessages = Record<string, Message[]>;

interface ChatStore {
  // Message history per agent type
  agentMessages: AgentMessages;
  
  // Get messages for a specific agent
  getMessages: (agentType: AgentType | null) => Message[];
  
  // Add a message to a specific agent's history
  addMessage: (agentType: AgentType | null, message: Message) => void;
  
  // Clear messages for a specific agent
  clearMessages: (agentType: AgentType | null) => void;
  
  // Clear all messages for all agents
  clearAllMessages: () => void;
  
  // Update a message (e.g., for streaming responses)
  updateMessage: (agentType: AgentType | null, messageId: string, updates: Partial<Message>) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      agentMessages: {},

      getMessages: (agentType) => {
        const key = agentType || 'general';
        return get().agentMessages[key] || [];
      },

      addMessage: (agentType, message) => {
        const key = agentType || 'general';
        set((state) => ({
          agentMessages: {
            ...state.agentMessages,
            [key]: [...(state.agentMessages[key] || []), message],
          },
        }));
      },

      clearMessages: (agentType) => {
        const key = agentType || 'general';
        set((state) => ({
          agentMessages: {
            ...state.agentMessages,
            [key]: [],
          },
        }));
      },

      clearAllMessages: () => {
        set({ agentMessages: {} });
      },

      updateMessage: (agentType, messageId, updates) => {
        const key = agentType || 'general';
        set((state) => ({
          agentMessages: {
            ...state.agentMessages,
            [key]: (state.agentMessages[key] || []).map((msg) =>
              msg.id === messageId ? { ...msg, ...updates } : msg
            ),
          },
        }));
      },
    }),
    {
      name: 'cpas4-chat-history',
    }
  )
);

