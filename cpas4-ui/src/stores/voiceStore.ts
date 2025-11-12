/**
 * Voice Store - Zustand State Management
 * Manages voice profiles, languages, voices, and STT/TTS operations
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { voiceApi } from '../lib/api';
import { toast } from 'sonner';

export interface VoiceProfile {
  id: string;
  name: string;
  language: string;
  voice: string;
  speed: number;
  pitch: number;
  volume: number;
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface WakeWordStats {
  total_detections: number;
  accuracy: number;
  last_detection: string | null;
  uptime_seconds: number;
}

export interface VoiceState {
  // State
  profiles: VoiceProfile[];
  languages: string[];
  voices: string[];
  currentProfile: VoiceProfile | null;
  isLoading: boolean;
  isProcessing: boolean;
  error: string | null;

  // Wake word detection state
  wakeWordEnabled: boolean;
  wakeWordKeywords: string[];
  wakeWordSensitivities: number[];
  isListeningForWakeWord: boolean;
  wakeWordStats: WakeWordStats | null;

  // Actions
  fetchProfiles: () => Promise<void>;
  fetchLanguages: () => Promise<void>;
  fetchVoices: (language?: string) => Promise<void>;
  createProfile: (profile: Omit<VoiceProfile, 'id' | 'createdAt' | 'updatedAt'>) => Promise<void>;
  updateProfile: (id: string, updates: Partial<VoiceProfile>) => Promise<void>;
  deleteProfile: (id: string) => Promise<void>;
  setCurrentProfile: (profile: VoiceProfile | null) => void;
  speechToText: (audioBlob: Blob, language?: string) => Promise<string>;
  textToSpeech: (text: string, voice?: string) => Promise<Blob>;

  // Wake word actions
  startWakeWordDetection: () => Promise<void>;
  stopWakeWordDetection: () => Promise<void>;
  getWakeWordStatus: () => Promise<void>;
  updateWakeWordSettings: (keywords: string[], sensitivities: number[]) => void;

  clearError: () => void;
  reset: () => void;
}

const initialState = {
  profiles: [],
  languages: [],
  voices: [],
  currentProfile: null,
  isLoading: false,
  isProcessing: false,
  error: null,

  // Wake word detection initial state
  wakeWordEnabled: false,
  wakeWordKeywords: ['greta'],
  wakeWordSensitivities: [0.5],
  isListeningForWakeWord: false,
  wakeWordStats: null,
};

export const useVoiceStore = create<VoiceState>()(
  persist(
    (set, get) => ({
      ...initialState,

      fetchProfiles: async () => {
        set({ isLoading: true, error: null });
        try {
          const profiles = await voiceApi.getVoiceProfiles();
          set({ profiles, isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to fetch voice profiles';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      fetchLanguages: async () => {
        set({ isLoading: true, error: null });
        try {
          const languages = await voiceApi.getSupportedLanguages();
          set({ languages, isLoading: false });
        } catch (error: any) {
          // Fallback to default languages if API fails
          const fallbackLanguages = [
            { code: 'en', name: 'English (US)' },
            { code: 'es', name: 'Spanish' },
            { code: 'fr', name: 'French' },
            { code: 'de', name: 'German' },
            { code: 'it', name: 'Italian' },
            { code: 'pt', name: 'Portuguese' },
            { code: 'ja', name: 'Japanese' },
            { code: 'zh', name: 'Chinese' },
          ];
          set({ languages: fallbackLanguages as any, isLoading: false });
          console.warn('Using fallback languages:', error.message);
        }
      },

      fetchVoices: async (language?: string) => {
        set({ isLoading: true, error: null });
        try {
          const voices = await voiceApi.getSupportedVoices(language);
          set({ voices, isLoading: false });
        } catch (error: any) {
          // Fallback to default voices if API fails
          const fallbackVoices = [
            { id: 'default', name: 'Default Voice' },
            { id: 'male-1', name: 'Male Voice 1' },
            { id: 'female-1', name: 'Female Voice 1' },
            { id: 'male-2', name: 'Male Voice 2' },
            { id: 'female-2', name: 'Female Voice 2' },
          ];
          set({ voices: fallbackVoices as any, isLoading: false });
          console.warn('Using fallback voices:', error.message);
        }
      },

      createProfile: async (profileData) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
          const tempId = `temp-${Date.now()}`;
          const optimisticProfile: VoiceProfile = {
            ...profileData,
            id: tempId,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          };
          set((state) => ({ profiles: [...state.profiles, optimisticProfile] }));

          // API call
          const newProfile = await voiceApi.createVoiceProfile(profileData);
          
          // Replace optimistic update with real data
          set((state) => ({
            profiles: state.profiles.map((p) => (p.id === tempId ? newProfile : p)),
            isLoading: false,
          }));
          
          toast.success('Voice profile created successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set((state) => ({
            profiles: state.profiles.filter((p) => !p.id.startsWith('temp-')),
            error: error.response?.data?.message || 'Failed to create profile',
            isLoading: false,
          }));
          toast.error(error.response?.data?.message || 'Failed to create profile');
        }
      },

      updateProfile: async (id, updates) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
// //           const previousProfiles = get().profiles;
          set((state) => ({
            profiles: state.profiles.map((p) =>
              p.id === id ? { ...p, ...updates, updatedAt: new Date().toISOString() } : p
            ),
          }));

          // API call
          const updatedProfile = await voiceApi.updateVoiceProfile(id, updates);
          
          // Update with real data
          set((state) => ({
            profiles: state.profiles.map((p) => (p.id === id ? updatedProfile : p)),
            isLoading: false,
          }));
          
          toast.success('Voice profile updated successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set({ profiles: get().profiles, error: error.response?.data?.message || 'Failed to update profile', isLoading: false });
          toast.error(error.response?.data?.message || 'Failed to update profile');
        }
      },

      deleteProfile: async (id) => {
        set({ isLoading: true, error: null });
        try {
          // Optimistic update
// //           const previousProfiles = get().profiles;
          set((state) => ({
            profiles: state.profiles.filter((p) => p.id !== id),
          }));

          // API call
          await voiceApi.deleteVoiceProfile(id);
          
          set({ isLoading: false });
          toast.success('Voice profile deleted successfully');
        } catch (error: any) {
          // Rollback optimistic update
          set({ profiles: get().profiles, error: error.response?.data?.message || 'Failed to delete profile', isLoading: false });
          toast.error(error.response?.data?.message || 'Failed to delete profile');
        }
      },

      setCurrentProfile: (profile) => {
        set({ currentProfile: profile });
      },

      speechToText: async (audioBlob, language) => {
        set({ isProcessing: true, error: null });
        try {
          const result = await voiceApi.speechToText(audioBlob, language);
          set({ isProcessing: false });
          return result.text;
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Speech-to-text failed';
          set({ error: errorMessage, isProcessing: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      textToSpeech: async (text, voice) => {
        set({ isProcessing: true, error: null });
        try {
          const audioBlob = await voiceApi.textToSpeech(text, voice);
          set({ isProcessing: false });
          return audioBlob;
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Text-to-speech failed';
          set({ error: errorMessage, isProcessing: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      // Wake word detection actions
      startWakeWordDetection: async () => {
        set({ isLoading: true, error: null });
        try {
          const { wakeWordKeywords, wakeWordSensitivities } = get();
          await voiceApi.startWakeWordDetection(wakeWordKeywords, wakeWordSensitivities);

          set({
            isListeningForWakeWord: true,
            wakeWordEnabled: true,
            isLoading: false
          });

          toast.success('Wake word detection started');
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to start wake word detection';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      stopWakeWordDetection: async () => {
        set({ isLoading: true, error: null });
        try {
          await voiceApi.stopWakeWordDetection();

          set({
            isListeningForWakeWord: false,
            wakeWordEnabled: false,
            isLoading: false
          });

          toast.success('Wake word detection stopped');
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to stop wake word detection';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      getWakeWordStatus: async () => {
        try {
          const stats = await voiceApi.getWakeWordStatus();
          set({ wakeWordStats: stats });
        } catch (error: any) {
          console.error('Failed to get wake word status:', error);
        }
      },

      updateWakeWordSettings: (keywords, sensitivities) => {
        set({
          wakeWordKeywords: keywords,
          wakeWordSensitivities: sensitivities
        });
      },

      clearError: () => set({ error: null }),

      reset: () => set(initialState),
    }),
    {
      name: 'voice-storage',
      partialize: (state) => ({
        profiles: state.profiles,
        currentProfile: state.currentProfile,
      }),
    }
  )
);
