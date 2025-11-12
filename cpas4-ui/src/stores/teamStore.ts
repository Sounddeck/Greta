/**
 * Team Store - Zustand State Management
 * Manages multi-agent teams and team executions
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from 'axios';
import { toast } from 'sonner';
import type {
  Team,
  TeamCreate,
  TeamUpdate,
  TeamExecutionRequest,
  TeamExecutionResult,
  TeamValidationResult,
  WorkflowPattern
} from '../types/team';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface TeamStore {
  // State
  teams: Team[];
  currentTeam: Team | null;
  executions: TeamExecutionResult[];
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchTeams: (status?: string, workflowPattern?: WorkflowPattern) => Promise<void>;
  fetchTeam: (teamId: string) => Promise<void>;
  createTeam: (team: TeamCreate) => Promise<Team>;
  updateTeam: (teamId: string, update: TeamUpdate) => Promise<Team>;
  deleteTeam: (teamId: string) => Promise<void>;
  validateTeam: (teamId: string) => Promise<TeamValidationResult>;
  executeTeam: (teamId: string, task: string, context?: Record<string, any>) => Promise<TeamExecutionResult>;
  fetchExecutions: (teamId: string) => Promise<void>;
  fetchExecution: (executionId: string) => Promise<TeamExecutionResult | null>;
  setCurrentTeam: (team: Team | null) => void;
  clearError: () => void;
  reset: () => void;
}

const initialState = {
  teams: [],
  currentTeam: null,
  executions: [],
  isLoading: false,
  error: null,
};

export const useTeamStore = create<TeamStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Fetch all teams
      fetchTeams: async (status?: string, workflowPattern?: WorkflowPattern) => {
        set({ isLoading: true, error: null });
        try {
          const params: Record<string, string> = {};
          if (status) params.status = status;
          if (workflowPattern) params.workflow_pattern = workflowPattern;

          const response = await axios.get(`${API_BASE_URL}/api/v1/teams`, { params });
          set({ teams: response.data || [], isLoading: false });
        } catch (error: any) {
          // Handle 404 gracefully - might be no teams yet or backend endpoint not available
          if (error.response?.status === 404) {
            console.warn('⚠️ No teams found or backend endpoint not available');
            set({
              teams: [],
              error: null, // Don't show error for empty state
              isLoading: false
            });
          } else {
            const errorMessage = error.response?.data?.detail || 'Failed to fetch teams';
            set({ error: errorMessage, isLoading: false });
            toast.error(errorMessage);
          }
        }
      },

      // Fetch single team
      fetchTeam: async (teamId: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await axios.get(`${API_BASE_URL}/api/v1/teams/${teamId}`);
          set({ currentTeam: response.data, isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to fetch team';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      // Create team
      createTeam: async (team: TeamCreate) => {
        set({ isLoading: true, error: null });
        try {
          const response = await axios.post(`${API_BASE_URL}/api/v1/teams`, team);
          const newTeam = response.data;
          
          set((state) => ({
            teams: [...state.teams, newTeam],
            currentTeam: newTeam,
            isLoading: false,
          }));
          
          toast.success('Team created successfully');
          return newTeam;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to create team';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      // Update team
      updateTeam: async (teamId: string, update: TeamUpdate) => {
        set({ isLoading: true, error: null });
        try {
          const response = await axios.patch(`${API_BASE_URL}/api/v1/teams/${teamId}`, update);
          const updatedTeam = response.data;
          
          set((state) => ({
            teams: state.teams.map((t) => (t.team_id === teamId ? updatedTeam : t)),
            currentTeam: state.currentTeam?.team_id === teamId ? updatedTeam : state.currentTeam,
            isLoading: false,
          }));
          
          toast.success('Team updated successfully');
          return updatedTeam;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to update team';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      // Delete team
      deleteTeam: async (teamId: string) => {
        set({ isLoading: true, error: null });
        try {
          await axios.delete(`${API_BASE_URL}/api/v1/teams/${teamId}`);
          
          set((state) => ({
            teams: state.teams.filter((t) => t.team_id !== teamId),
            currentTeam: state.currentTeam?.team_id === teamId ? null : state.currentTeam,
            isLoading: false,
          }));
          
          toast.success('Team deleted successfully');
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to delete team';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      // Validate team
      validateTeam: async (teamId: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await axios.post(`${API_BASE_URL}/api/v1/teams/${teamId}/validate`);
          set({ isLoading: false });
          
          const result: TeamValidationResult = response.data;
          
          if (!result.valid) {
            toast.error(`Validation failed: ${result.errors.join(', ')}`);
          } else if (result.warnings.length > 0) {
            toast.warning(`Warnings: ${result.warnings.join(', ')}`);
          } else {
            toast.success('Team configuration is valid');
          }
          
          return result;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to validate team';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      // Execute team
      executeTeam: async (teamId: string, task: string, context?: Record<string, any>) => {
        set({ isLoading: true, error: null });
        try {
          const request: TeamExecutionRequest = {
            team_id: teamId,
            task,
            context,
          };
          
          const response = await axios.post(`${API_BASE_URL}/api/v1/teams/${teamId}/execute`, request);
          const execution = response.data;
          
          set((state) => ({
            executions: [execution, ...state.executions],
            isLoading: false,
          }));
          
          toast.success('Team execution started');
          return execution;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to execute team';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
          throw error;
        }
      },

      // Fetch executions
      fetchExecutions: async (teamId: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await axios.get(`${API_BASE_URL}/api/v1/teams/${teamId}/executions`);
          set({ executions: response.data || [], isLoading: false });
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to fetch executions';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
        }
      },

      // Fetch single execution
      fetchExecution: async (executionId: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await axios.get(`${API_BASE_URL}/api/v1/teams/executions/${executionId}`);
          set({ isLoading: false });
          return response.data;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to fetch execution';
          set({ error: errorMessage, isLoading: false });
          toast.error(errorMessage);
          return null;
        }
      },

      // Set current team
      setCurrentTeam: (team: Team | null) => {
        set({ currentTeam: team });
      },

      // Clear error
      clearError: () => set({ error: null }),

      // Reset store
      reset: () => set(initialState),
    }),
    {
      name: 'team-storage',
      partialize: (state) => ({
        // Only persist teams and currentTeam, not executions or loading states
        teams: state.teams,
        currentTeam: state.currentTeam,
      }),
    }
  )
);

