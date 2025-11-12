import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { toast } from 'sonner';
import { api } from '../lib/api';

// Types
export interface Job {
  job_id: string;
  title: string;
  description: string;
  requirements: string[];
  responsibilities: string[];
  skills_required: string[];
  skills_preferred: string[];
  job_type: 'full_time' | 'part_time' | 'contract' | 'temporary' | 'internship';
  experience_level: 'entry' | 'junior' | 'mid' | 'senior' | 'lead' | 'executive';
  location: string;
  remote_allowed: boolean;
  salary_min?: number;
  salary_max?: number;
  salary_currency: string;
  department?: string;
  hiring_manager?: string;
  status: 'draft' | 'open' | 'closed' | 'on_hold' | 'filled';
  application_count: number;
  view_count: number;
  tags: string[];
  created_at: string;
  updated_at: string;
  posted_at?: string;
  closed_at?: string;
  custom_fields?: Record<string, any>;
}

export interface Candidate {
  candidate_id: string;
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
  location?: string;
  current_title?: string;
  current_company?: string;
  experience_years?: number;
  skills: string[];
  education: any[];
  work_history: any[];
  resume_url?: string;
  linkedin_url?: string;
  portfolio_url?: string;
  status: 'new' | 'active' | 'hired' | 'rejected' | 'withdrawn' | 'blacklisted';
  tags: string[];
  created_at: string;
  updated_at: string;
  custom_fields?: Record<string, any>;
}

export interface Application {
  application_id: string;
  job_id: string;
  candidate_id: string;
  status: 'applied' | 'screening' | 'phone_screen' | 'interview' | 'technical_test' | 'final_interview' | 'offer' | 'hired' | 'rejected' | 'withdrawn';
  match_score: number;
  applied_at: string;
  updated_at: string;
  notes?: string;
  custom_fields?: Record<string, any>;
}

export interface Interview {
  interview_id: string;
  application_id: string;
  candidate_id: string;
  job_id: string;
  type: 'phone' | 'video' | 'in_person' | 'technical' | 'panel' | 'behavioral';
  scheduled_at: string;
  duration_minutes: number;
  interviewer: string;
  location?: string;
  meeting_link?: string;
  status: 'scheduled' | 'completed' | 'cancelled' | 'no_show' | 'rescheduled';
  feedback?: string;
  rating?: number;
  created_at: string;
  updated_at: string;
}

export interface ATSStats {
  total_jobs: number;
  open_jobs: number;
  total_candidates: number;
  total_applications: number;
  applications_by_status: Record<string, number>;
  jobs_by_status: Record<string, number>;
  recent_applications: number;
  interviews_scheduled: number;
}

interface ATSStore {
  // State
  jobs: Job[];
  candidates: Candidate[];
  applications: Application[];
  interviews: Interview[];
  stats: ATSStats | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchJobs: (filters?: any) => Promise<void>;
  fetchCandidates: (filters?: any) => Promise<void>;
  fetchApplications: (filters?: any) => Promise<void>;
  fetchInterviews: (filters?: any) => Promise<void>;
  fetchStats: () => Promise<void>;
  
  createJob: (data: any) => Promise<Job | null>;
  updateJob: (id: string, data: any) => Promise<Job | null>;
  deleteJob: (id: string) => Promise<boolean>;
  
  createCandidate: (data: any) => Promise<Candidate | null>;
  updateCandidate: (id: string, data: any) => Promise<Candidate | null>;
  deleteCandidate: (id: string) => Promise<boolean>;
  
  createApplication: (data: any) => Promise<Application | null>;
  updateApplication: (id: string, data: any) => Promise<Application | null>;
  
  createInterview: (data: any) => Promise<Interview | null>;
  updateInterview: (id: string, data: any) => Promise<Interview | null>;
}

const extractErrorMessage = (error: any, defaultMessage: string): string => {
  if (error?.response?.data?.detail) return error.response.data.detail;
  if (error?.response?.data?.message) return error.response.data.message;
  if (error?.response?.data?.error) return error.response.data.error;
  if (error?.message) return error.message;
  return defaultMessage;
};

export const useATSStore = create<ATSStore>()(
  persist(
    (set, get) => ({
      // Initial state
      jobs: [],
      candidates: [],
      applications: [],
      interviews: [],
      stats: null,
      isLoading: false,
      error: null,

      // Fetch jobs
      fetchJobs: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/ats/jobs', { params: filters });
          set({ jobs: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch jobs');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch candidates
      fetchCandidates: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/ats/candidates', { params: filters });
          set({ candidates: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch candidates');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch applications
      fetchApplications: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/ats/applications', { params: filters });
          set({ applications: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch applications');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch interviews
      fetchInterviews: async (filters = {}) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.get('/ats/interviews', { params: filters });
          set({ interviews: response.data, isLoading: false });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch interviews');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
        }
      },

      // Fetch stats
      fetchStats: async () => {
        try {
          const response = await api.get('/ats/stats');
          set({ stats: response.data });
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to fetch stats');
          toast.error(errorMsg);
        }
      },

      // Create job
      createJob: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/ats/jobs', data);
          const newJob = response.data;
          set((state) => ({
            jobs: [newJob, ...state.jobs],
            isLoading: false,
          }));
          toast.success('Job created successfully');
          return newJob;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to create job');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Update job
      updateJob: async (id, data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.patch(`/ats/jobs/${id}`, data);
          const updatedJob = response.data;
          set((state) => ({
            jobs: state.jobs.map((j) => (j.job_id === id ? updatedJob : j)),
            isLoading: false,
          }));
          toast.success('Job updated successfully');
          return updatedJob;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to update job');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Delete job
      deleteJob: async (id) => {
        set({ isLoading: true, error: null });
        try {
          await api.delete(`/ats/jobs/${id}`);
          set((state) => ({
            jobs: state.jobs.filter((j) => j.job_id !== id),
            isLoading: false,
          }));
          toast.success('Job deleted successfully');
          return true;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to delete job');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return false;
        }
      },

      // Create candidate
      createCandidate: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/ats/candidates', data);
          const newCandidate = response.data;
          set((state) => ({
            candidates: [newCandidate, ...state.candidates],
            isLoading: false,
          }));
          toast.success('Candidate created successfully');
          return newCandidate;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to create candidate');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Update candidate
      updateCandidate: async (id, data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.patch(`/ats/candidates/${id}`, data);
          const updatedCandidate = response.data;
          set((state) => ({
            candidates: state.candidates.map((c) =>
              c.candidate_id === id ? updatedCandidate : c
            ),
            isLoading: false,
          }));
          toast.success('Candidate updated successfully');
          return updatedCandidate;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to update candidate');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Delete candidate
      deleteCandidate: async (id) => {
        set({ isLoading: true, error: null });
        try {
          await api.delete(`/ats/candidates/${id}`);
          set((state) => ({
            candidates: state.candidates.filter((c) => c.candidate_id !== id),
            isLoading: false,
          }));
          toast.success('Candidate deleted successfully');
          return true;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to delete candidate');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return false;
        }
      },

      // Create application
      createApplication: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/ats/applications', data);
          const newApplication = response.data;
          set((state) => ({
            applications: [newApplication, ...state.applications],
            isLoading: false,
          }));
          toast.success('Application created successfully');
          return newApplication;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to create application');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Update application
      updateApplication: async (id, data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.patch(`/ats/applications/${id}`, data);
          const updatedApplication = response.data;
          set((state) => ({
            applications: state.applications.map((a) =>
              a.application_id === id ? updatedApplication : a
            ),
            isLoading: false,
          }));
          toast.success('Application updated successfully');
          return updatedApplication;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to update application');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Create interview
      createInterview: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/ats/interviews', data);
          const newInterview = response.data;
          set((state) => ({
            interviews: [newInterview, ...state.interviews],
            isLoading: false,
          }));
          toast.success('Interview scheduled successfully');
          return newInterview;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to schedule interview');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // Update interview
      updateInterview: async (id, data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.patch(`/ats/interviews/${id}`, data);
          const updatedInterview = response.data;
          set((state) => ({
            interviews: state.interviews.map((i) =>
              i.interview_id === id ? updatedInterview : i
            ),
            isLoading: false,
          }));
          toast.success('Interview updated successfully');
          return updatedInterview;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to update interview');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },

      // CSV Import
      importCandidatesCSV: async (file: File) => {
        set({ isLoading: true, error: null });
        try {
          const formData = new FormData();
          formData.append('file', file);

          const response = await api.post('/ats/candidates/import-csv', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });

          const result = response.data;

          if (result.success) {
            toast.success(result.message);
            // Refresh candidates list
            get().fetchCandidates();
          } else {
            toast.warning(result.message);
            if (result.errors.length > 0) {
              console.error('Import errors:', result.errors);
            }
          }

          set({ isLoading: false });
          return result;
        } catch (error: any) {
          const errorMsg = extractErrorMessage(error, 'Failed to import candidates');
          set({ error: errorMsg, isLoading: false });
          toast.error(errorMsg);
          return null;
        }
      },
    }),
    {
      name: 'ats-storage',
      partialize: (state) => ({
        jobs: state.jobs,
        candidates: state.candidates,
        applications: state.applications,
      }),
    }
  )
);

