/**
 * Job List Component
 * Grid layout for all training jobs with filtering and real-time updates
 */

import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { JobCard } from './JobCard';
import { CreateJobDialog } from './CreateJobDialog';
import { JobLogsDialog } from './JobLogsDialog';
import { useTrainingStore } from '../../stores/trainingStore';
import type { TrainingJob } from '../../types/training';
import { toast } from 'sonner';

export function JobList() {
  const { jobs, jobsLoading, jobsError, fetchJobs, cancelJob } = useTrainingStore();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [logsDialogOpen, setLogsDialogOpen] = useState(false);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // Initial fetch
  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Polling for running jobs (every 5 seconds)
  useEffect(() => {
    const hasRunningJobs = jobs.some((job) => job.status === 'running');
    
    if (hasRunningJobs) {
      const interval = setInterval(() => {
        fetchJobs();
      }, 5000); // 5 seconds

      return () => clearInterval(interval);
    }
  }, [jobs, fetchJobs]);

  const handleCreateSuccess = () => {
    setCreateDialogOpen(false);
    fetchJobs();
  };

  const handleViewLogs = (job: TrainingJob) => {
    setSelectedJob(job);
    setLogsDialogOpen(true);
  };

  const handleCancel = async (job: TrainingJob) => {
    if (!confirm(`Are you sure you want to cancel job "${job.job_name}"?`)) {
      return;
    }

    try {
      await cancelJob(job.id);
      toast.success('Job cancelled successfully');
      fetchJobs();
    } catch (error: any) {
      toast.error(`Failed to cancel job: ${error.message}`);
    }
  };

  // Filter jobs
  const filteredJobs = filterStatus === 'all' 
    ? jobs 
    : jobs.filter((job) => job.status === filterStatus);

  // Sort jobs (running first, then by created date)
  const sortedJobs = [...filteredJobs].sort((a, b) => {
    if (a.status === 'running' && b.status !== 'running') return -1;
    if (a.status !== 'running' && b.status === 'running') return 1;
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

  if (jobsLoading && jobs.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading jobs...</p>
        </div>
      </div>
    );
  }

  if (jobsError) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-6xl mb-4">❌</div>
          <h3 className="text-lg font-semibold mb-2">Error Loading Jobs</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{jobsError}</p>
          <Button onClick={() => fetchJobs()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold">Training Jobs</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {jobs.length} total jobs
            {jobs.some((j) => j.status === 'running') && (
              <span className="ml-2 text-blue-500">• {jobs.filter((j) => j.status === 'running').length} running</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Filter */}
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
          >
            <option value="all">All Jobs</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <Button onClick={() => setCreateDialogOpen(true)}>Create Job</Button>
        </div>
      </div>

      {/* Job Grid */}
      {sortedJobs.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl mb-4">🔄</div>
            <h3 className="text-lg font-semibold mb-2">
              {filterStatus === 'all' ? 'No Training Jobs' : `No ${filterStatus} Jobs`}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {filterStatus === 'all' 
                ? 'Create your first training job to get started'
                : `No jobs with status "${filterStatus}"`}
            </p>
            {filterStatus === 'all' && (
              <Button onClick={() => setCreateDialogOpen(true)}>Create Job</Button>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {sortedJobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                onViewLogs={handleViewLogs}
                onCancel={handleCancel}
              />
            ))}
          </div>
        </div>
      )}

      {/* Dialogs */}
      <CreateJobDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onSuccess={handleCreateSuccess}
      />

      {selectedJob && (
        <JobLogsDialog
          job={selectedJob}
          open={logsDialogOpen}
          onOpenChange={setLogsDialogOpen}
        />
      )}
    </div>
  );
}

