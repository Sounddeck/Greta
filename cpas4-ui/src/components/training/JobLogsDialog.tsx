/**
 * Job Logs Dialog Component
 * View training logs in scrollable dialog
 */

import React, { useEffect, useRef } from 'react';
import { Dialog } from '../ui/dialog';
import { Button } from '../ui/button';
import type { TrainingJob } from '../../types/training';

interface JobLogsDialogProps {
  job: TrainingJob;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function JobLogsDialog({ job, open, onOpenChange }: JobLogsDialogProps) {
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when logs update
  useEffect(() => {
    if (open && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [job.logs, open]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <div className="fixed inset-0 z-50 bg-black/50" onClick={() => onOpenChange(false)} />
      <div className="fixed left-[50%] top-[50%] z-50 w-full max-w-4xl translate-x-[-50%] translate-y-[-50%] bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Training Logs: {job.job_name}</h2>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </div>

        {/* Logs Container */}
        <div className="flex-1 overflow-y-auto bg-gray-900 dark:bg-black rounded-md p-4 font-mono text-sm">
          {job.logs && job.logs.length > 0 ? (
            <div className="space-y-1">
              {job.logs.map((log, index) => (
                <div key={index} className="text-green-400">
                  {log}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          ) : (
            <div className="text-gray-500 text-center py-8">No logs available yet</div>
          )}
        </div>

        {/* Status Info */}
        <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
          <p>Status: {job.status}</p>
          {job.status === 'running' && <p className="text-blue-500">Logs update in real-time</p>}
        </div>
      </div>
    </Dialog>
  );
}

