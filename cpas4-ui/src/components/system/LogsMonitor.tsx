// @ts-nocheck
/**
 * Logs Monitor Component
 * Basic monitoring dashboard for viewing structured logs
 * Part of Issue #34: Structured Logging & Monitoring
 */

import React, { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Select } from '../ui/select';
import { Badge } from '../ui/badge';
import api from '../../lib/api';
import { toast } from 'sonner';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  module?: string;
  function?: string;
  line?: number;
  extra?: Record<string, any>;
}

interface LogStats {
  total_logs: number;
  by_level: Record<string, number>;
  by_event: Record<string, number>;
  time_range: {
    start: string;
    end: string;
    hours: number;
  };
}

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const LEVEL_COLORS = {
  DEBUG: 'bg-gray-500',
  INFO: 'bg-blue-500',
  WARNING: 'bg-yellow-500',
  ERROR: 'bg-red-500',
  CRITICAL: 'bg-purple-500',
};

export function LogsMonitor() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [stats, setStats] = useState<LogStats | null>(null);
  const [availableEvents, setAvailableEvents] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  
  // Filters
  const [selectedLevel, setSelectedLevel] = useState<string>('');
  const [selectedEvent, setSelectedEvent] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);

  // Fetch logs
  const fetchLogs = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (selectedLevel) params.append('level', selectedLevel);
      if (selectedEvent) params.append('event', selectedEvent);
      if (searchQuery) params.append('search', searchQuery);
      params.append('page', page.toString());
      params.append('page_size', '50');

      const response = await api.get(`/api/v1/logs/query?${params.toString()}`);
      setLogs(response.data.logs);
      setHasMore(response.data.has_more);
    } catch (error) {
      console.error('Failed to fetch logs:', error);
      toast.error('Failed to fetch logs');
    } finally {
      setLoading(false);
    }
  };

  // Fetch stats
  const fetchStats = async () => {
    try {
      const response = await api.get('/api/v1/logs/stats?hours=24');
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  // Fetch available events
  const fetchAvailableEvents = async () => {
    try {
      const response = await api.get('/api/v1/logs/events');
      setAvailableEvents(response.data.events);
    } catch (error) {
      console.error('Failed to fetch events:', error);
    }
  };

  // Initial load
  useEffect(() => {
    fetchLogs();
    fetchStats();
    fetchAvailableEvents();
  }, []);

  // Reload when filters change
  useEffect(() => {
    setPage(1);
    fetchLogs();
  }, [selectedLevel, selectedEvent, searchQuery]);

  // Reload when page changes
  useEffect(() => {
    fetchLogs();
  }, [page]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchLogs();
      fetchStats();
    }, 30000);
    return () => clearInterval(interval);
  }, [selectedLevel, selectedEvent, searchQuery, page]);

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">System Logs Monitor</h1>
          <p className="text-sm text-gray-500">View and filter structured application logs</p>
        </div>
        <Button onClick={() => { fetchLogs(); fetchStats(); }}>
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="text-sm text-gray-500">Total Logs (24h)</div>
            <div className="text-2xl font-bold">{stats.total_logs}</div>
          </Card>
          {Object.entries(stats.by_level).map(([level, count]) => (
            <Card key={level} className="p-4">
              <div className="text-sm text-gray-500">{level}</div>
              <div className="text-2xl font-bold">{count}</div>
            </Card>
          ))}
        </div>
      )}

      {/* Filters */}
      <Card className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Log Level</label>
            <select
              className="w-full p-2 border rounded"
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
            >
              <option value="">All Levels</option>
              {LOG_LEVELS.map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Event Type</label>
            <select
              className="w-full p-2 border rounded"
              value={selectedEvent}
              onChange={(e) => setSelectedEvent(e.target.value)}
            >
              <option value="">All Events</option>
              {availableEvents.map((event) => (
                <option key={event} value={event}>
                  {event}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Search</label>
            <Input
              type="text"
              placeholder="Search in messages..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>
      </Card>

      {/* Logs Table */}
      <Card className="p-4">
        <div className="space-y-2">
          {loading && <div className="text-center py-8">Loading logs...</div>}
          
          {!loading && logs.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No logs found matching your filters
            </div>
          )}

          {!loading && logs.map((log, index) => (
            <div
              key={index}
              className="border-b last:border-b-0 pb-3 mb-3 last:mb-0 last:pb-0"
            >
              <div className="flex items-start gap-3">
                <Badge className={`${LEVEL_COLORS[log.level]} text-white shrink-0`}>
                  {log.level}
                </Badge>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs text-gray-500">
                      {formatTimestamp(log.timestamp)}
                    </span>
                    {log.module && (
                      <span className="text-xs text-gray-400">
                        {log.module}:{log.function}:{log.line}
                      </span>
                    )}
                    {log.extra?.event && (
                      <Badge variant="outline" className="text-xs">
                        {log.extra.event}
                      </Badge>
                    )}
                  </div>
                  <div className="text-sm font-mono break-words">{log.message}</div>
                  {log.extra && Object.keys(log.extra).length > 0 && (
                    <details className="mt-2">
                      <summary className="text-xs text-blue-600 cursor-pointer">
                        Show details
                      </summary>
                      <pre className="text-xs bg-gray-50 p-2 rounded mt-1 overflow-x-auto">
                        {JSON.stringify(log.extra, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Pagination */}
        {!loading && logs.length > 0 && (
          <div className="flex items-center justify-between mt-4 pt-4 border-t">
            <Button
              variant="outline"
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
            >
              Previous
            </Button>
            <span className="text-sm text-gray-500">Page {page}</span>
            <Button
              variant="outline"
              onClick={() => setPage(page + 1)}
              disabled={!hasMore}
            >
              Next
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
}

