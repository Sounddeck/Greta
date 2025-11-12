/**
 * Activity Feed
 * 
 * Real-time feed of recent events and activities
 */

import React, { useState, useEffect, useRef } from 'react';
import { Activity, Filter, CheckCircle, AlertCircle, Info, TrendingUp } from 'lucide-react';
import { useWebSocketContext } from '../../contexts/WebSocketContext';

interface ActivityItem {
  id: string;
  type: string;
  title: string;
  description: string;
  timestamp: Date;
  icon: 'success' | 'error' | 'info' | 'activity';
  data?: any;
}

const EVENT_TYPE_FILTERS = [
  { id: 'all', label: 'All Events' },
  { id: 'lead', label: 'Leads' },
  { id: 'campaign', label: 'Campaigns' },
  { id: 'system', label: 'System' },
  { id: 'agent', label: 'Agents' },
];

export default function ActivityFeed() {
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [filter, setFilter] = useState('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const feedRef = useRef<HTMLDivElement>(null);
  const { lastMessage } = useWebSocketContext();

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    const activity = createActivityFromMessage(lastMessage);
    if (activity) {
      setActivities(prev => [activity, ...prev].slice(0, 100)); // Keep last 100
    }
  }, [lastMessage]);

  // Auto-scroll to top when new activity arrives
  useEffect(() => {
    if (autoScroll && feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [activities, autoScroll]);

  const createActivityFromMessage = (message: any): ActivityItem | null => {
    const timestamp = new Date(message.timestamp || Date.now());
    const id = message.event_id || `${Date.now()}-${Math.random()}`;

    switch (message.type) {
      case 'lead.scored':
        return {
          id,
          type: 'lead',
          title: 'Lead Scored',
          description: `Score: ${message.data.score.toFixed(0)} (${(message.data.confidence * 100).toFixed(0)}% confidence)`,
          timestamp,
          icon: 'success',
          data: message.data,
        };

      case 'lead.generated':
        return {
          id,
          type: 'lead',
          title: 'New Lead Generated',
          description: message.data.company_name || 'Lead added to campaign',
          timestamp,
          icon: 'info',
          data: message.data,
        };

      case 'lead.status_changed':
        return {
          id,
          type: 'lead',
          title: 'Lead Status Changed',
          description: `Status: ${message.data.status}`,
          timestamp,
          icon: 'activity',
          data: message.data,
        };

      case 'campaign.created':
        return {
          id,
          type: 'campaign',
          title: 'Campaign Created',
          description: message.data.campaign_name || 'New campaign',
          timestamp,
          icon: 'success',
          data: message.data,
        };

      case 'campaign.updated':
        return {
          id,
          type: 'campaign',
          title: 'Campaign Updated',
          description: message.data.campaign_name || 'Campaign modified',
          timestamp,
          icon: 'info',
          data: message.data,
        };

      case 'system.error':
        return {
          id,
          type: 'system',
          title: 'System Error',
          description: message.data.message || 'An error occurred',
          timestamp,
          icon: 'error',
          data: message.data,
        };

      default:
        return null;
    }
  };

  const getIcon = (icon: ActivityItem['icon']) => {
    switch (icon) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case 'activity':
        return <TrendingUp className="h-5 w-5 text-blue-500" />;
      default:
        return <Info className="h-5 w-5 text-gray-500" />;
    }
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);

    if (seconds < 10) return 'Just now';
    if (seconds < 60) return `${seconds}s ago`;
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  // Filter activities
  const filteredActivities = activities.filter(activity => {
    if (filter === 'all') return true;
    return activity.type === filter;
  });

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-gray-600 dark:text-gray-400" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Activity Feed
          </h3>
        </div>

        {/* Filter Dropdown */}
        <div className="relative">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {EVENT_TYPE_FILTERS.map(f => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Auto-scroll Toggle */}
      <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">
            Auto-scroll to latest
          </span>
        </label>
      </div>

      {/* Activity List */}
      <div
        ref={feedRef}
        className="flex-1 overflow-y-auto p-4 space-y-3"
      >
        {filteredActivities.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <Activity className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>No activities yet</p>
            <p className="text-sm mt-1">
              {filter !== 'all' ? 'Try changing the filter' : 'Activities will appear here in real-time'}
            </p>
          </div>
        ) : (
          filteredActivities.map(activity => (
            <div
              key={activity.id}
              className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-750 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex-shrink-0 mt-0.5">
                {getIcon(activity.icon)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-gray-900 dark:text-white">
                  {activity.title}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-0.5">
                  {activity.description}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                  {formatTimestamp(activity.timestamp)}
                </p>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-700 text-center">
        <p className="text-xs text-gray-500 dark:text-gray-500">
          Showing {filteredActivities.length} of {activities.length} activities
        </p>
      </div>
    </div>
  );
}

