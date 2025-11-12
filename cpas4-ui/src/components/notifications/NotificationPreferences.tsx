/**
 * Notification Preferences
 * 
 * User preferences for notification types and delivery methods
 */

import React, { useState, useEffect } from 'react';
import { Bell, BellOff, Volume2, VolumeX } from 'lucide-react';

export interface NotificationPreference {
  id: string;
  label: string;
  description: string;
  enabled: boolean;
  sound: boolean;
  toast: boolean;
}

const DEFAULT_PREFERENCES: NotificationPreference[] = [
  {
    id: 'lead_scored',
    label: 'Lead Scored',
    description: 'Notify when a lead is scored',
    enabled: true,
    sound: false,
    toast: true,
  },
  {
    id: 'lead_generated',
    label: 'Lead Generated',
    description: 'Notify when a new lead is generated',
    enabled: true,
    sound: false,
    toast: true,
  },
  {
    id: 'lead_status_changed',
    label: 'Lead Status Changed',
    description: 'Notify when a lead status changes',
    enabled: true,
    sound: false,
    toast: false,
  },
  {
    id: 'campaign_created',
    label: 'Campaign Created',
    description: 'Notify when a new campaign is created',
    enabled: true,
    sound: false,
    toast: true,
  },
  {
    id: 'campaign_updated',
    label: 'Campaign Updated',
    description: 'Notify when a campaign is updated',
    enabled: false,
    sound: false,
    toast: false,
  },
  {
    id: 'system_error',
    label: 'System Errors',
    description: 'Notify when system errors occur',
    enabled: true,
    sound: true,
    toast: true,
  },
];

export default function NotificationPreferences() {
  const [preferences, setPreferences] = useState<NotificationPreference[]>(DEFAULT_PREFERENCES);

  // Load preferences from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('notification_preferences');
    if (saved) {
      try {
        setPreferences(JSON.parse(saved));
      } catch (error) {
        console.error('Failed to load notification preferences:', error);
      }
    }
  }, []);

  // Save preferences to localStorage
  const savePreferences = (newPreferences: NotificationPreference[]) => {
    setPreferences(newPreferences);
    localStorage.setItem('notification_preferences', JSON.stringify(newPreferences));
  };

  const toggleEnabled = (id: string) => {
    const updated = preferences.map(pref =>
      pref.id === id ? { ...pref, enabled: !pref.enabled } : pref
    );
    savePreferences(updated);
  };

  const toggleSound = (id: string) => {
    const updated = preferences.map(pref =>
      pref.id === id ? { ...pref, sound: !pref.sound } : pref
    );
    savePreferences(updated);
  };

  const toggleToast = (id: string) => {
    const updated = preferences.map(pref =>
      pref.id === id ? { ...pref, toast: !pref.toast } : pref
    );
    savePreferences(updated);
  };

  const resetToDefaults = () => {
    savePreferences(DEFAULT_PREFERENCES);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Notification Preferences
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Customize how you receive notifications
          </p>
        </div>
        <button
          onClick={resetToDefaults}
          className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-750"
        >
          Reset to Defaults
        </button>
      </div>

      <div className="space-y-4">
        {preferences.map(pref => (
          <div
            key={pref.id}
            className="p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
                    {pref.label}
                  </h4>
                  {pref.enabled ? (
                    <Bell className="h-4 w-4 text-blue-500" />
                  ) : (
                    <BellOff className="h-4 w-4 text-gray-400" />
                  )}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {pref.description}
                </p>
              </div>
              <button
                onClick={() => toggleEnabled(pref.id)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  pref.enabled
                    ? 'bg-blue-600'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    pref.enabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {pref.enabled && (
              <div className="flex items-center gap-4 ml-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                {/* Toast Toggle */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={pref.toast}
                    onChange={() => toggleToast(pref.id)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Show toast
                  </span>
                </label>

                {/* Sound Toggle */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={pref.sound}
                    onChange={() => toggleSound(pref.id)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300 flex items-center gap-1">
                    {pref.sound ? (
                      <Volume2 className="h-3 w-3" />
                    ) : (
                      <VolumeX className="h-3 w-3" />
                    )}
                    Play sound
                  </span>
                </label>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <p className="text-sm text-blue-800 dark:text-blue-300">
          <strong>Note:</strong> Notification preferences are saved locally in your browser.
          They will be reset if you clear your browser data.
        </p>
      </div>
    </div>
  );
}

/**
 * Hook to get notification preferences
 */
export function useNotificationPreferences() {
  const [preferences, setPreferences] = useState<NotificationPreference[]>(DEFAULT_PREFERENCES);

  useEffect(() => {
    const saved = localStorage.getItem('notification_preferences');
    if (saved) {
      try {
        setPreferences(JSON.parse(saved));
      } catch (error) {
        console.error('Failed to load notification preferences:', error);
      }
    }
  }, []);

  const isEnabled = (id: string) => {
    const pref = preferences.find(p => p.id === id);
    return pref?.enabled ?? false;
  };

  const shouldShowToast = (id: string) => {
    const pref = preferences.find(p => p.id === id);
    return pref?.enabled && pref?.toast;
  };

  const shouldPlaySound = (id: string) => {
    const pref = preferences.find(p => p.id === id);
    return pref?.enabled && pref?.sound;
  };

  return {
    preferences,
    isEnabled,
    shouldShowToast,
    shouldPlaySound,
  };
}

