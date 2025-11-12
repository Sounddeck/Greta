/**
 * Connection Status Indicator
 * 
 * Displays WebSocket connection state with visual indicators
 */

import React, { useState, useEffect } from 'react';
import { Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { useWebSocketContext } from '../../contexts/WebSocketContext';

export default function ConnectionStatus() {
  const { isConnected, reconnectAttempts, connectionId } = useWebSocketContext();
  const [showDetails, setShowDetails] = useState(false);

  // Get status color and icon
  const getStatusInfo = () => {
    if (isConnected) {
      return {
        color: 'text-green-500',
        bgColor: 'bg-green-500',
        icon: <Wifi className="h-4 w-4" />,
        label: 'Connected',
        description: 'Real-time updates active',
      };
    } else if (reconnectAttempts > 0) {
      return {
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-500',
        icon: <RefreshCw className="h-4 w-4 animate-spin" />,
        label: 'Reconnecting',
        description: `Attempt ${reconnectAttempts}...`,
      };
    } else {
      return {
        color: 'text-red-500',
        bgColor: 'bg-red-500',
        icon: <WifiOff className="h-4 w-4" />,
        label: 'Disconnected',
        description: 'Real-time updates unavailable',
      };
    }
  };

  const status = getStatusInfo();

  return (
    <div className="relative">
      {/* Status Indicator */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        title={`${status.label} - ${status.description}`}
      >
        {/* Pulsing dot */}
        <div className="relative">
          <span className={`flex h-3 w-3 ${status.bgColor} rounded-full`}>
            {isConnected && (
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${status.bgColor} opacity-75`}></span>
            )}
          </span>
        </div>

        {/* Icon */}
        <span className={status.color}>
          {status.icon}
        </span>

        {/* Label (hidden on mobile) */}
        <span className={`hidden sm:inline text-sm font-medium ${status.color}`}>
          {status.label}
        </span>
      </button>

      {/* Details Popup */}
      {showDetails && (
        <div className="absolute right-0 mt-2 w-72 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-50">
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                Connection Status
              </h3>
              <button
                onClick={() => setShowDetails(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                ×
              </button>
            </div>

            <div className="space-y-3">
              {/* Status */}
              <div className="flex items-center gap-2">
                <span className={status.color}>{status.icon}</span>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {status.label}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {status.description}
                  </p>
                </div>
              </div>

              {/* Connection ID */}
              {connectionId && (
                <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                    Connection ID
                  </p>
                  <p className="text-xs font-mono text-gray-900 dark:text-white break-all">
                    {connectionId}
                  </p>
                </div>
              )}

              {/* Reconnection Info */}
              {reconnectAttempts > 0 && (
                <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                    Reconnection Attempts
                  </p>
                  <p className="text-sm font-medium text-yellow-600 dark:text-yellow-400">
                    {reconnectAttempts}
                  </p>
                </div>
              )}

              {/* Connection Info */}
              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Protocol</p>
                    <p className="font-medium text-gray-900 dark:text-white">WebSocket</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Auto-reconnect</p>
                    <p className="font-medium text-gray-900 dark:text-white">Enabled</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

