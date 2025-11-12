/**
 * WebSocket Context
 *
 * Global WebSocket context for real-time updates across the application.
 */

import React, { createContext, useContext, useCallback, ReactNode } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import type { WebSocketMessage, UseWebSocketReturn } from '../hooks/useWebSocket';
import { toast } from 'sonner';
import { useNotificationPreferences } from '../components/notifications/NotificationPreferences';

interface WebSocketContextValue extends UseWebSocketReturn {
  /** Subscribe to lead updates */
  subscribeToLeadUpdates: () => void;
  /** Subscribe to campaign updates */
  subscribeToCampaignUpdates: () => void;
  /** Subscribe to system status */
  subscribeToSystemStatus: () => void;
  /** Subscribe to agent updates */
  subscribeToAgentUpdates: () => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
  /** Whether to show connection status toasts */
  showToasts?: boolean;
}

export function WebSocketProvider({ children, showToasts = true }: WebSocketProviderProps) {
  const { shouldShowToast, shouldPlaySound } = useNotificationPreferences();

  // Play notification sound
  const playNotificationSound = useCallback(() => {
    try {
      // Create a simple beep sound using Web Audio API
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.frequency.value = 800;
      oscillator.type = 'sine';

      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.5);
    } catch (error) {
      console.error('Failed to play notification sound:', error);
    }
  }, []);

  // Handle connection events
  const handleConnect = useCallback((connectionId: string) => {
    console.log('✅ WebSocket connected:', connectionId);
    if (showToasts) {
      toast.success('Connected to real-time updates');
    }
  }, [showToasts]);

  const handleDisconnect = useCallback(() => {
    console.log('❌ WebSocket disconnected');
    if (showToasts) {
      toast.error('Disconnected from real-time updates');
    }
  }, [showToasts]);

  const handleError = useCallback((error: Event) => {
    console.error('WebSocket error:', error);
    if (showToasts) {
      toast.error('WebSocket connection error');
    }
  }, [showToasts]);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    console.log('📨 WebSocket message:', message);

    // Map event type to preference ID
    const getPreferenceId = (type: string) => {
      return type.replace('.', '_');
    };

    const preferenceId = getPreferenceId(message.type);

    // Handle specific message types
    switch (message.type) {
      case 'lead.scored':
        // Lead score updated
        if (showToasts && shouldShowToast(preferenceId)) {
          toast.info(`Lead scored: ${message.data.score}`, {
            description: `Confidence: ${(message.data.confidence * 100).toFixed(0)}%`,
          });
        }
        if (shouldPlaySound(preferenceId)) {
          playNotificationSound();
        }
        break;

      case 'lead.generated':
        // New lead generated
        if (showToasts && shouldShowToast(preferenceId)) {
          toast.info('New lead generated', {
            description: message.data.company_name || 'Lead added to campaign',
          });
        }
        if (shouldPlaySound(preferenceId)) {
          playNotificationSound();
        }
        break;

      case 'lead.status_changed':
        // Lead status changed
        if (showToasts && shouldShowToast(preferenceId)) {
          toast.info(`Lead status: ${message.data.status}`);
        }
        if (shouldPlaySound(preferenceId)) {
          playNotificationSound();
        }
        break;

      case 'campaign.created':
        // Campaign created
        if (showToasts && shouldShowToast(preferenceId)) {
          toast.success('New campaign created');
        }
        if (shouldPlaySound(preferenceId)) {
          playNotificationSound();
        }
        break;

      case 'campaign.updated':
        // Campaign updated
        if (showToasts && shouldShowToast(preferenceId)) {
          toast.info('Campaign updated');
        }
        if (shouldPlaySound(preferenceId)) {
          playNotificationSound();
        }
        break;

      case 'system.error':
        // System error
        if (showToasts && shouldShowToast(preferenceId)) {
          toast.error('System error', {
            description: message.data.message,
          });
        }
        if (shouldPlaySound(preferenceId)) {
          playNotificationSound();
        }
        break;

      default:
        // Log other messages
        console.log('Unhandled message type:', message.type);
    }
  }, [showToasts, shouldShowToast, shouldPlaySound, playNotificationSound]);

  // Initialize WebSocket with options
  const ws = useWebSocket(undefined, {
    autoConnect: true,
    autoReconnect: true,
    reconnectDelay: 3000,
    maxReconnectAttempts: 0, // Infinite reconnection attempts
    initialTopics: [],
    onConnect: handleConnect,
    onDisconnect: handleDisconnect,
    onError: handleError,
    onMessage: handleMessage,
  });

  // Convenience methods for subscribing to specific topics
  const subscribeToLeadUpdates = useCallback(() => {
    ws.subscribe('lead_updates');
  }, [ws]);

  const subscribeToCampaignUpdates = useCallback(() => {
    ws.subscribe('campaign_updates');
  }, [ws]);

  const subscribeToSystemStatus = useCallback(() => {
    ws.subscribe('system_status');
  }, [ws]);

  const subscribeToAgentUpdates = useCallback(() => {
    ws.subscribe('agent_updates');
  }, [ws]);

  const value: WebSocketContextValue = {
    ...ws,
    subscribeToLeadUpdates,
    subscribeToCampaignUpdates,
    subscribeToSystemStatus,
    subscribeToAgentUpdates,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

/**
 * Hook to access WebSocket context
 */
export function useWebSocketContext(): WebSocketContextValue {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocketContext must be used within WebSocketProvider');
  }
  return context;
}

/**
 * Hook to subscribe to lead updates
 */
export function useLeadUpdates(callback: (message: WebSocketMessage) => void) {
  const { lastMessage, subscribeToLeadUpdates, isConnected } = useWebSocketContext();

  React.useEffect(() => {
    if (isConnected) {
      subscribeToLeadUpdates();
    }
  }, [isConnected, subscribeToLeadUpdates]);

  React.useEffect(() => {
    if (lastMessage && lastMessage.topic === 'lead_updates') {
      callback(lastMessage);
    }
  }, [lastMessage, callback]);
}

/**
 * Hook to subscribe to campaign updates
 */
export function useCampaignUpdates(callback: (message: WebSocketMessage) => void) {
  const { lastMessage, subscribeToCampaignUpdates, isConnected } = useWebSocketContext();

  React.useEffect(() => {
    if (isConnected) {
      subscribeToCampaignUpdates();
    }
  }, [isConnected, subscribeToCampaignUpdates]);

  React.useEffect(() => {
    if (lastMessage && lastMessage.topic === 'campaign_updates') {
      callback(lastMessage);
    }
  }, [lastMessage, callback]);
}

/**
 * Hook to subscribe to system status
 */
export function useSystemStatus(callback: (message: WebSocketMessage) => void) {
  const { lastMessage, subscribeToSystemStatus, isConnected } = useWebSocketContext();

  React.useEffect(() => {
    if (isConnected) {
      subscribeToSystemStatus();
    }
  }, [isConnected, subscribeToSystemStatus]);

  React.useEffect(() => {
    if (lastMessage && lastMessage.topic === 'system_status') {
      callback(lastMessage);
    }
  }, [lastMessage, callback]);
}

