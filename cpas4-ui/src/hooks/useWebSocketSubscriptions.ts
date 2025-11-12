/**
 * WebSocket Subscription Hooks
 *
 * Custom hooks for subscribing to specific WebSocket topics.
 * These are separated from WebSocketContext.tsx to maintain Vite Fast Refresh compatibility.
 */

import { useEffect } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import type { WebSocketMessage } from './useWebSocket';

/**
 * Hook to subscribe to lead updates
 */
export function useLeadUpdates(callback: (message: WebSocketMessage) => void) {
  const { lastMessage, subscribeToLeadUpdates, isConnected } = useWebSocketContext();

  useEffect(() => {
    if (isConnected) {
      subscribeToLeadUpdates();
    }
  }, [isConnected, subscribeToLeadUpdates]);

  useEffect(() => {
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

  useEffect(() => {
    if (isConnected) {
      subscribeToCampaignUpdates();
    }
  }, [isConnected, subscribeToCampaignUpdates]);

  useEffect(() => {
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

  useEffect(() => {
    if (isConnected) {
      subscribeToSystemStatus();
    }
  }, [isConnected, subscribeToSystemStatus]);

  useEffect(() => {
    if (lastMessage && lastMessage.topic === 'system_status') {
      callback(lastMessage);
    }
  }, [lastMessage, callback]);
}

/**
 * Hook to subscribe to agent updates
 */
export function useAgentUpdates(callback: (message: WebSocketMessage) => void) {
  const { lastMessage, subscribeToAgentUpdates, isConnected } = useWebSocketContext();

  useEffect(() => {
    if (isConnected) {
      subscribeToAgentUpdates();
    }
  }, [isConnected, subscribeToAgentUpdates]);

  useEffect(() => {
    if (lastMessage && lastMessage.topic === 'agent_updates') {
      callback(lastMessage);
    }
  }, [lastMessage, callback]);
}

