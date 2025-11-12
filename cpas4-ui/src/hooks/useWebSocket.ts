/**
 * WebSocket Hook
 * 
 * Custom React hook for WebSocket connections with automatic reconnection,
 * topic subscription, and message handling.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  data: any;
  source?: string;
  event_id?: string;
  timestamp?: string;
  topic?: string;
}

export interface WebSocketOptions {
  /** Automatically connect on mount */
  autoConnect?: boolean;
  /** Automatically reconnect on disconnect */
  autoReconnect?: boolean;
  /** Reconnection delay in milliseconds */
  reconnectDelay?: number;
  /** Maximum reconnection attempts (0 = infinite) */
  maxReconnectAttempts?: number;
  /** Topics to subscribe to on connection */
  initialTopics?: string[];
  /** Callback for connection established */
  onConnect?: (connectionId: string) => void;
  /** Callback for disconnection */
  onDisconnect?: () => void;
  /** Callback for errors */
  onError?: (error: Event) => void;
  /** Callback for all messages */
  onMessage?: (message: WebSocketMessage) => void;
}

export interface UseWebSocketReturn {
  /** Whether the WebSocket is connected */
  isConnected: boolean;
  /** Connection ID from server */
  connectionId: string | null;
  /** Last received message */
  lastMessage: WebSocketMessage | null;
  /** Send a message to the server */
  send: (message: any) => void;
  /** Subscribe to a topic */
  subscribe: (topic: string) => void;
  /** Unsubscribe from a topic */
  unsubscribe: (topic: string) => void;
  /** Manually connect */
  connect: () => void;
  /** Manually disconnect */
  disconnect: () => void;
  /** Current subscribed topics */
  subscribedTopics: Set<string>;
  /** Reconnection attempt count */
  reconnectAttempts: number;
}

export function useWebSocket(
  url?: string,
  options: WebSocketOptions = {}
): UseWebSocketReturn {
  const {
    autoConnect = true,
    autoReconnect = true,
    reconnectDelay = 3000,
    maxReconnectAttempts = 0,
    initialTopics = [],
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [subscribedTopics, setSubscribedTopics] = useState<Set<string>>(new Set(initialTopics));
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef(true);

  // Get WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    if (url) return url;

    // Default to backend WebSocket endpoint
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
    const wsUrl = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');
    return `${wsUrl}/ws`;
  }, [url]);

  // Send message
  const send = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }, []);

  // Subscribe to topic
  const subscribe = useCallback((topic: string) => {
    setSubscribedTopics(prev => new Set(prev).add(topic));
    send({ type: 'subscribe', topic });
  }, [send]);

  // Unsubscribe from topic
  const unsubscribe = useCallback((topic: string) => {
    setSubscribedTopics(prev => {
      const newSet = new Set(prev);
      newSet.delete(topic);
      return newSet;
    });
    send({ type: 'unsubscribe', topic });
  }, [send]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    try {
      const wsUrl = getWebSocketUrl();
      console.log('Connecting to WebSocket:', wsUrl);

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setReconnectAttempts(0);

        // Subscribe to initial topics
        initialTopics.forEach(topic => {
          send({ type: 'subscribe', topic });
        });
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);

          // Handle connection established
          if (message.type === 'connection_established') {
            const connId = message.data?.connection_id;
            setConnectionId(connId);
            onConnect?.(connId);
          }

          // Call message callback
          onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setConnectionId(null);
        wsRef.current = null;
        onDisconnect?.();

        // Auto-reconnect
        if (shouldReconnectRef.current && autoReconnect) {
          if (maxReconnectAttempts === 0 || reconnectAttempts < maxReconnectAttempts) {
            console.log(`Reconnecting in ${reconnectDelay}ms... (attempt ${reconnectAttempts + 1})`);
            reconnectTimeoutRef.current = setTimeout(() => {
              setReconnectAttempts(prev => prev + 1);
              connect();
            }, reconnectDelay);
          } else {
            console.log('Max reconnection attempts reached');
          }
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, [
    getWebSocketUrl,
    autoReconnect,
    reconnectDelay,
    maxReconnectAttempts,
    reconnectAttempts,
    initialTopics,
    send,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  ]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setConnectionId(null);
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      shouldReconnectRef.current = true;
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect]); // Only run on mount/unmount

  return {
    isConnected,
    connectionId,
    lastMessage,
    send,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
    subscribedTopics,
    reconnectAttempts,
  };
}

