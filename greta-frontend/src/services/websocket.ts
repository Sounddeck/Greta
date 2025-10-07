import { apiService } from './api';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface SystemUpdate {
  type: 'system_update';
  memory_usage: string;
  active_tasks: number;
  timestamp: string;
}

export interface AgentUpdate {
  type: 'agent_update';
  agent_id: string;
  status: 'active' | 'inactive' | 'error';
  message: string;
  timestamp: string;
}

export interface ChatMessage {
  type: 'chat_message';
  message: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

export type WebSocketEvent = SystemUpdate | AgentUpdate | ChatMessage;

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private isConnecting = false;

  constructor() {
    this.connect();
  }

  private connect() {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isConnecting = true;

    try {
      // Try to connect to the live updates WebSocket endpoint
      this.ws = apiService.createWebSocket('/api/web-commands/ws/live-updates');
      
      if (!this.ws) {
        this.isConnecting = false;
        return;
      }

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.emit('connection', { status: 'connected' });
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.isConnecting = false;
        this.ws = null;
        this.emit('connection', { status: 'disconnected' });
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
        this.emit('connection', { status: 'error', error });
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      console.log('Max reconnection attempts reached');
      this.emit('connection', { status: 'failed' });
    }
  }

  private handleMessage(message: WebSocketMessage) {
    // Emit the message to all listeners for this message type
    this.emit(message.type, message.data);
    
    // Also emit to general message listeners
    this.emit('message', message);
  }

  private emit(event: string, data: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in WebSocket listener for ${event}:`, error);
        }
      });
    }
  }

  public on(event: string, listener: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);

    // Return unsubscribe function
    return () => {
      const eventListeners = this.listeners.get(event);
      if (eventListeners) {
        eventListeners.delete(listener);
        if (eventListeners.size === 0) {
          this.listeners.delete(event);
        }
      }
    };
  }

  public off(event: string, listener: (data: any) => void) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  public send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }

  public disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
  }

  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  public getConnectionState(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'disconnected';
      default: return 'unknown';
    }
  }
}

// Create and export singleton instance
export const webSocketService = new WebSocketService();

// React hook for using WebSocket in components
export const useWebSocket = (event: string, handler: (data: any) => void) => {
  React.useEffect(() => {
    const unsubscribe = webSocketService.on(event, handler);
    return unsubscribe;
  }, [event, handler]);
};

// React hook for WebSocket connection status
export const useWebSocketConnection = () => {
  const [connectionStatus, setConnectionStatus] = React.useState(
    webSocketService.getConnectionState()
  );

  React.useEffect(() => {
    const unsubscribe = webSocketService.on('connection', (data) => {
      setConnectionStatus(data.status);
    });

    return unsubscribe;
  }, []);

  return {
    status: connectionStatus,
    isConnected: connectionStatus === 'connected',
    send: webSocketService.send.bind(webSocketService),
    disconnect: webSocketService.disconnect.bind(webSocketService),
  };
};

// Import React for hooks
import React from 'react';

export default webSocketService;
