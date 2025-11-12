import { invoke } from '@tauri-apps/api/core';

export interface NotificationPayload {
  title: string;
  body: string;
  icon?: string;
}

/**
 * Send a custom notification
 */
export async function sendNotification(payload: NotificationPayload): Promise<void> {
  try {
    await invoke('send_notification', { payload });
  } catch (error) {
    console.error('Failed to send notification:', error);
  }
}

/**
 * Request notification permission (mainly for cross-platform compatibility)
 */
export async function requestNotificationPermission(): Promise<boolean> {
  try {
    return await invoke('request_notification_permission');
  } catch (error) {
    console.error('Failed to request notification permission:', error);
    return false;
  }
}

/**
 * Send notification when voice recognition is complete
 */
export async function notifyVoiceComplete(text: string): Promise<void> {
  try {
    await invoke('notify_voice_complete', { text });
  } catch (error) {
    console.error('Failed to send voice complete notification:', error);
  }
}

/**
 * Send notification for MCP server status change
 */
export async function notifyMCPStatus(serverName: string, status: string): Promise<void> {
  try {
    await invoke('notify_mcp_status', { serverName, status });
  } catch (error) {
    console.error('Failed to send MCP status notification:', error);
  }
}

/**
 * Send notification when a hook is executed
 */
export async function notifyHookExecuted(hookName: string, success: boolean): Promise<void> {
  try {
    await invoke('notify_hook_executed', { hookName, success });
  } catch (error) {
    console.error('Failed to send hook executed notification:', error);
  }
}

/**
 * Check if running in Tauri environment
 */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI__' in window;
}
