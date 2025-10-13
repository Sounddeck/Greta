/**
 * Hook API Client for CPAS4/Greta
 * Connects to Hook Service backend (Week 19-20)
 */

const API_BASE = 'http://localhost:8000/api/v1/hooks';

export interface Hook {
  hook_id: string;
  event_type: string;
  hook_type: 'pre' | 'post';
  priority: number;
  condition?: string;
  enabled: boolean;
  created_at: string;
}

export interface Plugin {
  plugin_id: string;
  name: string;
  description?: string;
  enabled: boolean;
  hooks_count: number;
  created_at: string;
}

export interface Middleware {
  middleware_id: string;
  priority: number;
  enabled: boolean;
}

export interface HookStats {
  total_hooks: number;
  total_plugins: number;
  total_events: number;
  total_executions: number;
  avg_execution_time: number;
  hooks_by_event: Record<string, number>;
  executions_by_event: Record<string, number>;
  failed_executions: number;
}

export interface EventEmit {
  event_type: string;
  event_data: Record<string, any>;
  context?: Record<string, any>;
}

// ============================================================================
// Hook Registration API
// ============================================================================

export async function registerHook(hook: {
  hook_id: string;
  event_type: string;
  hook_type?: 'pre' | 'post';
  priority?: number;
  condition?: string;
}): Promise<Hook> {
  const response = await fetch(`${API_BASE}/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(hook),
  });

  if (!response.ok) {
    throw new Error(`Hook registration failed: ${response.statusText}`);
  }

  return response.json();
}

export async function unregisterHook(hookId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/unregister/${hookId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Hook unregistration failed: ${response.statusText}`);
  }
}

export async function listHooks(filters?: {
  event_type?: string;
  hook_type?: 'pre' | 'post';
  enabled_only?: boolean;
}): Promise<Hook[]> {
  const params = new URLSearchParams();
  if (filters?.event_type) params.append('event_type', filters.event_type);
  if (filters?.hook_type) params.append('hook_type', filters.hook_type);
  if (filters?.enabled_only !== undefined) params.append('enabled_only', filters.enabled_only.toString());

  const url = params.toString() ? `${API_BASE}/list?${params}` : `${API_BASE}/list`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Hook listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getHook(hookId: string): Promise<Hook> {
  const response = await fetch(`${API_BASE}/${hookId}`);

  if (!response.ok) {
    throw new Error(`Hook retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function updateHook(
  hookId: string,
  updates: {
    priority?: number;
    condition?: string;
    enabled?: boolean;
  }
): Promise<Hook> {
  const response = await fetch(`${API_BASE}/${hookId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    throw new Error(`Hook update failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Hook Execution API
// ============================================================================

export async function emitEvent(event: EventEmit): Promise<{
  event_type: string;
  hooks_executed: number;
  modified_data: any;
  context: Record<string, any>;
}> {
  const response = await fetch(`${API_BASE}/emit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(event),
  });

  if (!response.ok) {
    throw new Error(`Event emission failed: ${response.statusText}`);
  }

  return response.json();
}

export async function testHook(
  hookId: string,
  testData: Record<string, any>
): Promise<{
  hook_id: string;
  success: boolean;
  output: any;
  execution_time: number;
  error?: string;
}> {
  const response = await fetch(`${API_BASE}/test/${hookId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(testData),
  });

  if (!response.ok) {
    throw new Error(`Hook test failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Plugin Management API
// ============================================================================

export async function registerPlugin(plugin: {
  plugin_id: string;
  name: string;
  description?: string;
  enabled?: boolean;
}): Promise<Plugin> {
  const response = await fetch(`${API_BASE}/plugins`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(plugin),
  });

  if (!response.ok) {
    throw new Error(`Plugin registration failed: ${response.statusText}`);
  }

  return response.json();
}

export async function listPlugins(enabledOnly?: boolean): Promise<Plugin[]> {
  const url = enabledOnly !== undefined
    ? `${API_BASE}/plugins?enabled_only=${enabledOnly}`
    : `${API_BASE}/plugins`;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Plugin listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getPlugin(pluginId: string): Promise<Plugin & { hooks: string[] }> {
  const response = await fetch(`${API_BASE}/plugins/${pluginId}`);

  if (!response.ok) {
    throw new Error(`Plugin retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function enablePlugin(pluginId: string): Promise<{ enabled: boolean }> {
  const response = await fetch(`${API_BASE}/plugins/${pluginId}/enable`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Plugin enable failed: ${response.statusText}`);
  }

  return response.json();
}

export async function disablePlugin(pluginId: string): Promise<{ disabled: boolean }> {
  const response = await fetch(`${API_BASE}/plugins/${pluginId}/disable`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Plugin disable failed: ${response.statusText}`);
  }

  return response.json();
}

export async function unregisterPlugin(pluginId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/plugins/${pluginId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Plugin unregistration failed: ${response.statusText}`);
  }
}

// ============================================================================
// Middleware Management API
// ============================================================================

export async function addMiddleware(middleware: {
  middleware_id: string;
  name: string;
  priority?: number;
}): Promise<Middleware & { name: string }> {
  const response = await fetch(`${API_BASE}/middleware`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(middleware),
  });

  if (!response.ok) {
    throw new Error(`Middleware add failed: ${response.statusText}`);
  }

  return response.json();
}

export async function listMiddleware(): Promise<Middleware[]> {
  const response = await fetch(`${API_BASE}/middleware`);

  if (!response.ok) {
    throw new Error(`Middleware listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function removeMiddleware(middlewareId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/middleware/${middlewareId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Middleware removal failed: ${response.statusText}`);
  }
}

// ============================================================================
// Statistics & Monitoring API
// ============================================================================

export async function getHookStats(): Promise<HookStats> {
  const response = await fetch(`${API_BASE}/stats`);

  if (!response.ok) {
    throw new Error(`Stats retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function resetHookStats(): Promise<void> {
  const response = await fetch(`${API_BASE}/stats/reset`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Stats reset failed: ${response.statusText}`);
  }
}

export async function getEventTypes(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/events`);

  if (!response.ok) {
    throw new Error(`Event types retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getHookHealth(): Promise<{
  status: string;
  total_hooks: number;
  enabled_hooks: number;
  total_plugins: number;
  enabled_plugins: number;
  uptime: number;
}> {
  const response = await fetch(`${API_BASE}/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }

  return response.json();
}

