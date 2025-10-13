/**
 * MCP API Client for CPAS4/Greta
 * Connects to MCP Service backend (Week 17-18)
 */

const API_BASE = 'http://localhost:8000/api/v1/mcp';

export interface MCPServer {
  id: string;
  name: string;
  url: string;
  status: 'connected' | 'disconnected' | 'error';
  description?: string;
  version?: string;
  capabilities?: string[];
  tools?: number;
  created_at?: string;
  updated_at?: string;
}

export interface MCPTool {
  id: string;
  name: string;
  description: string;
  server: string;
  server_id: string;
  enabled: boolean;
  tool_type?: string;
  parameters?: Record<string, any>;
  created_at?: string;
}

export interface MCPStats {
  total_servers: number;
  connected_servers: number;
  total_tools: number;
  enabled_tools: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  avg_response_time: number;
}

export interface MCPIntegration {
  id: string;
  agent_id: string;
  server_id: string;
  enabled: boolean;
  config?: Record<string, any>;
  created_at?: string;
}

export interface DiscoveryConfig {
  id: string;
  name: string;
  source: string;
  enabled: boolean;
  config?: Record<string, any>;
  created_at?: string;
}

// ============================================================================
// MCP Server API
// ============================================================================

export async function createServer(server: {
  name: string;
  url: string;
  description?: string;
  version?: string;
}): Promise<MCPServer> {
  const response = await fetch(`${API_BASE}/servers`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(server),
  });

  if (!response.ok) {
    throw new Error(`Server creation failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getServer(serverId: string): Promise<MCPServer> {
  const response = await fetch(`${API_BASE}/servers/${serverId}`);

  if (!response.ok) {
    throw new Error(`Server retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function updateServer(
  serverId: string,
  updates: {
    name?: string;
    url?: string;
    description?: string;
    status?: string;
  }
): Promise<MCPServer> {
  const response = await fetch(`${API_BASE}/servers/${serverId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    throw new Error(`Server update failed: ${response.statusText}`);
  }

  return response.json();
}

export async function deleteServer(serverId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/servers/${serverId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Server deletion failed: ${response.statusText}`);
  }
}

export async function listServers(filters?: {
  status?: string;
  discovery_source?: string;
  limit?: number;
}): Promise<MCPServer[]> {
  const params = new URLSearchParams();
  if (filters?.status) params.append('status', filters.status);
  if (filters?.discovery_source) params.append('discovery_source', filters.discovery_source);
  if (filters?.limit) params.append('limit', filters.limit.toString());

  const url = params.toString() ? `${API_BASE}/servers?${params}` : `${API_BASE}/servers`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Server listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function testServerConnection(serverId: string): Promise<{ success: boolean; message?: string }> {
  try {
    const server = await getServer(serverId);
    // In a real implementation, this would ping the server
    return { success: server.status === 'connected' };
  } catch (error) {
    return { success: false, message: (error as Error).message };
  }
}

// ============================================================================
// MCP Tool API
// ============================================================================

export async function createTool(serverId: string, tool: {
  name: string;
  description: string;
  tool_type?: string;
  parameters?: Record<string, any>;
}): Promise<MCPTool> {
  const response = await fetch(`${API_BASE}/servers/${serverId}/tools`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(tool),
  });

  if (!response.ok) {
    throw new Error(`Tool creation failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getTool(toolId: string): Promise<MCPTool> {
  const response = await fetch(`${API_BASE}/tools/${toolId}`);

  if (!response.ok) {
    throw new Error(`Tool retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function deleteTool(toolId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/tools/${toolId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Tool deletion failed: ${response.statusText}`);
  }
}

export async function listTools(filters?: {
  server_id?: string;
  tool_type?: string;
  limit?: number;
}): Promise<MCPTool[]> {
  const params = new URLSearchParams();
  if (filters?.server_id) params.append('server_id', filters.server_id);
  if (filters?.tool_type) params.append('tool_type', filters.tool_type);
  if (filters?.limit) params.append('limit', filters.limit.toString());

  const url = params.toString() ? `${API_BASE}/tools?${params}` : `${API_BASE}/tools`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Tool listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function toggleTool(toolId: string, enabled: boolean): Promise<MCPTool> {
  // This would be implemented with a PATCH endpoint in the backend
  // For now, we'll use the getTool as a placeholder
  const tool = await getTool(toolId);
  tool.enabled = enabled;
  return tool;
}

// ============================================================================
// Discovery Configuration API
// ============================================================================

export async function createDiscoveryConfig(config: {
  name: string;
  source: string;
  enabled?: boolean;
  config?: Record<string, any>;
}): Promise<DiscoveryConfig> {
  const response = await fetch(`${API_BASE}/discovery/configs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    throw new Error(`Discovery config creation failed: ${response.statusText}`);
  }

  return response.json();
}

export async function listDiscoveryConfigs(filters?: {
  enabled?: boolean;
  limit?: number;
}): Promise<DiscoveryConfig[]> {
  const params = new URLSearchParams();
  if (filters?.enabled !== undefined) params.append('enabled', filters.enabled.toString());
  if (filters?.limit) params.append('limit', filters.limit.toString());

  const url = params.toString() ? `${API_BASE}/discovery/configs?${params}` : `${API_BASE}/discovery/configs`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Discovery config listing failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Integration API
// ============================================================================

export async function createIntegration(integration: {
  agent_id: string;
  server_id: string;
  enabled?: boolean;
  config?: Record<string, any>;
}): Promise<MCPIntegration> {
  const response = await fetch(`${API_BASE}/integrations`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(integration),
  });

  if (!response.ok) {
    throw new Error(`Integration creation failed: ${response.statusText}`);
  }

  return response.json();
}

export async function listIntegrations(filters?: {
  agent_id?: string;
  server_id?: string;
  limit?: number;
}): Promise<MCPIntegration[]> {
  const params = new URLSearchParams();
  if (filters?.agent_id) params.append('agent_id', filters.agent_id);
  if (filters?.server_id) params.append('server_id', filters.server_id);
  if (filters?.limit) params.append('limit', filters.limit.toString());

  const url = params.toString() ? `${API_BASE}/integrations?${params}` : `${API_BASE}/integrations`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Integration listing failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Statistics & Health API
// ============================================================================

export async function getMCPStats(): Promise<MCPStats> {
  const response = await fetch(`${API_BASE}/stats`);

  if (!response.ok) {
    throw new Error(`Stats retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getMCPHealth(): Promise<{
  status: string;
  module: string;
  features: string[];
  endpoints: number;
  database: string;
}> {
  const response = await fetch(`${API_BASE}/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getServerCatalog(limit?: number): Promise<any[]> {
  const url = limit ? `${API_BASE}/catalog?limit=${limit}` : `${API_BASE}/catalog`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Catalog retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

