import React, { useEffect, useState } from 'react';
import { Server, Plus, Trash2, Check, X, Settings } from 'lucide-react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { toast } from 'sonner';
import api from '../../lib/api';

interface MCPServer {
  server_id: string;
  name: string;
  description: string;
  status: string;
  tool_count: number;
  tools?: string[];
}

interface MCPIntegration {
  integration_id: string;
  agent_id: string;
  server_id: string;
  enabled: boolean;
  tool_whitelist: string[];
  tool_blacklist: string[];
  priority: number;
}

interface MCPConfigurationStepProps {
  agentId: string;
  onComplete?: () => void;
}

export default function MCPConfigurationStep({ agentId, onComplete }: MCPConfigurationStepProps) {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [integrations, setIntegrations] = useState<MCPIntegration[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedServer, setSelectedServer] = useState<string | null>(null);
  const [toolWhitelist, setToolWhitelist] = useState<string[]>([]);
  const [priority, setPriority] = useState(50);

  useEffect(() => {
    loadData();
  }, [agentId]);

  const loadData = async () => {
    setLoading(true);
    try {
      // Load available MCP servers
      const serversResponse = await api.get('/mcp/servers', { params: { limit: 100 } });
      setServers(serversResponse.data || []);

      // Load existing integrations for this agent
      const integrationsResponse = await api.get(`/agent-builder/agents/${agentId}/mcp-config`);
      setIntegrations(integrationsResponse.data || []);
    } catch (error: any) {
      console.error('Error loading MCP data:', error);
      toast.error('Failed to load MCP configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleAddIntegration = async () => {
    if (!selectedServer) {
      toast.error('Please select an MCP server');
      return;
    }

    try {
      const response = await api.post(`/agent-builder/agents/${agentId}/mcp-config`, {
        agent_id: agentId,
        server_id: selectedServer,
        enabled: true,
        tool_whitelist: toolWhitelist,
        tool_blacklist: [],
        priority: priority,
      });

      setIntegrations([...integrations, response.data]);
      setSelectedServer(null);
      setToolWhitelist([]);
      setPriority(50);
      toast.success('MCP server added to agent');
    } catch (error: any) {
      console.error('Error adding integration:', error);
      toast.error('Failed to add MCP server');
    }
  };

  const handleRemoveIntegration = async (integrationId: string) => {
    try {
      await api.delete(`/agent-builder/agents/${agentId}/mcp-config/${integrationId}`);
      setIntegrations(integrations.filter((i) => i.integration_id !== integrationId));
      toast.success('MCP server removed from agent');
    } catch (error: any) {
      console.error('Error removing integration:', error);
      toast.error('Failed to remove MCP server');
    }
  };

  const handleToggleIntegration = async (integration: MCPIntegration) => {
    try {
      const response = await api.put(
        `/agent-builder/agents/${agentId}/mcp-config/${integration.integration_id}`,
        { enabled: !integration.enabled }
      );
      setIntegrations(
        integrations.map((i) =>
          i.integration_id === integration.integration_id ? response.data : i
        )
      );
      toast.success(`MCP server ${integration.enabled ? 'disabled' : 'enabled'}`);
    } catch (error: any) {
      console.error('Error toggling integration:', error);
      toast.error('Failed to update integration');
    }
  };

  const getServerName = (serverId: string) => {
    const server = servers.find((s) => s.server_id === serverId);
    return server?.name || serverId;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading MCP configuration...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">MCP Server Configuration</h2>
        <p className="text-muted-foreground">
          Configure which MCP servers and tools this agent can access
        </p>
      </div>

      {/* Current Integrations */}
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-3">
          Connected MCP Servers ({integrations.length})
        </h3>
        {integrations.length === 0 ? (
          <div className="bg-muted/50 border border-border rounded-lg p-6 text-center">
            <Server className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
            <p className="text-muted-foreground">No MCP servers connected yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            {integrations.map((integration) => (
              <div
                key={integration.integration_id}
                className="bg-card border border-border rounded-lg p-4 flex items-center justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <Server className="h-5 w-5 text-primary" />
                    <div>
                      <h4 className="font-semibold text-foreground">
                        {getServerName(integration.server_id)}
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        Priority: {integration.priority} | 
                        {integration.tool_whitelist.length > 0
                          ? ` ${integration.tool_whitelist.length} tools whitelisted`
                          : ' All tools enabled'}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant={integration.enabled ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => handleToggleIntegration(integration)}
                  >
                    {integration.enabled ? (
                      <>
                        <Check className="h-4 w-4 mr-2" />
                        Enabled
                      </>
                    ) : (
                      <>
                        <X className="h-4 w-4 mr-2" />
                        Disabled
                      </>
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveIntegration(integration.integration_id)}
                  >
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add New Integration */}
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-3">Add MCP Server</h3>
        <div className="bg-card border border-border rounded-lg p-4 space-y-4">
          {/* Server Selection */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Select MCP Server
            </label>
            <select
              value={selectedServer || ''}
              onChange={(e) => setSelectedServer(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="">Choose a server...</option>
              {servers
                .filter((s) => !integrations.some((i) => i.server_id === s.server_id))
                .map((server) => (
                  <option key={server.server_id} value={server.server_id}>
                    {server.name} ({server.tool_count} tools)
                  </option>
                ))}
            </select>
          </div>

          {/* Priority */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Priority (0-100)
            </label>
            <Input
              type="number"
              min="0"
              max="100"
              value={priority}
              onChange={(e) => setPriority(parseInt(e.target.value) || 0)}
              placeholder="50"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Higher priority servers are used first
            </p>
          </div>

          {/* Tool Whitelist (Optional) */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Tool Whitelist (Optional)
            </label>
            <Input
              type="text"
              placeholder="tool1, tool2, tool3 (leave empty for all tools)"
              value={toolWhitelist.join(', ')}
              onChange={(e) =>
                setToolWhitelist(
                  e.target.value
                    .split(',')
                    .map((t) => t.trim())
                    .filter((t) => t)
                )
              }
            />
            <p className="text-xs text-muted-foreground mt-1">
              Comma-separated list of tool names. Leave empty to allow all tools.
            </p>
          </div>

          {/* Add Button */}
          <Button onClick={handleAddIntegration} className="w-full" disabled={!selectedServer}>
            <Plus className="h-4 w-4 mr-2" />
            Add MCP Server
          </Button>
        </div>
      </div>

      {/* Complete Button */}
      {onComplete && (
        <div className="flex justify-end pt-4 border-t border-border">
          <Button onClick={onComplete} size="lg">
            Continue
          </Button>
        </div>
      )}
    </div>
  );
}

