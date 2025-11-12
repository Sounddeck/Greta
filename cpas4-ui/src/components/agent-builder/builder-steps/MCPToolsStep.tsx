// @ts-nocheck
import { useState, useEffect } from 'react';
import { Search, Plus, Trash2, Settings2, Server, Wrench, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import { api } from '../../../lib/api';

interface MCPToolsStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation?: any;
}

export function MCPToolsStep({ draft, updateDraft, validation }: MCPToolsStepProps) {
  const [mcpServers, setMcpServers] = useState<any[]>([]);
  const [mcpTools, setMcpTools] = useState<any[]>([]);
  const [selectedServer, setSelectedServer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const selectedMcpTools = draft.mcp_tools || [];

  // Load MCP servers and tools
  useEffect(() => {
    loadMcpServers();
  }, []);

  useEffect(() => {
    if (selectedServer) {
      loadMcpTools(selectedServer);
    }
  }, [selectedServer]);

  const loadMcpServers = async () => {
    setLoading(true);
    try {
      const response = await api.get('/mcp/servers');
      // Handle API response structure: {success: true, servers: [], count: 0}
      const servers = response.data?.servers || response.data || [];
      // Ensure servers is always an array
      setMcpServers(Array.isArray(servers) ? servers : []);
    } catch (error) {
      console.error('Failed to load MCP servers:', error);
      toast.error('Failed to load MCP servers');
      setMcpServers([]); // Set empty array on error
    } finally {
      setLoading(false);
    }
  };

  const loadMcpTools = async (serverId: string) => {
    setLoading(true);
    try {
      const response = await api.get(`/mcp/servers/${serverId}/tools`);
      // Handle API response - could be array or object with tools property
      const tools = response.data?.tools || response.data || [];
      // Ensure tools is always an array
      setMcpTools(Array.isArray(tools) ? tools : []);
    } catch (error) {
      console.error('Failed to load MCP tools:', error);
      toast.error('Failed to load MCP tools');
      setMcpTools([]); // Set empty array on error
    } finally {
      setLoading(false);
    }
  };

  const addMcpTool = (tool: any) => {
    const isAlreadyAdded = selectedMcpTools.some(
      (t: any) => t.tool_id === tool.tool_id && t.server_id === selectedServer
    );

    if (isAlreadyAdded) {
      toast.error('Tool already added');
      return;
    }

    const newTool = {
      tool_id: tool.tool_id,
      server_id: selectedServer,
      configuration: {},
      is_enabled: true,
    };

    updateDraft({
      mcp_tools: [...selectedMcpTools, newTool],
    });

    toast.success(`Added ${tool.name}`);
  };

  const removeMcpTool = (toolId: string, serverId: string) => {
    updateDraft({
      mcp_tools: selectedMcpTools.filter(
        (t: any) => !(t.tool_id === toolId && t.server_id === serverId)
      ),
    });
    toast.success('Tool removed');
  };

  const toggleToolEnabled = (toolId: string, serverId: string) => {
    updateDraft({
      mcp_tools: selectedMcpTools.map((t: any) =>
        t.tool_id === toolId && t.server_id === serverId
          ? { ...t, is_enabled: !t.is_enabled }
          : t
      ),
    });
  };

  // Ensure mcpTools is an array before filtering
  const safeTools = Array.isArray(mcpTools) ? mcpTools : [];
  const filteredTools = safeTools.filter((tool) =>
    tool.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    tool.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">MCP Tools</h2>
        <p className="text-muted-foreground">
          Select MCP (Model Context Protocol) tools to enhance your agent's capabilities.
        </p>
      </div>

      {/* Server Selection */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">
          Select MCP Server
        </label>
        <select
          value={selectedServer || ''}
          onChange={(e) => setSelectedServer(e.target.value || null)}
          className="w-full px-3 py-2 border border-border rounded-md bg-background"
          disabled={loading || !Array.isArray(mcpServers) || mcpServers.length === 0}
        >
          <option value="">-- Select a server --</option>
          {Array.isArray(mcpServers) && mcpServers.map((server) => (
            <option key={server.server_id} value={server.server_id}>
              {server.name} - {server.description}
            </option>
          ))}
        </select>
        {Array.isArray(mcpServers) && mcpServers.length === 0 && !loading && (
          <p className="text-sm text-yellow-600">
            No MCP servers available. Please configure MCP servers first.
          </p>
        )}
      </div>

      {/* Available Tools */}
      {selectedServer && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Available Tools</h3>
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search tools..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-3 py-2 border border-border rounded-md bg-background text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-96 overflow-y-auto">
            {loading ? (
              <div className="col-span-2 text-center py-8 text-muted-foreground">
                Loading tools...
              </div>
            ) : filteredTools.length === 0 ? (
              <div className="col-span-2 text-center py-8 text-muted-foreground">
                No tools found
              </div>
            ) : (
              filteredTools.map((tool) => {
                const isAdded = selectedMcpTools.some(
                  (t: any) => t.tool_id === tool.tool_id && t.server_id === selectedServer
                );

                return (
                  <div
                    key={tool.tool_id}
                    className={`p-4 border rounded-lg ${
                      isAdded
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:border-primary/50'
                    } transition-colors`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Wrench className="h-4 w-4 text-primary" />
                        <h4 className="font-medium">{tool.name}</h4>
                      </div>
                      <button
                        onClick={() => isAdded ? removeMcpTool(tool.tool_id, selectedServer) : addMcpTool(tool)}
                        className={`p-1 rounded ${
                          isAdded
                            ? 'text-destructive hover:bg-destructive/10'
                            : 'text-primary hover:bg-primary/10'
                        }`}
                      >
                        {isAdded ? <Trash2 className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
                      </button>
                    </div>
                    <p className="text-sm text-muted-foreground">{tool.description}</p>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}

      {/* Selected Tools */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">
          Selected Tools ({selectedMcpTools.length})
        </h3>

        {selectedMcpTools.length === 0 ? (
          <div className="text-center py-8 border border-dashed border-border rounded-lg">
            <Wrench className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
            <p className="text-muted-foreground">No MCP tools selected</p>
            <p className="text-sm text-muted-foreground mt-1">
              Select a server and add tools to enhance your agent
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {selectedMcpTools.map((tool: any) => {
              const safeMcpServers = Array.isArray(mcpServers) ? mcpServers : [];
              const safeMcpTools = Array.isArray(mcpTools) ? mcpTools : [];
              const server = safeMcpServers.find((s) => s.server_id === tool.server_id);
              const toolInfo = safeMcpTools.find((t) => t.tool_id === tool.tool_id);

              return (
                <div
                  key={`${tool.server_id}-${tool.tool_id}`}
                  className="flex items-center justify-between p-3 border border-border rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={tool.is_enabled}
                      onChange={() => toggleToolEnabled(tool.tool_id, tool.server_id)}
                      className="h-4 w-4"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{toolInfo?.name || tool.tool_id}</span>
                        <span className="text-xs text-muted-foreground">
                          ({server?.name || tool.server_id})
                        </span>
                      </div>
                      {toolInfo?.description && (
                        <p className="text-sm text-muted-foreground">{toolInfo.description}</p>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => removeMcpTool(tool.tool_id, tool.server_id)}
                    className="p-2 text-destructive hover:bg-destructive/10 rounded"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Validation Messages */}
      {validation?.warnings?.find((w: any) => w.field === 'mcp_tools') && (
        <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
          <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
          <p className="text-sm text-yellow-600">
            {validation.warnings.find((w: any) => w.field === 'mcp_tools').message}
          </p>
        </div>
      )}
    </div>
  );
}

export default MCPToolsStep;

