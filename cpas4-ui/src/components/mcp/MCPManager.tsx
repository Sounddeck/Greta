import { useEffect, useState } from 'react';
import { Server, Tool, Plus, Trash2, TestTube, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { Button } from '../ui/button';
import { useMCPStore } from '../../stores/mcpStore';

export function MCPManager() {
  // Use Zustand store
  const {
    servers,
    tools,
    isLoading,
    isExecuting,
    fetchServers,
    fetchTools,
    createServer,
    deleteServer,
    testConnection,
    executeTool,
  } = useMCPStore();

  // Local state for form
  const [serverName, setServerName] = useState('');
  const [serverUrl, setServerUrl] = useState('');
  const [serverDescription, setServerDescription] = useState('');
  const [selectedServerId, setSelectedServerId] = useState<string | null>(null);

  // Load servers on mount
  useEffect(() => {
    fetchServers();
  }, [fetchServers]);

  // Load tools when server is selected
  useEffect(() => {
    if (selectedServerId) {
      fetchTools(selectedServerId);
    }
  }, [selectedServerId, fetchTools]);

  const handleCreateServer = async () => {
    if (!serverName || !serverUrl) return;

    await createServer({
      name: serverName,
      url: serverUrl,
      description: serverDescription,
      isEnabled: true,
    });

    // Reset form
    setServerName('');
    setServerUrl('');
    setServerDescription('');
  };

  const handleDeleteServer = async (id: string) => {
    if (confirm('Are you sure you want to delete this server?')) {
      await deleteServer(id);
      if (selectedServerId === id) {
        setSelectedServerId(null);
      }
    }
  };

  const handleTestConnection = async (id: string) => {
    await testConnection(id);
  };

  if (isLoading && servers.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading MCP servers...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">MCP Server Manager</h2>
        <p className="text-muted-foreground">
          Manage Model Context Protocol servers and tools
        </p>
      </div>

      {/* Server List */}
      {servers.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold">Connected Servers</h3>
          <div className="grid gap-2">
            {servers.map((server) => (
              <div
                key={server.id}
                className={`p-3 border rounded-md cursor-pointer transition-colors ${
                  selectedServerId === server.id
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-primary/50'
                }`}
                onClick={() => setSelectedServerId(server.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Server className="h-4 w-4" />
                      <p className="font-medium">{server.name}</p>
                      {server.status === 'connected' && (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      )}
                      {server.status === 'error' && (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">{server.url}</p>
                    {server.description && (
                      <p className="text-sm text-muted-foreground mt-1">{server.description}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleTestConnection(server.id);
                      }}
                      disabled={isLoading}
                    >
                      <TestTube className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteServer(server.id);
                      }}
                      disabled={isLoading}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tools List */}
      {selectedServerId && tools.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold">Available Tools</h3>
          <div className="grid gap-2">
            {tools.map((tool) => (
              <div key={tool.id} className="p-3 bg-muted rounded-md">
                <div className="flex items-center gap-2">
                  <Tool className="h-4 w-4 text-primary" />
                  <p className="font-medium">{tool.name}</p>
                </div>
                {tool.description && (
                  <p className="text-sm text-muted-foreground mt-1">{tool.description}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Add New Server */}
      <div className="space-y-4 p-4 border border-border rounded-lg">
        <h3 className="font-semibold">Add New Server</h3>

        <div className="space-y-2">
          <label className="text-sm font-medium">Server Name</label>
          <input
            type="text"
            value={serverName}
            onChange={(e) => setServerName(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            placeholder="My MCP Server"
            disabled={isLoading}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Server URL</label>
          <input
            type="url"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            placeholder="http://localhost:3000"
            disabled={isLoading}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Description (Optional)</label>
          <textarea
            value={serverDescription}
            onChange={(e) => setServerDescription(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            placeholder="Server description..."
            rows={3}
            disabled={isLoading}
          />
        </div>

        <Button
          onClick={handleCreateServer}
          disabled={isLoading || !serverName || !serverUrl}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Adding...
            </>
          ) : (
            <>
              <Plus className="h-4 w-4 mr-2" />
              Add Server
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
