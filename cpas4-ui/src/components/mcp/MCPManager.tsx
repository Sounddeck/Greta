import { useEffect, useState } from 'react';
import { Server, Plus, Trash2, TestTube, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { Button } from '../ui/button';
import { useMCPStore } from '../../stores/mcpStore';
import { MCPServerSkeletonList } from '../skeletons/MCPServerSkeleton';

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
  } = useMCPStore();

  // Local state for form
  const [serverName, setServerName] = useState('');
  const [serverUrl, setServerUrl] = useState('');
  const [serverDescription, setServerDescription] = useState('');
  const [selectedServerId, __setSelectedServerId] = useState<string | null>(null);

  useEffect(() => {
    fetchServers();
  }, [fetchServers]);

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
    }
  };

  const handleTestConnection = async (id: string) => {
    await testConnection(id);
  };

  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">MCP Manager</h2>
        <p className="text-muted-foreground">
          Manage Model Context Protocol servers and tools
        </p>
      </div>

      {/* MCP Servers - Show skeleton during initial load */}
      {isLoading && servers.length === 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">MCP Servers</h3>
          <MCPServerSkeletonList count={3} />
        </div>
      ) : servers.length > 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">MCP Servers</h3>
          <div className="grid gap-3">
            {servers.map((server) => (
              <div key={server.id} className="p-4 border border-border rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Server className="h-5 w-5 text-primary" />
                      <p className="font-medium">{server.name}</p>
                      {server.status === 'connected' && (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      )}
                      {server.status === 'disconnected' && (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      {server.isEnabled && (
                        <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-600 rounded-full">
                          Enabled
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">{server.url}</p>
                    {server.description && (
                      <p className="text-sm text-muted-foreground mt-1">{server.description}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleTestConnection(server.id)}
                      disabled={isExecuting}
                    >
                      {isExecuting ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <TestTube className="h-4 w-4" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteServer(server.id)}
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
      ) : null}

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
            type="text"
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
            rows={2}
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

      {/* Available Tools */}
      {tools.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold">Available Tools</h3>
          <div className="grid md:grid-cols-2 gap-2">
            {tools.map((tool) => (
              <div key={tool.id} className="p-3 bg-muted rounded-md">
                <p className="font-medium">{tool.name}</p>
                {tool.description && (
                  <p className="text-sm text-muted-foreground mt-1">{tool.description}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
