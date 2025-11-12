import React, { useState, useEffect } from 'react';
import { Plus, Server, Trash2, Edit, CheckCircle, XCircle, RefreshCw, AlertCircle } from 'lucide-react';
import { mcpApi } from '../lib/api';
import { toast } from 'sonner';

interface MCPServer {
  id: string;
  name: string;
  description: string;
  base_url: string;
  api_key?: string;
  capabilities: string[];
  is_active: boolean;
  config: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

interface MCPServerFormData {
  name: string;
  description: string;
  base_url: string;
  api_key: string;
  capabilities: string;
  is_active: boolean;
  config: string;
}

export default function MCPServerManagement() {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingServer, setEditingServer] = useState<MCPServer | null>(null);
  const [testingServerId, setTestingServerId] = useState<string | null>(null);
  
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: '',
    description: '',
    base_url: '',
    api_key: '',
    capabilities: '',
    is_active: true,
    config: '{}',
  });

  useEffect(() => {
    loadServers();
  }, []);

  const loadServers = async () => {
    setLoading(true);
    try {
      const response = await mcpApi.getServers();
      const serverList = response?.servers || response || [];
      setServers(Array.isArray(serverList) ? serverList : []);
    } catch (error) {
      console.error('Failed to load MCP servers:', error);
      toast.error('Failed to load MCP servers');
      setServers([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      // Parse capabilities and config
      const capabilities = formData.capabilities
        .split(',')
        .map(c => c.trim())
        .filter(c => c.length > 0);
      
      let config = {};
      try {
        config = JSON.parse(formData.config);
      } catch (err) {
        toast.error('Invalid JSON in config field');
        return;
      }

      const serverData = {
        name: formData.name,
        description: formData.description,
        base_url: formData.base_url,
        api_key: formData.api_key || undefined,
        capabilities,
        is_active: formData.is_active,
        config,
      };

      if (editingServer) {
        await mcpApi.updateServer(editingServer.id, serverData);
        toast.success('MCP server updated successfully');
      } else {
        await mcpApi.createServer(serverData);
        toast.success('MCP server created successfully');
      }

      resetForm();
      loadServers();
    } catch (error: any) {
      console.error('Failed to save MCP server:', error);
      toast.error(error.response?.data?.detail || 'Failed to save MCP server');
    }
  };

  const handleEdit = (server: MCPServer) => {
    setEditingServer(server);
    setFormData({
      name: server.name,
      description: server.description,
      base_url: server.base_url,
      api_key: server.api_key || '',
      capabilities: server.capabilities.join(', '),
      is_active: server.is_active,
      config: JSON.stringify(server.config, null, 2),
    });
    setShowForm(true);
  };

  const handleDelete = async (serverId: string) => {
    if (!confirm('Are you sure you want to delete this MCP server?')) {
      return;
    }

    try {
      await mcpApi.deleteServer(serverId);
      toast.success('MCP server deleted successfully');
      loadServers();
    } catch (error) {
      console.error('Failed to delete MCP server:', error);
      toast.error('Failed to delete MCP server');
    }
  };

  const handleTestConnection = async (serverId: string) => {
    setTestingServerId(serverId);
    try {
      const result = await mcpApi.testConnection(serverId);
      if (result.success) {
        toast.success('Connection test successful!');
      } else {
        toast.error(`Connection test failed: ${result.message || 'Unknown error'}`);
      }
    } catch (error: any) {
      console.error('Connection test failed:', error);
      toast.error(error.response?.data?.detail || 'Connection test failed');
    } finally {
      setTestingServerId(null);
    }
  };

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      base_url: '',
      api_key: '',
      capabilities: '',
      is_active: true,
      config: '{}',
    });
    setEditingServer(null);
    setShowForm(false);
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Server className="h-8 w-8 text-primary" />
            MCP Server Management
          </h1>
          <p className="text-muted-foreground mt-1">
            Manage Model Context Protocol (MCP) servers and their configurations
          </p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
        >
          <Plus className="h-4 w-4" />
          Add MCP Server
        </button>
      </div>

      {/* Server Form */}
      {showForm && (
        <div className="bg-card border border-border rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">
            {editingServer ? 'Edit MCP Server' : 'Add New MCP Server'}
          </h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Server Name *</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background"
                  placeholder="My MCP Server"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Base URL *</label>
                <input
                  type="url"
                  value={formData.base_url}
                  onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background"
                  placeholder="http://localhost:9000"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Description</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded-lg bg-background"
                placeholder="Description of this MCP server"
                rows={2}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">API Key (Optional)</label>
              <input
                type="password"
                value={formData.api_key}
                onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded-lg bg-background"
                placeholder="Optional API key for authentication"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Capabilities (comma-separated)</label>
              <input
                type="text"
                value={formData.capabilities}
                onChange={(e) => setFormData({ ...formData, capabilities: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded-lg bg-background"
                placeholder="web_search, content_reading, data_analysis"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Configuration (JSON)</label>
              <textarea
                value={formData.config}
                onChange={(e) => setFormData({ ...formData, config: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded-lg bg-background font-mono text-sm"
                placeholder='{"timeout": 30, "retry_count": 3}'
                rows={4}
              />
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="is_active"
                checked={formData.is_active}
                onChange={(e) => setFormData({ ...formData, is_active: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="is_active" className="text-sm font-medium">
                Server is active
              </label>
            </div>

            <div className="flex gap-2">
              <button
                type="submit"
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
              >
                {editingServer ? 'Update Server' : 'Create Server'}
              </button>
              <button
                type="button"
                onClick={resetForm}
                className="px-4 py-2 border border-border rounded-lg hover:bg-accent"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Server List */}
      <div className="bg-card border border-border rounded-lg">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Registered MCP Servers ({servers.length})</h2>
        </div>
        
        {loading ? (
          <div className="p-8 text-center text-muted-foreground">
            <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
            Loading servers...
          </div>
        ) : servers.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground">
            <AlertCircle className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="text-lg font-medium">No MCP servers configured</p>
            <p className="text-sm mt-1">Click "Add MCP Server" to register your first server</p>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {servers.map((server) => (
              <div key={server.id} className="p-4 hover:bg-accent/50 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold text-lg">{server.name}</h3>
                      {server.is_active ? (
                        <CheckCircle className="h-4 w-4 text-green-500" title="Active" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" title="Inactive" />
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{server.description}</p>
                    <div className="flex flex-wrap gap-2 text-sm">
                      <span className="px-2 py-1 bg-accent rounded">
                        <strong>URL:</strong> {server.base_url}
                      </span>
                      {server.capabilities.length > 0 && (
                        <span className="px-2 py-1 bg-accent rounded">
                          <strong>Capabilities:</strong> {server.capabilities.join(', ')}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <button
                      onClick={() => handleTestConnection(server.id)}
                      disabled={testingServerId === server.id}
                      className="p-2 hover:bg-accent rounded-lg transition-colors disabled:opacity-50"
                      title="Test Connection"
                    >
                      <RefreshCw className={`h-4 w-4 ${testingServerId === server.id ? 'animate-spin' : ''}`} />
                    </button>
                    <button
                      onClick={() => handleEdit(server)}
                      className="p-2 hover:bg-accent rounded-lg transition-colors"
                      title="Edit"
                    >
                      <Edit className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(server.id)}
                      className="p-2 hover:bg-destructive/10 text-destructive rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

