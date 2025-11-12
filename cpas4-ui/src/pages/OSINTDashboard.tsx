/**
 * OSINT Dashboard
 * Main dashboard for OSINT tools, execution, and monitoring
 */
import { useEffect, useState } from 'react';
import { Play, Eye, TrendingUp, Clock, CheckCircle, XCircle, Loader2, Shield, Zap } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '../components/ui/dialog';
import { useOSINTStore } from '../stores/osintStore';
import type { OSINTTool, ToolExecutionRequest } from '../stores/osintStore';
import { toast } from 'sonner';

export default function OSINTDashboard() {
  const {
    tools,
    executions,
    metrics,
    selectedTool,
    isLoading,
    isExecuting,
    error,
    filterCategory,
    searchQuery,
    fetchTools,
    fetchExecutions,
    fetchMetrics,
    executeTool,
    setSelectedTool,
    setFilterCategory,
    setSearchQuery,
  } = useOSINTStore();

  const [isExecuteDialogOpen, setIsExecuteDialogOpen] = useState(false);
  const [executionParams, setExecutionParams] = useState<Record<string, any>>({});
  const [selectedToolId, setSelectedToolId] = useState<string>('');

  // Load data on mount
  useEffect(() => {
    fetchTools();
    fetchExecutions();
    fetchMetrics();
  }, []);

  // Filter tools
  const filteredTools = tools.filter((tool) => {
    const matchesCategory = filterCategory === 'all' || tool.category === filterCategory;
    const matchesSearch = !searchQuery || 
      tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tool.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tool.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  // Get unique categories
  const categories = ['all', ...Array.from(new Set(tools.map(t => t.category)))];

  // Handle tool execution
  const handleExecuteTool = (tool: OSINTTool) => {
    setSelectedTool(tool);
    setExecutionParams({});
    setIsExecuteDialogOpen(true);
  };

  // Handle tool selection from dropdown
  const handleToolSelection = (toolId: string) => {
    setSelectedToolId(toolId);
    if (toolId) {
      const tool = tools.find(t => t.tool_id === toolId);
      if (tool) {
        handleExecuteTool(tool);
      }
    }
  };

  // Submit execution
  const handleSubmitExecution = async () => {
    if (!selectedTool) return;

    try {
      const request: ToolExecutionRequest = {
        tool_id: selectedTool.tool_id,
        parameters: executionParams,
        agent_id: 'osint_dashboard'
      };

      await executeTool(request);
      setIsExecuteDialogOpen(false);
      setExecutionParams({});
      setSelectedToolId(''); // Reset dropdown
    } catch (error) {
      // Error already handled in store
    }
  };

  // Update parameter value
  const updateParam = (paramName: string, value: any) => {
    setExecutionParams(prev => ({
      ...prev,
      [paramName]: value
    }));
  };

  return (
    <div className="h-full overflow-auto bg-background">
      <div className="space-y-6 p-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold mb-2">OSINT Dashboard</h1>
          <p className="text-muted-foreground">
            Open Source Intelligence tools for research, lead generation, and competitive analysis
          </p>
        </div>

        {/* Metrics Cards */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Total Executions</p>
                  <p className="text-2xl font-bold">{metrics.total_executions}</p>
                </div>
                <TrendingUp className="h-8 w-8 text-primary" />
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold">
                    {metrics.total_executions > 0
                      ? Math.round((metrics.successful_executions / metrics.total_executions) * 100)
                      : 0}%
                  </p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Avg Time</p>
                  <p className="text-2xl font-bold">{metrics.avg_execution_time.toFixed(2)}s</p>
                </div>
                <Clock className="h-8 w-8 text-blue-500" />
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Tools Available</p>
                  <p className="text-2xl font-bold">{tools.length}</p>
                </div>
                <Zap className="h-8 w-8 text-yellow-500" />
              </div>
            </Card>
          </div>
        )}

        {/* Tool Selection Dropdown */}
        <div className="space-y-2">
          <Label htmlFor="tool-select" className="text-base font-semibold">
            Select OSINT Tool
          </Label>
          <select
            id="tool-select"
            value={selectedToolId}
            onChange={(e) => handleToolSelection(e.target.value)}
            className="w-full px-4 py-3 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            disabled={isLoading || tools.length === 0}
          >
            <option value="">-- Select a tool to execute --</option>
            {tools.map((tool) => (
              <option key={tool.tool_id} value={tool.tool_id}>
                {tool.name} {tool.permission_level === 'RESTRICTED' ? '🔒' : ''}
              </option>
            ))}
          </select>
          {tools.length > 0 && (
            <p className="text-sm text-muted-foreground">
              {tools.length} OSINT tools available
            </p>
          )}
        </div>

        {/* Tools Grid */}
        <div>
          <h2 className="text-xl font-semibold mb-4">Available Tools</h2>
          
          {isLoading && tools.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : filteredTools.length === 0 ? (
            <Card className="p-8 text-center">
              <p className="text-muted-foreground">No tools found matching your criteria</p>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTools.map((tool) => (
                <Card key={tool.tool_id} className="p-4 hover:shadow-lg transition-shadow">
                  <div className="space-y-3">
                    {/* Header */}
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="font-semibold text-lg">{tool.name}</h3>
                        <p className="text-sm text-muted-foreground mt-1">{tool.description}</p>
                      </div>
                      {tool.permission_level === 'RESTRICTED' && (
                        <Shield className="h-4 w-4 text-yellow-500 ml-2" />
                      )}
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1">
                      {tool.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-primary/10 text-primary"
                        >
                          {tag}
                        </span>
                      ))}
                      {tool.tags.length > 3 && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-muted text-muted-foreground">
                          +{tool.tags.length - 3}
                        </span>
                      )}
                    </div>

                    {/* Metadata */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      {tool.rate_limit && (
                        <span>Limit: {tool.rate_limit}/min</span>
                      )}
                      {tool.timeout && (
                        <span>Timeout: {tool.timeout}s</span>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex gap-2 pt-2">
                      <Button
                        size="sm"
                        onClick={() => handleExecuteTool(tool)}
                        disabled={!tool.is_available}
                        className="flex-1"
                      >
                        <Play className="h-3 w-3 mr-1" />
                        Execute
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          setSelectedTool(tool);
                          // Could open a details dialog here
                        }}
                      >
                        <Eye className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Recent Executions */}
        <div>
          <h2 className="text-xl font-semibold mb-4">Recent Executions</h2>
          
          {executions.length === 0 ? (
            <Card className="p-8 text-center">
              <p className="text-muted-foreground">No executions yet</p>
            </Card>
          ) : (
            <Card className="overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-medium">Tool</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Time</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {executions.slice(0, 10).map((execution) => (
                      <tr key={execution.execution_id} className="hover:bg-muted/30">
                        <td className="px-4 py-3 text-sm">{execution.tool_name || execution.tool_id}</td>
                        <td className="px-4 py-3">
                          {execution.success ? (
                            <span className="inline-flex items-center gap-1 text-green-600">
                              <CheckCircle className="h-4 w-4" />
                              Success
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-red-600">
                              <XCircle className="h-4 w-4" />
                              Failed
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm">{execution.execution_time.toFixed(2)}s</td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {new Date(execution.timestamp).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Execute Tool Dialog */}
      <Dialog open={isExecuteDialogOpen} onOpenChange={setIsExecuteDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Execute Tool: {selectedTool?.name}</DialogTitle>
            <DialogDescription>{selectedTool?.description}</DialogDescription>
          </DialogHeader>

          {selectedTool && (
            <div className="space-y-4 mt-4">
              {/* Parameters */}
              {Object.entries(selectedTool.parameters || {}).map(([paramName, paramDef]: [string, any]) => (
                <div key={paramName} className="space-y-2">
                  <Label htmlFor={paramName}>
                    {paramName}
                    {paramDef.required && <span className="text-red-500 ml-1">*</span>}
                  </Label>

                  {paramDef.type === 'string' && (
                    <Input
                      id={paramName}
                      placeholder={paramDef.description || `Enter ${paramName}`}
                      value={executionParams[paramName] || ''}
                      onChange={(e) => updateParam(paramName, e.target.value)}
                    />
                  )}

                  {paramDef.type === 'integer' && (
                    <Input
                      id={paramName}
                      type="number"
                      placeholder={paramDef.description || `Enter ${paramName}`}
                      value={executionParams[paramName] || paramDef.default || ''}
                      onChange={(e) => updateParam(paramName, parseInt(e.target.value) || 0)}
                    />
                  )}

                  {paramDef.type === 'boolean' && (
                    <div className="flex items-center gap-2">
                      <input
                        id={paramName}
                        type="checkbox"
                        checked={executionParams[paramName] || paramDef.default || false}
                        onChange={(e) => updateParam(paramName, e.target.checked)}
                        className="h-4 w-4"
                      />
                      <label htmlFor={paramName} className="text-sm text-muted-foreground">
                        {paramDef.description}
                      </label>
                    </div>
                  )}

                  {paramDef.type === 'object' && (
                    <textarea
                      id={paramName}
                      placeholder={`Enter JSON object for ${paramName}`}
                      value={typeof executionParams[paramName] === 'object'
                        ? JSON.stringify(executionParams[paramName], null, 2)
                        : executionParams[paramName] || '{}'}
                      onChange={(e) => {
                        try {
                          const parsed = JSON.parse(e.target.value);
                          updateParam(paramName, parsed);
                        } catch {
                          updateParam(paramName, e.target.value);
                        }
                      }}
                      className="w-full min-h-[100px] p-2 border rounded-md font-mono text-sm"
                    />
                  )}

                  {paramDef.description && (
                    <p className="text-xs text-muted-foreground">{paramDef.description}</p>
                  )}
                </div>
              ))}

              {/* Actions */}
              <div className="flex justify-end gap-2 pt-4 border-t">
                <Button
                  variant="outline"
                  onClick={() => setIsExecuteDialogOpen(false)}
                  disabled={isExecuting}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmitExecution}
                  disabled={isExecuting}
                >
                  {isExecuting ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Execute
                    </>
                  )}
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

