import { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Wrench, ChevronDown, ChevronUp, Plus, Settings, X } from 'lucide-react';
import { toast } from 'sonner';

interface ToolNodeData {
  label: string;
  tools?: ToolConfig[];
}

interface ToolConfig {
  name: string;
  type: string;
  config: Record<string, any>;
}

const availableToolTypes = [
  { name: 'Code Execution', type: 'code_execution', category: 'Development' },
  { name: 'File Operations', type: 'file_operations', category: 'System' },
  { name: 'Database Query', type: 'database_query', category: 'Data' },
  { name: 'API Call', type: 'api_call', category: 'Integration' },
  { name: 'Email Send', type: 'email_send', category: 'Communication' },
  { name: 'Calendar', type: 'calendar', category: 'Productivity' },
  { name: 'Document Analysis', type: 'document_analysis', category: 'AI' },
];

function ToolNode({ data, selected }: { data: ToolNodeData; selected?: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedTools, setSelectedTools] = useState<ToolConfig[]>(data.tools || []);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configuringTool, setConfiguringTool] = useState<{ name: string; type: string } | null>(null);

  // Tool configuration states
  const [toolConfig, setToolConfig] = useState<Record<string, any>>({});

  const addTool = (toolType: { name: string; type: string }) => {
    setConfiguringTool(toolType);
    setToolConfig({});
    setShowConfigModal(true);
  };

  const saveTool = () => {
    if (!configuringTool) return;

    const newTool: ToolConfig = {
      name: configuringTool.name,
      type: configuringTool.type,
      config: toolConfig
    };

    setSelectedTools((prev) => [...prev, newTool]);
    setShowConfigModal(false);
    setConfiguringTool(null);
    setToolConfig({});
    toast.success(`${configuringTool.name} configured successfully`);
  };

  const removeTool = (index: number) => {
    setSelectedTools((prev) => prev.filter((_, i) => i !== index));
  };

  const renderConfigForm = () => {
    if (!configuringTool) return null;

    switch (configuringTool.type) {
      case 'code_execution':
        return (
          <div className="space-y-3">
            <div>
              <label className="text-sm text-green-300 block mb-1">Language</label>
              <select
                value={toolConfig.language || 'python'}
                onChange={(e) => setToolConfig({ ...toolConfig, language: e.target.value })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              >
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
                <option value="bash">Bash</option>
                <option value="typescript">TypeScript</option>
              </select>
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">Timeout (seconds)</label>
              <input
                type="number"
                value={toolConfig.timeout || 30}
                onChange={(e) => setToolConfig({ ...toolConfig, timeout: parseInt(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">Environment</label>
              <select
                value={toolConfig.environment || 'sandbox'}
                onChange={(e) => setToolConfig({ ...toolConfig, environment: e.target.value })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              >
                <option value="sandbox">Sandbox</option>
                <option value="docker">Docker Container</option>
                <option value="local">Local</option>
              </select>
            </div>
          </div>
        );

      case 'file_operations':
        return (
          <div className="space-y-3">
            <div>
              <label className="text-sm text-green-300 block mb-1">Allowed Operations</label>
              <div className="space-y-1">
                {['read', 'write', 'delete', 'list'].map((op) => (
                  <label key={op} className="flex items-center gap-2 text-sm text-white">
                    <input
                      type="checkbox"
                      checked={toolConfig.operations?.includes(op) || false}
                      onChange={(e) => {
                        const ops = toolConfig.operations || [];
                        setToolConfig({
                          ...toolConfig,
                          operations: e.target.checked
                            ? [...ops, op]
                            : ops.filter((o: string) => o !== op)
                        });
                      }}
                      className="rounded border-green-600"
                    />
                    {op.charAt(0).toUpperCase() + op.slice(1)}
                  </label>
                ))}
              </div>
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">Base Directory</label>
              <input
                type="text"
                value={toolConfig.baseDir || '/workspace'}
                onChange={(e) => setToolConfig({ ...toolConfig, baseDir: e.target.value })}
                placeholder="/workspace"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">File Pattern Restrictions</label>
              <input
                type="text"
                value={toolConfig.patterns || ''}
                onChange={(e) => setToolConfig({ ...toolConfig, patterns: e.target.value })}
                placeholder="*.txt, *.json"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              />
            </div>
          </div>
        );

      case 'database_query':
        return (
          <div className="space-y-3">
            <div>
              <label className="text-sm text-green-300 block mb-1">Database Type</label>
              <select
                value={toolConfig.dbType || 'postgresql'}
                onChange={(e) => setToolConfig({ ...toolConfig, dbType: e.target.value })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              >
                <option value="postgresql">PostgreSQL</option>
                <option value="mysql">MySQL</option>
                <option value="mongodb">MongoDB</option>
                <option value="sqlite">SQLite</option>
              </select>
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">Connection String</label>
              <input
                type="text"
                value={toolConfig.connectionString || ''}
                onChange={(e) => setToolConfig({ ...toolConfig, connectionString: e.target.value })}
                placeholder="postgresql://user:pass@localhost:5432/db"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm font-mono"
              />
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">Allowed Tables</label>
              <input
                type="text"
                value={toolConfig.allowedTables || ''}
                onChange={(e) => setToolConfig({ ...toolConfig, allowedTables: e.target.value })}
                placeholder="users, products, orders"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm text-white">
                <input
                  type="checkbox"
                  checked={toolConfig.readOnly || false}
                  onChange={(e) => setToolConfig({ ...toolConfig, readOnly: e.target.checked })}
                  className="rounded border-green-600"
                />
                Read-only mode
              </label>
            </div>
          </div>
        );

      case 'api_call':
        return (
          <div className="space-y-3">
            <div>
              <label className="text-sm text-green-300 block mb-1">API Endpoint</label>
              <input
                type="text"
                value={toolConfig.endpoint || ''}
                onChange={(e) => setToolConfig({ ...toolConfig, endpoint: e.target.value })}
                placeholder="https://api.example.com/v1"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">HTTP Method</label>
              <select
                value={toolConfig.method || 'GET'}
                onChange={(e) => setToolConfig({ ...toolConfig, method: e.target.value })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="DELETE">DELETE</option>
                <option value="PATCH">PATCH</option>
              </select>
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">Headers (JSON)</label>
              <textarea
                value={toolConfig.headers || ''}
                onChange={(e) => setToolConfig({ ...toolConfig, headers: e.target.value })}
                placeholder='{"Authorization": "Bearer token"}'
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm font-mono resize-none"
                rows={3}
              />
            </div>
            <div>
              <label className="text-sm text-green-300 block mb-1">Authentication Type</label>
              <select
                value={toolConfig.authType || 'none'}
                onChange={(e) => setToolConfig({ ...toolConfig, authType: e.target.value })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
              >
                <option value="none">None</option>
                <option value="bearer">Bearer Token</option>
                <option value="basic">Basic Auth</option>
                <option value="api_key">API Key</option>
              </select>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-sm text-gray-400">
            No configuration needed for this tool type.
          </div>
        );
    }
  };

  return (
    <>
      <div
        className={`
          min-w-[250px] rounded-lg border-2 bg-gradient-to-br from-green-900/90 to-green-800/90 backdrop-blur-sm
          ${selected ? 'border-green-400 shadow-lg shadow-green-500/50' : 'border-green-600'}
          transition-all duration-200
        `}
      >
        {/* Input Handle */}
        <Handle
          type="target"
          position={Position.Top}
          className="w-3 h-3 !bg-green-400 border-2 border-white"
        />

        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b border-green-700/50">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded bg-green-500/30">
              <Wrench className="h-4 w-4 text-green-300" />
            </div>
            <div>
              <div className="text-sm font-semibold text-white">{data.label}</div>
              <div className="text-xs text-green-300">
                {selectedTools.length} tool{selectedTools.length !== 1 ? 's' : ''} configured
              </div>
            </div>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-green-700/30 rounded transition-colors"
          >
            {isExpanded ? (
              <ChevronUp className="h-4 w-4 text-green-300" />
            ) : (
              <ChevronDown className="h-4 w-4 text-green-300" />
            )}
          </button>
        </div>

        {/* Tool Selection (Expanded) */}
        {isExpanded && (
          <div className="p-3 space-y-2">
            <div className="text-xs text-green-300 mb-2">Available Tools:</div>
            <div className="max-h-48 overflow-y-auto space-y-1">
              {availableToolTypes.map((tool) => (
                <button
                  key={tool.type}
                  onClick={() => addTool(tool)}
                  className="w-full flex items-center gap-2 p-2 hover:bg-green-700/20 rounded cursor-pointer transition-colors text-left"
                >
                  <Plus className="h-3 w-3 text-green-400" />
                  <div className="flex-1">
                    <div className="text-xs text-white">{tool.name}</div>
                    <div className="text-[10px] text-green-400">{tool.category}</div>
                  </div>
                </button>
              ))}
            </div>

            {/* Configured Tools List */}
            {selectedTools.length > 0 && (
              <div className="mt-3 pt-3 border-t border-green-700/30">
                <div className="text-xs text-green-300 mb-2">Configured Tools:</div>
                <div className="space-y-1">
                  {selectedTools.map((tool, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-2 bg-green-700/20 rounded"
                    >
                      <div className="flex-1">
                        <div className="text-xs text-white">{tool.name}</div>
                        <div className="text-[10px] text-green-400">
                          {Object.keys(tool.config).length} settings
                        </div>
                      </div>
                      <button
                        onClick={() => removeTool(index)}
                        className="p-1 hover:bg-red-500/20 rounded transition-colors"
                      >
                        <X className="h-3 w-3 text-red-400" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Selected Tools Preview */}
        {!isExpanded && selectedTools.length > 0 && (
          <div className="px-3 py-2">
            <div className="flex flex-wrap gap-1">
              {selectedTools.slice(0, 3).map((tool, index) => (
                <span
                  key={index}
                  className="px-2 py-0.5 bg-green-700/30 text-green-200 text-[10px] rounded"
                >
                  {tool.name}
                </span>
              ))}
              {selectedTools.length > 3 && (
                <span className="px-2 py-0.5 bg-green-700/30 text-green-200 text-[10px] rounded">
                  +{selectedTools.length - 3} more
                </span>
              )}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="px-3 py-2 bg-green-950/30 rounded-b-lg">
          <div className="flex items-center justify-between text-xs text-green-300">
            <span>Tool Node</span>
            <div className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
              <span>Ready</span>
            </div>
          </div>
        </div>

        {/* Output Handle */}
        <Handle
          type="source"
          position={Position.Bottom}
          className="w-3 h-3 !bg-green-400 border-2 border-white"
        />
      </div>

      {/* Tool Configuration Modal */}
      {showConfigModal && configuringTool && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowConfigModal(false)}>
          <div className="bg-gray-900 rounded-lg p-6 max-w-lg w-full mx-4 border border-green-600 max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-white">Configure {configuringTool.name}</h3>
                <p className="text-sm text-gray-400">Set up the tool parameters</p>
              </div>
              <button onClick={() => setShowConfigModal(false)} className="text-gray-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>

            {renderConfigForm()}

            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setShowConfigModal(false)}
                className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveTool}
                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded transition-colors flex items-center justify-center gap-2"
              >
                <Settings className="h-4 w-4" />
                Save Tool
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default memo(ToolNode);

