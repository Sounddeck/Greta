import { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Wrench, ChevronDown, ChevronUp, Plus } from 'lucide-react';

interface ToolNodeData {
  label: string;
  tools?: string[];
}

const availableTools = [
  { name: 'Code Execution', category: 'Development' },
  { name: 'File Operations', category: 'System' },
  { name: 'Database Query', category: 'Data' },
  { name: 'API Call', category: 'Integration' },
  { name: 'Email Send', category: 'Communication' },
  { name: 'Calendar', category: 'Productivity' },
  { name: 'Document Analysis', category: 'AI' },
];

function ToolNode({ data, selected }: { data: ToolNodeData; selected?: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedTools, setSelectedTools] = useState<string[]>(data.tools || []);

  const toggleTool = (toolName: string) => {
    setSelectedTools((prev) =>
      prev.includes(toolName)
        ? prev.filter((t) => t !== toolName)
        : [...prev, toolName]
    );
  };

  return (
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
              {selectedTools.length} tool{selectedTools.length !== 1 ? 's' : ''} selected
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
            {availableTools.map((tool) => (
              <label
                key={tool.name}
                className="flex items-center gap-2 p-2 hover:bg-green-700/20 rounded cursor-pointer transition-colors"
              >
                <input
                  type="checkbox"
                  checked={selectedTools.includes(tool.name)}
                  onChange={() => toggleTool(tool.name)}
                  className="rounded border-green-600"
                />
                <div className="flex-1">
                  <div className="text-xs text-white">{tool.name}</div>
                  <div className="text-[10px] text-green-400">{tool.category}</div>
                </div>
              </label>
            ))}
          </div>

          <button className="w-full px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors flex items-center justify-center gap-1 mt-2">
            <Plus className="h-3 w-3" />
            Add Custom Tool
          </button>
        </div>
      )}

      {/* Selected Tools Preview */}
      {!isExpanded && selectedTools.length > 0 && (
        <div className="px-3 py-2">
          <div className="flex flex-wrap gap-1">
            {selectedTools.slice(0, 3).map((tool) => (
              <span
                key={tool}
                className="px-2 py-0.5 bg-green-700/30 text-green-200 text-[10px] rounded"
              >
                {tool}
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
  );
}

export default memo(ToolNode);

