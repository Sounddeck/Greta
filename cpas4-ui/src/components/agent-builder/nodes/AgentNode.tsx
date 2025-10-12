import { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Bot, Settings, ChevronDown, ChevronUp } from 'lucide-react';

interface AgentNodeData {
  label: string;
  agentType?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
}

function AgentNode({ data, selected }: { data: AgentNodeData; selected?: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={`
        min-w-[250px] rounded-lg border-2 bg-gradient-to-br from-blue-900/90 to-blue-800/90 backdrop-blur-sm
        ${selected ? 'border-blue-400 shadow-lg shadow-blue-500/50' : 'border-blue-600'}
        transition-all duration-200
      `}
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-blue-400 border-2 border-white"
      />

      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-blue-700/50">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-blue-500/30">
            <Bot className="h-4 w-4 text-blue-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white">{data.label}</div>
            <div className="text-xs text-blue-300">{data.agentType || 'General'}</div>
          </div>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 hover:bg-blue-700/30 rounded transition-colors"
        >
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-blue-300" />
          ) : (
            <ChevronDown className="h-4 w-4 text-blue-300" />
          )}
        </button>
      </div>

      {/* Configuration (Expanded) */}
      {isExpanded && (
        <div className="p-3 space-y-2 text-xs">
          <div>
            <label className="text-blue-300 block mb-1">Model</label>
            <select className="w-full px-2 py-1 bg-blue-950/50 border border-blue-700 rounded text-white text-xs">
              <option value="gpt-4">{data.model || 'GPT-4'}</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              <option value="claude-3">Claude 3</option>
            </select>
          </div>

          <div>
            <label className="text-blue-300 block mb-1">
              Temperature: {data.temperature || 0.7}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              defaultValue={data.temperature || 0.7}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-blue-300 block mb-1">Max Tokens</label>
            <input
              type="number"
              defaultValue={data.maxTokens || 2000}
              className="w-full px-2 py-1 bg-blue-950/50 border border-blue-700 rounded text-white text-xs"
            />
          </div>

          <div>
            <label className="text-blue-300 block mb-1">System Prompt</label>
            <textarea
              defaultValue={data.systemPrompt || ''}
              placeholder="Enter system prompt..."
              className="w-full px-2 py-1 bg-blue-950/50 border border-blue-700 rounded text-white text-xs resize-none"
              rows={3}
            />
          </div>

          <button className="w-full px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors flex items-center justify-center gap-1">
            <Settings className="h-3 w-3" />
            Advanced Settings
          </button>
        </div>
      )}

      {/* Footer */}
      <div className="px-3 py-2 bg-blue-950/30 rounded-b-lg">
        <div className="flex items-center justify-between text-xs text-blue-300">
          <span>Agent Node</span>
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
            <span>Active</span>
          </div>
        </div>
      </div>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-blue-400 border-2 border-white"
      />
    </div>
  );
}

export default memo(AgentNode);

