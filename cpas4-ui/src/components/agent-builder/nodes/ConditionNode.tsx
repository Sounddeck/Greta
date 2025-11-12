import { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { GitBranch, ChevronDown, ChevronUp } from 'lucide-react';

interface ConditionNodeData {
  label: string;
  condition?: string;
  operator?: string;
  value?: string;
}

function ConditionNode({ data, selected }: { data: ConditionNodeData; selected?: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={`
        min-w-[250px] rounded-lg border-2 bg-gradient-to-br from-yellow-900/90 to-yellow-800/90 backdrop-blur-sm
        ${selected ? 'border-yellow-400 shadow-lg shadow-yellow-500/50' : 'border-yellow-600'}
        transition-all duration-200
      `}
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-yellow-400 border-2 border-white"
      />

      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-yellow-700/50">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-yellow-500/30">
            <GitBranch className="h-4 w-4 text-yellow-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white">{data.label}</div>
            <div className="text-xs text-yellow-300">Conditional Logic</div>
          </div>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 hover:bg-yellow-700/30 rounded transition-colors"
        >
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-yellow-300" />
          ) : (
            <ChevronDown className="h-4 w-4 text-yellow-300" />
          )}
        </button>
      </div>

      {/* Condition Configuration (Expanded) */}
      {isExpanded && (
        <div className="p-3 space-y-2 text-xs">
          <div>
            <label className="text-yellow-300 block mb-1">Condition Type</label>
            <select className="w-full px-2 py-1 bg-yellow-950/50 border border-yellow-700 rounded text-white text-xs">
              <option value="if">If</option>
              <option value="else-if">Else If</option>
              <option value="switch">Switch</option>
            </select>
          </div>

          <div>
            <label className="text-yellow-300 block mb-1">Variable</label>
            <input
              type="text"
              defaultValue={data.condition || ''}
              placeholder="e.g., response.status"
              className="w-full px-2 py-1 bg-yellow-950/50 border border-yellow-700 rounded text-white text-xs"
            />
          </div>

          <div>
            <label className="text-yellow-300 block mb-1">Operator</label>
            <select className="w-full px-2 py-1 bg-yellow-950/50 border border-yellow-700 rounded text-white text-xs">
              <option value="equals">Equals (==)</option>
              <option value="not-equals">Not Equals (!=)</option>
              <option value="greater">Greater Than (&gt;)</option>
              <option value="less">Less Than (&lt;)</option>
              <option value="contains">Contains</option>
              <option value="regex">Regex Match</option>
            </select>
          </div>

          <div>
            <label className="text-yellow-300 block mb-1">Value</label>
            <input
              type="text"
              defaultValue={data.value || ''}
              placeholder="Comparison value"
              className="w-full px-2 py-1 bg-yellow-950/50 border border-yellow-700 rounded text-white text-xs"
            />
          </div>

          <div className="pt-2 border-t border-yellow-700/30">
            <label className="text-yellow-300 block mb-1">Description</label>
            <textarea
              placeholder="Describe what this condition checks..."
              className="w-full px-2 py-1 bg-yellow-950/50 border border-yellow-700 rounded text-white text-xs resize-none"
              rows={2}
            />
          </div>
        </div>
      )}

      {/* Condition Preview */}
      {!isExpanded && (
        <div className="px-3 py-2">
          <div className="text-xs text-yellow-200 font-mono">
            {data.condition || 'variable'} {data.operator || '=='} {data.value || 'value'}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="px-3 py-2 bg-yellow-950/30 rounded-b-lg">
        <div className="flex items-center justify-between text-xs text-yellow-300">
          <span>Condition Node</span>
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
            <span>Ready</span>
          </div>
        </div>
      </div>

      {/* Output Handles - True and False paths */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="true"
        style={{ left: '33%' }}
        className="w-3 h-3 !bg-green-400 border-2 border-white"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        style={{ left: '66%' }}
        className="w-3 h-3 !bg-red-400 border-2 border-white"
      />
    </div>
  );
}

export default memo(ConditionNode);

