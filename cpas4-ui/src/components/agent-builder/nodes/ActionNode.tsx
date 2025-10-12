import { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Zap, ChevronDown, ChevronUp } from 'lucide-react';

interface ActionNodeData {
  label: string;
  actionType?: string;
  parameters?: Record<string, any>;
}

const actionTypes = [
  { value: 'send-message', label: 'Send Message', category: 'Communication' },
  { value: 'api-call', label: 'API Call', category: 'Integration' },
  { value: 'database-query', label: 'Database Query', category: 'Data' },
  { value: 'file-operation', label: 'File Operation', category: 'System' },
  { value: 'email', label: 'Send Email', category: 'Communication' },
  { value: 'webhook', label: 'Trigger Webhook', category: 'Integration' },
  { value: 'transform-data', label: 'Transform Data', category: 'Processing' },
  { value: 'log', label: 'Log Event', category: 'Monitoring' },
];

function ActionNode({ data, selected }: { data: ActionNodeData; selected?: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedAction, setSelectedAction] = useState(data.actionType || 'send-message');

  return (
    <div
      className={`
        min-w-[250px] rounded-lg border-2 bg-gradient-to-br from-purple-900/90 to-purple-800/90 backdrop-blur-sm
        ${selected ? 'border-purple-400 shadow-lg shadow-purple-500/50' : 'border-purple-600'}
        transition-all duration-200
      `}
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-purple-400 border-2 border-white"
      />

      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-purple-700/50">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-purple-500/30">
            <Zap className="h-4 w-4 text-purple-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white">{data.label}</div>
            <div className="text-xs text-purple-300">
              {actionTypes.find((a) => a.value === selectedAction)?.label || 'Action'}
            </div>
          </div>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 hover:bg-purple-700/30 rounded transition-colors"
        >
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-purple-300" />
          ) : (
            <ChevronDown className="h-4 w-4 text-purple-300" />
          )}
        </button>
      </div>

      {/* Action Configuration (Expanded) */}
      {isExpanded && (
        <div className="p-3 space-y-2 text-xs">
          <div>
            <label className="text-purple-300 block mb-1">Action Type</label>
            <select
              value={selectedAction}
              onChange={(e) => setSelectedAction(e.target.value)}
              className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs"
            >
              {actionTypes.map((action) => (
                <option key={action.value} value={action.value}>
                  {action.label} ({action.category})
                </option>
              ))}
            </select>
          </div>

          {/* Dynamic parameters based on action type */}
          {selectedAction === 'send-message' && (
            <>
              <div>
                <label className="text-purple-300 block mb-1">Message</label>
                <textarea
                  placeholder="Enter message content..."
                  className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs resize-none"
                  rows={3}
                />
              </div>
              <div>
                <label className="text-purple-300 block mb-1">Recipient</label>
                <input
                  type="text"
                  placeholder="user, channel, or email"
                  className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs"
                />
              </div>
            </>
          )}

          {selectedAction === 'api-call' && (
            <>
              <div>
                <label className="text-purple-300 block mb-1">Endpoint URL</label>
                <input
                  type="text"
                  placeholder="https://api.example.com/endpoint"
                  className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs"
                />
              </div>
              <div>
                <label className="text-purple-300 block mb-1">Method</label>
                <select className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs">
                  <option value="GET">GET</option>
                  <option value="POST">POST</option>
                  <option value="PUT">PUT</option>
                  <option value="DELETE">DELETE</option>
                </select>
              </div>
              <div>
                <label className="text-purple-300 block mb-1">Request Body (JSON)</label>
                <textarea
                  placeholder='{"key": "value"}'
                  className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs font-mono resize-none"
                  rows={3}
                />
              </div>
            </>
          )}

          {selectedAction === 'email' && (
            <>
              <div>
                <label className="text-purple-300 block mb-1">To</label>
                <input
                  type="email"
                  placeholder="recipient@example.com"
                  className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs"
                />
              </div>
              <div>
                <label className="text-purple-300 block mb-1">Subject</label>
                <input
                  type="text"
                  placeholder="Email subject"
                  className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs"
                />
              </div>
              <div>
                <label className="text-purple-300 block mb-1">Body</label>
                <textarea
                  placeholder="Email content..."
                  className="w-full px-2 py-1 bg-purple-950/50 border border-purple-700 rounded text-white text-xs resize-none"
                  rows={3}
                />
              </div>
            </>
          )}

          <div className="pt-2 border-t border-purple-700/30">
            <label className="flex items-center gap-2 text-purple-300">
              <input type="checkbox" className="rounded border-purple-600" />
              <span>Execute asynchronously</span>
            </label>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="px-3 py-2 bg-purple-950/30 rounded-b-lg">
        <div className="flex items-center justify-between text-xs text-purple-300">
          <span>Action Node</span>
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-purple-400" />
            <span>Ready</span>
          </div>
        </div>
      </div>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-purple-400 border-2 border-white"
      />
    </div>
  );
}

export default memo(ActionNode);

