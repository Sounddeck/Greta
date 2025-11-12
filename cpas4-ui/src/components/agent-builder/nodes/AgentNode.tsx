import { memo, useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Bot, Settings, ChevronDown, ChevronUp, X } from 'lucide-react';
import { api } from '../../../lib/api';
import { toast } from 'sonner';

interface AgentNodeData {
  label: string;
  agentType?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
}

interface Model {
  model_id: string;
  name: string;
  provider: string;
  type: string;
  description: string;
  available: boolean;
  default_temperature: number;
  default_max_tokens: number;
}

function AgentNode({ data, selected }: { data: AgentNodeData; selected?: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [selectedModel, setSelectedModel] = useState(data.model || '');
  const [temperature, setTemperature] = useState(data.temperature || 0.7);
  const [maxTokens, setMaxTokens] = useState(data.maxTokens || 2000);
  const [systemPrompt, setSystemPrompt] = useState(data.systemPrompt || '');

  // Advanced settings
  const [topP, setTopP] = useState(1.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  const [stopSequences, setStopSequences] = useState('');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoadingModels(true);
      const response = await api.get('/agent-builder/models');
      setModels(response.data.models || []);

      // Set first available model as default if none selected
      if (!selectedModel && response.data.models.length > 0) {
        const firstAvailable = response.data.models.find((m: Model) => m.available);
        if (firstAvailable) {
          setSelectedModel(firstAvailable.model_id);
        }
      }
    } catch (error: any) {
      console.error('Failed to fetch models:', error);
      toast.error('Failed to load available models');
    } finally {
      setLoadingModels(false);
    }
  };

  return (
    <>
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
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={loadingModels}
                className="w-full px-2 py-1 bg-blue-950/50 border border-blue-700 rounded text-white text-xs"
              >
                {loadingModels ? (
                  <option>Loading models...</option>
                ) : models.length === 0 ? (
                  <option>No models available</option>
                ) : (
                  models.map((model) => (
                    <option
                      key={model.model_id}
                      value={model.model_id}
                      disabled={!model.available}
                    >
                      {model.name} {!model.available && '(Not installed)'}
                    </option>
                  ))
                )}
              </select>
              {models.length > 0 && (
                <p className="text-[10px] text-blue-400 mt-1">
                  {models.find(m => m.model_id === selectedModel)?.description || ''}
                </p>
              )}
            </div>

            <div>
              <label className="text-blue-300 block mb-1">
                Temperature: {temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-blue-300 block mb-1">Max Tokens</label>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 2000)}
                className="w-full px-2 py-1 bg-blue-950/50 border border-blue-700 rounded text-white text-xs"
              />
            </div>

            <div>
              <label className="text-blue-300 block mb-1">System Prompt</label>
              <textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="Enter system prompt..."
                className="w-full px-2 py-1 bg-blue-950/50 border border-blue-700 rounded text-white text-xs resize-none"
                rows={3}
              />
            </div>

            <button
              onClick={() => setShowAdvancedSettings(true)}
              className="w-full px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors flex items-center justify-center gap-1"
            >
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

      {/* Advanced Settings Modal */}
      {showAdvancedSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowAdvancedSettings(false)}>
          <div className="bg-gray-900 rounded-lg p-6 max-w-md w-full mx-4 border border-blue-600" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Advanced Settings</h3>
              <button onClick={() => setShowAdvancedSettings(false)} className="text-gray-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-sm text-blue-300 block mb-1">
                  Top P: {topP.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-400 mt-1">Controls diversity via nucleus sampling</p>
              </div>

              <div>
                <label className="text-sm text-blue-300 block mb-1">
                  Frequency Penalty: {frequencyPenalty.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={frequencyPenalty}
                  onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-400 mt-1">Penalizes repeated tokens</p>
              </div>

              <div>
                <label className="text-sm text-blue-300 block mb-1">
                  Presence Penalty: {presencePenalty.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={presencePenalty}
                  onChange={(e) => setPresencePenalty(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-400 mt-1">Penalizes new tokens based on presence</p>
              </div>

              <div>
                <label className="text-sm text-blue-300 block mb-1">Stop Sequences</label>
                <textarea
                  value={stopSequences}
                  onChange={(e) => setStopSequences(e.target.value)}
                  placeholder="Enter stop sequences (comma-separated)"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm resize-none"
                  rows={3}
                />
                <p className="text-xs text-gray-400 mt-1">Sequences where the model will stop generating</p>
              </div>

              <button
                onClick={() => {
                  setShowAdvancedSettings(false);
                  toast.success('Advanced settings saved');
                }}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default memo(AgentNode);

