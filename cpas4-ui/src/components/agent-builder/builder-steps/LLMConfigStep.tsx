// @ts-nocheck
import { useState, useEffect } from 'react';
import { Settings, Cpu, Zap, Info } from 'lucide-react';
import { agentBuilderAPI } from '../../../lib/api';
import { toast } from 'sonner';

const LLM_PROVIDERS = [
  { value: 'llama_cpp', label: 'llama.cpp (Local)', description: 'Local GGUF models' },
  { value: 'ollama', label: 'Ollama (Local)', description: 'Local Ollama models' },
  { value: 'openai', label: 'OpenAI', description: 'GPT-4, GPT-3.5' },
  { value: 'anthropic', label: 'Anthropic', description: 'Claude models' },
  { value: 'google', label: 'Google', description: 'Gemini models' },
  { value: 'custom', label: 'Custom', description: 'Custom endpoint' },
];

interface LLMConfigStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation: any;
}

export function LLMConfigStep({ draft, updateDraft, validation }: LLMConfigStepProps) {
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  const llmConfig = draft.configuration?.llm || {
    provider: 'llama_cpp',
    model_id: 'llama_cpp:llama3-8b',
    model_name: 'Llama 3 8B (llama.cpp)',
    temperature: 0.7,
    max_tokens: 2000,
    top_p: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    context_window: 8192,
  };

  useEffect(() => {
    loadAvailableModels();
  }, [llmConfig.provider]);

  const loadAvailableModels = async () => {
    // Only load models for local providers (llama.cpp and ollama)
    if (llmConfig.provider !== 'llama_cpp' && llmConfig.provider !== 'ollama') return;

    setLoadingModels(true);
    try {
      const response = await agentBuilderAPI.getAvailableModels();
      // Backend returns {models: [...]}
      const models = response.data.models || [];
      setAvailableModels(models.filter((m: any) => m.provider === llmConfig.provider));
    } catch (error) {
      console.error('Failed to load models:', error);
      toast.error('Failed to load available models');
    } finally {
      setLoadingModels(false);
    }
  };

  const updateLLMConfig = (updates: any) => {
    updateDraft({
      configuration: {
        ...draft.configuration,
        llm: {
          ...llmConfig,
          ...updates,
        },
      },
    });
  };

  const selectModel = (model: any) => {
    updateLLMConfig({
      model_id: model.model_id,
      model_name: model.name,
      context_window: model.context_window,
      temperature: model.default_temperature || 0.7,
      max_tokens: model.default_max_tokens || 2000,
    });
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">LLM Configuration</h2>
        </div>

        <div className="space-y-6">
          {/* Provider Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">
              LLM Provider <span className="text-destructive">*</span>
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {LLM_PROVIDERS.map((provider) => (
                <button
                  key={provider.value}
                  onClick={() => updateLLMConfig({ provider: provider.value })}
                  className={`p-3 border rounded-md text-left transition-colors ${
                    llmConfig.provider === provider.value
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:border-primary/50'
                  }`}
                >
                  <div className="font-medium text-sm">{provider.label}</div>
                  <div className="text-xs text-muted-foreground mt-1">{provider.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Model Selection (Ollama) */}
          {llmConfig.provider === 'ollama' && (
            <div>
              <label className="block text-sm font-medium mb-2">
                Model <span className="text-destructive">*</span>
              </label>
              {loadingModels ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                  Loading models...
                </div>
              ) : availableModels.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {availableModels.map((model) => (
                    <button
                      key={model.model_id}
                      onClick={() => selectModel(model)}
                      className={`p-3 border rounded-md text-left transition-colors ${
                        llmConfig.model_id === model.model_id
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-primary" />
                        <div className="font-medium text-sm">{model.name}</div>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {model.description || 'Local model'}
                      </div>
                      {model.context_window && (
                        <div className="text-xs text-muted-foreground mt-1">
                          Context: {model.context_window.toLocaleString()} tokens
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="p-4 border border-yellow-500 bg-yellow-500/10 rounded-md">
                  <p className="text-sm text-yellow-600">
                    No Ollama models found. Make sure Ollama is running and has models installed.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Manual Model Input (for other providers) */}
          {llmConfig.provider !== 'ollama' && (
            <div>
              <label className="block text-sm font-medium mb-2">
                Model ID <span className="text-destructive">*</span>
              </label>
              <input
                type="text"
                value={llmConfig.model_id}
                onChange={(e) => updateLLMConfig({ model_id: e.target.value })}
                placeholder="e.g., gpt-4, claude-3-opus, gemini-pro"
                className="w-full px-3 py-2 border border-border rounded-md bg-background"
              />
            </div>
          )}

          {/* Temperature */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Temperature: {llmConfig.temperature}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={llmConfig.temperature}
              onChange={(e) => updateLLMConfig({ temperature: parseFloat(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Precise (0.0)</span>
              <span>Balanced (1.0)</span>
              <span>Creative (2.0)</span>
            </div>
            <p className="text-xs text-muted-foreground mt-2 flex items-start gap-1">
              <Info className="h-3 w-3 mt-0.5" />
              Lower values make output more focused and deterministic. Higher values make it more creative and random.
            </p>
          </div>

          {/* Max Tokens */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Max Tokens: {llmConfig.max_tokens}
            </label>
            <input
              type="range"
              min="100"
              max="10000"
              step="100"
              value={llmConfig.max_tokens}
              onChange={(e) => updateLLMConfig({ max_tokens: parseInt(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>100</span>
              <span>5000</span>
              <span>10000</span>
            </div>
            <p className="text-xs text-muted-foreground mt-2 flex items-start gap-1">
              <Info className="h-3 w-3 mt-0.5" />
              Maximum number of tokens to generate in the response.
            </p>
          </div>

          {/* Top P */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Top P: {llmConfig.top_p}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={llmConfig.top_p}
              onChange={(e) => updateLLMConfig({ top_p: parseFloat(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>0.0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
            <p className="text-xs text-muted-foreground mt-2 flex items-start gap-1">
              <Info className="h-3 w-3 mt-0.5" />
              Controls diversity via nucleus sampling. 1.0 means no restrictions.
            </p>
          </div>

          {/* Advanced Settings */}
          <details className="border border-border rounded-md">
            <summary className="px-4 py-3 cursor-pointer font-medium text-sm hover:bg-accent">
              Advanced Settings
            </summary>
            <div className="px-4 py-3 space-y-4 border-t border-border">
              {/* Frequency Penalty */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Frequency Penalty: {llmConfig.frequency_penalty}
                </label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={llmConfig.frequency_penalty}
                  onChange={(e) => updateLLMConfig({ frequency_penalty: parseFloat(e.target.value) })}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Reduces repetition of token sequences.
                </p>
              </div>

              {/* Presence Penalty */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Presence Penalty: {llmConfig.presence_penalty}
                </label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={llmConfig.presence_penalty}
                  onChange={(e) => updateLLMConfig({ presence_penalty: parseFloat(e.target.value) })}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Encourages talking about new topics.
                </p>
              </div>

              {/* API Key (for cloud providers) */}
              {llmConfig.provider !== 'ollama' && (
                <div>
                  <label className="block text-sm font-medium mb-2">
                    API Key
                  </label>
                  <input
                    type="password"
                    value={llmConfig.api_key || ''}
                    onChange={(e) => updateLLMConfig({ api_key: e.target.value })}
                    placeholder="Enter your API key"
                    className="w-full px-3 py-2 border border-border rounded-md bg-background"
                  />
                </div>
              )}
            </div>
          </details>
        </div>
      </div>

      {/* Configuration Preview */}
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Zap className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold">Configuration Preview</h3>
        </div>
        <pre className="bg-muted p-4 rounded-md text-xs overflow-auto">
          {JSON.stringify(llmConfig, null, 2)}
        </pre>
      </div>
    </div>
  );
}

export default LLMConfigStep;

