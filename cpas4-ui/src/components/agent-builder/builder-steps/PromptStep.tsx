// @ts-nocheck
import { useState } from 'react';
import { Sparkles, Plus, X, Lightbulb } from 'lucide-react';

const PROMPT_TEMPLATES = [
  {
    name: 'Research Assistant',
    prompt: 'You are a helpful research assistant. Your role is to gather, analyze, and synthesize information from various sources. Always cite your sources and provide comprehensive, well-structured responses.',
  },
  {
    name: 'Code Helper',
    prompt: 'You are an expert programming assistant. Help users write clean, efficient, and well-documented code. Explain your solutions clearly and suggest best practices.',
  },
  {
    name: 'Data Analyst',
    prompt: 'You are a data analysis expert. Help users understand their data through statistical analysis, visualization, and clear insights. Always explain your methodology.',
  },
  {
    name: 'Customer Support',
    prompt: 'You are a friendly and professional customer support agent. Help users solve their problems efficiently while maintaining a positive and empathetic tone.',
  },
];

const CAPABILITY_SUGGESTIONS = [
  'Information Retrieval',
  'Data Analysis',
  'Code Generation',
  'Problem Solving',
  'Creative Writing',
  'Task Planning',
  'Decision Making',
  'Summarization',
  'Translation',
  'Explanation',
];

interface PromptStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation: any;
}

export function PromptStep({ draft, updateDraft, validation }: PromptStepProps) {
  const [capabilityInput, setCapabilityInput] = useState('');
  const [showTemplates, setShowTemplates] = useState(false);

  const addCapability = (capability: string) => {
    if (capability.trim() && !draft.capabilities.includes(capability.trim())) {
      updateDraft({
        capabilities: [...draft.capabilities, capability.trim()],
      });
      setCapabilityInput('');
    }
  };

  const removeCapability = (capability: string) => {
    updateDraft({
      capabilities: draft.capabilities.filter((c: string) => c !== capability),
    });
  };

  const applyTemplate = (template: any) => {
    updateDraft({
      system_prompt: template.prompt,
    });
    setShowTemplates(false);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">System Prompt & Capabilities</h2>
        </div>

        <div className="space-y-6">
          {/* System Prompt */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium">
                System Prompt
              </label>
              <button
                onClick={() => setShowTemplates(!showTemplates)}
                className="text-sm text-primary hover:text-primary/80 flex items-center gap-1"
              >
                <Lightbulb className="h-4 w-4" />
                {showTemplates ? 'Hide' : 'Show'} Templates
              </button>
            </div>

            {/* Template Selector */}
            {showTemplates && (
              <div className="mb-4 p-4 bg-muted rounded-md space-y-2">
                <p className="text-sm font-medium mb-2">Choose a template:</p>
                {PROMPT_TEMPLATES.map((template) => (
                  <button
                    key={template.name}
                    onClick={() => applyTemplate(template)}
                    className="w-full text-left p-3 border border-border rounded-md hover:border-primary hover:bg-primary/5 transition-colors"
                  >
                    <div className="font-medium text-sm">{template.name}</div>
                    <div className="text-xs text-muted-foreground mt-1">{template.prompt}</div>
                  </button>
                ))}
              </div>
            )}

            <textarea
              value={draft.system_prompt}
              onChange={(e) => updateDraft({ system_prompt: e.target.value })}
              placeholder="Define the agent's behavior, personality, and guidelines. Be specific about what the agent should and shouldn't do..."
              rows={10}
              className={`w-full px-3 py-2 border rounded-md bg-background resize-none font-mono text-sm ${
                validation?.field_validations?.system_prompt === false
                  ? 'border-yellow-500'
                  : 'border-border'
              }`}
            />
            
            <div className="flex items-center justify-between mt-2">
              <p className="text-xs text-muted-foreground">
                {draft.system_prompt.length} characters
              </p>
              {validation?.warnings?.find((w: any) => w.field === 'system_prompt') && (
                <p className="text-xs text-yellow-600">
                  {validation.warnings.find((w: any) => w.field === 'system_prompt').message}
                </p>
              )}
            </div>

            {validation?.suggestions?.find((s: any) => s.field === 'system_prompt') && (
              <p className="text-sm text-blue-600 mt-2 flex items-center gap-1">
                <Lightbulb className="h-3 w-3" />
                {validation.suggestions.find((s: any) => s.field === 'system_prompt').message}
              </p>
            )}
          </div>

          {/* Capabilities */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Capabilities
            </label>
            
            {/* Capability Input */}
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={capabilityInput}
                onChange={(e) => setCapabilityInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addCapability(capabilityInput);
                  }
                }}
                placeholder="Add a capability (press Enter)"
                className="flex-1 px-3 py-2 border border-border rounded-md bg-background"
              />
              <button
                onClick={() => addCapability(capabilityInput)}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
              >
                <Plus className="h-4 w-4" />
              </button>
            </div>

            {/* Capability Suggestions */}
            <div className="mb-3">
              <p className="text-xs text-muted-foreground mb-2">Quick add:</p>
              <div className="flex flex-wrap gap-2">
                {CAPABILITY_SUGGESTIONS.filter(
                  (cap) => !draft.capabilities.includes(cap)
                ).map((capability) => (
                  <button
                    key={capability}
                    onClick={() => addCapability(capability)}
                    className="px-3 py-1 text-xs border border-border rounded-full hover:border-primary hover:bg-primary/5 transition-colors"
                  >
                    + {capability}
                  </button>
                ))}
              </div>
            </div>

            {/* Selected Capabilities */}
            {draft.capabilities.length > 0 ? (
              <div className="space-y-2">
                {draft.capabilities.map((capability: string) => (
                  <div
                    key={capability}
                    className="flex items-center justify-between p-3 bg-primary/10 border border-primary/20 rounded-md"
                  >
                    <span className="text-sm font-medium">{capability}</span>
                    <button
                      onClick={() => removeCapability(capability)}
                      className="text-destructive hover:text-destructive/80"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 border border-dashed border-border rounded-md text-center">
                <p className="text-sm text-muted-foreground">
                  No capabilities added yet. Add capabilities to define what your agent can do.
                </p>
              </div>
            )}

            {validation?.warnings?.find((w: any) => w.field === 'capabilities') && (
              <p className="text-sm text-yellow-600 mt-2">
                {validation.warnings.find((w: any) => w.field === 'capabilities').message}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Prompt Tips */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-primary" />
          Prompt Writing Tips
        </h3>
        <ul className="space-y-2 text-sm text-muted-foreground">
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            <span>Be specific about the agent's role and expertise</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            <span>Define the tone and personality (professional, friendly, technical, etc.)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            <span>Include guidelines for what the agent should and shouldn't do</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            <span>Mention any specific formats or structures for responses</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            <span>Add examples of good responses if applicable</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">•</span>
            <span>Keep it concise but comprehensive (200-500 words is usually good)</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default PromptStep;

