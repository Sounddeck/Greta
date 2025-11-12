// @ts-nocheck
import { Check, AlertCircle, Info, FileText, Settings, Wrench, Sparkles, Download } from 'lucide-react';

interface ReviewStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation: any;
}

export function ReviewStep({ draft, updateDraft, validation }: ReviewStepProps) {
  const llmConfig = draft.configuration?.llm || {};

  const exportPreview = () => {
    const config = {
      name: draft.name,
      description: draft.description,
      agent_type: draft.agent_type,
      capabilities: draft.capabilities,
      system_prompt: draft.system_prompt,
      tools: draft.tools,
      configuration: draft.configuration,
      tags: draft.tags,
      metadata: draft.metadata,
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${draft.name || 'agent'}-preview.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Next Steps Info */}
      <div className="bg-blue-500/10 border border-blue-500 rounded-lg p-6">
        <div className="flex items-start gap-3">
          <Info className="h-6 w-6 text-blue-500 mt-0.5" />
          <div>
            <h3 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
              Ready to Save Your Agent?
            </h3>
            <p className="text-sm text-blue-600 dark:text-blue-400 mb-3">
              Review your configuration below. When ready, click the <strong>"Save Agent"</strong> button
              at the bottom right to create your agent.
            </p>
            <div className="space-y-1 text-sm text-blue-600 dark:text-blue-400">
              <p>• <strong>Save Agent:</strong> Creates the agent with status "DRAFT" (bottom right button)</p>
              <p>• <strong>Export Preview:</strong> Downloads configuration as JSON for backup/sharing</p>
              <p>• After saving, you can deploy the agent from the Agent List page</p>
            </div>
          </div>
        </div>
      </div>

      {/* Validation Status */}
      <div className={`border rounded-lg p-6 ${
        validation?.is_valid
          ? 'bg-green-500/10 border-green-500'
          : 'bg-yellow-500/10 border-yellow-500'
      }`}>
        <div className="flex items-center gap-2 mb-4">
          {validation?.is_valid ? (
            <Check className="h-6 w-6 text-green-500" />
          ) : (
            <AlertCircle className="h-6 w-6 text-yellow-500" />
          )}
          <h2 className="text-xl font-semibold">
            {validation?.is_valid ? 'Configuration Valid ✓' : 'Review Required'}
          </h2>
        </div>

        {validation && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Completion:</span>
              <span className="text-sm font-bold">{validation.completion_percentage}%</span>
            </div>

            <div className="w-full bg-muted rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  validation.completion_percentage === 100
                    ? 'bg-green-500'
                    : 'bg-yellow-500'
                }`}
                style={{ width: `${validation.completion_percentage}%` }}
              />
            </div>

            {validation.errors.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-medium text-destructive mb-2">Errors:</h3>
                <ul className="space-y-1">
                  {validation.errors.map((error: any, index: number) => (
                    <li key={index} className="text-sm text-destructive flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 mt-0.5" />
                      <span>{error.message}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {validation.warnings.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-medium text-yellow-600 mb-2">Warnings:</h3>
                <ul className="space-y-1">
                  {validation.warnings.map((warning: any, index: number) => (
                    <li key={index} className="text-sm text-yellow-600 flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 mt-0.5" />
                      <span>{warning.message}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {validation.suggestions.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-medium text-blue-600 mb-2">Suggestions:</h3>
                <ul className="space-y-1">
                  {validation.suggestions.map((suggestion: any, index: number) => (
                    <li key={index} className="text-sm text-blue-600 flex items-start gap-2">
                      <Info className="h-4 w-4 mt-0.5" />
                      <span>{suggestion.message}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Configuration Summary */}
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Configuration Summary</h2>
          <button
            onClick={exportPreview}
            className="flex items-center gap-2 px-3 py-2 text-sm border border-border rounded-md hover:bg-accent transition-colors"
          >
            <Download className="h-4 w-4" />
            Export Preview
          </button>
        </div>

        <div className="space-y-6">
          {/* Basic Info */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <FileText className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Basic Information</h3>
            </div>
            <div className="space-y-2 pl-7">
              <div className="flex items-start gap-2">
                <span className="text-sm font-medium min-w-[120px]">Name:</span>
                <span className="text-sm text-muted-foreground">{draft.name || 'Not set'}</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-sm font-medium min-w-[120px]">Type:</span>
                <span className="text-sm text-muted-foreground">
                  {draft.agent_type?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Not set'}
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-sm font-medium min-w-[120px]">Description:</span>
                <span className="text-sm text-muted-foreground">{draft.description || 'Not set'}</span>
              </div>
              {draft.tags.length > 0 && (
                <div className="flex items-start gap-2">
                  <span className="text-sm font-medium min-w-[120px]">Tags:</span>
                  <div className="flex flex-wrap gap-1">
                    {draft.tags.map((tag: string) => (
                      <span key={tag} className="px-2 py-0.5 bg-primary/10 text-primary rounded-full text-xs">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* LLM Configuration */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Settings className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">LLM Configuration</h3>
            </div>
            <div className="space-y-2 pl-7">
              <div className="flex items-start gap-2">
                <span className="text-sm font-medium min-w-[120px]">Provider:</span>
                <span className="text-sm text-muted-foreground">
                  {llmConfig.provider?.toUpperCase() || 'Not set'}
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-sm font-medium min-w-[120px]">Model:</span>
                <span className="text-sm text-muted-foreground">{llmConfig.model_name || 'Not set'}</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-sm font-medium min-w-[120px]">Temperature:</span>
                <span className="text-sm text-muted-foreground">{llmConfig.temperature ?? 'Not set'}</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-sm font-medium min-w-[120px]">Max Tokens:</span>
                <span className="text-sm text-muted-foreground">{llmConfig.max_tokens ?? 'Not set'}</span>
              </div>
            </div>
          </div>

          {/* Tools */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Wrench className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Tools ({draft.tools?.length || 0})</h3>
            </div>
            {draft.tools && draft.tools.length > 0 ? (
              <div className="space-y-2 pl-7">
                {draft.tools.map((tool: any) => (
                  <div key={tool.tool_id} className="flex items-start gap-2">
                    <span className="text-sm font-medium min-w-[120px]">{tool.tool_id}:</span>
                    <span className="text-sm text-muted-foreground">
                      {tool.permissions?.join(', ') || 'No permissions'}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground pl-7">No tools selected</p>
            )}
          </div>

          {/* MCP Tools */}
          {draft.mcp_tools && draft.mcp_tools.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Wrench className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">MCP Tools ({draft.mcp_tools.length})</h3>
              </div>
              <div className="space-y-2 pl-7">
                {draft.mcp_tools.map((tool: any) => (
                  <div key={tool.tool_id} className="flex items-start gap-2">
                    <span className="text-sm font-medium min-w-[120px]">{tool.tool_id}:</span>
                    <span className="text-sm text-muted-foreground">
                      {tool.server_name || 'MCP Tool'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Skills */}
          {draft.skills && draft.skills.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Skills ({draft.skills.length})</h3>
              </div>
              <div className="space-y-2 pl-7">
                {draft.skills.map((skill: any) => (
                  <div key={skill.skill_id} className="flex items-start gap-2">
                    <span className="text-sm font-medium min-w-[120px]">{skill.skill_id}:</span>
                    <span className="text-sm text-muted-foreground">
                      {skill.is_enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Memory Configuration */}
          {draft.memory_config && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Settings className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Memory Systems</h3>
              </div>
              <div className="space-y-2 pl-7">
                {draft.memory_config.enable_vector_search && (
                  <div className="flex items-start gap-2">
                    <span className="text-sm font-medium min-w-[120px]">Vector Search:</span>
                    <span className="text-sm text-muted-foreground">
                      Enabled (Namespace: {draft.memory_config.vector_search_namespace || 'default'},
                      Limit: {draft.memory_config.vector_search_limit || 10})
                    </span>
                  </div>
                )}
                {draft.memory_config.enable_reasoning_bank && (
                  <div className="flex items-start gap-2">
                    <span className="text-sm font-medium min-w-[120px]">ReasoningBank:</span>
                    <span className="text-sm text-muted-foreground">
                      Enabled (Limit: {draft.memory_config.reasoning_bank_limit || 5})
                    </span>
                  </div>
                )}
                {draft.memory_config.enable_functional_memory && (
                  <div className="flex items-start gap-2">
                    <span className="text-sm font-medium min-w-[120px]">Functional Memory:</span>
                    <span className="text-sm text-muted-foreground">
                      Enabled ({draft.memory_config.memory_types?.join(', ') || 'episodic, semantic'})
                    </span>
                  </div>
                )}
                {draft.memory_config.enable_memory_orchestrator && (
                  <div className="flex items-start gap-2">
                    <span className="text-sm font-medium min-w-[120px]">Memory Orchestrator:</span>
                    <span className="text-sm text-muted-foreground">Enabled</span>
                  </div>
                )}
                {!draft.memory_config.enable_vector_search &&
                 !draft.memory_config.enable_reasoning_bank &&
                 !draft.memory_config.enable_functional_memory &&
                 !draft.memory_config.enable_memory_orchestrator && (
                  <p className="text-sm text-muted-foreground">No memory systems enabled</p>
                )}
              </div>
            </div>
          )}

          {/* System Prompt & Capabilities */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">System Prompt & Capabilities</h3>
            </div>
            <div className="space-y-3 pl-7">
              <div>
                <span className="text-sm font-medium">System Prompt:</span>
                <div className="mt-1 p-3 bg-muted rounded-md">
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                    {draft.system_prompt || 'Not set'}
                  </p>
                </div>
              </div>
              <div>
                <span className="text-sm font-medium">Capabilities ({draft.capabilities.length}):</span>
                {draft.capabilities.length > 0 ? (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {draft.capabilities.map((capability: string) => (
                      <span
                        key={capability}
                        className="px-3 py-1 bg-primary/10 text-primary rounded-full text-xs"
                      >
                        {capability}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground mt-1">No capabilities defined</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* JSON Preview */}
      <details className="bg-card border border-border rounded-lg">
        <summary className="px-6 py-4 cursor-pointer font-semibold hover:bg-accent">
          View JSON Configuration
        </summary>
        <div className="px-6 py-4 border-t border-border">
          <pre className="bg-muted p-4 rounded-md text-xs overflow-auto max-h-96">
            {JSON.stringify({
              name: draft.name,
              description: draft.description,
              agent_type: draft.agent_type,
              capabilities: draft.capabilities,
              system_prompt: draft.system_prompt,
              tools: draft.tools,
              configuration: draft.configuration,
              tags: draft.tags,
              metadata: draft.metadata,
            }, null, 2)}
          </pre>
        </div>
      </details>
    </div>
  );
}

export default ReviewStep;

