// @ts-nocheck
import { useState } from 'react';
import { Database, Brain, Clock, Zap, AlertCircle, Info } from 'lucide-react';

interface MemoryStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation?: any;
}

export function MemoryStep({ draft, updateDraft, validation }: MemoryStepProps) {
  const memoryConfig = draft.memory_config || {
    enable_vector_search: false,
    vector_search_namespace: null,
    vector_search_limit: 10,
    enable_reasoning_bank: false,
    reasoning_bank_pattern_types: [],
    reasoning_bank_limit: 5,
    enable_functional_memory: false,
    memory_types: ['episodic', 'semantic'],
    enable_memory_orchestrator: false,
    auto_store_patterns: true,
    memory_retention_days: 90,
  };

  const updateMemoryConfig = (updates: any) => {
    updateDraft({
      memory_config: {
        ...memoryConfig,
        ...updates,
      },
    });
  };

  const toggleMemoryType = (type: string) => {
    const currentTypes = memoryConfig.memory_types || [];
    const newTypes = currentTypes.includes(type)
      ? currentTypes.filter((t: string) => t !== type)
      : [...currentTypes, type];
    updateMemoryConfig({ memory_types: newTypes });
  };

  const togglePatternType = (type: string) => {
    const currentTypes = memoryConfig.reasoning_bank_pattern_types || [];
    const newTypes = currentTypes.includes(type)
      ? currentTypes.filter((t: string) => t !== type)
      : [...currentTypes, type];
    updateMemoryConfig({ reasoning_bank_pattern_types: newTypes });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">Memory Systems</h2>
        <p className="text-muted-foreground">
          Configure memory systems to help your agent learn from past experiences and improve over time.
        </p>
      </div>

      {/* Vector Search */}
      <div className="border border-border rounded-lg p-4 space-y-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <Database className="h-5 w-5 text-primary" />
            <div>
              <h3 className="font-semibold">Vector Search</h3>
              <p className="text-sm text-muted-foreground">
                Semantic memory storage and retrieval using ChromaDB
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={memoryConfig.enable_vector_search}
              onChange={(e) => updateMemoryConfig({ enable_vector_search: e.target.checked })}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
          </label>
        </div>

        {memoryConfig.enable_vector_search && (
          <div className="space-y-3 pl-8">
            <div>
              <label className="block text-sm font-medium mb-1">
                Namespace (Optional)
              </label>
              <input
                type="text"
                value={memoryConfig.vector_search_namespace || ''}
                onChange={(e) => updateMemoryConfig({ vector_search_namespace: e.target.value || null })}
                placeholder={`agent_${draft.draft_id || 'default'}`}
                className="w-full px-3 py-2 border border-border rounded-md bg-background text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Leave empty to use default namespace based on agent ID
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                Search Limit
              </label>
              <input
                type="number"
                min="1"
                max="50"
                value={memoryConfig.vector_search_limit}
                onChange={(e) => updateMemoryConfig({ vector_search_limit: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-md bg-background text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Maximum number of memories to retrieve (1-50)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ReasoningBank */}
      <div className="border border-border rounded-lg p-4 space-y-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <Brain className="h-5 w-5 text-primary" />
            <div>
              <h3 className="font-semibold">ReasoningBank</h3>
              <p className="text-sm text-muted-foreground">
                Pattern storage and retrieval for similar problems
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={memoryConfig.enable_reasoning_bank}
              onChange={(e) => updateMemoryConfig({ enable_reasoning_bank: e.target.checked })}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
          </label>
        </div>

        {memoryConfig.enable_reasoning_bank && (
          <div className="space-y-3 pl-8">
            <div>
              <label className="block text-sm font-medium mb-2">
                Pattern Types
              </label>
              <div className="grid grid-cols-2 gap-2">
                {['problem_solving', 'decision_making', 'optimization', 'agent_execution'].map((type) => (
                  <label key={type} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={memoryConfig.reasoning_bank_pattern_types?.includes(type)}
                      onChange={() => togglePatternType(type)}
                      className="h-4 w-4"
                    />
                    <span className="text-sm">{type.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Leave empty to search all pattern types
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                Pattern Limit
              </label>
              <input
                type="number"
                min="1"
                max="20"
                value={memoryConfig.reasoning_bank_limit}
                onChange={(e) => updateMemoryConfig({ reasoning_bank_limit: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-md bg-background text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Maximum number of patterns to retrieve (1-20)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Functional Memory */}
      <div className="border border-border rounded-lg p-4 space-y-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <Clock className="h-5 w-5 text-primary" />
            <div>
              <h3 className="font-semibold">Functional Memory</h3>
              <p className="text-sm text-muted-foreground">
                Episodic, semantic, procedural, and working memory
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={memoryConfig.enable_functional_memory}
              onChange={(e) => updateMemoryConfig({ enable_functional_memory: e.target.checked })}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
          </label>
        </div>

        {memoryConfig.enable_functional_memory && (
          <div className="space-y-3 pl-8">
            <div>
              <label className="block text-sm font-medium mb-2">
                Memory Types
              </label>
              <div className="grid grid-cols-2 gap-2">
                {['episodic', 'semantic', 'procedural', 'working'].map((type) => (
                  <label key={type} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={memoryConfig.memory_types?.includes(type)}
                      onChange={() => toggleMemoryType(type)}
                      className="h-4 w-4"
                    />
                    <span className="text-sm capitalize">{type}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Memory Orchestrator */}
      <div className="border border-border rounded-lg p-4 space-y-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <Zap className="h-5 w-5 text-primary" />
            <div>
              <h3 className="font-semibold">Memory Orchestrator</h3>
              <p className="text-sm text-muted-foreground">
                Intelligent memory management and synthesis
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={memoryConfig.enable_memory_orchestrator}
              onChange={(e) => updateMemoryConfig({ enable_memory_orchestrator: e.target.checked })}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
          </label>
        </div>
      </div>

      {/* General Settings */}
      <div className="border border-border rounded-lg p-4 space-y-4">
        <h3 className="font-semibold">General Settings</h3>

        <div className="space-y-3">
          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <span className="text-sm font-medium">Auto-store Patterns</span>
              <p className="text-xs text-muted-foreground">
                Automatically store successful executions as patterns
              </p>
            </div>
            <input
              type="checkbox"
              checked={memoryConfig.auto_store_patterns}
              onChange={(e) => updateMemoryConfig({ auto_store_patterns: e.target.checked })}
              className="h-4 w-4"
            />
          </label>

          <div>
            <label className="block text-sm font-medium mb-1">
              Memory Retention (Days)
            </label>
            <input
              type="number"
              min="1"
              max="365"
              value={memoryConfig.memory_retention_days}
              onChange={(e) => updateMemoryConfig({ memory_retention_days: parseInt(e.target.value) })}
              className="w-full px-3 py-2 border border-border rounded-md bg-background text-sm"
            />
            <p className="text-xs text-muted-foreground mt-1">
              How long to retain memories (1-365 days)
            </p>
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="flex items-start gap-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <Info className="h-5 w-5 text-blue-600 mt-0.5" />
        <div className="text-sm text-blue-600">
          <p className="font-medium mb-1">Memory Systems</p>
          <p>
            Memory systems help your agent learn from past experiences. Enable the systems that best
            fit your use case. More systems provide better learning but may increase execution time.
          </p>
        </div>
      </div>

      {/* Validation Messages */}
      {validation?.warnings?.find((w: any) => w.field === 'memory_config') && (
        <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
          <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
          <p className="text-sm text-yellow-600">
            {validation.warnings.find((w: any) => w.field === 'memory_config').message}
          </p>
        </div>
      )}
    </div>
  );
}

export default MemoryStep;

