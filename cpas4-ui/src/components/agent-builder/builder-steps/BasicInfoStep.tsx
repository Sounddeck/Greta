// @ts-nocheck
import { useState, useEffect } from 'react';
import { FileText, Tag, Info } from 'lucide-react';

const AGENT_TYPES = [
  { value: 'research', label: 'Research', description: 'Information gathering and analysis' },
  { value: 'coding', label: 'Coding', description: 'Code generation and debugging' },
  { value: 'data_analysis', label: 'Data Analysis', description: 'Data processing and insights' },
  { value: 'content_creation', label: 'Content Creation', description: 'Writing and content generation' },
  { value: 'task_automation', label: 'Task Automation', description: 'Workflow automation' },
  { value: 'customer_support', label: 'Customer Support', description: 'Customer service and support' },
  { value: 'sales', label: 'Sales', description: 'Sales and lead generation' },
  { value: 'marketing', label: 'Marketing', description: 'Marketing campaigns and content' },
  { value: 'finance', label: 'Finance', description: 'Financial analysis and reporting' },
  { value: 'general', label: 'General', description: 'General purpose assistant' },
  { value: 'custom', label: 'Custom', description: 'Custom agent type' },
];

interface BasicInfoStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation: any;
}

export function BasicInfoStep({ draft, updateDraft, validation }: BasicInfoStepProps) {
  const [tagInput, setTagInput] = useState('');

  const addTag = () => {
    if (tagInput.trim() && !draft.tags.includes(tagInput.trim())) {
      updateDraft({
        tags: [...draft.tags, tagInput.trim()],
      });
      setTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    updateDraft({
      tags: draft.tags.filter((t: string) => t !== tag),
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTag();
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">Basic Information</h2>
        </div>

        <div className="space-y-4">
          {/* Agent Name */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Agent Name <span className="text-destructive">*</span>
            </label>
            <input
              type="text"
              value={draft.name}
              onChange={(e) => updateDraft({ name: e.target.value })}
              placeholder="e.g., Research Assistant, Code Helper, Data Analyzer"
              className={`w-full px-3 py-2 border rounded-md bg-background ${
                validation?.field_validations?.name === false
                  ? 'border-destructive'
                  : 'border-border'
              }`}
            />
            {validation?.errors?.find((e: any) => e.field === 'name') && (
              <p className="text-sm text-destructive mt-1">
                {validation.errors.find((e: any) => e.field === 'name').message}
              </p>
            )}
            {validation?.warnings?.find((w: any) => w.field === 'name') && (
              <p className="text-sm text-yellow-600 mt-1">
                {validation.warnings.find((w: any) => w.field === 'name').message}
              </p>
            )}
          </div>

          {/* Agent Type */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Agent Type <span className="text-destructive">*</span>
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {AGENT_TYPES.map((type) => (
                <button
                  key={type.value}
                  onClick={() => updateDraft({ agent_type: type.value })}
                  className={`p-3 border rounded-md text-left transition-colors ${
                    draft.agent_type === type.value
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:border-primary/50'
                  }`}
                >
                  <div className="font-medium text-sm">{type.label}</div>
                  <div className="text-xs text-muted-foreground mt-1">{type.description}</div>
                </button>
              ))}
            </div>
            {validation?.errors?.find((e: any) => e.field === 'agent_type') && (
              <p className="text-sm text-destructive mt-2">
                {validation.errors.find((e: any) => e.field === 'agent_type').message}
              </p>
            )}
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Description
            </label>
            <textarea
              value={draft.description}
              onChange={(e) => updateDraft({ description: e.target.value })}
              placeholder="Describe what this agent does and its purpose..."
              rows={4}
              className={`w-full px-3 py-2 border rounded-md bg-background resize-none ${
                validation?.field_validations?.description === false
                  ? 'border-yellow-500'
                  : 'border-border'
              }`}
            />
            {validation?.warnings?.find((w: any) => w.field === 'description') && (
              <p className="text-sm text-yellow-600 mt-1">
                {validation.warnings.find((w: any) => w.field === 'description').message}
              </p>
            )}
            {validation?.suggestions?.find((s: any) => s.field === 'description') && (
              <p className="text-sm text-blue-600 mt-1 flex items-center gap-1">
                <Info className="h-3 w-3" />
                {validation.suggestions.find((s: any) => s.field === 'description').message}
              </p>
            )}
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Tags
            </label>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Add tags (press Enter)"
                className="flex-1 px-3 py-2 border border-border rounded-md bg-background"
              />
              <button
                onClick={addTag}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
              >
                Add
              </button>
            </div>
            {draft.tags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {draft.tags.map((tag: string) => (
                  <span
                    key={tag}
                    className="inline-flex items-center gap-1 px-3 py-1 bg-primary/10 text-primary rounded-full text-sm"
                  >
                    <Tag className="h-3 w-3" />
                    {tag}
                    <button
                      onClick={() => removeTag(tag)}
                      className="ml-1 hover:text-destructive"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Validation Summary */}
      {validation && (
        <div className="bg-card border border-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Validation Summary</h3>
          
          {validation.errors.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-destructive mb-2">Errors</h4>
              <ul className="space-y-1">
                {validation.errors.map((error: any, index: number) => (
                  <li key={index} className="text-sm text-destructive flex items-start gap-2">
                    <span className="text-destructive">•</span>
                    <span>{error.message}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {validation.warnings.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-yellow-600 mb-2">Warnings</h4>
              <ul className="space-y-1">
                {validation.warnings.map((warning: any, index: number) => (
                  <li key={index} className="text-sm text-yellow-600 flex items-start gap-2">
                    <span className="text-yellow-600">•</span>
                    <span>{warning.message}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {validation.suggestions.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-blue-600 mb-2">Suggestions</h4>
              <ul className="space-y-1">
                {validation.suggestions.map((suggestion: any, index: number) => (
                  <li key={index} className="text-sm text-blue-600 flex items-start gap-2">
                    <span className="text-blue-600">•</span>
                    <span>{suggestion.message}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default BasicInfoStep;

