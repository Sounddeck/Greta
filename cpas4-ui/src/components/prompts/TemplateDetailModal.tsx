import React, { useState } from 'react';
import { X, Copy, Send, Star, Edit, Sparkles, Check } from 'lucide-react';
import { usePromptTemplateStore } from '../../stores/promptTemplateStore';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { toast } from 'sonner';

interface TemplateDetailModalProps {
  onClose: () => void;
  onEdit: (template: any) => void;
}

export default function TemplateDetailModal({ onClose, onEdit }: TemplateDetailModalProps) {
  const { selectedTemplate, renderTemplate, rateTemplate } = usePromptTemplateStore();
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [renderedPrompt, setRenderedPrompt] = useState('');
  const [rating, setRating] = useState(0);
  const [copied, setCopied] = useState(false);

  // Early return if no template selected
  if (!selectedTemplate) {
    console.warn('[TemplateDetailModal] No template selected');
    return null;
  }

  // Defensive checks for template properties
  const templateName = selectedTemplate.name || 'Untitled Template';
  const templateDescription = selectedTemplate.description || 'No description available';
  const templateCategory = selectedTemplate.category || 'general';
  const templateVersion = selectedTemplate.version || '1.0';
  const templateRating = selectedTemplate.rating || 0;
  const templateVariables = selectedTemplate.variables || [];
  const templateTags = selectedTemplate.tags || [];
  const templateContent = selectedTemplate.template || '';
  const isSystemTemplate = selectedTemplate.is_system || false;
  const createdAt = selectedTemplate.created_at || new Date().toISOString();
  const updatedAt = selectedTemplate.updated_at || new Date().toISOString();
  const usageCount = selectedTemplate.usage_count || 0;

  const handleVariableChange = (varName: string, value: string) => {
    setVariables((prev) => ({ ...prev, [varName]: value }));
  };

  const handleRender = async () => {
    try {
      const rendered = await renderTemplate(selectedTemplate.id, variables);
      setRenderedPrompt(rendered);
      toast.success('Template rendered successfully!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to render template');
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(renderedPrompt);
    setCopied(true);
    toast.success('Copied to clipboard!');
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRate = async (newRating: number) => {
    try {
      await rateTemplate(selectedTemplate.id, newRating);
      setRating(newRating);
      toast.success('Rating submitted!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to rate template');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-border flex items-start justify-between">
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-foreground mb-1">{templateName}</h2>
            <p className="text-muted-foreground">{templateDescription}</p>
            <div className="flex items-center gap-3 mt-2">
              <span className="inline-block px-2 py-1 bg-primary/10 text-primary rounded text-sm">
                {templateCategory.replace('_', ' ')}
              </span>
              <span className="text-sm text-muted-foreground">v{templateVersion}</span>
              <div className="flex items-center gap-1">
                <Star className="h-4 w-4 fill-yellow-500 text-yellow-500" />
                <span className="text-sm">{templateRating.toFixed(1)}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!isSystemTemplate && (
              <Button variant="outline" size="sm" onClick={() => onEdit(selectedTemplate)}>
                <Edit className="h-4 w-4 mr-2" />
                Edit
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Template Preview */}
          <div>
            <h3 className="text-sm font-semibold text-foreground mb-2">Template</h3>
            <div className="bg-muted/50 border border-border rounded-lg p-4">
              <pre className="text-sm text-foreground whitespace-pre-wrap font-mono">
                {templateContent}
              </pre>
            </div>
          </div>

          {/* Variables Input */}
          {templateVariables.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                Variables ({templateVariables.length})
              </h3>
              <div className="space-y-3">
                {templateVariables.map((varName) => (
                  <div key={varName}>
                    <label className="block text-sm font-medium text-foreground mb-1">
                      {varName.replace(/_/g, ' ')}
                    </label>
                    <Input
                      type="text"
                      placeholder={`Enter ${varName}...`}
                      value={variables[varName] || ''}
                      onChange={(e) => handleVariableChange(varName, e.target.value)}
                    />
                  </div>
                ))}
              </div>
              <Button onClick={handleRender} className="mt-4 w-full">
                Render Template
              </Button>
            </div>
          )}

          {/* Rendered Output */}
          {renderedPrompt && (
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Rendered Prompt</h3>
              <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
                <pre className="text-sm text-foreground whitespace-pre-wrap">
                  {renderedPrompt}
                </pre>
              </div>
              <div className="flex gap-2 mt-3">
                <Button onClick={handleCopy} variant="outline" className="flex-1">
                  {copied ? (
                    <>
                      <Check className="h-4 w-4 mr-2" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4 mr-2" />
                      Copy
                    </>
                  )}
                </Button>
                <Button className="flex-1">
                  <Send className="h-4 w-4 mr-2" />
                  Send to Agent
                </Button>
              </div>
            </div>
          )}

          {/* Tags */}
          {templateTags.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {templateTags.map((tag) => (
                  <span
                    key={tag}
                    className="inline-block px-3 py-1 bg-muted text-muted-foreground rounded-full text-sm"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Rating */}
          <div>
            <h3 className="text-sm font-semibold text-foreground mb-2">Rate this template</h3>
            <div className="flex gap-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => handleRate(star)}
                  className="transition-transform hover:scale-110"
                >
                  <Star
                    className={`h-6 w-6 ${
                      star <= (rating || templateRating)
                        ? 'fill-yellow-500 text-yellow-500'
                        : 'text-muted-foreground'
                    }`}
                  />
                </button>
              ))}
            </div>
          </div>

          {/* Metadata */}
          <div className="text-xs text-muted-foreground space-y-1">
            <p>Created: {new Date(createdAt).toLocaleDateString()}</p>
            <p>Last Updated: {new Date(updatedAt).toLocaleDateString()}</p>
            <p>Usage Count: {usageCount}</p>
            {isSystemTemplate && <p className="text-primary">System Template (Read-only)</p>}
          </div>
        </div>
      </div>
    </div>
  );
}

