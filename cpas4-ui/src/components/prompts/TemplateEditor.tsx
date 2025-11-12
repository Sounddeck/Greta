import React, { useState, useEffect } from 'react';
import { X, Save, Sparkles, Plus, Trash2 } from 'lucide-react';
import { usePromptTemplateStore } from '../../stores/promptTemplateStore';
import type { PromptTemplate } from '../../stores/promptTemplateStore';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { toast } from 'sonner';

interface TemplateEditorProps {
  template: PromptTemplate | null;
  onClose: () => void;
}

const CATEGORIES = [
  'research',
  'coding',
  'data_analysis',
  'writing',
  'business',
  'creative',
  'technical',
  'general',
];

export default function TemplateEditor({ template, onClose }: TemplateEditorProps) {
  const { createTemplate, updateTemplate } = usePromptTemplateStore();
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    category: 'general',
    template: '',
    tags: [] as string[],
  });
  const [newTag, setNewTag] = useState('');
  const [saving, setSaving] = useState(false);
  const [detectedVariables, setDetectedVariables] = useState<string[]>([]);

  useEffect(() => {
    if (template) {
      setFormData({
        name: template.name,
        description: template.description,
        category: template.category,
        template: template.template,
        tags: template.tags,
      });
    }
  }, [template]);

  useEffect(() => {
    // Auto-detect variables in template
    const matches = formData.template.match(/\{([^}]+)\}/g);
    if (matches) {
      const vars = matches.map((m) => m.slice(1, -1));
      setDetectedVariables([...new Set(vars)]);
    } else {
      setDetectedVariables([]);
    }
  }, [formData.template]);

  const handleChange = (field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleAddTag = () => {
    if (newTag.trim() && !formData.tags.includes(newTag.trim())) {
      handleChange('tags', [...formData.tags, newTag.trim()]);
      setNewTag('');
    }
  };

  const handleRemoveTag = (tag: string) => {
    handleChange('tags', formData.tags.filter((t) => t !== tag));
  };

  const handleSave = async () => {
    // Validation
    if (!formData.name.trim()) {
      toast.error('Please enter a template name');
      return;
    }
    if (!formData.description.trim()) {
      toast.error('Please enter a description');
      return;
    }
    if (!formData.template.trim()) {
      toast.error('Please enter a template');
      return;
    }

    setSaving(true);
    try {
      const data = {
        ...formData,
        variables: detectedVariables,
      };

      if (template) {
        await updateTemplate(template.id, data);
        toast.success('Template updated successfully!');
      } else {
        await createTemplate(data);
        toast.success('Template created successfully!');
      }
      onClose();
    } catch (error: any) {
      // Handle 404 gracefully - backend endpoints not yet implemented
      if (error.response?.status === 404) {
        toast.error('Template creation is not yet available - backend endpoints are being implemented');
      } else {
        toast.error(error.message || 'Failed to save template');
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-border flex items-center justify-between">
          <h2 className="text-2xl font-bold text-foreground">
            {template ? 'Edit Template' : 'Create New Template'}
          </h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Template Name *
            </label>
            <Input
              type="text"
              placeholder="e.g., Research Paper Analysis"
              value={formData.name}
              onChange={(e) => handleChange('name', e.target.value)}
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Description *
            </label>
            <Input
              type="text"
              placeholder="Brief description of what this template does"
              value={formData.description}
              onChange={(e) => handleChange('description', e.target.value)}
            />
          </div>

          {/* Category */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Category *
            </label>
            <select
              value={formData.category}
              onChange={(e) => handleChange('category', e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {cat.replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>

          {/* Template */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Template Content *
            </label>
            <p className="text-xs text-muted-foreground mb-2">
              Use {'{variable_name}'} for dynamic placeholders
            </p>
            <textarea
              value={formData.template}
              onChange={(e) => handleChange('template', e.target.value)}
              placeholder="Enter your template here. Use {variable_name} for placeholders..."
              className="w-full h-64 px-3 py-2 bg-background border border-border rounded-lg text-foreground font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary resize-none"
            />
          </div>

          {/* Detected Variables */}
          {detectedVariables.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-foreground mb-2 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                Detected Variables ({detectedVariables.length})
              </label>
              <div className="flex flex-wrap gap-2 p-3 bg-primary/5 border border-primary/20 rounded-lg">
                {detectedVariables.map((varName) => (
                  <span
                    key={varName}
                    className="inline-block px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-mono"
                  >
                    {'{' + varName + '}'}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Tags
            </label>
            <div className="flex gap-2 mb-2">
              <Input
                type="text"
                placeholder="Add a tag..."
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddTag())}
              />
              <Button onClick={handleAddTag} variant="outline">
                <Plus className="h-4 w-4" />
              </Button>
            </div>
            {formData.tags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {formData.tags.map((tag) => (
                  <span
                    key={tag}
                    className="inline-flex items-center gap-1 px-3 py-1 bg-muted text-muted-foreground rounded-full text-sm"
                  >
                    #{tag}
                    <button
                      onClick={() => handleRemoveTag(tag)}
                      className="hover:text-destructive transition-colors"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-border flex items-center justify-end gap-3">
          <Button variant="outline" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                {template ? 'Update' : 'Create'} Template
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

