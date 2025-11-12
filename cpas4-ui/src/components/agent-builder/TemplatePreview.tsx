/**
 * Template Preview Component
 * 
 * Modal dialog for previewing agent template details before using
 */

import React from 'react';
import { Zap, X, Code, Wrench, Tag, TrendingUp } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import type { AgentTemplate } from '../../stores/agentBuilderStore';

interface TemplatePreviewProps {
  template: AgentTemplate | null;
  isOpen: boolean;
  onClose: () => void;
  onUse: (templateId: string) => void;
}

export function TemplatePreview({ template, isOpen, onClose, onUse }: TemplatePreviewProps) {
  if (!template) return null;

  const handleUse = () => {
    onUse(template.template_id);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl">{template.name}</DialogTitle>
          <DialogDescription className="text-base">
            {template.description}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Agent Type */}
          <div>
            <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
              <Tag className="h-4 w-4" />
              Agent Type
            </h3>
            <Badge variant="outline" className="text-sm">
              {template.agent_type.replace('_', ' ')}
            </Badge>
          </div>

          {/* Capabilities */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Capabilities ({template.capabilities.length})
            </h3>
            <div className="flex flex-wrap gap-2">
              {template.capabilities.map((cap) => (
                <Badge key={cap} variant="secondary">
                  {cap}
                </Badge>
              ))}
            </div>
          </div>

          {/* Default Tools */}
          {template.default_tools.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <Wrench className="h-4 w-4" />
                Included Tools ({template.default_tools.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {template.default_tools.map((tool) => (
                  <Badge key={tool} variant="outline" className="font-mono text-xs">
                    {tool}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* System Prompt */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Code className="h-4 w-4" />
              System Prompt
            </h3>
            <div className="bg-muted rounded-lg p-4 border border-border">
              <pre className="text-sm whitespace-pre-wrap font-mono text-muted-foreground">
                {template.default_prompt}
              </pre>
            </div>
          </div>

          {/* Configuration */}
          {Object.keys(template.configuration).length > 0 && (
            <div>
              <h3 className="text-sm font-semibold mb-3">Default Configuration</h3>
              <div className="bg-muted rounded-lg p-4 border border-border">
                <pre className="text-sm whitespace-pre-wrap font-mono text-muted-foreground">
                  {JSON.stringify(template.configuration, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Tags */}
          {template.tags.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold mb-3">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {template.tags.map((tag) => (
                  <span
                    key={tag}
                    className="text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Usage Statistics */}
          <div className="border-t border-border pt-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <TrendingUp className="h-4 w-4" />
              <span>
                This template has been used <strong>{template.usage_count}</strong> time
                {template.usage_count !== 1 ? 's' : ''}
              </span>
            </div>
          </div>

          {/* Metadata */}
          <div className="border-t border-border pt-4 text-xs text-muted-foreground space-y-1">
            <p>Created: {new Date(template.created_at).toLocaleDateString()}</p>
            <p>Last Updated: {new Date(template.updated_at).toLocaleDateString()}</p>
            <p>Template ID: {template.template_id}</p>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleUse} className="gap-2">
            <Zap className="h-4 w-4" />
            Use This Template
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

