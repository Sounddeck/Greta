import React from 'react';
import { Star, TrendingUp, Lock, Edit, Sparkles } from 'lucide-react';
import { Button } from '../ui/button';
import type { PromptTemplate } from '../../stores/promptTemplateStore';

interface TemplateCardProps {
  template: PromptTemplate;
  onClick: () => void;
  onEdit: () => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  research: 'bg-blue-500/10 text-blue-500 border-blue-500/20',
  coding: 'bg-purple-500/10 text-purple-500 border-purple-500/20',
  data_analysis: 'bg-green-500/10 text-green-500 border-green-500/20',
  writing: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20',
  business: 'bg-orange-500/10 text-orange-500 border-orange-500/20',
  creative: 'bg-pink-500/10 text-pink-500 border-pink-500/20',
  technical: 'bg-cyan-500/10 text-cyan-500 border-cyan-500/20',
  general: 'bg-gray-500/10 text-gray-500 border-gray-500/20',
};

export default function TemplateCard({ template, onClick, onEdit }: TemplateCardProps) {
  const categoryColor = CATEGORY_COLORS[template.category] || CATEGORY_COLORS.general;

  return (
    <div
      className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-all cursor-pointer group"
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
              {template.name}
            </h3>
            {template.is_system && (
              <Lock className="h-3 w-3 text-muted-foreground" title="System Template" />
            )}
          </div>
          <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${categoryColor}`}>
            {template.category.replace('_', ' ')}
          </span>
        </div>
        {!template.is_system && (
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onEdit();
            }}
            className="opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <Edit className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Description */}
      <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
        {template.description}
      </p>

      {/* Tags */}
      {template.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {template.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="inline-block px-2 py-0.5 bg-muted text-muted-foreground rounded text-xs"
            >
              #{tag}
            </span>
          ))}
          {template.tags.length > 3 && (
            <span className="inline-block px-2 py-0.5 text-muted-foreground text-xs">
              +{template.tags.length - 3} more
            </span>
          )}
        </div>
      )}

      {/* Variables */}
      {template.variables.length > 0 && (
        <div className="mb-3">
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Sparkles className="h-3 w-3" />
            <span>{template.variables.length} variable{template.variables.length !== 1 ? 's' : ''}</span>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-border">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {/* Rating */}
          <div className="flex items-center gap-1">
            <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
            <span>{template.rating.toFixed(1)}</span>
          </div>

          {/* Usage Count */}
          <div className="flex items-center gap-1">
            <TrendingUp className="h-3 w-3" />
            <span>{template.usage_count} uses</span>
          </div>
        </div>

        {/* Version */}
        <span className="text-xs text-muted-foreground">
          v{template.version}
        </span>
      </div>
    </div>
  );
}

