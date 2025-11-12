import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Sparkles, TrendingUp, Zap, Code, BarChart, Palette, Headphones, DollarSign, Megaphone } from 'lucide-react';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Select } from '../../components/ui/select';
import { useAgentBuilderStore } from '../../stores/agentBuilderStore';
import type { AgentType } from '../../stores/agentBuilderStore';
import { CardSkeleton } from '../../components/skeletons/TableSkeleton';

export function TemplateGallery() {
  console.log('[TemplateGallery] 🔵 Component render started');
  const navigate = useNavigate();

  // Zustand store
  const {
    templates,
    isLoadingTemplates,
    fetchTemplates,
    setSelectedTemplate,
  } = useAgentBuilderStore();

  console.log('[TemplateGallery] 📊 Store state:', {
    templatesCount: templates?.length || 0,
    isLoadingTemplates,
    templatesArray: templates,
    templatesType: typeof templates,
    isArray: Array.isArray(templates)
  });

  // Local state for filters
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<AgentType | 'ALL'>('ALL');

  console.log('[TemplateGallery] 🔍 Filter state:', {
    searchQuery,
    filterType
  });

  // Load templates on mount
  useEffect(() => {
    console.log('[TemplateGallery] ⚡ Component mounted, calling fetchTemplates...');
    const result = fetchTemplates();
    console.log('[TemplateGallery] 📞 fetchTemplates() called, result:', result);
  }, [fetchTemplates]);

  // Debug: Log templates state changes
  useEffect(() => {
    console.log('[TemplateGallery] 🔄 Templates state changed:', {
      total: templates?.length || 0,
      isLoading: isLoadingTemplates,
      templatesData: templates
    });
  }, [templates, isLoadingTemplates]);

  // Filter templates
  const filteredTemplates = templates.filter((template) => {
    const matchesSearch =
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesType = filterType === 'ALL' || template.agent_type === filterType;

    return matchesSearch && matchesType;
  });

  console.log('[TemplateGallery] 🔎 Filtered templates:', {
    totalTemplates: templates.length,
    filteredCount: filteredTemplates.length,
    filteredTemplates: filteredTemplates
  });

  // Handle template selection
  const handleUseTemplate = (templateId: string) => {
    const template = templates.find(t => t.template_id === templateId);
    if (template) {
      setSelectedTemplate(template);
      navigate('/agent-builder/agents/new', { state: { templateId } });
    }
  };

  // Agent type icons
  const getTypeIcon = (type: AgentType) => {
    const icons = {
      GENERAL: Sparkles,
      RESEARCH: TrendingUp,
      CODING: Code,
      DATA_ANALYSIS: BarChart,
      CREATIVE: Palette,
      CUSTOMER_SERVICE: Headphones,
      SALES: DollarSign,
      MARKETING: Megaphone,
    };
    return icons[type] || Sparkles;
  };

  // Agent type colors
  const getTypeColor = (type: AgentType) => {
    const colors = {
      GENERAL: 'bg-purple-100 text-purple-700',
      RESEARCH: 'bg-blue-100 text-blue-700',
      CODING: 'bg-green-100 text-green-700',
      DATA_ANALYSIS: 'bg-orange-100 text-orange-700',
      CREATIVE: 'bg-pink-100 text-pink-700',
      CUSTOMER_SERVICE: 'bg-cyan-100 text-cyan-700',
      SALES: 'bg-emerald-100 text-emerald-700',
      MARKETING: 'bg-indigo-100 text-indigo-700',
    };
    return colors[type] || 'bg-gray-100 text-gray-700';
  };

  console.log('[TemplateGallery] 🎨 RENDERING COMPONENT - Return statement reached');

  return (
    <div className="h-full overflow-auto">
      <div className="space-y-6 p-6">
        {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">Template Gallery</h2>
        <p className="text-muted-foreground">
          Choose a pre-built template to quickly create your agent
        </p>
      </div>

      {/* Filters and Search */}
      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search templates..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <Select value={filterType} onValueChange={(value) => setFilterType(value as AgentType | 'ALL')}>
          <option value="ALL">All Types</option>
          <option value="GENERAL">General</option>
          <option value="RESEARCH">Research</option>
          <option value="CODING">Coding</option>
          <option value="DATA_ANALYSIS">Data Analysis</option>
          <option value="CREATIVE">Creative</option>
          <option value="CUSTOMER_SERVICE">Customer Service</option>
          <option value="SALES">Sales</option>
          <option value="MARKETING">Marketing</option>
        </Select>
      </div>

      {/* Results count */}
      <div className="text-sm text-muted-foreground">
        {filteredTemplates.length} template{filteredTemplates.length !== 1 ? 's' : ''} found
      </div>

      {/* Template Grid */}
      {(() => {
        const showSkeleton = isLoadingTemplates && templates.length === 0;
        const showEmpty = filteredTemplates.length === 0;
        const showGrid = !showSkeleton && !showEmpty;

        console.log('[TemplateGallery] 🎨 Rendering decision:', {
          isLoadingTemplates,
          templatesLength: templates.length,
          filteredLength: filteredTemplates.length,
          showSkeleton,
          showEmpty,
          showGrid
        });

        return null;
      })()}
      {isLoadingTemplates && templates.length === 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : filteredTemplates.length === 0 ? (
        <div className="border border-border rounded-lg p-12 text-center">
          <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
            <Sparkles className="h-6 w-6 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">No templates found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery || filterType !== 'ALL'
              ? 'Try adjusting your filters or search query'
              : 'No templates available yet'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredTemplates.map((template) => {
            const TypeIcon = getTypeIcon(template.agent_type);
            const typeColor = getTypeColor(template.agent_type);
            
            return (
              <div
                key={template.template_id}
                className="group border border-border rounded-lg p-6 hover:border-primary hover:shadow-lg transition-all cursor-pointer bg-card"
                onClick={() => handleUseTemplate(template.template_id)}
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-3 rounded-lg ${typeColor}`}>
                    <TypeIcon className="h-6 w-6" />
                  </div>
                  <span className="text-xs px-2 py-1 bg-muted text-muted-foreground rounded-full">
                    Used {template.usage_count}x
                  </span>
                </div>

                {/* Content */}
                <div className="space-y-3">
                  <div>
                    <h3 className="font-semibold text-lg mb-1 group-hover:text-primary transition-colors">
                      {template.name}
                    </h3>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {template.description}
                    </p>
                  </div>

                  {/* Type Badge */}
                  <div>
                    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium ${typeColor}`}>
                      {template.agent_type.replace('_', ' ')}
                    </span>
                  </div>

                  {/* Capabilities */}
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">Capabilities:</p>
                    <div className="flex flex-wrap gap-1">
                      {template.capabilities.slice(0, 3).map((cap) => (
                        <span
                          key={cap}
                          className="text-xs px-2 py-0.5 bg-primary/10 text-primary rounded-full"
                        >
                          {cap}
                        </span>
                      ))}
                      {template.capabilities.length > 3 && (
                        <span className="text-xs px-2 py-0.5 bg-muted text-muted-foreground rounded-full">
                          +{template.capabilities.length - 3} more
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Tools */}
                  {template.default_tools.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-muted-foreground">Tools:</p>
                      <div className="flex flex-wrap gap-1">
                        {template.default_tools.slice(0, 3).map((tool) => (
                          <span
                            key={tool}
                            className="text-xs px-2 py-0.5 bg-secondary text-secondary-foreground rounded-full"
                          >
                            {tool}
                          </span>
                        ))}
                        {template.default_tools.length > 3 && (
                          <span className="text-xs px-2 py-0.5 bg-muted text-muted-foreground rounded-full">
                            +{template.default_tools.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Tags */}
                  {template.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 pt-2 border-t border-border">
                      {template.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="text-xs text-muted-foreground"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Action Button */}
                <div className="mt-4 pt-4 border-t border-border">
                  <Button 
                    className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                    variant="outline"
                  >
                    <Zap className="h-4 w-4 mr-2" />
                    Use This Template
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Create from Scratch Option */}
      <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary transition-colors">
        <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
          <Sparkles className="h-6 w-6 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">Start from Scratch</h3>
        <p className="text-muted-foreground mb-4">
          Create a completely custom agent without using a template
        </p>
        <Button onClick={() => navigate('/agent-builder/agents/new')}>
          Create Custom Agent
        </Button>
      </div>
      </div>
    </div>
  );
}

