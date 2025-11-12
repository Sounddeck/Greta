// @ts-nocheck
import { useState, useEffect } from 'react';
import { Wrench, Search, Check, X, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { api } from '../../../lib/api';

// Tool interface matching backend ToolDefinition
interface Tool {
  tool_id: string;
  name: string;
  description: string;
  category: string;
  permission_level: string;
  parameters?: Record<string, any>;
  rate_limit?: number;
  timeout?: number;
  tags?: string[];
  is_available?: boolean;
}

// Tool categories from backend
const TOOL_CATEGORIES = [
  { value: 'all', label: 'All Categories' },
  { value: 'osint', label: 'OSINT' },
  { value: 'data_processing', label: 'Data Processing' },
  { value: 'web_api', label: 'Web API' },
  { value: 'database', label: 'Database' },
  { value: 'communication', label: 'Communication' },
  { value: 'file_system', label: 'File System' },
  { value: 'code_execution', label: 'Code Execution' },
  { value: 'ai_ml', label: 'AI/ML' },
  { value: 'business', label: 'Business' },
  { value: 'analytics', label: 'Analytics' },
  { value: 'security', label: 'Security' },
];

interface ToolsStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation: any;
}

export function ToolsStep({ draft, updateDraft, validation }: ToolsStepProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [availableTools, setAvailableTools] = useState<Tool[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const selectedTools = draft.tools || [];

  // Fetch tools from Tool Registry API
  useEffect(() => {
    fetchTools();
  }, []);

  const fetchTools = async () => {
    setIsLoading(true);
    try {
      const response = await api.get('/tools/discover', {
        params: {
          limit: 1000 // Get all tools
        }
      });

      const tools = response.data.tools || [];
      setAvailableTools(tools);

      if (tools.length > 0) {
        console.log(`✅ Loaded ${tools.length} tools from Tool Registry`);
      }
    } catch (error: any) {
      console.error('Failed to fetch tools:', error);
      toast.error('Failed to load tools from Tool Registry');
      setAvailableTools([]);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredTools = availableTools.filter((tool) => {
    const matchesSearch = tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         tool.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || tool.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const isToolSelected = (toolId: string) => {
    return selectedTools.some((t: any) => t.tool_id === toolId);
  };

  const toggleTool = (tool: Tool) => {
    console.log('🔨 toggleTool called for:', tool.tool_id, tool.name);
    console.log('🔨 Currently selected tools:', selectedTools);
    console.log('🔨 Is tool selected?', isToolSelected(tool.tool_id));

    if (isToolSelected(tool.tool_id)) {
      // Remove tool
      console.log('🔨 Removing tool:', tool.tool_id);
      updateDraft({
        tools: selectedTools.filter((t: any) => t.tool_id !== tool.tool_id),
      });
    } else {
      // Add tool with permission level from Tool Registry
      console.log('🔨 Adding tool:', tool.tool_id);
      const newTools = [
        ...selectedTools,
        {
          tool_id: tool.tool_id,
          name: tool.name,
          description: tool.description,
          category: tool.category,
          permission_level: tool.permission_level,
          configuration: {},
          is_enabled: true,
        },
      ];
      console.log('🔨 New tools array:', newTools);
      updateDraft({
        tools: newTools,
      });
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Wrench className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">Tools & Capabilities</h2>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search tools..."
              className="w-full pl-10 pr-3 py-2 border border-border rounded-md bg-background"
              disabled={isLoading}
            />
          </div>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-border rounded-md bg-background min-w-[200px]"
            disabled={isLoading}
          >
            {TOOL_CATEGORIES.map((category) => (
              <option key={category.value} value={category.value}>
                {category.label}
              </option>
            ))}
          </select>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <span className="ml-3 text-muted-foreground">Loading tools from Tool Registry...</span>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && availableTools.length === 0 && (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No tools available. Please check Tool Registry configuration.</p>
          </div>
        )}

        {/* Available Tools */}
        {!isLoading && availableTools.length > 0 && (
          <>
            <div className="mb-4 text-sm text-muted-foreground">
              Showing {filteredTools.length} of {availableTools.length} tools
              {selectedTools.length > 0 && ` • ${selectedTools.length} selected`}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTools.map((tool) => {
            const selected = isToolSelected(tool.tool_id);
            const selectedTool = selectedTools.find((t: any) => t.tool_id === tool.tool_id);

            return (
              <div
                key={tool.tool_id}
                onClick={() => {
                  console.log('🖱️ Tool card clicked:', tool.name);
                  toggleTool(tool);
                }}
                className={`border rounded-lg p-4 transition-all cursor-pointer ${
                  selected
                    ? 'border-green-500 bg-green-500/10 shadow-lg'
                    : 'border-border hover:border-primary/50 hover:shadow-md'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h3 className="font-medium text-sm flex items-center gap-2">
                      {tool.name}
                      {selected && <Check className="h-4 w-4 text-green-600" />}
                      {tool.permission_level === 'RESTRICTED' && (
                        <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded">
                          🔒 Restricted
                        </span>
                      )}
                    </h3>
                    <p className="text-xs text-muted-foreground mt-1">{tool.description}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs bg-muted px-2 py-0.5 rounded">
                        {tool.category}
                      </span>
                      {tool.tags && tool.tags.length > 0 && (
                        <span className="text-xs text-muted-foreground">
                          {tool.tags.slice(0, 2).join(', ')}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent double-triggering
                      console.log('🖱️ Toggle button clicked:', tool.name);
                      toggleTool(tool);
                    }}
                    className={`ml-2 p-2 rounded-md transition-all ${
                      selected
                        ? 'bg-green-600 text-white hover:bg-green-700'
                        : 'bg-gray-300 hover:bg-gray-400 text-gray-700'
                    }`}
                    title={selected ? 'Remove tool' : 'Add tool'}
                  >
                    {selected ? <Check className="h-5 w-5" /> : <X className="h-5 w-5" />}
                  </button>
                </div>

                {/* Tool Info (only show if selected) */}
                {selected && selectedTool && (
                  <div className="mt-3 pt-3 border-t border-border">
                    <p className="text-xs text-muted-foreground">
                      Permission Level: <span className="font-medium">{tool.permission_level}</span>
                    </p>
                    {tool.rate_limit && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Rate Limit: {tool.rate_limit} calls/min
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
            </div>
          </>
        )}
      </div>

      {/* Selected Tools Summary */}
      {selectedTools.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">
            Selected Tools ({selectedTools.length})
          </h3>
          <div className="space-y-2">
            {selectedTools.map((tool: any) => {
              const toolInfo = availableTools.find((t) => t.tool_id === tool.tool_id);
              return (
                <div
                  key={tool.tool_id}
                  className="flex items-center justify-between p-3 bg-muted rounded-md"
                >
                  <div>
                    <span className="font-medium text-sm">{toolInfo?.name || tool.tool_id}</span>
                    <span className="text-xs text-muted-foreground ml-2">
                      ({toolInfo?.category || 'unknown'})
                    </span>
                  </div>
                  <button
                    onClick={() => toolInfo && toggleTool(toolInfo)}
                    className="text-destructive hover:text-destructive/80"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Validation Info */}
      {validation?.suggestions?.find((s: any) => s.field === 'tools') && (
        <div className="bg-blue-500/10 border border-blue-500 rounded-lg p-4">
          <p className="text-sm text-blue-600">
            {validation.suggestions.find((s: any) => s.field === 'tools').message}
          </p>
        </div>
      )}
    </div>
  );
}

export default ToolsStep;

