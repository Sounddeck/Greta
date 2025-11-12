/**
 * Team Templates Component
 * 
 * Browse and deploy pre-built team templates
 */

import React, { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Dialog } from '../ui/dialog';
import { Input } from '../ui/input';
import { Label } from '../ui/label';

interface TeamTemplate {
  template_id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  agents: any[];
  workflow_pattern: string;
  termination: any;
  metadata: {
    estimated_execution_time?: string;
    success_rate?: number;
    total_deployments?: number;
    average_rating?: number;
    tags?: string[];
    difficulty?: string;
  };
  is_featured: boolean;
}

interface TeamTemplatesProps {
  onDeploy?: (template: TeamTemplate, customizations?: any) => void;
  category?: string;
}

export default function TeamTemplates({ onDeploy, category }: TeamTemplatesProps) {
  const [templates, setTemplates] = useState<TeamTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedTemplate, setSelectedTemplate] = useState<TeamTemplate | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showCustomize, setShowCustomize] = useState(false);
  const [filterCategory, setFilterCategory] = useState(category || 'all');
  const [searchQuery, setSearchQuery] = useState('');

  // Customization state
  const [customName, setCustomName] = useState('');
  const [customDescription, setCustomDescription] = useState('');

  useEffect(() => {
    fetchTemplates();
  }, [filterCategory]);

  const fetchTemplates = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filterCategory && filterCategory !== 'all') {
        params.append('category', filterCategory);
      }
      params.append('is_featured', 'true');

      const response = await fetch(`/api/v1/teams/templates/list?${params}`);
      const data = await response.json();
      
      if (data.success) {
        setTemplates(data.templates);
      }
    } catch (error) {
      console.error('Failed to fetch templates:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePreview = (template: TeamTemplate) => {
    setSelectedTemplate(template);
    setShowPreview(true);
  };

  const handleCustomize = (template: TeamTemplate) => {
    setSelectedTemplate(template);
    setCustomName(template.name);
    setCustomDescription(template.description);
    setShowCustomize(true);
  };

  const handleDeploy = async (template: TeamTemplate, customize: boolean = false) => {
    try {
      const customizations = customize ? {
        name: customName,
        description: customDescription,
      } : undefined;

      const response = await fetch(`/api/v1/teams/templates/${template.template_id}/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ customizations }),
      });

      const data = await response.json();
      
      if (response.ok) {
        onDeploy?.(template, customizations);
        setShowCustomize(false);
        alert(`Team "${data.name}" deployed successfully!`);
      } else {
        alert('Failed to deploy template');
      }
    } catch (error) {
      console.error('Failed to deploy template:', error);
      alert('Failed to deploy template');
    }
  };

  const filteredTemplates = templates.filter(template => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        template.name.toLowerCase().includes(query) ||
        template.description.toLowerCase().includes(query) ||
        template.metadata.tags?.some(tag => tag.toLowerCase().includes(query))
      );
    }
    return true;
  });

  const categories = [
    { value: 'all', label: 'All Templates', icon: '📋' },
    { value: 'sales', label: 'Sales', icon: '💼' },
    { value: 'dev', label: 'Development', icon: '💻' },
    { value: 'support', label: 'Support', icon: '🎧' },
    { value: 'marketing', label: 'Marketing', icon: '📢' },
    { value: 'research', label: 'Research', icon: '🔬' },
  ];

  const difficultyColors = {
    beginner: 'bg-green-100 text-green-800',
    intermediate: 'bg-yellow-100 text-yellow-800',
    advanced: 'bg-red-100 text-red-800',
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Team Templates</h2>
        <p className="text-gray-600">Deploy pre-built agent teams with one click</p>
      </div>

      {/* Filters */}
      <div className="mb-6 flex gap-4">
        <div className="flex-1">
          <Input
            type="search"
            placeholder="Search templates..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full"
          />
        </div>
        <div className="flex gap-2">
          {categories.map((cat) => (
            <Button
              key={cat.value}
              onClick={() => setFilterCategory(cat.value)}
              variant={filterCategory === cat.value ? 'default' : 'outline'}
              size="sm"
            >
              {cat.icon} {cat.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Templates Grid */}
      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin text-4xl mb-4">⚙️</div>
          <p className="text-gray-600">Loading templates...</p>
        </div>
      ) : filteredTemplates.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-600">No templates found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredTemplates.map((template) => (
            <Card key={template.template_id} className="p-6 hover:shadow-lg transition-shadow">
              {/* Template Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <span className="text-4xl">{template.icon}</span>
                  <div>
                    <h3 className="font-semibold text-lg">{template.name}</h3>
                    <Badge variant="outline" className="mt-1">
                      {template.category}
                    </Badge>
                  </div>
                </div>
                {template.is_featured && (
                  <Badge className="bg-yellow-100 text-yellow-800">⭐ Featured</Badge>
                )}
              </div>

              {/* Description */}
              <p className="text-sm text-gray-600 mb-4">{template.description}</p>

              {/* Metadata */}
              <div className="space-y-2 mb-4 text-sm">
                <div className="flex items-center gap-2">
                  <span>👥</span>
                  <span>{template.agents.length} agents</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>📊</span>
                  <span className="capitalize">{template.workflow_pattern}</span>
                </div>
                {template.metadata.estimated_execution_time && (
                  <div className="flex items-center gap-2">
                    <span>⏱️</span>
                    <span>{template.metadata.estimated_execution_time}</span>
                  </div>
                )}
                {template.metadata.success_rate && (
                  <div className="flex items-center gap-2">
                    <span>✅</span>
                    <span>{(template.metadata.success_rate * 100).toFixed(0)}% success rate</span>
                  </div>
                )}
                {template.metadata.difficulty && (
                  <Badge className={difficultyColors[template.metadata.difficulty as keyof typeof difficultyColors]}>
                    {template.metadata.difficulty}
                  </Badge>
                )}
              </div>

              {/* Tags */}
              {template.metadata.tags && template.metadata.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-4">
                  {template.metadata.tags.map((tag) => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2">
                <Button
                  onClick={() => handleDeploy(template)}
                  variant="default"
                  size="sm"
                  className="flex-1"
                >
                  Deploy
                </Button>
                <Button
                  onClick={() => handlePreview(template)}
                  variant="outline"
                  size="sm"
                >
                  Preview
                </Button>
                <Button
                  onClick={() => handleCustomize(template)}
                  variant="outline"
                  size="sm"
                >
                  Customize
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Preview Dialog */}
      {showPreview && selectedTemplate && (
        <Dialog open={showPreview} onOpenChange={setShowPreview}>
          <div className="p-6">
            <h2 className="text-2xl font-bold mb-4">
              {selectedTemplate.icon} {selectedTemplate.name}
            </h2>
            <p className="text-gray-600 mb-6">{selectedTemplate.description}</p>

            <h3 className="font-semibold mb-3">Agents ({selectedTemplate.agents.length}):</h3>
            <div className="space-y-3 mb-6">
              {selectedTemplate.agents.map((agent, index) => (
                <div key={index} className="border rounded p-3">
                  <div className="font-medium">
                    {index + 1}. {agent.role}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{agent.description}</p>
                  {agent.tools && (
                    <div className="mt-2 text-xs text-gray-500">
                      Tools: {agent.tools.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <Label>Workflow Pattern</Label>
                <p className="capitalize">{selectedTemplate.workflow_pattern}</p>
              </div>
              <div>
                <Label>Termination</Label>
                <p>{selectedTemplate.termination.type}: {selectedTemplate.termination.value}</p>
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={() => handleDeploy(selectedTemplate)} variant="default">
                Deploy Team
              </Button>
              <Button onClick={() => {
                setShowPreview(false);
                handleCustomize(selectedTemplate);
              }} variant="outline">
                Customize
              </Button>
              <Button onClick={() => setShowPreview(false)} variant="outline">
                Close
              </Button>
            </div>
          </div>
        </Dialog>
      )}

      {/* Customize Dialog */}
      {showCustomize && selectedTemplate && (
        <Dialog open={showCustomize} onOpenChange={setShowCustomize}>
          <div className="p-6">
            <h2 className="text-2xl font-bold mb-4">Customize Template</h2>
            
            <div className="space-y-4 mb-6">
              <div>
                <Label htmlFor="custom-name">Team Name</Label>
                <Input
                  id="custom-name"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                  placeholder="Enter team name"
                />
              </div>
              <div>
                <Label htmlFor="custom-description">Description</Label>
                <Input
                  id="custom-description"
                  value={customDescription}
                  onChange={(e) => setCustomDescription(e.target.value)}
                  placeholder="Enter team description"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={() => handleDeploy(selectedTemplate, true)} variant="default">
                Deploy with Customizations
              </Button>
              <Button onClick={() => setShowCustomize(false)} variant="outline">
                Cancel
              </Button>
            </div>
          </div>
        </Dialog>
      )}
    </div>
  );
}

