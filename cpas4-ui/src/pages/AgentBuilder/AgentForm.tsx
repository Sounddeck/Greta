import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Save, Loader2, Sparkles, X } from 'lucide-react';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import { Textarea } from '../../components/ui/textarea';
import { Select } from '../../components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../components/ui/dialog';
import { useAgentBuilderStore } from '../../stores/agentBuilderStore';
import type { AgentType, AgentStatus } from '../../stores/agentBuilderStore';
import { toast } from 'sonner';

export function AgentForm() {
  const navigate = useNavigate();
  const { agentId } = useParams<{ agentId: string }>();
  const isEditMode = Boolean(agentId && agentId !== 'new');

  // Zustand store
  const {
    selectedAgent,
    selectedTemplate,
    fetchAgent,
    createAgent,
    updateAgent,
    templates,
    fetchTemplates,
    setSelectedTemplate,
  } = useAgentBuilderStore();

  // Form state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [agentType, setAgentType] = useState<AgentType>('GENERAL');
  const [capabilities, setCapabilities] = useState<string[]>([]);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [configuration, setConfiguration] = useState('{}');
  const [tools, setTools] = useState<string[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [status, setStatus] = useState<AgentStatus>('DRAFT');
  
  // UI state
  const [isSaving, setIsSaving] = useState(false);
  const [showTemplateDialog, setShowTemplateDialog] = useState(false);
  const [newCapability, setNewCapability] = useState('');
  const [newTool, setNewTool] = useState('');
  const [newTag, setNewTag] = useState('');
  const [configError, setConfigError] = useState('');

  // Load agent data in edit mode
  useEffect(() => {
    if (isEditMode && agentId) {
      fetchAgent(agentId);
    }
  }, [isEditMode, agentId]);

  // Populate form when agent is loaded
  useEffect(() => {
    if (isEditMode && selectedAgent) {
      setName(selectedAgent.name);
      setDescription(selectedAgent.description);
      setAgentType(selectedAgent.agent_type);
      setCapabilities(selectedAgent.capabilities);
      setSystemPrompt(selectedAgent.system_prompt);
      setConfiguration(JSON.stringify(selectedAgent.configuration, null, 2));
      setTools(selectedAgent.tools);
      setTags(selectedAgent.tags);
      setStatus(selectedAgent.status);
    }
  }, [isEditMode, selectedAgent]);

  // Load templates
  useEffect(() => {
    fetchTemplates();
  }, []);

  // Populate form when template is selected (from TemplateGallery)
  useEffect(() => {
    if (!isEditMode && selectedTemplate) {
      setName(selectedTemplate.name);
      setDescription(selectedTemplate.description);
      setAgentType(selectedTemplate.agent_type);
      setCapabilities(selectedTemplate.capabilities);
      setSystemPrompt(selectedTemplate.default_prompt);
      setConfiguration(JSON.stringify(selectedTemplate.configuration, null, 2));
      setTools(selectedTemplate.default_tools);
      setTags(selectedTemplate.tags);
      toast.success(`Template "${selectedTemplate.name}" loaded`);
      // Clear the selected template after loading
      setSelectedTemplate(null);
    }
  }, [isEditMode, selectedTemplate, setSelectedTemplate]);

  // Validate configuration JSON
  const validateConfiguration = (value: string): boolean => {
    try {
      JSON.parse(value);
      setConfigError('');
      return true;
    } catch (error) {
      setConfigError('Invalid JSON format');
      return false;
    }
  };

  // Handle template selection
  const handleSelectTemplate = (templateId: string) => {
    const template = templates.find(t => t.template_id === templateId);
    if (!template) return;

    setName(template.name);
    setDescription(template.description);
    setAgentType(template.agent_type);
    setCapabilities(template.capabilities);
    setSystemPrompt(template.default_prompt);
    setConfiguration(JSON.stringify(template.configuration, null, 2));
    setTools(template.default_tools);
    setTags(template.tags);
    
    setShowTemplateDialog(false);
    toast.success(`Template "${template.name}" applied`);
  };

  // Add/remove capabilities
  const addCapability = () => {
    if (newCapability && !capabilities.includes(newCapability)) {
      setCapabilities([...capabilities, newCapability]);
      setNewCapability('');
    }
  };

  const removeCapability = (capability: string) => {
    setCapabilities(capabilities.filter(c => c !== capability));
  };

  // Add/remove tools
  const addTool = () => {
    if (newTool && !tools.includes(newTool)) {
      setTools([...tools, newTool]);
      setNewTool('');
    }
  };

  const removeTool = (tool: string) => {
    setTools(tools.filter(t => t !== tool));
  };

  // Add/remove tags
  const addTag = () => {
    if (newTag && !tags.includes(newTag)) {
      setTags([...tags, newTag]);
      setNewTag('');
    }
  };

  const removeTag = (tag: string) => {
    setTags(tags.filter(t => t !== tag));
  };

  // Form validation
  const validateForm = (): boolean => {
    if (!name.trim()) {
      toast.error('Agent name is required');
      return false;
    }
    if (!description.trim()) {
      toast.error('Agent description is required');
      return false;
    }
    if (!systemPrompt.trim()) {
      toast.error('System prompt is required');
      return false;
    }
    if (!validateConfiguration(configuration)) {
      toast.error('Configuration must be valid JSON');
      return false;
    }
    return true;
  };

  // Handle save
  const handleSave = async (saveStatus: AgentStatus) => {
    if (!validateForm()) return;

    setIsSaving(true);
    try {
      const agentData = {
        name,
        description,
        agent_type: agentType,
        capabilities,
        system_prompt: systemPrompt,
        configuration: JSON.parse(configuration),
        tools,
        tags,
        status: saveStatus,
      };

      if (isEditMode && agentId) {
        await updateAgent(agentId, agentData);
        toast.success('Agent updated successfully');
      } else {
        const created = await createAgent(agentData);
        if (created) {
          toast.success('Agent created successfully');
          navigate(`/agent-builder/agents/${created.agent_id}`);
        }
      }
    } catch (error) {
      console.error('Failed to save agent:', error);
      toast.error('Failed to save agent');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="h-full overflow-auto bg-background">
      <div className="space-y-6 p-6 max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold mb-2">
            {isEditMode ? 'Edit Agent' : 'Create New Agent'}
          </h2>
          <p className="text-muted-foreground">
            {isEditMode 
              ? 'Update your agent configuration'
              : 'Build a custom AI agent from scratch or use a template'}
          </p>
        </div>
        {!isEditMode && (
          <Button variant="outline" onClick={() => setShowTemplateDialog(true)}>
            <Sparkles className="h-4 w-4 mr-2" />
            Use Template
          </Button>
        )}
      </div>

      {/* Form */}
      <div className="space-y-6 border border-border rounded-lg p-6">
        {/* Basic Information */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Basic Information</h3>
          
          <div className="space-y-2">
            <Label htmlFor="name">Name *</Label>
            <Input
              id="name"
              placeholder="My Custom Agent"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description *</Label>
            <Textarea
              id="description"
              placeholder="Describe what this agent does..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="type">Agent Type *</Label>
              <Select
                id="type"
                value={agentType}
                onChange={(e) => setAgentType(e.target.value as AgentType)}
              >
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

            {isEditMode && (
              <div className="space-y-2">
                <Label htmlFor="status">Status</Label>
                <Select
                  id="status"
                  value={status}
                  onChange={(e) => setStatus(e.target.value as AgentStatus)}
                >
                  <option value="DRAFT">Draft</option>
                  <option value="ACTIVE">Active</option>
                  <option value="DEPLOYED">Deployed</option>
                  <option value="ARCHIVED">Archived</option>
                </Select>
              </div>
            )}
          </div>
        </div>

        {/* System Prompt */}
        <div className="space-y-2">
          <Label htmlFor="prompt">System Prompt *</Label>
          <Textarea
            id="prompt"
            placeholder="You are a helpful AI assistant that..."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={6}
          />
          <p className="text-xs text-muted-foreground">
            Define the agent's behavior, personality, and instructions
          </p>
        </div>

        {/* Capabilities */}
        <div className="space-y-2">
          <Label>Capabilities</Label>
          <div className="flex gap-2">
            <Input
              placeholder="Add capability (e.g., 'web_search', 'code_execution')"
              value={newCapability}
              onChange={(e) => setNewCapability(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addCapability())}
            />
            <Button type="button" onClick={addCapability}>Add</Button>
          </div>
          <div className="flex flex-wrap gap-2 mt-2">
            {capabilities.map((cap) => (
              <span
                key={cap}
                className="inline-flex items-center gap-1 px-3 py-1 bg-primary/10 text-primary rounded-full text-sm"
              >
                {cap}
                <button onClick={() => removeCapability(cap)} className="hover:text-destructive">
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        </div>

        {/* Tools */}
        <div className="space-y-2">
          <Label>Tools</Label>
          <div className="flex gap-2">
            <Input
              placeholder="Add tool (e.g., 'calculator', 'web_browser')"
              value={newTool}
              onChange={(e) => setNewTool(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTool())}
            />
            <Button type="button" onClick={addTool}>Add</Button>
          </div>
          <div className="flex flex-wrap gap-2 mt-2">
            {tools.map((tool) => (
              <span
                key={tool}
                className="inline-flex items-center gap-1 px-3 py-1 bg-secondary text-secondary-foreground rounded-full text-sm"
              >
                {tool}
                <button onClick={() => removeTool(tool)} className="hover:text-destructive">
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        </div>

        {/* Tags */}
        <div className="space-y-2">
          <Label>Tags</Label>
          <div className="flex gap-2">
            <Input
              placeholder="Add tag"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
            />
            <Button type="button" onClick={addTag}>Add</Button>
          </div>
          <div className="flex flex-wrap gap-2 mt-2">
            {tags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center gap-1 px-3 py-1 bg-muted text-muted-foreground rounded-full text-sm"
              >
                {tag}
                <button onClick={() => removeTag(tag)} className="hover:text-destructive">
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        </div>

        {/* Configuration */}
        <div className="space-y-2">
          <Label htmlFor="config">Configuration (JSON)</Label>
          <Textarea
            id="config"
            placeholder='{"temperature": 0.7, "max_tokens": 2000}'
            value={configuration}
            onChange={(e) => {
              setConfiguration(e.target.value);
              validateConfiguration(e.target.value);
            }}
            rows={8}
            className={configError ? 'border-destructive' : ''}
          />
          {configError && (
            <p className="text-xs text-destructive">{configError}</p>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <Button variant="outline" onClick={() => navigate('/agent-builder/agents')}>
          Cancel
        </Button>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => handleSave('DRAFT')}
            disabled={isSaving}
          >
            {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Save as Draft
          </Button>
          <Button
            onClick={() => handleSave('ACTIVE')}
            disabled={isSaving}
          >
            {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            {isEditMode ? 'Update Agent' : 'Create & Activate'}
          </Button>
        </div>
      </div>

      {/* Template Selection Dialog */}
      <Dialog open={showTemplateDialog} onOpenChange={setShowTemplateDialog}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Select a Template</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 mt-4">
            {templates.map((template) => (
              <div
                key={template.template_id}
                className="p-4 border border-border rounded-lg hover:border-primary cursor-pointer transition-colors"
                onClick={() => handleSelectTemplate(template.template_id)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-semibold">{template.name}</h4>
                  <span className="text-xs px-2 py-1 bg-primary/10 text-primary rounded-full">
                    {template.agent_type.replace('_', ' ')}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mb-3">{template.description}</p>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>Used {template.usage_count} times</span>
                  <span>•</span>
                  <span>{template.capabilities.length} capabilities</span>
                  <span>•</span>
                  <span>{template.default_tools.length} tools</span>
                </div>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
      </div>
    </div>
  );
}

