// @ts-nocheck
import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Save,
  Play,
  Download,
  Upload,
  ChevronLeft,
  ChevronRight,
  Check,
  AlertCircle,
  Sparkles,
  Settings,
  Wrench,
  FileText,
  Tag,
  Server,
  Brain,
  Database,
  BookOpen
} from 'lucide-react';
import { toast } from 'sonner';
import { agentBuilderAPI } from '../../lib/api';

// Step components
import BasicInfoStep from './builder-steps/BasicInfoStep';
import LLMConfigStep from './builder-steps/LLMConfigStep';
import ToolsStep from './builder-steps/ToolsStep';
import MCPToolsStep from './builder-steps/MCPToolsStep';
import SkillsStep from './builder-steps/SkillsStep';
import MemoryStep from './builder-steps/MemoryStep';
import PromptStep from './builder-steps/PromptStep';
import ReviewStep from './builder-steps/ReviewStep';

// Tutorial components
import { AgentBuilderTutorial } from './AgentBuilderTutorial';
import { HelpButton } from './HelpButton';

// Types
interface AgentDraft {
  draft_id: string;
  name: string;
  description: string;
  agent_type: string | null;
  capabilities: string[];
  system_prompt: string;
  tools: any[];
  mcp_tools?: any[];
  skills?: any[];
  memory_config?: any;
  configuration: any;
  tags: string[];
  metadata: any;
  template_id: string | null;
  current_step: number;
  created_at: string;
  updated_at: string;
}

interface ValidationResult {
  is_valid: boolean;
  errors: Array<{ field: string; message: string }>;
  warnings: Array<{ field: string; message: string }>;
  suggestions: Array<{ field: string; message: string }>;
  field_validations: Record<string, boolean>;
  completion_percentage: number;
}

const STEPS = [
  { id: 0, name: 'Basic Info', icon: FileText, description: 'Name, type, and description' },
  { id: 1, name: 'LLM Config', icon: Settings, description: 'Model and parameters' },
  { id: 2, name: 'Tools', icon: Wrench, description: 'Select agent tools' },
  { id: 3, name: 'MCP Tools', icon: Server, description: 'MCP server tools' },
  { id: 4, name: 'Skills', icon: Brain, description: 'Specialized skills' },
  { id: 5, name: 'Memory', icon: Database, description: 'Memory systems' },
  { id: 6, name: 'Prompt', icon: Sparkles, description: 'System prompt and capabilities' },
  { id: 7, name: 'Review', icon: Check, description: 'Review and save' },
];

export function VisualAgentBuilder() {
  const navigate = useNavigate();

  // State
  const [currentStep, setCurrentStep] = useState(0);
  const [draft, setDraft] = useState<AgentDraft | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(true);
  const [showTutorial, setShowTutorial] = useState(false);

  // Initialize draft
  useEffect(() => {
    initializeDraft();
  }, []);

  // Auto-save every 30 seconds
  useEffect(() => {
    if (!autoSaveEnabled || !draft) return;

    const interval = setInterval(() => {
      saveDraft(false); // Silent save
    }, 30000);

    return () => clearInterval(interval);
  }, [autoSaveEnabled, draft]);

  // Debug logging
  useEffect(() => {
    console.log('🔍 Visual Agent Builder Debug:', {
      currentStep,
      totalSteps: STEPS.length,
      isLastStep: currentStep === STEPS.length - 1,
      stepName: STEPS[currentStep]?.name,
      validation: validation,
      loading,
    });
  }, [currentStep, validation, loading]);

  const initializeDraft = async () => {
    setLoading(true);
    try {
      // Check if there's a draft in localStorage
      const savedDraftId = localStorage.getItem('current_agent_draft_id');
      
      if (savedDraftId) {
        // Load existing draft
        const response = await agentBuilderAPI.getDraft(savedDraftId);
        setDraft(response.data);
        setCurrentStep(response.data.current_step || 0);
        toast.success('Draft loaded');
      } else {
        // Create new draft
        const response = await agentBuilderAPI.createDraft({
          name: '',
          description: '',
          agent_type: null,
          capabilities: [],
          system_prompt: '',
          tools: [],
          configuration: {
            llm: {
              provider: 'llama_cpp',
              model_id: 'llama_cpp:llama3-8b',
              model_name: 'Llama 3 8B (llama.cpp)',
              temperature: 0.7,
              max_tokens: 2000,
              top_p: 1.0,
              frequency_penalty: 0.0,
              presence_penalty: 0.0,
              context_window: 8192,
            }
          },
          tags: [],
          metadata: {},
          template_id: null,
          current_step: 0,
        });
        
        setDraft(response.data);
        localStorage.setItem('current_agent_draft_id', response.data.draft_id);
        toast.success('New draft created');
      }
    } catch (error) {
      console.error('Failed to initialize draft:', error);
      toast.error('Failed to initialize draft');
    } finally {
      setLoading(false);
    }
  };

  const saveDraft = async (showToast = true) => {
    if (!draft) return;

    setSaving(true);
    try {
      await agentBuilderAPI.updateDraft(draft.draft_id, {
        ...draft,
        current_step: currentStep,
      });
      
      if (showToast) {
        toast.success('Draft saved');
      }
    } catch (error) {
      console.error('Failed to save draft:', error);
      if (showToast) {
        toast.error('Failed to save draft');
      }
    } finally {
      setSaving(false);
    }
  };

  const validateDraft = async () => {
    if (!draft) return;

    try {
      const response = await agentBuilderAPI.validateEnhanced(null, draft.draft_id);
      setValidation(response.data);
      return response.data;
    } catch (error) {
      console.error('Failed to validate draft:', error);
      toast.error('Failed to validate draft');
      return null;
    }
  };

  const updateDraft = useCallback((updates: Partial<AgentDraft>) => {
    console.log('🔧 updateDraft called with:', updates);
    setDraft(prev => {
      const newDraft = prev ? { ...prev, ...updates } : null;
      console.log('🔧 Draft updated:', { prev, updates, newDraft });
      return newDraft;
    });
  }, []);

  const nextStep = async () => {
    // Validate current step before proceeding
    const validationResult = await validateDraft();
    
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
      await saveDraft(false);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const goToStep = (step: number) => {
    setCurrentStep(step);
  };

  const saveAgent = async () => {
    if (!draft) return;

    // Final validation
    const validationResult = await validateDraft();
    
    if (!validationResult?.is_valid) {
      toast.error('Please fix validation errors before saving');
      return;
    }

    setLoading(true);
    try {
      // Convert draft to agent
      const response = await agentBuilderAPI.convertDraftToAgent(draft.draft_id);
      
      // Clear draft from localStorage
      localStorage.removeItem('current_agent_draft_id');
      
      toast.success(`Agent "${response.data.name}" created successfully!`);
      navigate('/agent-builder');
    } catch (error) {
      console.error('Failed to save agent:', error);
      toast.error('Failed to save agent');
    } finally {
      setLoading(false);
    }
  };

  const exportConfig = () => {
    if (!draft) return;

    const config = {
      name: draft.name,
      description: draft.description,
      agent_type: draft.agent_type,
      capabilities: draft.capabilities,
      system_prompt: draft.system_prompt,
      tools: draft.tools,
      configuration: draft.configuration,
      tags: draft.tags,
      metadata: draft.metadata,
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${draft.name || 'agent'}-config.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success('Configuration exported');
  };

  const importConfig = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const config = JSON.parse(text);
      
      updateDraft({
        name: config.name || '',
        description: config.description || '',
        agent_type: config.agent_type || null,
        capabilities: config.capabilities || [],
        system_prompt: config.system_prompt || '',
        tools: config.tools || [],
        configuration: config.configuration || {},
        tags: config.tags || [],
        metadata: config.metadata || {},
      });
      
      await saveDraft();
      toast.success('Configuration imported');
    } catch (error) {
      console.error('Failed to import config:', error);
      toast.error('Failed to import configuration');
    }
  };

  const cancelBuilder = () => {
    if (confirm('Are you sure you want to cancel? Unsaved changes will be lost.')) {
      if (draft) {
        agentBuilderAPI.deleteDraft(draft.draft_id).catch(console.error);
        localStorage.removeItem('current_agent_draft_id');
      }
      navigate('/agent-builder');
    }
  };

  if (loading && !draft) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading builder...</p>
        </div>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-destructive mx-auto mb-4" />
          <p className="text-muted-foreground">Failed to load draft</p>
        </div>
      </div>
    );
  }

  const CurrentStepComponent = [
    BasicInfoStep,
    LLMConfigStep,
    ToolsStep,
    MCPToolsStep,
    SkillsStep,
    MemoryStep,
    PromptStep,
    ReviewStep
  ][currentStep];

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Visual Agent Builder</h1>
            <p className="text-sm text-muted-foreground mt-1">
              {draft.name || 'Untitled Agent'} • Step {currentStep + 1} of {STEPS.length}
            </p>
          </div>

          <div className="flex items-center gap-2">
            {/* Tutorial Button */}
            <button
              onClick={() => setShowTutorial(true)}
              className="flex items-center gap-2 px-3 py-2 text-sm border border-border rounded-md hover:bg-accent transition-colors"
              title="Open Tutorial"
            >
              <BookOpen className="h-4 w-4" />
              Tutorial
            </button>
            {/* Auto-save indicator */}
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              {saving ? (
                <span className="flex items-center gap-1">
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-primary"></div>
                  Saving...
                </span>
              ) : (
                <span className="flex items-center gap-1">
                  <Check className="h-3 w-3 text-green-500" />
                  Saved
                </span>
              )}
            </div>

            {/* Export/Import */}
            <button
              onClick={exportConfig}
              className="flex items-center gap-2 px-3 py-2 text-sm border border-border rounded-md hover:bg-accent transition-colors"
              title="Export Configuration"
            >
              <Download className="h-4 w-4" />
              Export
            </button>

            <label className="flex items-center gap-2 px-3 py-2 text-sm border border-border rounded-md hover:bg-accent transition-colors cursor-pointer">
              <Upload className="h-4 w-4" />
              Import
              <input
                type="file"
                accept=".json"
                onChange={importConfig}
                className="hidden"
              />
            </label>

            {/* Cancel */}
            <button
              onClick={cancelBuilder}
              className="px-3 py-2 text-sm border border-border rounded-md hover:bg-accent transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center gap-2 mt-6">
          {STEPS.map((step, index) => {
            const StepIcon = step.icon;
            const isActive = index === currentStep;
            const isCompleted = index < currentStep;
            const isValid = validation?.field_validations?.[step.name.toLowerCase().replace(' ', '_')] ?? true;

            return (
              <div key={step.id} className="flex items-center flex-1">
                <button
                  onClick={() => goToStep(index)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors w-full ${
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : isCompleted
                      ? 'bg-primary/20 text-primary'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                  }`}
                >
                  <StepIcon className="h-4 w-4" />
                  <div className="text-left flex-1">
                    <div className="text-sm font-medium">{step.name}</div>
                    <div className="text-xs opacity-80">{step.description}</div>
                  </div>
                  {isCompleted && <Check className="h-4 w-4" />}
                  {!isValid && !isCompleted && <AlertCircle className="h-4 w-4 text-destructive" />}
                </button>
                {index < STEPS.length - 1 && (
                  <ChevronRight className="h-4 w-4 text-muted-foreground mx-1" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {/* Step Header with Help Button */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold">{STEPS[currentStep].name}</h2>
            <p className="text-sm text-muted-foreground">{STEPS[currentStep].description}</p>
          </div>
          <HelpButton
            topic={
              currentStep === 0 ? 'basic-info' :
              currentStep === 1 ? 'llm-config' :
              currentStep === 2 ? 'tools' :
              currentStep === 3 ? 'mcp-tools' :
              currentStep === 4 ? 'skills' :
              currentStep === 5 ? 'memory' :
              currentStep === 6 ? 'prompt' :
              'review'
            }
          />
        </div>

        <CurrentStepComponent
          draft={draft}
          updateDraft={updateDraft}
          validation={validation}
        />
      </div>

      {/* Footer - ALWAYS VISIBLE */}
      <div className="border-t-4 border-blue-500 bg-card px-6 py-4 shadow-lg">
        {/* DEBUG: Show current state */}
        <div className="mb-2 p-2 bg-blue-100 dark:bg-blue-900 rounded text-xs">
          <strong>DEBUG:</strong> Step {currentStep + 1}/{STEPS.length} |
          {currentStep < STEPS.length - 1 ? ' Showing NEXT button' : ' Showing SAVE AGENT button'} |
          Valid: {validation?.is_valid ? 'YES' : 'NO'}
        </div>

        <div className="flex items-center justify-between">
          <button
            onClick={prevStep}
            disabled={currentStep === 0}
            className="flex items-center gap-2 px-4 py-2 text-sm border border-border rounded-md hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </button>

          <div className="flex items-center gap-3">
            {/* Debug info */}
            <div className="text-xs text-muted-foreground">
              Step {currentStep + 1}/{STEPS.length}
            </div>
            {validation && (
              <div className="text-sm text-muted-foreground">
                Completion: {validation.completion_percentage}%
              </div>
            )}
          </div>

          {currentStep < STEPS.length - 1 ? (
            <button
              onClick={nextStep}
              className="flex items-center gap-2 px-4 py-2 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </button>
          ) : (
            <div className="flex items-center gap-2">
              {/* Show validation status */}
              {!validation?.is_valid && (
                <span className="text-xs text-yellow-600 dark:text-yellow-400">
                  Fix validation errors to enable save
                </span>
              )}
              <button
                onClick={saveAgent}
                disabled={loading || !validation?.is_valid}
                className="flex items-center gap-2 px-6 py-3 text-base font-bold bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-gray-400"
                title={!validation?.is_valid ? 'Please fix validation errors first' : 'Save agent to database'}
              >
                <Save className="h-5 w-5" />
                {loading ? 'Saving...' : 'SAVE AGENT'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Tutorial Modal */}
      {showTutorial && (
        <AgentBuilderTutorial
          currentStep={currentStep + 1}
          onClose={() => setShowTutorial(false)}
        />
      )}
    </div>
  );
}

export default VisualAgentBuilder;

