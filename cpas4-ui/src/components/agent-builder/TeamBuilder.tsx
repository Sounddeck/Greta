/**
 * Team Builder Component
 * Create and edit multi-agent teams with workflow patterns
 */
import React, { useState, useEffect } from 'react';
import { Plus, Trash2, GripVertical, AlertCircle, ArrowUp, ArrowDown } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';
import type {
  Team,
  TeamCreate,
  TeamAgent,
  WorkflowPattern,
  TerminationType,
  WORKFLOW_PATTERN_LABELS,
  WORKFLOW_PATTERN_DESCRIPTIONS,
  TERMINATION_TYPE_LABELS
} from '../../types/team';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Import constants
const WORKFLOW_LABELS: Record<WorkflowPattern, string> = {
  sequential: 'Sequential Chat',
  autonomous: 'Autonomous Chat',
  round_robin: 'Round Robin',
  selector: 'Selector'
};

const WORKFLOW_DESCRIPTIONS: Record<WorkflowPattern, string> = {
  sequential: 'Agents execute in the order you define',
  autonomous: 'An LLM decides which agent speaks next based on context',
  round_robin: 'Agents take turns speaking in rotation',
  selector: 'A model intelligently selects the next speaker'
};

const TERMINATION_LABELS: Record<TerminationType, string> = {
  max_turns: 'Max Turns',
  keyword: 'Keyword Detection',
  max_consecutive: 'Max Consecutive Auto-Reply',
  function: 'Custom Function'
};

interface Agent {
  agent_id: string;
  name: string;
  status: string;
}

interface TeamBuilderProps {
  team?: Team;
  onSave?: (team: Team) => void;
  onCancel?: () => void;
}

export function TeamBuilder({ team, onSave, onCancel }: TeamBuilderProps) {
  const [name, setName] = useState(team?.name || '');
  const [description, setDescription] = useState(team?.description || '');
  const [workflowPattern, setWorkflowPattern] = useState<WorkflowPattern>(
    team?.workflow_pattern || 'sequential'
  );
  const [terminationType, setTerminationType] = useState<TerminationType>(
    team?.termination.type || 'max_turns'
  );
  const [terminationValue, setTerminationValue] = useState<string>(
    team?.termination.value?.toString() || '10'
  );
  const [teamAgents, setTeamAgents] = useState<TeamAgent[]>(team?.agents || []);
  const [availableAgents, setAvailableAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Load available agents
  useEffect(() => {
    loadAvailableAgents();
  }, []);

  const loadAvailableAgents = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/agent-builder/agents`);
      setAvailableAgents(response.data || []);
    } catch (error) {
      toast.error('Failed to load agents');
    }
  };

  const addAgent = () => {
    setTeamAgents([
      ...teamAgents,
      {
        role: '',
        agent_id: '',
        order: workflowPattern === 'sequential' ? teamAgents.length + 1 : undefined,
        description: ''
      }
    ]);
  };

  const removeAgent = (index: number) => {
    const updated = teamAgents.filter((_, i) => i !== index);
    // Reorder if sequential
    if (workflowPattern === 'sequential') {
      updated.forEach((agent, i) => {
        agent.order = i + 1;
      });
    }
    setTeamAgents(updated);
  };

  const updateAgent = (index: number, field: keyof TeamAgent, value: any) => {
    const updated = [...teamAgents];
    updated[index] = { ...updated[index], [field]: value };
    setTeamAgents(updated);
  };

  const moveAgent = (index: number, direction: 'up' | 'down') => {
    if (
      (direction === 'up' && index === 0) ||
      (direction === 'down' && index === teamAgents.length - 1)
    ) {
      return;
    }

    const updated = [...teamAgents];
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    [updated[index], updated[newIndex]] = [updated[newIndex], updated[index]];
    
    // Update order
    if (workflowPattern === 'sequential') {
      updated.forEach((agent, i) => {
        agent.order = i + 1;
      });
    }
    
    setTeamAgents(updated);
  };

  const validate = (): boolean => {
    const errors: string[] = [];

    if (!name.trim()) {
      errors.push('Team name is required');
    }

    if (!description.trim()) {
      errors.push('Team description is required');
    }

    if (teamAgents.length === 0) {
      errors.push('Team must have at least one agent');
    }

    teamAgents.forEach((agent, index) => {
      if (!agent.role.trim()) {
        errors.push(`Agent ${index + 1}: Role is required`);
      }
      if (!agent.agent_id) {
        errors.push(`Agent ${index + 1}: Agent selection is required`);
      }
    });

    if (terminationType === 'max_turns' || terminationType === 'max_consecutive') {
      const maxTurns = parseInt(terminationValue);
      if (isNaN(maxTurns) || maxTurns <= 0) {
        errors.push('Value must be a positive number');
      }
    }

    setValidationErrors(errors);
    return errors.length === 0;
  };

  const handleSave = async () => {
    if (!validate()) {
      toast.error('Please fix validation errors');
      return;
    }

    setLoading(true);
    try {
      const payload: TeamCreate = {
        name,
        description,
        agents: teamAgents,
        workflow_pattern: workflowPattern,
        termination: {
          type: terminationType,
          value: terminationType === 'max_turns' || terminationType === 'max_consecutive' 
            ? parseInt(terminationValue) 
            : terminationValue
        }
      };

      let savedTeam: Team;
      if (team) {
        // Update existing team
        const response = await axios.patch(`${API_BASE_URL}/api/v1/teams/${team.team_id}`, payload);
        savedTeam = response.data;
        toast.success('Team updated successfully');
      } else {
        // Create new team
        const response = await axios.post(`${API_BASE_URL}/api/v1/teams`, payload);
        savedTeam = response.data;
        toast.success('Team created successfully');
      }

      onSave?.(savedTeam);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save team');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full overflow-auto bg-background">
      <div className="space-y-6 p-6 max-w-5xl mx-auto">
        {/* Header */}
        <div>
          <h2 className="text-2xl font-bold">
            {team ? 'Edit Team' : 'Create Team'}
          </h2>
          <p className="text-muted-foreground mt-1">
            Configure a multi-agent team with workflow patterns and termination conditions
          </p>
        </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="bg-red-900/20 border border-red-600 rounded-lg p-4">
          <div className="flex items-start gap-2">
            <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-red-400 font-semibold">Validation Errors</h3>
              <ul className="list-disc list-inside text-red-300 text-sm mt-2">
                {validationErrors.map((error, i) => (
                  <li key={i}>{error}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Basic Info */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Team Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., Research Team, Content Creation Team"
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe what this team does..."
            rows={3}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          />
        </div>
      </div>

      {/* Workflow Pattern */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Workflow Pattern
        </label>
        <select
          value={workflowPattern}
          onChange={(e) => setWorkflowPattern(e.target.value as WorkflowPattern)}
          className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {Object.entries(WORKFLOW_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">
          {WORKFLOW_DESCRIPTIONS[workflowPattern]}
        </p>
      </div>

      {/* Termination Condition */}
      <div className="space-y-4">
        <label className="block text-sm font-medium text-gray-300">
          Termination Condition
        </label>
        <div className="grid grid-cols-2 gap-4">
          <select
            value={terminationType}
            onChange={(e) => setTerminationType(e.target.value as TerminationType)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {Object.entries(TERMINATION_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <input
            type={terminationType === 'max_turns' || terminationType === 'max_consecutive' ? 'number' : 'text'}
            value={terminationValue}
            onChange={(e) => setTerminationValue(e.target.value)}
            placeholder={terminationType === 'keyword' ? 'e.g., TERMINATE' : '10'}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Team Agents - Continued in next part */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-300">
            Team Agents ({teamAgents.length})
          </label>
          <button
            onClick={addAgent}
            className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
          >
            <Plus className="h-4 w-4" />
            Add Agent
          </button>
        </div>

        <div className="space-y-3">
          {teamAgents.map((teamAgent, index) => (
            <div
              key={index}
              className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {workflowPattern === 'sequential' && (
                    <div className="flex flex-col gap-1">
                      <button
                        onClick={() => moveAgent(index, 'up')}
                        disabled={index === 0}
                        className="p-1 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        <ArrowUp className="h-4 w-4 text-gray-400" />
                      </button>
                      <button
                        onClick={() => moveAgent(index, 'down')}
                        disabled={index === teamAgents.length - 1}
                        className="p-1 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        <ArrowDown className="h-4 w-4 text-gray-400" />
                      </button>
                    </div>
                  )}
                  <span className="text-sm font-medium text-gray-400">
                    {workflowPattern === 'sequential' ? `#${index + 1}` : `Agent ${index + 1}`}
                  </span>
                </div>
                <button
                  onClick={() => removeAgent(index)}
                  className="p-1.5 hover:bg-red-900/20 rounded transition-colors"
                >
                  <Trash2 className="h-4 w-4 text-red-400" />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Role</label>
                  <input
                    type="text"
                    value={teamAgent.role}
                    onChange={(e) => updateAgent(index, 'role', e.target.value)}
                    placeholder="e.g., researcher, writer"
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-white text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Agent</label>
                  <select
                    value={teamAgent.agent_id}
                    onChange={(e) => updateAgent(index, 'agent_id', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select agent...</option>
                    {availableAgents.map((agent) => (
                      <option key={agent.agent_id} value={agent.agent_id}>
                        {agent.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs text-gray-400 mb-1">Description (Optional)</label>
                <input
                  type="text"
                  value={teamAgent.description || ''}
                  onChange={(e) => updateAgent(index, 'description', e.target.value)}
                  placeholder="Describe this agent's role in the team..."
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-white text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          ))}

          {teamAgents.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No agents added yet. Click "Add Agent" to get started.
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-700">
        {onCancel && (
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            Cancel
          </button>
        )}
        <button
          onClick={handleSave}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Saving...' : team ? 'Update Team' : 'Create Team'}
        </button>
      </div>
      </div>
    </div>
  );
}

