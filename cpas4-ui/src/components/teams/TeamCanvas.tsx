/**
 * Team Canvas Component
 * 
 * Visual team builder using React Flow for drag-and-drop agent team creation
 */

import React, { useCallback, useState, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  Edge,
  Node,
  BackgroundVariant,
  Panel,
  useReactFlow,
  ReactFlowProvider,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// Import custom node type
import TeamAgentNode from './TeamAgentNode';

// Import UI components
import { Button } from '../ui/button';
import { Select } from '../ui/select';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Card } from '../ui/card';

// Define custom node types
const nodeTypes = {
  teamAgent: TeamAgentNode,
};

// Default viewport settings
const defaultViewport = { x: 0, y: 0, zoom: 1 };

interface TeamCanvasProps {
  team?: any;
  onSave?: (team: any) => void;
  onCancel?: () => void;
  mode?: 'create' | 'edit' | 'view';
}

function TeamCanvasInner({
  team,
  onSave,
  onCancel,
  mode = 'create',
}: TeamCanvasProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [teamName, setTeamName] = useState(team?.name || '');
  const [teamDescription, setTeamDescription] = useState(team?.description || '');
  const [workflowPattern, setWorkflowPattern] = useState(team?.workflow_pattern || 'sequential');
  const [terminationType, setTerminationType] = useState(team?.termination?.type || 'max_turns');
  const [terminationValue, setTerminationValue] = useState(team?.termination?.value || 10);
  const [availableAgents, setAvailableAgents] = useState<any[]>([]);
  const reactFlowInstance = useReactFlow();

  // Load available agents
  useEffect(() => {
    fetchAvailableAgents();
  }, []);

  // Load team data if editing
  useEffect(() => {
    if (team && mode === 'edit') {
      loadTeamData(team);
    }
  }, [team, mode]);

  const fetchAvailableAgents = async () => {
    try {
      const response = await fetch('/api/v1/agents');
      const data = await response.json();
      setAvailableAgents(data.agents || []);
    } catch (error) {
      console.error('Failed to fetch agents:', error);
    }
  };

  const loadTeamData = (teamData: any) => {
    // Convert team agents to nodes
    const teamNodes: Node[] = teamData.agents.map((agent: any, index: number) => ({
      id: `agent-${index}`,
      type: 'teamAgent',
      position: { x: 250, y: 100 + (index * 150) },
      data: {
        agentId: agent.agent_id,
        role: agent.role,
        order: agent.order,
        description: agent.description,
      },
    }));

    setNodes(teamNodes);

    // Create edges based on workflow pattern
    if (workflowPattern === 'sequential' && teamNodes.length > 1) {
      const teamEdges: Edge[] = [];
      for (let i = 0; i < teamNodes.length - 1; i++) {
        teamEdges.push({
          id: `edge-${i}`,
          source: teamNodes[i].id,
          target: teamNodes[i + 1].id,
          type: 'default',
          markerEnd: { type: MarkerType.ArrowClosed },
        });
      }
      setEdges(teamEdges);
    }
  };

  // Handle new connections between nodes
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge({
        ...params,
        markerEnd: { type: MarkerType.ArrowClosed },
      }, eds));
    },
    [setEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    console.log('Node clicked:', node);
  }, []);

  // Add agent to canvas
  const addAgentToCanvas = (agentId: string) => {
    const agent = availableAgents.find(a => a.agent_id === agentId);
    if (!agent) return;

    const newNode: Node = {
      id: `agent-${nodes.length}`,
      type: 'teamAgent',
      position: { x: 250, y: 100 + (nodes.length * 150) },
      data: {
        agentId: agent.agent_id,
        role: agent.name,
        order: nodes.length + 1,
        description: agent.description || '',
      },
    };

    setNodes((nds) => [...nds, newNode]);

    // Auto-connect in sequential mode
    if (workflowPattern === 'sequential' && nodes.length > 0) {
      const lastNode = nodes[nodes.length - 1];
      const newEdge: Edge = {
        id: `edge-${edges.length}`,
        source: lastNode.id,
        target: newNode.id,
        type: 'default',
        markerEnd: { type: MarkerType.ArrowClosed },
      };
      setEdges((eds) => [...eds, newEdge]);
    }
  };

  // Handle save
  const handleSave = () => {
    if (!teamName.trim()) {
      alert('Please enter a team name');
      return;
    }

    if (nodes.length === 0) {
      alert('Please add at least one agent to the team');
      return;
    }

    const teamData = {
      name: teamName,
      description: teamDescription,
      agents: nodes.map((node, index) => ({
        role: node.data.role,
        agent_id: node.data.agentId,
        order: index + 1,
        description: node.data.description,
      })),
      workflow_pattern: workflowPattern,
      termination: {
        type: terminationType,
        value: terminationType === 'max_turns' ? parseInt(terminationValue.toString()) : terminationValue,
      },
    };

    onSave?.(teamData);
  };

  // Handle workflow pattern change
  const handlePatternChange = (newPattern: string) => {
    setWorkflowPattern(newPattern);

    // Update edges based on pattern
    if (newPattern === 'sequential') {
      // Create sequential edges
      const newEdges: Edge[] = [];
      for (let i = 0; i < nodes.length - 1; i++) {
        newEdges.push({
          id: `edge-${i}`,
          source: nodes[i].id,
          target: nodes[i + 1].id,
          type: 'default',
          markerEnd: { type: MarkerType.ArrowClosed },
        });
      }
      setEdges(newEdges);
    } else if (newPattern === 'autonomous' || newPattern === 'round_robin') {
      // Clear edges for autonomous/round_robin (managed by LLM)
      setEdges([]);
    }
  };

  const readOnly = mode === 'view';

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b bg-white">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <Label htmlFor="team-name">Team Name</Label>
            <Input
              id="team-name"
              value={teamName}
              onChange={(e) => setTeamName(e.target.value)}
              placeholder="Enter team name"
              disabled={readOnly}
            />
          </div>
          <div>
            <Label htmlFor="team-description">Description</Label>
            <Input
              id="team-description"
              value={teamDescription}
              onChange={(e) => setTeamDescription(e.target.value)}
              placeholder="Enter team description"
              disabled={readOnly}
            />
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <Label htmlFor="workflow-pattern">Workflow Pattern</Label>
            <select
              id="workflow-pattern"
              value={workflowPattern}
              onChange={(e) => handlePatternChange(e.target.value)}
              className="w-full p-2 border rounded"
              disabled={readOnly}
            >
              <option value="sequential">Sequential</option>
              <option value="autonomous">Autonomous</option>
              <option value="round_robin">Round Robin</option>
              <option value="selector">Selector</option>
            </select>
          </div>
          <div>
            <Label htmlFor="termination-type">Termination Type</Label>
            <select
              id="termination-type"
              value={terminationType}
              onChange={(e) => setTerminationType(e.target.value)}
              className="w-full p-2 border rounded"
              disabled={readOnly}
            >
              <option value="max_turns">Max Turns</option>
              <option value="keyword">Keyword</option>
              <option value="max_consecutive">Max Consecutive</option>
            </select>
          </div>
          <div>
            <Label htmlFor="termination-value">Termination Value</Label>
            <Input
              id="termination-value"
              type={terminationType === 'max_turns' || terminationType === 'max_consecutive' ? 'number' : 'text'}
              value={terminationValue}
              onChange={(e) => setTerminationValue(e.target.value)}
              placeholder={terminationType === 'keyword' ? 'Enter keyword' : 'Enter number'}
              disabled={readOnly}
            />
          </div>
        </div>

        {!readOnly && (
          <div className="mt-4 flex gap-2">
            <Button onClick={handleSave} variant="default">
              Save Team
            </Button>
            <Button onClick={onCancel} variant="outline">
              Cancel
            </Button>
          </div>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          defaultViewport={defaultViewport}
          fitView
          attributionPosition="bottom-left"
        >
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          <Controls />
          <MiniMap />

          {/* Agent Library Panel */}
          {!readOnly && (
            <Panel position="top-left" className="bg-white p-4 rounded shadow-lg">
              <h3 className="font-semibold mb-2">Agent Library</h3>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {availableAgents.map((agent) => (
                  <Button
                    key={agent.agent_id}
                    onClick={() => addAgentToCanvas(agent.agent_id)}
                    variant="outline"
                    size="sm"
                    className="w-full justify-start"
                  >
                    {agent.name}
                  </Button>
                ))}
              </div>
            </Panel>
          )}
        </ReactFlow>
      </div>
    </div>
  );
}

// Wrapper with ReactFlowProvider
export default function TeamCanvas(props: TeamCanvasProps) {
  return (
    <ReactFlowProvider>
      <TeamCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

