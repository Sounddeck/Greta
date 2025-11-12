import { useCallback, useState } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Plus, Save, Play, Download, Upload } from 'lucide-react';
import ValidationPanel from './ValidationPanel';
import TestRunConsole from './TestRunConsole';
import VersionSwitcher from './VersionSwitcher';
import { toast } from 'sonner';
import { agentBuilderAPI } from '../../lib/api';

// Custom node types
import AgentNode from './nodes/AgentNode';
import ToolNode from './nodes/ToolNode';
import ConditionNode from './nodes/ConditionNode';
import ActionNode from './nodes/ActionNode';

const nodeTypes = {
  agent: AgentNode as any,
  tool: ToolNode as any,
  condition: ConditionNode as any,
  action: ActionNode as any,
};

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'agent',
    position: { x: 250, y: 50 },
    data: {
      label: 'Master Agent',
      agentType: 'general',
      model: 'gpt-4',
      temperature: 0.7,
    },
  },
];

const initialEdges: Edge[] = [];

export default function AgentBuilderWorkflow() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [_selectedNodeType, _setSelectedNodeType] = useState<string>('agent');

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const addNode = useCallback(
    (type: string) => {
      const newNode: Node = {
        id: `${nodes.length + 1}`,
        type,
        position: {
          x: Math.random() * 400 + 100,
          y: Math.random() * 400 + 100,
        },
        data: {
          label: `New ${type.charAt(0).toUpperCase() + type.slice(1)}`,
        },
      };
      setNodes((nds) => [...nds, newNode]);
      toast.success(`Added ${type} node`);
    },
    [nodes, setNodes]
  );

  const saveWorkflow = useCallback(() => {
    const workflow = {
      nodes,
      edges,
      metadata: {
        name: 'My Agent Workflow',
        version: '1.0.0',
        createdAt: new Date().toISOString(),
      },
    };

    // Save to localStorage for now
    localStorage.setItem('agent-workflow', JSON.stringify(workflow));
    toast.success('Workflow saved successfully');
  }, [nodes, edges]);

  const loadWorkflow = useCallback(() => {
    const saved = localStorage.getItem('agent-workflow');
    if (saved) {
      const workflow = JSON.parse(saved);
      setNodes(workflow.nodes);
      setEdges(workflow.edges);
      toast.success('Workflow loaded successfully');
    } else {
      toast.error('No saved workflow found');
    }
  }, [setNodes, setEdges]);

  const exportWorkflow = useCallback(() => {
    const workflow = {
      nodes,
      edges,
      metadata: {
        name: 'My Agent Workflow',
        version: '1.0.0',
        exportedAt: new Date().toISOString(),
      },
    };

    const blob = new Blob([JSON.stringify(workflow, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'agent-workflow.json';
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Workflow exported');
  }, [nodes, edges]);

  const [validationResult, setValidationResult] = useState<any | null>(null);
  const [showTestRun, setShowTestRun] = useState(false);
  const [lastDeployedAgentId, setLastDeployedAgentId] = useState<string>('');

  const deployWorkflow = useCallback(async () => {
    try {
      // Basic client-side validation
      if (!Array.isArray(nodes) || nodes.length === 0) {
        toast.error('Add at least one node to the workflow');
        return;
      }
      const agentNodes = nodes.filter((n) => n.type === 'agent');
      if (agentNodes.length === 0) {
        toast.error('At least one Agent node is required');
        return;
      }
      const nodeIds = new Set(nodes.map((n) => String(n.id)));
      for (const e of edges) {
        if (!nodeIds.has(String((e as any).source)) || !nodeIds.has(String((e as any).target))) {
          toast.error('An edge references a missing node');
          return;
        }
      }

      const workflow = {
        nodes,
        edges,
        metadata: {
          name: 'My Agent Workflow',
          version: '1.0.0',
          createdAt: new Date().toISOString(),
        },
      };

      // Server-side validation
      const v = await agentBuilderAPI.validateWorkflow(workflow);
      setValidationResult(v.data);
      if (v.data?.is_valid === false) {
        const first = v.data?.errors?.[0] || v.data?.node_errors?.[0]?.message || 'Validation failed';
        toast.error(first);
        return;
      }

      toast.info('Deploying workflow...');
      const res = await agentBuilderAPI.deployWorkflow(workflow);
      const agentId = res.data?.agent_id;
      if (agentId) {
        setLastDeployedAgentId(agentId);
        toast.success(`Deployed! Agent ID: ${agentId}`);
      } else {
        toast.error(res.data?.message || 'Deployment failed');
      }
    } catch (error: any) {
      const msg = error?.response?.data?.detail || error?.message || 'Failed to deploy workflow';
      toast.error(msg);
    }
  }, [nodes, edges]);

  return (
    <div className="h-full w-full flex flex-col bg-gradient-to-br from-[#0a0e27] via-[#1a1d3a] to-[#0f1419] overflow-hidden">
      {/* Toolbar - Fixed at top */}
      <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 bg-black/20 border-b border-white/10">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-white">Agent Workflow Builder</h2>
          <span className="text-xs text-gray-400">Visual Designer</span>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Add Node Buttons */}
          <button
            onClick={() => addNode('agent')}
            className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-md transition-colors"
            title="Add Agent Node"
          >
            <Plus className="h-4 w-4" />
            Agent
          </button>

          <button
            onClick={() => addNode('tool')}
            className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md transition-colors"
            title="Add Tool Node"
          >
            <Plus className="h-4 w-4" />
            Tool
          </button>

          <button
            onClick={() => addNode('condition')}
            className="flex items-center gap-2 px-3 py-1.5 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded-md transition-colors"
            title="Add Condition Node"
          >
            <Plus className="h-4 w-4" />
            Condition
          </button>

          <button
            onClick={() => addNode('action')}
            className="flex items-center gap-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-md transition-colors"
            title="Add Action Node"
          >
            <Plus className="h-4 w-4" />
            Action
          </button>

          <div className="w-px h-6 bg-white/20 mx-2" />

          {/* Workflow Actions */}
          <button
            onClick={saveWorkflow}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-md transition-colors"
            title="Save Workflow"
          >
            <Save className="h-4 w-4" />
            Save
          </button>

          <button
            onClick={loadWorkflow}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-md transition-colors"
            title="Load Workflow"
          >
            <Upload className="h-4 w-4" />
            Load
          </button>

          <button
            onClick={exportWorkflow}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-md transition-colors"
            title="Export Workflow"
          >
            <Download className="h-4 w-4" />
            Export
          </button>

          <button
            onClick={deployWorkflow}
            className="flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white text-sm rounded-md transition-colors shadow-lg"
            title="Deploy Workflow"
          >
            <Play className="h-4 w-4" />
            Deploy
          </button>
        </div>
      </div>

      {/* React Flow Canvas - Scrollable content area */}
      <div className="flex-1 overflow-hidden relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          className="bg-transparent"
        >
          <Controls className="bg-black/40 border border-white/10" />
          <MiniMap
            className="bg-black/40 border border-white/10"
            nodeColor={(node) => {
              switch (node.type) {
                case 'agent':
                  return '#3b82f6';
                case 'tool':
                  return '#10b981';
                case 'condition':
                  return '#f59e0b';
                case 'action':
                  return '#8b5cf6';
                default:
                  return '#6b7280';
              }
            }}
          />
          <Background
            variant={BackgroundVariant.Dots}
            gap={12}
            size={1}
            color="#ffffff20"
          />
        </ReactFlow>
      </div>

      {/* Validation Panel - Positioned absolutely or as overlay */}
      <ValidationPanel result={validationResult} />

      {/* Optional Test Run Console */}

      {/* Version Switcher (after deploy) */}
      {lastDeployedAgentId && <VersionSwitcher agentId={lastDeployedAgentId} />}

      {showTestRun && <TestRunConsole agentIdHint={lastDeployedAgentId} />}

      {/* Status Bar - Fixed at bottom */}
      <div className="flex-shrink-0 flex items-center justify-between px-4 py-2 bg-black/20 border-t border-white/10 text-xs text-gray-400">
        <div className="flex items-center gap-4">
          <span>Nodes: {nodes.length}</span>
          <span>Connections: {edges.length}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTestRun((s) => !s)}
            className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white"
          >
            {showTestRun ? 'Hide Test Run' : 'Show Test Run'}
          </button>
        </div>
      </div>
    </div>
  );
}

