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
import { toast } from 'sonner';

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

  const deployWorkflow = useCallback(async () => {
    try {
      // TODO: Implement actual deployment to backend
      toast.info('Deploying workflow...', { duration: 2000 });

      // Simulate deployment
      await new Promise((resolve) => setTimeout(resolve, 2000));

      toast.success('Workflow deployed successfully!');
    } catch (error) {
      toast.error('Failed to deploy workflow');
    }
  }, [nodes, edges]);

  return (
    <div className="h-full w-full flex flex-col bg-gradient-to-br from-[#0a0e27] via-[#1a1d3a] to-[#0f1419]">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-3 bg-black/20 border-b border-white/10">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-white">Agent Workflow Builder</h2>
          <span className="text-xs text-gray-400">Visual Designer</span>
        </div>
        
        <div className="flex items-center gap-2">
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

      {/* React Flow Canvas */}
      <div className="flex-1">
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

      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-black/20 border-t border-white/10 text-xs text-gray-400">
        <div className="flex items-center gap-4">
          <span>Nodes: {nodes.length}</span>
          <span>Connections: {edges.length}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            Ready
          </span>
        </div>
      </div>
    </div>
  );
}

