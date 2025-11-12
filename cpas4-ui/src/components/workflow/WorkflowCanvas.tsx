/**
 * Workflow Canvas Component
 * 
 * Main canvas for visual workflow builder using React Flow
 */

import React, { useCallback, useState, useRef } from 'react';
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
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// Import custom node types
import TriggerNode from './nodes/TriggerNode';
import ActionNode from './nodes/ActionNode';
import ConditionNode from './nodes/ConditionNode';
import AgentNode from './nodes/AgentNode';

// Import custom edge types
import DefaultEdge from './edges/DefaultEdge';
import ConditionalEdge from './edges/ConditionalEdge';

// Import node palette and editor
import NodePalette from './NodePalette';
import NodeEditor from './NodeEditor';

// Define custom node types
const nodeTypes = {
  trigger: TriggerNode,
  action: ActionNode,
  condition: ConditionNode,
  agent: AgentNode,
};

// Define custom edge types
const edgeTypes = {
  default: DefaultEdge,
  conditional: ConditionalEdge,
};

// Initial nodes for demo
const initialNodes: Node[] = [
  {
    id: '1',
    type: 'trigger',
    position: { x: 250, y: 50 },
    data: { label: 'Workflow Start', triggerType: 'manual' },
  },
];

const initialEdges: Edge[] = [];

// Default viewport settings
const defaultViewport = { x: 0, y: 0, zoom: 1 };

interface WorkflowCanvasProps {
  workflowId?: string;
  readOnly?: boolean;
  onSave?: (nodes: Node[], edges: Edge[]) => void;
}

function WorkflowCanvasInner({
  workflowId,
  readOnly = false,
  onSave,
}: WorkflowCanvasProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [showMinimap, setShowMinimap] = useState(true);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const reactFlowInstance = useReactFlow();

  // Handle new connections between nodes
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges]
  );

  // Validate connections
  const isValidConnection = useCallback((connection: Connection) => {
    // Trigger nodes can't have inputs
    const targetNode = nodes.find((n) => n.id === connection.target);
    if (targetNode?.type === 'trigger') {
      return false;
    }

    // Prevent self-connections
    if (connection.source === connection.target) {
      return false;
    }

    // Prevent duplicate connections
    const existingEdge = edges.find(
      (e) =>
        e.source === connection.source &&
        e.target === connection.target &&
        e.sourceHandle === connection.sourceHandle &&
        e.targetHandle === connection.targetHandle
    );
    if (existingEdge) {
      return false;
    }

    return true;
  }, [nodes, edges]);

  // Handle node drag from palette
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle node drop on canvas
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      if (!type) return;

      const reactFlowBounds = event.currentTarget.getBoundingClientRect();
      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: { label: `New ${type}` },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes]
  );

  // Handle save
  const handleSave = useCallback(() => {
    if (onSave) {
      onSave(nodes, edges);
    }
  }, [nodes, edges, onSave]);

  // Handle node double-click to open editor
  const onNodeDoubleClick = useCallback((event: React.MouseEvent, node: Node) => {
    if (!readOnly) {
      setSelectedNode(node);
      setIsEditorOpen(true);
    }
  }, [readOnly]);

  // Handle node editor save
  const handleNodeEditorSave = useCallback((nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          return { ...node, data: { ...node.data, ...data } };
        }
        return node;
      })
    );
  }, [setNodes]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    reactFlowInstance.zoomIn();
  }, [reactFlowInstance]);

  const handleZoomOut = useCallback(() => {
    reactFlowInstance.zoomOut();
  }, [reactFlowInstance]);

  const handleFitView = useCallback(() => {
    reactFlowInstance.fitView({ padding: 0.2 });
  }, [reactFlowInstance]);

  const handleResetView = useCallback(() => {
    reactFlowInstance.setViewport({ x: 0, y: 0, zoom: 1 });
  }, [reactFlowInstance]);

  return (
    <div className="w-full h-full relative bg-gray-50 dark:bg-gray-900">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onNodeDoubleClick={onNodeDoubleClick}
        isValidConnection={isValidConnection}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultViewport={defaultViewport}
        fitView
        attributionPosition="bottom-left"
        className="bg-gray-50 dark:bg-gray-900"
        deleteKeyCode={readOnly ? null : 'Delete'}
        multiSelectionKeyCode={readOnly ? null : 'Shift'}
      >
        {/* Background pattern */}
        <Background
          variant={BackgroundVariant.Dots}
          gap={16}
          size={1}
          color="#94a3b8"
          className="bg-gray-50 dark:bg-gray-900"
        />

        {/* Controls (zoom, fit view, etc.) */}
        <Controls
          showZoom={true}
          showFitView={true}
          showInteractive={!readOnly}
          className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg"
        />

        {/* Minimap */}
        {showMinimap && (
          <MiniMap
            nodeColor={(node) => {
              switch (node.type) {
                case 'trigger':
                  return '#10b981'; // green
                case 'action':
                  return '#3b82f6'; // blue
                case 'condition':
                  return '#f59e0b'; // amber
                case 'agent':
                  return '#8b5cf6'; // purple
                default:
                  return '#6b7280'; // gray
              }
            }}
            className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg"
            maskColor="rgba(0, 0, 0, 0.1)"
          />
        )}

        {/* Top toolbar panel */}
        <Panel position="top-left" className="m-4">
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 flex items-center gap-2">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white px-2">
              Workflow Builder
            </h3>
            <div className="h-4 w-px bg-gray-300 dark:bg-gray-600" />
            <button
              onClick={() => setShowMinimap(!showMinimap)}
              className="px-3 py-1 text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            >
              {showMinimap ? 'Hide' : 'Show'} Minimap
            </button>
            {!readOnly && onSave && (
              <>
                <div className="h-4 w-px bg-gray-300 dark:bg-gray-600" />
                <button
                  onClick={handleSave}
                  className="px-3 py-1 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded transition-colors"
                >
                  Save Workflow
                </button>
              </>
            )}
          </div>
        </Panel>

        {/* Node palette panel */}
        {!readOnly && (
          <Panel position="top-right" className="m-4">
            <NodePalette />
          </Panel>
        )}
      </ReactFlow>

      {/* Node Editor Modal */}
      <NodeEditor
        node={selectedNode}
        isOpen={isEditorOpen}
        onClose={() => setIsEditorOpen(false)}
        onSave={handleNodeEditorSave}
      />
    </div>
  );
}

// Wrapper component with ReactFlowProvider
export default function WorkflowCanvas(props: WorkflowCanvasProps) {
  return (
    <ReactFlowProvider>
      <WorkflowCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

