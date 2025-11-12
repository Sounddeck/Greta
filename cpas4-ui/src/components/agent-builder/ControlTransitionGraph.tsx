/**
 * Control Transition Graph Component
 * Visualizes real-time message flow between agents during team execution
 */
import React, { useEffect, useMemo } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  BackgroundVariant,
  MarkerType,
  Position,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Users, MessageSquare, CheckCircle } from 'lucide-react';
import type { TeamMessage, Team } from '../../types/team';

interface ControlTransitionGraphProps {
  team: Team;
  messages: TeamMessage[];
  currentTurn?: number;
}

// Custom node component for agents
function AgentGraphNode({ data }: { data: any }) {
  const isActive = data.isActive;
  const isCompleted = data.isCompleted;
  const messageCount = data.messageCount || 0;

  return (
    <div
      className={`px-6 py-4 rounded-lg border-2 shadow-lg transition-all duration-300 min-w-[200px] ${
        isActive
          ? 'border-blue-500 bg-blue-500/20 shadow-blue-500/50'
          : isCompleted
          ? 'border-green-500 bg-green-500/10'
          : 'border-gray-600 bg-gray-800'
      }`}
    >
      <div className="flex items-center gap-3 mb-2">
        <div
          className={`p-2 rounded-lg ${
            isActive
              ? 'bg-blue-500'
              : isCompleted
              ? 'bg-green-500'
              : 'bg-gray-700'
          }`}
        >
          <Users className="h-5 w-5 text-white" />
        </div>
        <div className="flex-1">
          <div className="font-semibold text-white text-sm">{data.role}</div>
          <div className="text-xs text-gray-400">{data.agentName}</div>
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs text-gray-400">
        <MessageSquare className="h-3 w-3" />
        <span>{messageCount} message{messageCount !== 1 ? 's' : ''}</span>
        {isCompleted && (
          <>
            <CheckCircle className="h-3 w-3 text-green-400 ml-2" />
            <span className="text-green-400">Complete</span>
          </>
        )}
      </div>

      {isActive && (
        <div className="mt-2 flex items-center gap-2">
          <div className="flex gap-1">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse delay-75"></div>
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse delay-150"></div>
          </div>
          <span className="text-xs text-blue-400">Active</span>
        </div>
      )}
    </div>
  );
}

const nodeTypes = {
  agentNode: AgentGraphNode,
};

export function ControlTransitionGraph({
  team,
  messages,
  currentTurn,
}: ControlTransitionGraphProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Calculate node positions and create nodes
  const calculatedNodes = useMemo(() => {
    const agentCount = team.agents.length;
    const spacing = 300;
    const startX = 100;
    const startY = 100;

    // Count messages per agent
    const messageCounts = new Map<string, number>();
    messages.forEach((msg) => {
      messageCounts.set(msg.agent_id, (messageCounts.get(msg.agent_id) || 0) + 1);
    });

    // Get last message to determine active agent
    const lastMessage = messages[messages.length - 1];
    const activeAgentId = lastMessage?.agent_id;

    // Get all agents that have sent messages
    const completedAgentIds = new Set(messages.map((m) => m.agent_id));

    return team.agents.map((agent, index) => {
      const isActive = agent.agent_id === activeAgentId;
      const isCompleted = completedAgentIds.has(agent.agent_id) && !isActive;
      const messageCount = messageCounts.get(agent.agent_id) || 0;

      // Layout: horizontal flow for sequential, circular for others
      let x = startX + index * spacing;
      let y = startY;

      if (team.workflow_pattern === 'round_robin' || team.workflow_pattern === 'autonomous') {
        // Circular layout
        const angle = (index / agentCount) * 2 * Math.PI;
        const radius = 200;
        x = startX + 400 + Math.cos(angle) * radius;
        y = startY + 200 + Math.sin(angle) * radius;
      }

      return {
        id: agent.agent_id,
        type: 'agentNode',
        position: { x, y },
        data: {
          role: agent.role,
          agentName: `Agent ${index + 1}`,
          agentId: agent.agent_id,
          isActive,
          isCompleted,
          messageCount,
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      } as Node;
    });
  }, [team, messages]);

  // Calculate edges based on message flow
  const calculatedEdges = useMemo(() => {
    const edgeMap = new Map<string, { count: number; lastTurn: number }>();

    messages.forEach((msg, index) => {
      if (index === 0) return; // Skip first message (no previous agent)

      const prevMsg = messages[index - 1];
      const edgeId = `${prevMsg.agent_id}-${msg.agent_id}`;

      if (!edgeMap.has(edgeId)) {
        edgeMap.set(edgeId, { count: 0, lastTurn: 0 });
      }

      const edge = edgeMap.get(edgeId)!;
      edge.count += 1;
      edge.lastTurn = msg.turn_number || index + 1;
    });

    return Array.from(edgeMap.entries()).map(([edgeId, data]) => {
      const [source, target] = edgeId.split('-');
      const isRecent = currentTurn && data.lastTurn >= currentTurn - 1;

      return {
        id: edgeId,
        source,
        target,
        type: 'smoothstep',
        animated: isRecent,
        label: data.count > 1 ? `${data.count}x` : undefined,
        labelStyle: {
          fill: '#9CA3AF',
          fontSize: 12,
          fontWeight: 600,
        },
        labelBgStyle: {
          fill: '#1F2937',
          fillOpacity: 0.9,
        },
        style: {
          stroke: isRecent ? '#3B82F6' : '#4B5563',
          strokeWidth: isRecent ? 3 : 2,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: isRecent ? '#3B82F6' : '#4B5563',
          width: 20,
          height: 20,
        },
      } as Edge;
    });
  }, [messages, currentTurn]);

  // Update nodes and edges when calculated values change
  useEffect(() => {
    setNodes(calculatedNodes);
  }, [calculatedNodes, setNodes]);

  useEffect(() => {
    setEdges(calculatedEdges);
  }, [calculatedEdges, setEdges]);

  return (
    <div className="h-full w-full bg-gray-900 rounded-lg border border-gray-700">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{
          padding: 0.2,
          includeHiddenNodes: false,
        }}
        minZoom={0.1}
        maxZoom={2}
        defaultEdgeOptions={{
          type: 'smoothstep',
        }}
      >
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} color="#374151" />
        <Controls className="bg-gray-800 border-gray-700" />
        <MiniMap
          className="bg-gray-800 border-gray-700"
          nodeColor={(node) => {
            if (node.data.isActive) return '#3B82F6';
            if (node.data.isCompleted) return '#10B981';
            return '#4B5563';
          }}
          maskColor="rgba(0, 0, 0, 0.6)"
        />
      </ReactFlow>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs">
        <div className="font-semibold text-white mb-2">Legend</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded"></div>
            <span className="text-gray-300">Active Agent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-gray-300">Completed</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-600 rounded"></div>
            <span className="text-gray-300">Pending</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-blue-500"></div>
            <span className="text-gray-300">Recent Flow</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-gray-600"></div>
            <span className="text-gray-300">Previous Flow</span>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="absolute top-4 left-4 bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs">
        <div className="font-semibold text-white mb-2">Workflow Stats</div>
        <div className="space-y-1 text-gray-300">
          <div>Pattern: <span className="text-blue-400">{team.workflow_pattern}</span></div>
          <div>Agents: <span className="text-blue-400">{team.agents.length}</span></div>
          <div>Messages: <span className="text-blue-400">{messages.length}</span></div>
          {currentTurn && <div>Turn: <span className="text-blue-400">{currentTurn}</span></div>}
        </div>
      </div>
    </div>
  );
}

