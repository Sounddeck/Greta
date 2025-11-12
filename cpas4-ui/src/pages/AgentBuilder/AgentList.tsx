import { useEffect, useState } from 'react';
import { 
  Search, 
  Filter, 
  Plus, 
  Eye, 
  Edit, 
  Copy, 
  Trash2, 
  Rocket, 
  Play,
  MoreVertical,
  CheckCircle2,
  Clock,
  FileText,
  Archive
} from 'lucide-react';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Select } from '../../components/ui/select';
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../../components/ui/dropdown-menu';
import { useAgentBuilderStore } from '../../stores/agentBuilderStore';
import type { AgentType, AgentStatus } from '../../stores/agentBuilderStore';
import { TableSkeleton } from '../../components/skeletons/TableSkeleton';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';

export function AgentList() {
  console.log('[AgentList] 🔵 Component render started');
  const navigate = useNavigate();

  // Zustand store
  const {
    agents,
    isLoadingAgents,
    fetchAgents,
    deleteAgent,
    cloneAgent,
    setSelectedAgent,
  } = useAgentBuilderStore();

  console.log('[AgentList] 📊 Store state:', {
    agentsCount: agents?.length || 0,
    isLoadingAgents,
    agentsArray: agents,
    agentsType: typeof agents,
    isArray: Array.isArray(agents)
  });

  // Local state for filters and pagination
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<AgentType | 'ALL'>('ALL');
  const [filterStatus, setFilterStatus] = useState<AgentStatus | 'ALL'>('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  console.log('[AgentList] 🔍 Filter state:', {
    searchQuery,
    filterType,
    filterStatus,
    currentPage,
    itemsPerPage
  });

  // Load agents on mount
  useEffect(() => {
    console.log('[AgentList] ⚡ Component mounted, calling fetchAgents...');
    const result = fetchAgents();
    console.log('[AgentList] 📞 fetchAgents() called, result:', result);
  }, [fetchAgents]);

  // Debug: Log agents state changes
  useEffect(() => {
    console.log('[AgentList] 🔄 Agents state changed:', {
      total: agents?.length || 0,
      isLoading: isLoadingAgents,
      agentsData: agents
    });
  }, [agents, isLoadingAgents]);

  // Filter and search agents
  const filteredAgents = agents.filter((agent) => {
    const matchesSearch =
      agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      agent.description.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesType = filterType === 'ALL' || agent.agent_type === filterType;
    const matchesStatus = filterStatus === 'ALL' || agent.status === filterStatus;

    return matchesSearch && matchesType && matchesStatus;
  });

  console.log('[AgentList] 🔎 Filtered agents:', {
    totalAgents: agents.length,
    filteredCount: filteredAgents.length,
    filteredAgents: filteredAgents
  });

  // Pagination
  const totalPages = Math.ceil(filteredAgents.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedAgents = filteredAgents.slice(startIndex, endIndex);

  console.log('[AgentList] 📄 Pagination:', {
    totalPages,
    currentPage,
    startIndex,
    endIndex,
    paginatedCount: paginatedAgents.length,
    paginatedAgents: paginatedAgents
  });

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, filterType, filterStatus]);

  // Action handlers
  const handleView = (agentId: string) => {
    const agent = agents.find(a => a.agent_id === agentId);
    if (agent) {
      setSelectedAgent(agent);
      navigate(`/agent-builder/agents/${agentId}`);
    }
  };

  const handleEdit = (agentId: string) => {
    const agent = agents.find(a => a.agent_id === agentId);
    if (agent) {
      setSelectedAgent(agent);
      navigate(`/agent-builder/agents/${agentId}/edit`);
    }
  };

  const handleClone = async (agentId: string) => {
    const agent = agents.find(a => a.agent_id === agentId);
    if (!agent) return;
    
    const newName = prompt(`Clone "${agent.name}" as:`, `${agent.name} (Copy)`);
    if (newName) {
      const cloned = await cloneAgent(agentId, newName);
      if (cloned) {
        toast.success(`Agent cloned as "${newName}"`);
      }
    }
  };

  const handleDelete = async (agentId: string) => {
    const agent = agents.find(a => a.agent_id === agentId);
    if (!agent) return;
    
    if (confirm(`Are you sure you want to delete "${agent.name}"? This action cannot be undone.`)) {
      await deleteAgent(agentId);
    }
  };

  const handleDeploy = (agentId: string) => {
    const agent = agents.find(a => a.agent_id === agentId);
    if (agent) {
      setSelectedAgent(agent);
      navigate(`/agent-builder/deploy/${agentId}`);
    }
  };

  const handleExecute = (agentId: string) => {
    const agent = agents.find(a => a.agent_id === agentId);
    if (agent) {
      setSelectedAgent(agent);
      navigate(`/agent-builder/execute/${agentId}`);
    }
  };

  // Status badge component
  const StatusBadge = ({ status }: { status: AgentStatus }) => {
    const statusConfig = {
      DRAFT: { icon: FileText, color: 'text-gray-500 bg-gray-100', label: 'Draft' },
      ACTIVE: { icon: CheckCircle2, color: 'text-green-600 bg-green-100', label: 'Active' },
      DEPLOYED: { icon: Rocket, color: 'text-blue-600 bg-blue-100', label: 'Deployed' },
      ARCHIVED: { icon: Archive, color: 'text-orange-600 bg-orange-100', label: 'Archived' },
    };

    // Fallback for unknown status values
    const config = statusConfig[status] || { icon: FileText, color: 'text-gray-500 bg-gray-100', label: status || 'Unknown' };
    const Icon = config.icon;

    return (
      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${config.color}`}>
        <Icon className="h-3 w-3" />
        {config.label}
      </span>
    );
  };

  // Type badge component
  const TypeBadge = ({ type }: { type: AgentType }) => {
    return (
      <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary/10 text-primary">
        {type.replace('_', ' ')}
      </span>
    );
  };

  console.log('[AgentList] 🎨 RENDERING COMPONENT - Return statement reached');

  return (
    <div className="h-full overflow-auto bg-background">
      <div className="space-y-6 p-6">
        {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold mb-2">Agents</h2>
          <p className="text-muted-foreground">
            Manage your custom AI agents
          </p>
        </div>
        <Button onClick={() => navigate('/agent-builder/agents/new')}>
          <Plus className="h-4 w-4 mr-2" />
          Create Agent
        </Button>
      </div>

      {/* Filters and Search */}
      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search agents by name or description..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <Select value={filterType} onValueChange={(value) => setFilterType(value as AgentType | 'ALL')}>
          <option value="ALL">All Types</option>
          <option value="GENERAL">General</option>
          <option value="RESEARCH">Research</option>
          <option value="CODING">Coding</option>
          <option value="DATA_ANALYSIS">Data Analysis</option>
          <option value="CREATIVE">Creative</option>
          <option value="CUSTOMER_SERVICE">Customer Service</option>
          <option value="SALES">Sales</option>
          <option value="MARKETING">Marketing</option>
        </Select>
        
        <Select value={filterStatus} onValueChange={(value) => setFilterStatus(value as AgentStatus | 'ALL')}>
          <option value="ALL">All Statuses</option>
          <option value="DRAFT">Draft</option>
          <option value="ACTIVE">Active</option>
          <option value="DEPLOYED">Deployed</option>
          <option value="ARCHIVED">Archived</option>
        </Select>
      </div>

      {/* Results count */}
      <div className="text-sm text-muted-foreground">
        Showing {paginatedAgents.length} of {filteredAgents.length} agents
        {filteredAgents.length !== agents.length && ` (filtered from ${agents.length} total)`}
      </div>

      {/* Table */}
      {(() => {
        const showSkeleton = isLoadingAgents && agents.length === 0;
        const showEmpty = paginatedAgents.length === 0;
        const showTable = !showSkeleton && !showEmpty;

        console.log('[AgentList] 🎨 Rendering decision:', {
          isLoadingAgents,
          agentsLength: agents.length,
          paginatedLength: paginatedAgents.length,
          showSkeleton,
          showEmpty,
          showTable
        });

        return null;
      })()}
      {isLoadingAgents && agents.length === 0 ? (
        <TableSkeleton rows={10} columns={7} />
      ) : paginatedAgents.length === 0 ? (
        <div className="border border-border rounded-lg p-12 text-center">
          <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
            <FileText className="h-6 w-6 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">No agents found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery || filterType !== 'ALL' || filterStatus !== 'ALL'
              ? 'Try adjusting your filters or search query'
              : 'Get started by creating your first agent'}
          </p>
          {!searchQuery && filterType === 'ALL' && filterStatus === 'ALL' && (
            <Button onClick={() => navigate('/agent-builder/agents/new')}>
              <Plus className="h-4 w-4 mr-2" />
              Create Your First Agent
            </Button>
          )}
        </div>
      ) : (
        <>
          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-muted border-b border-border">
                <tr>
                  <th className="text-left p-3 font-semibold text-sm">Name</th>
                  <th className="text-left p-3 font-semibold text-sm">Type</th>
                  <th className="text-left p-3 font-semibold text-sm">Status</th>
                  <th className="text-left p-3 font-semibold text-sm">Created</th>
                  <th className="text-left p-3 font-semibold text-sm">Last Executed</th>
                  <th className="text-center p-3 font-semibold text-sm">Executions</th>
                  <th className="text-right p-3 font-semibold text-sm">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {paginatedAgents.map((agent) => (
                  <tr key={agent.agent_id} className="hover:bg-muted/50 transition-colors">
                    <td className="p-3">
                      <div>
                        <p className="font-medium">{agent.name}</p>
                        <p className="text-sm text-muted-foreground line-clamp-1">
                          {agent.description}
                        </p>
                      </div>
                    </td>
                    <td className="p-3">
                      <TypeBadge type={agent.agent_type} />
                    </td>
                    <td className="p-3">
                      <StatusBadge status={agent.status} />
                    </td>
                    <td className="p-3 text-sm text-muted-foreground">
                      {new Date(agent.created_at).toLocaleDateString()}
                    </td>
                    <td className="p-3 text-sm text-muted-foreground">
                      {agent.last_executed 
                        ? new Date(agent.last_executed).toLocaleDateString()
                        : 'Never'}
                    </td>
                    <td className="p-3 text-center">
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted text-sm font-medium">
                        {agent.execution_count}
                      </span>
                    </td>
                    <td className="p-3">
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleView(agent.agent_id)}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm">
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleEdit(agent.agent_id)}>
                              <Edit className="h-4 w-4 mr-2" />
                              Edit
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleClone(agent.agent_id)}>
                              <Copy className="h-4 w-4 mr-2" />
                              Clone
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleExecute(agent.agent_id)}>
                              <Play className="h-4 w-4 mr-2" />
                              Execute
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleDeploy(agent.agent_id)}>
                              <Rocket className="h-4 w-4 mr-2" />
                              Deploy
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem 
                              onClick={() => handleDelete(agent.agent_id)}
                              className="text-destructive"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Page {currentPage} of {totalPages}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </>
      )}
      </div>
    </div>
  );
}

