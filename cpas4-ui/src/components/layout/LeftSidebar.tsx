import { ChevronLeft, ChevronRight, Bot, History, Zap, Plus, FileText, Workflow, Mic, Server, Webhook, BookTemplate, Users, Target, Settings } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { toast } from 'sonner';
import { Button } from '../ui/button';
import { useUIStore } from '../../stores/uiStore';
import { cn } from '../../lib/utils';
import type { AgentType } from '../../types';

const AGENT_TYPES = [
  { id: 'research', name: 'Research', icon: '🔍' },
  { id: 'coding', name: 'Coding', icon: '💻' },
  { id: 'data_analysis', name: 'Data Analysis', icon: '📊' },
  { id: 'content_creation', name: 'Content Creation', icon: '✍️' },
  { id: 'task_automation', name: 'Task Automation', icon: '⚙️' },
  { id: 'customer_support', name: 'Customer Support', icon: '💬' },
  { id: 'sales', name: 'Sales', icon: '💰' },
  { id: 'marketing', name: 'Marketing', icon: '📢' },
  { id: 'finance', name: 'Finance', icon: '💵' },
  { id: 'general', name: 'General', icon: '🤖' },
];

export function LeftSidebar() {
  const { leftSidebarCollapsed, toggleLeftSidebar, setCurrentAgentType } = useUIStore();
  const navigate = useNavigate();
  const location = useLocation();

  // Quick Action Handlers
  const handleNewAgent = () => {
    // Navigate to general agent with fresh conversation
    setCurrentAgentType('general');
    navigate('/agent/general');
    toast.success('Starting new conversation');
  };

  const handleExecuteTask = () => {
    // Navigate to home (MasterAgentContext)
    navigate('/');
    
    // Focus the input after navigation
    setTimeout(() => {
      const input = document.querySelector('textarea[placeholder*="message"]') as HTMLTextAreaElement;
      if (input) {
        input.focus();
        toast.info('Ready to execute task');
      }
    }, 100);
  };

  const handleGenerateReport = () => {
    // Show "Coming Soon" notification
    toast.info('Report generation coming soon!', {
      description: 'This feature is currently in development',
    });
  };

  return (
    <aside
      className={cn(
        'relative h-full border-r border-border bg-card transition-all duration-300',
        leftSidebarCollapsed ? 'w-16' : 'w-72'
      )}
    >
      {/* Collapse/Expand Button */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute -right-3 top-4 z-10 h-6 w-6 rounded-full border border-border bg-background"
        onClick={toggleLeftSidebar}
      >
        {leftSidebarCollapsed ? (
          <ChevronRight className="h-4 w-4" />
        ) : (
          <ChevronLeft className="h-4 w-4" />
        )}
      </Button>

      <div className="p-4 space-y-6">
        {/* Agent Types Section */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Bot className="h-5 w-5 text-primary" />
            {!leftSidebarCollapsed && (
              <h2 className="font-semibold text-sm">Agent Types</h2>
            )}
          </div>

          <div className="space-y-1">
            {AGENT_TYPES.map((agent) => {
              const isActive = location.pathname === `/agent/${agent.id}`;
              return (
                <button
                  key={agent.id}
                  onClick={() => {
                    setCurrentAgentType(agent.id as AgentType);
                    navigate(`/agent/${agent.id}`);
                  }}
                  className={cn(
                    'w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors',
                    'hover:bg-accent hover:text-accent-foreground',
                    'text-left',
                    isActive && 'bg-primary/10 text-primary font-medium'
                  )}
                  title={leftSidebarCollapsed ? agent.name : undefined}
                >
                  <span className="text-lg">{agent.icon}</span>
                  {!leftSidebarCollapsed && (
                    <span className="truncate">{agent.name}</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Quick Actions Section */}
        {!leftSidebarCollapsed && (
          <>
            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="h-5 w-5 text-secondary" />
                <h2 className="font-semibold text-sm">Quick Actions</h2>
              </div>

              <div className="space-y-1">
                <button 
                  onClick={handleNewAgent}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Start a new agent conversation"
                  aria-label="Start a new agent conversation"
                >
                  <Plus className="h-4 w-4 text-primary" />
                  <span>New Agent</span>
                </button>
                
                <button 
                  onClick={handleExecuteTask}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Execute a task with the master agent"
                  aria-label="Execute a task with the master agent"
                >
                  <Zap className="h-4 w-4 text-primary" />
                  <span>Execute Task</span>
                </button>
                
                <button 
                  onClick={handleGenerateReport}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Generate a system report (coming soon)"
                  aria-label="Generate a system report"
                >
                  <FileText className="h-4 w-4 text-primary" />
                  <span>Generate Report</span>
                </button>
        <button
          onClick={() => navigate('/agent-builder')}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
          title="Open visual agent workflow builder"
          aria-label="Open visual agent workflow builder"
        >
          <Workflow className="h-4 w-4 text-primary" />
          <span>Agent Builder</span>
        </button>
              </div>
            </div>

            {/* Advanced Features Section */}
            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="h-5 w-5 text-secondary" />
                <h2 className="font-semibold text-sm">Advanced</h2>
              </div>

              <div className="space-y-1">
                <button
                  onClick={() => navigate('/voice')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Voice integration settings"
                  aria-label="Voice integration settings"
                >
                  <Mic className="h-4 w-4 text-primary" />
                  <span>Voice Control</span>
                </button>

                <button
                  onClick={() => navigate('/mcp')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="MCP server and tool management"
                  aria-label="MCP server and tool management"
                >
                  <Server className="h-4 w-4 text-primary" />
                  <span>MCP Manager</span>
                </button>

                <button
                  onClick={() => navigate('/hooks')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Hook system and plugin management"
                  aria-label="Hook system and plugin management"
                >
                  <Webhook className="h-4 w-4 text-primary" />
                  <span>Hook System</span>
                </button>

                <button
                  onClick={() => navigate('/prompts/library')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Browse and use prompt templates"
                  aria-label="Browse and use prompt templates"
                >
                  <BookTemplate className="h-4 w-4 text-primary" />
                  <span>Prompt Library</span>
                </button>

                <button
                  onClick={() => navigate('/leads')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Lead generation and research"
                  aria-label="Lead generation and research"
                >
                  <Target className="h-4 w-4 text-primary" />
                  <span>Lead Generation</span>
                </button>

                <button
                  onClick={() => navigate('/settings/google-accounts')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Manage Google Workspace accounts"
                  aria-label="Manage Google Workspace accounts"
                >
                  <Users className="h-4 w-4 text-primary" />
                  <span>Google Accounts</span>
                </button>

                <button
                  onClick={() => navigate('/settings')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Application settings and configuration"
                  aria-label="Application settings"
                >
                  <Settings className="h-4 w-4 text-primary" />
                  <span>Settings</span>
                </button>
              </div>
            </div>

            {/* Recent Section */}
            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-2 mb-3">
                <History className="h-5 w-5 text-muted-foreground" />
                <h2 className="font-semibold text-sm">Recent</h2>
              </div>

              <div className="space-y-1">
                <div className="px-3 py-2 text-sm text-muted-foreground">
                  No recent activity
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
