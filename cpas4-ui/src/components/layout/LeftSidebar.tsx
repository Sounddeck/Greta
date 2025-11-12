import { ChevronLeft, ChevronRight, Bot, History, Zap, Workflow, Mic, Server, Webhook, BookTemplate, Users, Target, Settings, BarChart3, Palette, Boxes, LayoutTemplate, Building2, GraduationCap, Sparkles } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { toast } from 'sonner';
import { Button } from '../ui/button';
import { useUIStore } from '../../stores/uiStore';
import { cn } from '../../lib/utils';
import type { AgentType } from '../../types';

export function LeftSidebar() {
  const { leftSidebarCollapsed, toggleLeftSidebar, setCurrentAgentType } = useUIStore();
  const navigate = useNavigate();
  const location = useLocation();



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

      <div className="h-full overflow-y-auto p-4 space-y-6 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent hover:scrollbar-thumb-gray-400">
        {/* Master Agent Section */}
        {!leftSidebarCollapsed && (
          <>
            <div className="pt-2">
              <button
                onClick={() => navigate('/')}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                  location.pathname === '/' && 'bg-primary/10 text-primary font-medium'
                )}
                title="Master Agent - AI Assistant"
                aria-label="Master Agent - AI Assistant"
              >
                <Bot className="h-5 w-5 text-primary" />
                <span className="font-semibold">Master Agent</span>
              </button>
            </div>

            {/* Agent Builder Section */}
            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-2 mb-3">
                <Boxes className="h-5 w-5 text-secondary" />
                <h2 className="font-semibold text-sm">Agent Builder</h2>
              </div>

              <div className="space-y-1">
                <button
                  onClick={() => navigate('/agent-builder/agents')}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                    location.pathname.startsWith('/agent-builder/agents') && 'bg-primary/10 text-primary font-medium'
                  )}
                  title="Manage custom agents"
                  aria-label="Manage custom agents"
                >
                  <Bot className="h-4 w-4 text-primary" />
                  <span>Agent List</span>
                </button>

                <button
                  onClick={() => navigate('/agent-builder/templates')}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                    location.pathname === '/agent-builder/templates' && 'bg-primary/10 text-primary font-medium'
                  )}
                  title="Browse agent templates"
                  aria-label="Browse agent templates"
                >
                  <LayoutTemplate className="h-4 w-4 text-primary" />
                  <span>Template Gallery</span>
                </button>

                <button
                  onClick={() => navigate('/agent-builder')}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                    location.pathname === '/agent-builder' && 'bg-primary/10 text-primary font-medium'
                  )}
                  title="Visual workflow designer"
                  aria-label="Visual workflow designer"
                >
                  <Workflow className="h-4 w-4 text-primary" />
                  <span>Workflow Designer</span>
                </button>

                <button
                  onClick={() => navigate('/workflows')}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                    location.pathname === '/workflows' && 'bg-primary/10 text-primary font-medium'
                  )}
                  title="Visual workflow builder"
                  aria-label="Visual workflow builder"
                >
                  <Workflow className="h-4 w-4 text-primary" />
                  <span>Workflow Builder</span>
                </button>

                <button
                  onClick={() => navigate('/teams')}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                    location.pathname.startsWith('/teams') && 'bg-primary/10 text-primary font-medium'
                  )}
                  title="Multi-agent teams"
                  aria-label="Multi-agent teams"
                >
                  <Users className="h-4 w-4 text-primary" />
                  <span>Teams</span>
                </button>
              </div>
            </div>

            {/* Business Management Section */}
            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-2 mb-3">
                <Building2 className="h-5 w-5 text-secondary" />
                <h2 className="font-semibold text-sm">Business</h2>
              </div>

              <div className="space-y-1">
                <button
                  onClick={() => navigate('/businesses')}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                    location.pathname === '/businesses' && 'bg-primary/10 text-primary font-medium'
                  )}
                  title="Manage autonomous AI-powered businesses"
                  aria-label="Business Management"
                >
                  <Building2 className="h-4 w-4 text-primary" />
                  <span>Business Management</span>
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
                  onClick={() => navigate('/training')}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer",
                    location.pathname === '/training' && 'bg-primary/10 text-primary font-medium'
                  )}
                  title="LLM training automation"
                  aria-label="LLM training automation"
                >
                  <GraduationCap className="h-4 w-4 text-primary" />
                  <span>LLM Training</span>
                </button>

                <button
                  onClick={() => navigate('/figma')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Figma design-to-code integration"
                  aria-label="Figma design-to-code integration"
                >
                  <Palette className="h-4 w-4 text-primary" />
                  <span>Figma to Code</span>
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
                  onClick={() => navigate('/prompts/enhance')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="Enhance your prompts with AI"
                  aria-label="Enhance your prompts with AI"
                >
                  <Sparkles className="h-4 w-4 text-primary" />
                  <span>Prompt Enhancement</span>
                </button>

                <button
                  onClick={() => navigate('/prompts/history')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="View prompt enhancement history"
                  aria-label="View prompt enhancement history"
                >
                  <History className="h-4 w-4 text-primary" />
                  <span>Enhancement History</span>
                </button>

                <button
                  onClick={() => navigate('/prompts/analytics')}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-accent hover:text-accent-foreground text-left transition-colors cursor-pointer"
                  title="View enhancement analytics"
                  aria-label="View enhancement analytics"
                >
                  <BarChart3 className="h-4 w-4 text-primary" />
                  <span>Enhancement Analytics</span>
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
