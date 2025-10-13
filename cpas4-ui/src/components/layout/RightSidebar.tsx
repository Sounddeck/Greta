import { ChevronLeft, ChevronRight, Clock, Activity, CheckCircle2 } from 'lucide-react';
import { Button } from '../ui/button';
import { VoiceControl } from '../voice/VoiceControl';
import { useUIStore } from '../../stores/uiStore';
import { cn } from '../../lib/utils';

export function RightSidebar() {
  const { rightSidebarCollapsed, toggleRightSidebar } = useUIStore();

  return (
    <aside
      className={cn(
        'relative h-full border-l border-border bg-card transition-all duration-300',
        rightSidebarCollapsed ? 'w-16' : 'w-72'
      )}
    >
      {/* Collapse/Expand Button */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute -left-3 top-4 z-10 h-6 w-6 rounded-full border border-border bg-background"
        onClick={toggleRightSidebar}
      >
        {rightSidebarCollapsed ? (
          <ChevronLeft className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
      </Button>

      <div className="p-4 space-y-6 overflow-y-auto">
        {/* Voice Control Section */}
        {!rightSidebarCollapsed && (
          <VoiceControl />
        )}

        {/* Task History Section */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Clock className="h-5 w-5 text-primary" />
            {!rightSidebarCollapsed && (
              <h2 className="font-semibold text-sm">Task History</h2>
            )}
          </div>

          {!rightSidebarCollapsed && (
            <div className="space-y-2">
              <div className="px-3 py-2 text-sm text-muted-foreground">
                No tasks executed yet
              </div>
            </div>
          )}
        </div>

        {/* Active Jobs Section */}
        {!rightSidebarCollapsed && (
          <div className="border-t border-border pt-4">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="h-5 w-5 text-secondary" />
              <h2 className="font-semibold text-sm">Active Jobs</h2>
            </div>

            <div className="space-y-2">
              <div className="px-3 py-2 text-sm text-muted-foreground">
                No active jobs
              </div>
            </div>
          </div>
        )}

        {/* System Status Section */}
        {!rightSidebarCollapsed && (
          <div className="border-t border-border pt-4">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="h-5 w-5 text-green-500" />
              <h2 className="font-semibold text-sm">System Status</h2>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between px-3 py-2 rounded-md bg-accent/50">
                <span className="text-sm">Backend</span>
                <span className="text-xs text-green-500 font-medium">Online</span>
              </div>
              <div className="flex items-center justify-between px-3 py-2 rounded-md bg-accent/50">
                <span className="text-sm">Database</span>
                <span className="text-xs text-green-500 font-medium">Connected</span>
              </div>
              <div className="flex items-center justify-between px-3 py-2 rounded-md bg-accent/50">
                <span className="text-sm">Ollama</span>
                <span className="text-xs text-green-500 font-medium">Ready</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}

