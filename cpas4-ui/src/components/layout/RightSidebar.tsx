import { ChevronLeft, ChevronRight, Clock, Activity, CheckCircle2 } from 'lucide-react';
import { Button } from '../ui/button';
import { VoiceControl } from '../voice/VoiceControl';
import { useUIStore } from '../../stores/uiStore';
import { cn } from '../../lib/utils';
import ActivityFeed from '../status/ActivityFeed';

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

      <div className="h-full flex flex-col overflow-hidden">
        {/* Voice Control Section */}
        {!rightSidebarCollapsed && (
          <div className="p-4 border-b border-border">
            <VoiceControl />
          </div>
        )}

        {/* Activity Feed - Takes remaining space */}
        {!rightSidebarCollapsed && (
          <div className="flex-1 p-4 overflow-hidden">
            <ActivityFeed />
          </div>
        )}

      </div>
    </aside>
  );
}

