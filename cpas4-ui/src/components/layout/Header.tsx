import { Moon, Sun, Settings } from 'lucide-react';
import { Button } from '../ui/button';
import { useUIStore } from '../../stores/uiStore';
import { SettingsDialog } from '../shared/SettingsDialog';
import { useState } from 'react';
import NotificationCenter from '../notifications/NotificationCenter';
import ConnectionStatus from '../status/ConnectionStatus';

export function Header() {
  const { theme, toggleTheme } = useUIStore();
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <>
      <header className="h-16 border-b border-border bg-card px-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
            CPAS4
          </h1>
          <span className="text-sm text-muted-foreground">
            Comprehensive Personal Automation System
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Connection Status */}
          <ConnectionStatus />

          {/* Notification Center */}
          <NotificationCenter />

          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          >
            {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSettingsOpen(true)}
            title="Settings"
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </header>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  );
}

