import { useEffect, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'sonner';
import { Header } from './components/layout/Header';
import { LeftSidebar } from './components/layout/LeftSidebar';
import { RightSidebar } from './components/layout/RightSidebar';
import { MasterAgentContext } from './components/agents/MasterAgentContext';
import SystemStatusBar from './components/system/SystemStatusBar';
import { useUIStore } from './stores/uiStore';

// Lazy load large components for better code splitting
const AgentBuilderWorkflow = lazy(() => import('./components/agent-builder/AgentBuilderWorkflow'));
const VoiceSettings = lazy(() => import('./components/voice/VoiceSettings').then(m => ({ default: m.VoiceSettings })));
const MCPManager = lazy(() => import('./components/mcp/MCPManager').then(m => ({ default: m.MCPManager })));
const HookManager = lazy(() => import('./components/hooks/HookManager').then(m => ({ default: m.HookManager })));
const PromptLibrary = lazy(() => import('./pages/PromptLibrary'));

// Loading fallback component
function LoadingFallback() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading...</p>
      </div>
    </div>
  );
}

function App() {
  const { theme } = useUIStore();

  // Apply theme on mount
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  return (
    <BrowserRouter>
      {/* Toast Notifications */}
      <Toaster position="top-right" richColors />
      
      <div className="h-screen flex flex-col overflow-hidden">
        {/* Header */}
        <Header />

        {/* Main Content Area with Three Columns */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left Sidebar */}
          <LeftSidebar />

          {/* Center Area - Routes */}
          <main className="flex-1 overflow-hidden">
            <Suspense fallback={<LoadingFallback />}>
              <Routes>
                <Route path="/" element={<MasterAgentContext />} />
                <Route path="/agent/:agentType" element={<MasterAgentContext />} />
                <Route path="/agent-builder" element={<AgentBuilderWorkflow />} />
                <Route path="/voice" element={<VoiceSettings />} />
                <Route path="/mcp" element={<MCPManager />} />
                <Route path="/hooks" element={<HookManager />} />
                <Route path="/prompts/library" element={<PromptLibrary />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </main>

          {/* Right Sidebar */}
          <RightSidebar />
        </div>

        {/* System Status Bar */}
        <SystemStatusBar />
      </div>
    </BrowserRouter>
  );
}

export default App;
