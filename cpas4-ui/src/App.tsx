console.log('🚀 APP.TSX IS LOADING - Top of file');

import { useEffect, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'sonner';
import { Header } from './components/layout/Header';
import { LeftSidebar } from './components/layout/LeftSidebar';
import { RightSidebar } from './components/layout/RightSidebar';
import { MasterAgentContext } from './components/agents/MasterAgentContext';
import SystemStatusBar from './components/system/SystemStatusBar';
import { useUIStore } from './stores/uiStore';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { MCPStoreInitializer } from './components/MCPStoreInitializer';
import ErrorBoundary from './components/ErrorBoundary';

console.log('✅ APP.TSX - All imports successful');

// Lazy load large components for better code splitting
console.log('🔄 APP.TSX - Starting lazy imports...');
const AgentBuilderWorkflow = lazy(() => import('./components/agent-builder/AgentBuilderWorkflow'));
const VisualAgentBuilder = lazy(() => import('./components/agent-builder/VisualAgentBuilder').then(m => ({ default: m.VisualAgentBuilder })));
const VoiceSettings = lazy(() => import('./components/voice/VoiceSettings').then(m => ({ default: m.VoiceSettings })));
const MCPManager = lazy(() => import('./components/mcp/MCPManager').then(m => ({ default: m.MCPManager })));
const HookManager = lazy(() => import('./components/hooks/HookManager').then(m => ({ default: m.HookManager })));
const FigmaIntegration = lazy(() => import('./components/figma/FigmaIntegration').then(m => ({ default: m.FigmaIntegration })));
const PromptLibrary = lazy(() => import('./pages/PromptLibrary'));
const PromptEnhancement = lazy(() => import('./pages/PromptEnhancement'));
const GoogleAccountSettings = lazy(() => import('./pages/GoogleAccountSettings'));
const LeadGenerationDashboard = lazy(() => import('./pages/LeadGenerationDashboard'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const EnhancementHistory = lazy(() => import('./pages/EnhancementHistory'));
const EnhancementAnalytics = lazy(() => import('./pages/EnhancementAnalytics'));
const TestPage = lazy(() => import('./pages/TestPage'));

// Agent Builder Pages
const AgentList = lazy(() => import('./pages/AgentBuilder/AgentList').then(m => ({ default: m.AgentList })));
const AgentForm = lazy(() => import('./pages/AgentBuilder/AgentForm').then(m => ({ default: m.AgentForm })));
const AgentDetail = lazy(() => import('./pages/AgentBuilder/AgentDetail'));
const DeploymentDashboard = lazy(() => import('./pages/AgentBuilder/DeploymentDashboard'));
const TemplateGallery = lazy(() => import('./pages/AgentBuilder/TemplateGallery').then(m => ({ default: m.TemplateGallery })));

// Team Management Pages
const TeamList = lazy(() => import('./components/agent-builder/TeamList').then(m => ({ default: m.TeamList })));
const TeamBuilder = lazy(() => import('./components/agent-builder/TeamBuilder').then(m => ({ default: m.TeamBuilder })));
const TeamExecutionViewer = lazy(() => import('./components/agent-builder/TeamExecutionViewer').then(m => ({ default: m.TeamExecutionViewer })));

// Analytics Pages
const AgentAnalyticsDashboard = lazy(() => import('./components/agent-builder/AgentAnalyticsDashboard').then(m => ({ default: m.AgentAnalyticsDashboard })));

// Business Management Pages
console.log('🔄 APP.TSX - Loading Business Management pages...');
const BusinessManagement = lazy(() => import('./pages/BusinessManagement'));
const BusinessDetail = lazy(() => import('./pages/BusinessDetail'));

// Workflow Builder Page
const WorkflowBuilder = lazy(() => import('./pages/WorkflowBuilder'));

// Workflow Dashboard Page (Recommendation 4)
const WorkflowDashboard = lazy(() => import('./pages/WorkflowDashboard'));

// Training Page
const Training = lazy(() => import('./pages/Training'));

// OSINT Dashboard Page
const OSINTDashboard = lazy(() => import('./pages/OSINTDashboard'));

// CRM and ATS Dashboard Pages
const CRMDashboard = lazy(() => import('./pages/CRMDashboard'));
const ATSDashboard = lazy(() => import('./pages/ATSDashboard'));

console.log('✅ APP.TSX - All lazy imports defined successfully');

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
  console.log('🎯 APP FUNCTION IS EXECUTING');
  const { theme } = useUIStore();
  console.log('✅ APP - useUIStore hook executed, theme:', theme);

  // Apply theme on mount
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <WebSocketProvider showToasts={true}>
          {/* Initialize MCP Store from MongoDB on app startup */}
          <MCPStoreInitializer />

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

                {/* Agent Builder Routes */}
                <Route path="/agent-builder" element={<AgentBuilderWorkflow />} />
                <Route path="/agent-builder/visual" element={<VisualAgentBuilder />} />
                <Route path="/agent-builder/agents" element={<AgentList />} />
                <Route path="/agent-builder/agents/new" element={<AgentForm />} />
                <Route path="/agent-builder/agents/:agentId" element={<AgentDetail />} />
                <Route path="/agent-builder/agents/:agentId/edit" element={<AgentForm />} />
                <Route path="/agent-builder/templates" element={<TemplateGallery />} />
                <Route path="/agent-builder/deployments" element={<DeploymentDashboard />} />

                {/* Team Management Routes */}
                <Route path="/teams" element={<TeamList />} />
                <Route path="/teams/new" element={<TeamBuilder />} />
                <Route path="/teams/:teamId" element={<TeamExecutionViewer />} />
                <Route path="/teams/:teamId/edit" element={<TeamBuilder />} />
                <Route path="/teams/:teamId/execute" element={<TeamExecutionViewer />} />

                {/* Analytics Routes */}
                <Route path="/agent-analytics" element={<AgentAnalyticsDashboard />} />

                {/* Business Management Routes */}
                <Route path="/businesses" element={<BusinessManagement />} />
                <Route path="/businesses/:id" element={<BusinessDetail />} />

                {/* Lead Generation Routes */}
                <Route path="/leads" element={<LeadGenerationDashboard />} />

                {/* Workflow Builder Routes */}
                <Route path="/workflows" element={<WorkflowBuilder />} />
                <Route path="/workflow-dashboard" element={<WorkflowDashboard />} />

                {/* Training Routes */}
                <Route path="/training" element={<Training />} />

                {/* OSINT Dashboard */}
                <Route path="/osint" element={<OSINTDashboard />} />

                {/* CRM and ATS Routes */}
                <Route path="/crm" element={<CRMDashboard />} />
                <Route path="/ats" element={<ATSDashboard />} />

                {/* Other Routes */}
                <Route path="/voice" element={<VoiceSettings />} />
                <Route path="/mcp" element={<MCPManager />} />
                <Route path="/hooks" element={<HookManager />} />
                <Route path="/figma" element={<FigmaIntegration />} />
                <Route path="/prompts/library" element={<PromptLibrary />} />
                <Route path="/prompts/enhance" element={<PromptEnhancement />} />
                <Route path="/prompts/history" element={<EnhancementHistory />} />
                <Route path="/prompts/analytics" element={<EnhancementAnalytics />} />
                <Route path="/settings/google-accounts" element={<GoogleAccountSettings />} />
                <Route path="/test" element={<TestPage />} />
                <Route path="/settings" element={<SettingsPage />} />
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
      </WebSocketProvider>
    </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
