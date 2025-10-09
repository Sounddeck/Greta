import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Chat } from './pages/Chat';
import { Agents } from './pages/Agents';
import { Settings } from './pages/Settings';
import { Workflows } from './pages/Workflows';

// New PAI System Pages
import { PAIHubManagement } from './pages/PAIHubManagement';
import { MasterAgentEnhanced } from './pages/MasterAgentEnhanced';
import { Reasoning } from './pages/Reasoning';
import { PredictiveAnalyticsEngine } from './pages/PredictiveAnalyticsEngine';
import { Autonomous } from './pages/Autonomous';
import AgentBuilder from './pages/AgentBuilder';
import AgentManagementFramework from './pages/AgentManagementFramework';
import AutonomousOperationsCenter from './pages/AutonomousOperationsCenter';
import AgentBuilderDevelopmentEnvironment from './pages/AgentBuilderDevelopmentEnvironment';
import MCPHubManagementSystem from './pages/MCPHubManagementSystem';
import OSINTManagementPlatform from './pages/OSINTManagementPlatform';
import MultiModalProcessing from './pages/MultiModalProcessing';
import FunctionalMemorySystem from './pages/FunctionalMemorySystem';
import TrainingPlatformManagement from './pages/TrainingPlatformManagement';
import IntelligentKnowledgeBase from './pages/IntelligentKnowledgeBase';
import { ResearchTools } from './pages/ResearchTools';
import { OSINTAgent } from './pages/OSINTAgent';
import { MultiModal } from './pages/MultiModal';
import { Memory } from './pages/Memory';
import { TrainingPlatform } from './pages/TrainingPlatform';
import { Knowledge } from './pages/Knowledge';
import { SystemMonitor } from './pages/SystemMonitor';
import { Enterprise } from './pages/Enterprise';
import EnhancedDashboard from './pages/EnhancedDashboard';
import MasterAgentChat from './pages/MasterAgentChat';
import MCPHub from './pages/MCPHub';
import Business from './pages/Business';
import EnhancedMasterAgent from './pages/EnhancedMasterAgent';
import CPAS4 from './pages/CPAS4';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div style={{ minHeight: '100vh' }}>
          <Layout>
            <Routes>
              {/* Core Routes */}
              <Route path="/" element={<MasterAgentChat />} />
              <Route path="/dashboard" element={<EnhancedDashboard />} />
              <Route path="/system-overview" element={<Dashboard />} />
              <Route path="/chat" element={<Chat />} />

              {/* Intelligence Routes */}
              <Route path="/pai-hub" element={<PAIHubManagement />} />
              <Route path="/master-agent" element={<EnhancedMasterAgent />} />
              <Route path="/master-agent-enhanced" element={<MasterAgentEnhanced />} />
              <Route path="/reasoning" element={<Reasoning />} />
              <Route path="/predictive" element={<PredictiveAnalyticsEngine />} />

              <Route path="/cpas4" element={<CPAS4 />} />

              {/* Agent Routes */}
              <Route path="/agents" element={<AgentManagementFramework />} />
              <Route path="/autonomous" element={<AutonomousOperationsCenter />} />
              <Route path="/agent-builder" element={<AgentBuilderDevelopmentEnvironment />} />

              {/* Tools Routes */}
              <Route path="/mcp-hub" element={<MCPHubManagementSystem />} />
              <Route path="/research-tools" element={<ResearchTools />} />
              <Route path="/osint" element={<OSINTManagementPlatform />} />
              <Route path="/multi-modal" element={<MultiModalProcessing />} />

              {/* Learning Routes */}
              <Route path="/memory" element={<FunctionalMemorySystem />} />
              <Route path="/training" element={<TrainingPlatformManagement />} />
              <Route path="/knowledge" element={<IntelligentKnowledgeBase />} />

              {/* Business Routes */}
              <Route path="/business" element={<Business />} />

              {/* Operations Routes */}
              <Route path="/workflows" element={<Workflows />} />
              <Route path="/system-monitor" element={<SystemMonitor />} />
              <Route path="/enterprise" element={<Enterprise />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Layout>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
