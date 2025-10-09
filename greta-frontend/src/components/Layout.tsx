import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Home,
  Bot,
  Workflow,
  Settings,
  Menu,
  X,
  Activity,
  Zap,
  Brain,
  Crown,
  GraduationCap,
  Search,
  TrendingUp,
  Layers,
  Wrench,
  Shield,
  Eye,
  ChevronDown,
  ChevronRight,
  HardDrive,
  Users,
  BookOpen,
  Building
} from 'lucide-react';

// Add shimmer animation styles
const shimmerStyles = `
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
  .animate-shimmer {
    position: relative;
    overflow: hidden;
  }
  .animate-shimmer::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    animation: shimmer 3s infinite;
  }
`;

// Inject styles
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = shimmerStyles;
  document.head.appendChild(styleSheet);
}

interface LayoutProps {
  children: React.ReactNode;
}

interface NavigationSection {
  name: string;
  items: NavigationItem[];
  icon: React.ComponentType<any>;
}

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<any>;
  description?: string;
}

const navigationSections: NavigationSection[] = [
  {
    name: 'Core',
    icon: Home,
    items: [
      { name: 'Master Agent', href: '/', icon: Bot, description: 'Main chat interface' },
      { name: 'System Overview', href: '/system-overview', icon: Activity, description: 'System metrics' },
      { name: 'Dashboard', href: '/dashboard', icon: Home, description: 'Enhanced dashboard' },
    ]
  },
  {
    name: 'Intelligence',
    icon: Brain,
    items: [
      { name: 'PAI Hub', href: '/pai-hub', icon: Brain, description: 'PAI orchestrator control' },
      { name: 'Master Agent', href: '/master-agent', icon: Crown, description: 'Greta master control' },
      { name: 'Reasoning', href: '/reasoning', icon: Zap, description: 'Hierarchical reasoning' },
      { name: 'Predictive', href: '/predictive', icon: TrendingUp, description: 'Analytics & forecasting' },
      { name: 'CPAS4', href: '/cpas4', icon: Layers, description: 'Cognitive Processing System' },
    ]
  },
  {
    name: 'Agents',
    icon: Users,
    items: [
      { name: 'Agent Management', href: '/agents', icon: Bot, description: 'Multi-agent coordination' },
      { name: 'Autonomous', href: '/autonomous', icon: Zap, description: 'Self-directed operations' },
      { name: 'Agent Builder', href: '/agent-builder', icon: Wrench, description: 'Create custom agents' },
    ]
  },
  {
    name: 'Tools',
    icon: Wrench,
    items: [
      { name: 'MCP Hub', href: '/mcp-hub', icon: Wrench, description: 'MCP servers & tools' },
      { name: 'Research Tools', href: '/research', icon: Search, description: 'Jina MCP integration' },
      { name: 'OSINT', href: '/osint', icon: Eye, description: 'Intelligence gathering' },
      { name: 'Multi-Modal', href: '/multimodal', icon: Layers, description: 'Voice, text, media' },
      { name: 'Memory', href: '/memory', icon: HardDrive, description: 'Memory orchestration' },
    ]
  },
  {
    name: 'Learning',
    icon: GraduationCap,
    items: [
      { name: 'Training Platform', href: '/training', icon: GraduationCap, description: 'AI learning curriculum' },
      { name: 'Knowledge Base', href: '/knowledge', icon: BookOpen, description: 'System knowledge' },
    ]
  },
  {
    name: 'Business',
    icon: Building,
    items: [
      { name: 'Business Hub', href: '/business', icon: Building, description: 'CRM, ATS & integrations' },
      { name: 'Enterprise', href: '/enterprise', icon: Shield, description: 'Commercial features' },
    ]
  },
  {
    name: 'Operations',
    icon: Activity,
    items: [
      { name: 'Workflows', href: '/workflows', icon: Workflow, description: 'Automation pipelines' },
      { name: 'System Monitor', href: '/system', icon: Activity, description: 'Performance monitoring' },
      { name: 'Settings', href: '/settings', icon: Settings, description: 'System configuration' },
    ]
  }
];

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [expandedSections, setExpandedSections] = useState<string[]>(['Core', 'Intelligence']);
  const location = useLocation();

  const toggleSection = (sectionName: string) => {
    setExpandedSections(prev =>
      prev.includes(sectionName)
        ? prev.filter(s => s !== sectionName)
        : [...prev, sectionName]
    );
  };

  return (
    <div className="flex h-screen relative overflow-hidden" style={{
      background: 'linear-gradient(135deg, #0a0e27 0%, #1a1d3a 25%, #2d3561 50%, #1e2749 75%, #0f1419 100%)'
    }}>
      {/* Sci-fi background effects */}
      <div className="absolute inset-0 pointer-events-none" style={{
        background: `
          radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, rgba(147, 51, 234, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 40% 40%, rgba(16, 185, 129, 0.05) 0%, transparent 50%)
        `
      }}></div>
      {/* Mobile sidebar */}
      <motion.div
        initial={false}
        animate={{ x: sidebarOpen ? 0 : '-100%' }}
        className="fixed inset-y-0 left-0 z-50 w-64 lg:hidden"
        style={{
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 25%, rgba(51, 65, 85, 0.85) 50%, rgba(30, 41, 59, 0.9) 75%, rgba(15, 23, 42, 0.95) 100%)',
          backdropFilter: 'blur(24px)',
          borderRight: '1px solid rgba(59, 130, 246, 0.2)'
        }}
      >
        <div className="flex items-center justify-between h-16 px-6">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-blue-800 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">GRETA</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="text-white hover:text-gray-300"
          >
            <X className="w-6 h-6" />
          </button>
        </div>
        <nav className="mt-8">
          {navigationSections.map((section) => (
            <div key={section.name} className="mb-4">
              <button
                onClick={() => toggleSection(section.name)}
                className="flex items-center justify-between w-full px-6 py-2 text-sm font-semibold text-gray-400 hover:text-white transition-colors"
              >
                <div className="flex items-center">
                  <section.icon className="w-4 h-4 mr-2" />
                  {section.name}
                </div>
                {expandedSections.includes(section.name) ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
              </button>
              {expandedSections.includes(section.name) && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-1"
                >
                  {section.items.map((item) => {
                    const Icon = item.icon;
                    const isActive = location.pathname === item.href;
                    return (
                      <Link
                        key={item.name}
                        to={item.href}
                        onClick={() => setSidebarOpen(false)}
                        className={`flex items-center px-8 py-2 text-sm font-medium transition-colors ${
                          isActive
                            ? 'text-white bg-blue-600/20 border-r-2 border-blue-500'
                            : 'text-gray-300 hover:text-white hover:bg-blue-500/10'
                        }`}
                      >
                        <Icon className="w-4 h-4 mr-3" />
                        <div>
                          <div>{item.name}</div>
                          {item.description && (
                            <div className="text-xs text-gray-500">{item.description}</div>
                          )}
                        </div>
                      </Link>
                    );
                  })}
                </motion.div>
              )}
            </div>
          ))}
        </nav>
      </motion.div>

      {/* Desktop sidebar */}
      <div className="hidden lg:flex lg:flex-shrink-0">
        <div className="flex flex-col w-64">
          <div className="flex flex-col flex-grow pt-5 pb-4 overflow-y-auto relative" style={{
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 25%, rgba(51, 65, 85, 0.85) 50%, rgba(30, 41, 59, 0.9) 75%, rgba(15, 23, 42, 0.95) 100%)',
            backdropFilter: 'blur(24px)',
            borderRight: '1px solid rgba(59, 130, 246, 0.2)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
          }}>
            <div className="flex items-center flex-shrink-0 px-6">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-blue-800 rounded-2xl flex items-center justify-center relative overflow-hidden">
                  <Zap className="w-7 h-7 text-white relative z-10" />
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 to-transparent animate-shimmer"></div>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">⚡ GRETA</h1>
                  <p className="text-xs text-blue-300">Personal AI System</p>
                </div>
              </div>
            </div>
            <nav className="mt-8 flex-1 px-3 space-y-2">
              {navigationSections.map((section) => (
                <div key={section.name} className="mb-6">
                  <button
                    onClick={() => toggleSection(section.name)}
                    className="flex items-center justify-between w-full px-3 py-2 text-sm font-semibold text-gray-400 hover:text-white transition-colors rounded-lg hover:bg-blue-500/10"
                  >
                    <div className="flex items-center">
                      <section.icon className="w-4 h-4 mr-2" />
                      {section.name}
                    </div>
                    {expandedSections.includes(section.name) ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </button>
                  {expandedSections.includes(section.name) && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-2 space-y-1"
                    >
                      {section.items.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname === item.href;
                        return (
                          <Link
                            key={item.name}
                            to={item.href}
                            className={`group flex items-center px-6 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                              isActive
                                ? 'text-white bg-gradient-to-r from-blue-600/20 to-blue-800/20 border border-blue-500/30'
                                : 'text-gray-300 hover:text-white hover:bg-blue-500/15'
                            }`}
                          >
                            <Icon className={`w-4 h-4 mr-3 transition-colors ${
                              isActive ? 'text-blue-400' : 'text-gray-400 group-hover:text-white'
                            }`} />
                            <div className="flex-1">
                              <div className="text-sm">{item.name}</div>
                              {item.description && (
                                <div className="text-xs text-gray-500 mt-0.5">{item.description}</div>
                              )}
                            </div>
                          </Link>
                        );
                      })}
                    </motion.div>
                  )}
                </div>
              ))}
            </nav>
            
            {/* System Status */}
            <div className="px-3 mt-6">
              <div className="glass p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">System Status</span>
                  <Activity className="w-4 h-4 text-green-400" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Backend</span>
                    <span className="text-green-400">Online</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Agents</span>
                    <span className="text-blue-400">7 Active</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Memory</span>
                    <span className="text-yellow-400">74%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Mobile header */}
        <div className="lg:hidden">
          <div className="flex items-center justify-between h-16 px-4" style={{
            background: 'rgba(15, 23, 42, 0.8)',
            backdropFilter: 'blur(24px)',
            borderBottom: '1px solid rgba(59, 130, 246, 0.2)'
          }}>
            <button
              onClick={() => setSidebarOpen(true)}
              className="text-white hover:text-gray-300"
            >
              <Menu className="w-6 h-6" />
            </button>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-blue-800 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg font-bold text-white">GRETA</span>
            </div>
            <div className="w-6" /> {/* Spacer */}
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="py-6"
          >
            {children}
          </motion.div>
        </main>
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
};
