import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Brain, 
  Users, 
  Settings, 
  Play, 
  Pause, 
  Square, 
  BarChart3, 
  Zap, 
  Target, 
  Globe, 
  Database, 
  MessageSquare, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  TrendingUp,
  Layers,
  Network,
  Eye,
  Shield,
  Cpu,
  HardDrive,
  Wifi,
  Server,
  Monitor,
  Terminal,
  Code,
  FileText,
  Search,
  Filter,
  Download,
  Upload,
  RefreshCw,
  Plus,
  Minus,
  Edit,
  Trash2,
  Save,
  X,
  ChevronRight,
  ChevronDown,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  Rocket,
  GitBranch,
  History,
  Star,
  Award,
  Wrench,
  Activity,
  DollarSign,
  Calendar,
  User,
  Copy,
  ExternalLink,
  Package,
  Workflow,
  TestTube,
  LineChart,
  PieChart,
  RotateCcw,
  FastForward,
  Rewind,
  SkipForward,
  SkipBack,
  Volume2,
  VolumeX,
  Mic,
  MicOff,
  Camera,
  CameraOff,
  Video,
  VideoOff,
  Phone,
  PhoneOff,
  Mail,
  Send,
  Inbox,
  Archive,
  Bookmark,
  BookmarkCheck,
  Flag,
  Tag,
  Tags,
  Link,
  Unlink,
  Lock,
  Unlock,
  Key,
  UserCheck,
  UserX,
  UserPlus,
  UserMinus,
  Crown,
  Briefcase,
  Building,
  Home,
  MapPin,
  Navigation,
  Compass,
  Map,
  Route,
  Car,
  Truck,
  Plane,
  Ship,
  Train,
  Bike,
  Footprints,
  TreePine,
  Mountain,
  Sun,
  Moon,
  Cloud,
  CloudRain,
  CloudSnow,
  CloudLightning,
  Umbrella,
  Thermometer,
  Wind,
  Droplets,
  Flame,
  Snowflake,
  Zap as Lightning,
  Battery,
  BatteryLow,
  Plug,
  Power,
  PowerOff,
  Lightbulb,
  Flashlight,
  Radio,
  Tv,
  Smartphone,
  Tablet,
  Laptop,
  Monitor as Desktop,
  Watch,
  Headphones,
  Speaker,
  Mic as Microphone,
  Camera as CameraIcon,
  Video as VideoIcon,
  Image,
  Film,
  Music,
  Disc,
  Music as Cassette,
  Disc as Vinyl,
  Music as Guitar,
  Music as Piano,
  Music as Drum,
  Music as Trumpet,
  Music as Violin,
  Music as Saxophone,
  Music as Flute,
  Music as Harp,
  Music as Banjo,
  Music as Accordion,
  Music as Harmonica,
  Music as Xylophone,
  Bell,
  Volume2 as Megaphone,
  Volume2 as Volume,
  Volume1 as VolumeDown,
  Volume2 as VolumeUp,
  VolumeX as Mute,
  Volume2 as Unmute,
  Speaker as SpeakerLoud,
  Speaker as SpeakerQuiet,
  Headphones as Headset,
  EarOff,
  Volume2 as Ear,
  User as Hearing,
  User as Deaf,
  User as SignLanguage,
  User as Braille,
  Accessibility,
  User as Wheelchair,
  User as WheelchairMove,
  User as Cane,
  User as Guide,
  User as ServiceDog,
  Glasses,
  Glasses as Sunglasses,
  Glasses as Monocle,
  Telescope,
  Microscope,
  Search as Magnifier,
  Search as MagnifyingGlass,
  Eye as Binoculars,
  Eye as Periscope,
  Radar,
  Satellite,
  Antenna,
  Signal,
  Signal as SignalHigh,
  Signal as SignalLow,
  Signal as SignalMedium,
  Signal as SignalZero,
  Bluetooth,
  Bluetooth as BluetoothConnected,
  Bluetooth as BluetoothSearching,
  BluetoothOff,
  WifiOff,
  Wifi as WifiHigh,
  Wifi as WifiLow,
  Wifi as WifiMedium,
  WifiOff as WifiZero,
  Router,
  Router as Modem,
  Cable as Ethernet,
  Cable,
  Usb,
  Usb as UsbC,
  Monitor as Hdmi,
  Monitor as Vga,
  Monitor as DisplayPort,
  Zap as Thunderbolt,
  Cable as FireWire,
  Cable as Serial,
  Cable as Parallel,
  HardDrive as Scsi,
  HardDrive as Ide,
  HardDrive as Sata,
  HardDrive as Nvme,
  HardDrive as SsdSolid,
  HardDrive as HddSolid,
  Disc as OpticalDisk,
  HardDrive as FloppyDisk,
  Music as CassetteIcon,
  Package as Cartridge,
  CreditCard as MemoryCard,
  CreditCard as SimCard,
  CreditCard,
  CreditCard as DebitCard,
  CreditCard as GiftCard,
  CreditCard as IdCard,
  FileText as Passport,
  FileText as Ticket,
  FileText as Receipt,
  FileText as Invoice,
  FileText as Bill,
  DollarSign as Money,
  DollarSign as Coins,
  DollarSign as Banknote,
  Wallet,
  Wallet as Purse,
  Lock as Safe,
  Lock as Vault,
  Building as Atm,
  Building as CashRegister,
  Building as PointOfSale,
  Barcode,
  QrCode,
  Search as Scanner,
  Printer,
  Printer as Fax,
  Printer as Copier,
  Trash2 as Shredder,
  Package as Laminator,
  Paperclip as Stapler,
  Paperclip as PaperClip,
  Pin as Pushpin,
  Pin as Thumbtack,
  Paperclip,
  Package as Binder,
  Folder,
  FolderOpen,
  Folder as FolderClosed,
  FolderPlus,
  FolderMinus,
  FolderX,
  FolderCheck,
  Folder as FolderHeart,
  Folder as FolderStar,
  FolderLock,
  FolderKey,
  FolderSearch,
  Folder as FolderEdit,
  Folder as FolderCopy,
  Folder as FolderMove,
  Folder as FolderSync,
  Folder as FolderCloud,
  FolderDown as FolderDownload,
  FolderUp as FolderUpload,
  FolderArchive,
  Folder as FolderRestore,
  Folder as FolderDelete,
  Folder as FolderEmpty,
  Folder as FolderFull,
  Folder as FolderHalf,
  Folder as FolderQuarter,
  Folder as FolderThreeQuarters
} from 'lucide-react';
import { apiService } from '../services/api';

// Agent Management Framework Interfaces
interface Agent {
  id: string;
  name: string;
  description: string;
  type: 'conversational' | 'task' | 'research' | 'analysis' | 'automation' | 'monitoring';
  status: 'active' | 'inactive' | 'training' | 'testing' | 'deployed' | 'error';
  version: string;
  created_at: string;
  updated_at: string;
  performance_metrics: {
    success_rate: number;
    avg_response_time: number;
    total_interactions: number;
    user_satisfaction: number;
    error_rate: number;
  };
  configuration: {
    model: string;
    temperature: number;
    max_tokens: number;
    system_prompt: string;
    tools: string[];
    memory_enabled: boolean;
    learning_enabled: boolean;
  };
  deployment: {
    environment: 'development' | 'staging' | 'production';
    replicas: number;
    auto_scaling: boolean;
    health_checks: boolean;
    rollback_available: boolean;
  };
  team: {
    owner: string;
    collaborators: string[];
    permissions: { [key: string]: string[] };
  };
  experiments: {
    active_experiments: number;
    ab_tests: number;
    champion_variant: string;
  };
}

interface Experiment {
  id: string;
  name: string;
  agent_id: string;
  type: 'ab_test' | 'multivariate' | 'champion_challenger' | 'canary';
  status: 'draft' | 'running' | 'completed' | 'paused' | 'cancelled';
  variants: {
    id: string;
    name: string;
    configuration: any;
    traffic_percentage: number;
    performance: {
      success_rate: number;
      avg_response_time: number;
      user_satisfaction: number;
      conversion_rate: number;
    };
  }[];
  metrics: {
    total_interactions: number;
    statistical_significance: number;
    confidence_level: number;
    winner: string | null;
  };
  created_at: string;
  duration_days: number;
}

interface AgentRegistry {
  id: string;
  name: string;
  category: string;
  description: string;
  tags: string[];
  author: string;
  downloads: number;
  rating: number;
  reviews: number;
  version: string;
  license: string;
  documentation_url: string;
  repository_url: string;
  demo_url: string;
  price: number;
  is_free: boolean;
  is_verified: boolean;
  last_updated: string;
}

interface Team {
  id: string;
  name: string;
  description: string;
  members: {
    user_id: string;
    username: string;
    role: 'owner' | 'admin' | 'developer' | 'viewer';
    joined_at: string;
  }[];
  agents: string[];
  experiments: string[];
  permissions: {
    can_create_agents: boolean;
    can_deploy_agents: boolean;
    can_run_experiments: boolean;
    can_manage_team: boolean;
  };
  created_at: string;
}

// Mock Data
const mockAgents: Agent[] = [
  {
    id: 'agent-1',
    name: 'Customer Support Assistant',
    description: 'AI agent for handling customer inquiries and support tickets',
    type: 'conversational',
    status: 'deployed',
    version: '2.1.0',
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-01-20T14:30:00Z',
    performance_metrics: {
      success_rate: 94.5,
      avg_response_time: 1.2,
      total_interactions: 15420,
      user_satisfaction: 4.6,
      error_rate: 2.1
    },
    configuration: {
      model: 'gpt-4-turbo',
      temperature: 0.7,
      max_tokens: 2048,
      system_prompt: 'You are a helpful customer support assistant...',
      tools: ['knowledge_base', 'ticket_system', 'escalation'],
      memory_enabled: true,
      learning_enabled: true
    },
    deployment: {
      environment: 'production',
      replicas: 3,
      auto_scaling: true,
      health_checks: true,
      rollback_available: true
    },
    team: {
      owner: 'john.doe',
      collaborators: ['jane.smith', 'mike.wilson'],
      permissions: {
        'jane.smith': ['read', 'write'],
        'mike.wilson': ['read']
      }
    },
    experiments: {
      active_experiments: 2,
      ab_tests: 1,
      champion_variant: 'variant-a'
    }
  }
];

const AgentManagementFramework: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'agents' | 'experiments' | 'registry' | 'teams' | 'analytics' | 'settings'>('agents');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [showCreateAgent, setShowCreateAgent] = useState(false);
  const [showCreateExperiment, setShowCreateExperiment] = useState(false);

  const queryClient = useQueryClient();

  // Queries
  const { data: agents = mockAgents, isLoading: agentsLoading } = useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: async (): Promise<Agent[]> => {
      try {
        const response = await apiService.request<Agent[]>('/api/v1/agents');
        return response.data || mockAgents;
      } catch (error) {
        return mockAgents;
      }
    }
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-blue-600 rounded-xl">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Agent Management Framework</h1>
              <p className="text-gray-400">LangSmith-inspired agent development and experimentation platform</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowCreateAgent(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Create Agent</span>
            </button>
            <button
              onClick={() => setShowCreateExperiment(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
            >
              <TestTube className="w-4 h-4" />
              <span>New Experiment</span>
            </button>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-400">Framework Active</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-white/5 p-1 rounded-lg">
          {[
            { id: 'agents', label: 'Agents', icon: Brain },
            { id: 'experiments', label: 'Experiments', icon: TestTube },
            { id: 'registry', label: 'Registry', icon: Package },
            { id: 'teams', label: 'Teams', icon: Users },
            { id: 'analytics', label: 'Analytics', icon: BarChart3 },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/10'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Agents Tab */}
        {activeTab === 'agents' && (
          <div className="space-y-6">
            {/* Filters and Search */}
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search agents..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="all">All Status</option>
                  <option value="deployed">Deployed</option>
                  <option value="active">Active</option>
                  <option value="training">Training</option>
                  <option value="testing">Testing</option>
                  <option value="inactive">Inactive</option>
                  <option value="error">Error</option>
                </select>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="all">All Types</option>
                  <option value="conversational">Conversational</option>
                  <option value="task">Task</option>
                  <option value="research">Research</option>
                  <option value="analysis">Analysis</option>
                  <option value="automation">Automation</option>
                  <option value="monitoring">Monitoring</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['agents'] })}
                  className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Agents Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {agents.map((agent) => (
                <div key={agent.id} className="glass p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        agent.status === 'deployed' ? 'bg-green-500/20 text-green-400' :
                        agent.status === 'active' ? 'bg-blue-500/20 text-blue-400' :
                        agent.status === 'training' ? 'bg-yellow-500/20 text-yellow-400' :
                        agent.status === 'testing' ? 'bg-purple-500/20 text-purple-400' :
                        agent.status === 'error' ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        <Brain className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
                        <p className="text-sm text-gray-400">{agent.type} • v{agent.version}</p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      agent.status === 'deployed' ? 'bg-green-500/20 text-green-400' :
                      agent.status === 'active' ? 'bg-blue-500/20 text-blue-400' :
                      agent.status === 'training' ? 'bg-yellow-500/20 text-yellow-400' :
                      agent.status === 'testing' ? 'bg-purple-500/20 text-purple-400' :
                      agent.status === 'error' ? 'bg-red-500/20 text-red-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      {agent.status}
                    </span>
                  </div>

                  <p className="text-gray-300 text-sm mb-4 line-clamp-2">{agent.description}</p>

                  {/* Performance Metrics */}
                  <div className="space-y-3 mb-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Success Rate</p>
                        <p className="text-white font-semibold">{agent.performance_metrics.success_rate}%</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Satisfaction</p>
                        <p className="text-white font-semibold">{agent.performance_metrics.user_satisfaction}/5</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Interactions</p>
                        <p className="text-white font-semibold">{agent.performance_metrics.total_interactions.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Avg Response</p>
                        <p className="text-white font-semibold">{agent.performance_metrics.avg_response_time}s</p>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                    <div className="flex items-center space-x-2">
                      {agent.status === 'inactive' ? (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                          <Play className="w-3 h-3" />
                          <span>Start</span>
                        </button>
                      ) : (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs transition-colors">
                          <Pause className="w-3 h-3" />
                          <span>Stop</span>
                        </button>
                      )}
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <TestTube className="w-3 h-3" />
                        <span>Test</span>
                      </button>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button 
                        onClick={() => setSelectedAgent(agent.id)}
                        className="p-1 hover:bg-white/10 rounded"
                      >
                        <Eye className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded">
                        <Edit className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded">
                        <Copy className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded">
                        <Settings className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Experiments Tab */}
        {activeTab === 'experiments' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">A/B Testing & Experiments</h2>
              <div className="flex items-center space-x-2">
                <select className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500">
                  <option value="all">All Experiments</option>
                  <option value="running">Running</option>
                  <option value="completed">Completed</option>
                  <option value="draft">Draft</option>
                </select>
                <button className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors">
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-purple-500/20 text-purple-400 rounded-lg">
                      <TestTube className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Response Style A/B Test</h3>
                      <p className="text-sm text-gray-400">Customer Support Assistant</p>
                    </div>
                  </div>
                  <span className="text-xs px-2 py-1 rounded-full bg-green-500/20 text-green-400">Running</span>
                </div>

                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-white/5 rounded-lg">
                      <p className="text-sm text-gray-400">Variant A (Control)</p>
                      <p className="text-white font-semibold">Formal tone</p>
                      <p className="text-sm text-blue-400">50% traffic</p>
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg">
                      <p className="text-sm text-gray-400">Variant B</p>
                      <p className="text-white font-semibold">Casual tone</p>
                      <p className="text-sm text-blue-400">50% traffic</p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Statistical Significance</span>
                      <span className="text-white">87%</span>
                    </div>
                    <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                      <div className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full" style={{ width: '87%' }}></div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Duration</p>
                      <p className="text-white">14 days</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Interactions</p>
                      <p className="text-white">2,847</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Confidence</p>
                      <p className="text-white">95%</p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <div className="flex items-center space-x-2">
                    <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                      <BarChart3 className="w-3 h-3" />
                      <span>View Results</span>
                    </button>
                    <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                      <CheckCircle className="w-3 h-3" />
                      <span>Promote Winner</span>
                    </button>
                  </div>
                  <div className="flex items-center space-x-1">
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Pause className="w-4 h-4 text-gray-400" />
                    </button>
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Settings className="w-4 h-4 text-gray-400" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Registry Tab */}
        {activeTab === 'registry' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Agent Registry & Marketplace</h2>
              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search registry..."
                    className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <select className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500">
                  <option value="all">All Categories</option>
                  <option value="customer-service">Customer Service</option>
                  <option value="sales">Sales</option>
                  <option value="research">Research</option>
                  <option value="automation">Automation</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
                      <Package className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">E-commerce Assistant</h3>
                      <p className="text-sm text-gray-400">by OpenAI</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400 fill-current" />
                    <span className="text-sm text-white">4.8</span>
                  </div>
                </div>

                <p className="text-gray-300 text-sm mb-4">Advanced e-commerce assistant with product recommendations, order tracking, and customer support capabilities.</p>

                <div className="flex flex-wrap gap-1 mb-4">
                  <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded">E-commerce</span>
                  <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">Verified</span>
                  <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded">Premium</span>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                  <div>
                    <p className="text-gray-400">Downloads</p>
                    <p className="text-white">12.5K</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Version</p>
                    <p className="text-white">v3.2.1</p>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <div className="flex items-center space-x-2">
                    <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                      <Download className="w-3 h-3" />
                      <span>Install</span>
                    </button>
                    <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                      <Eye className="w-3 h-3" />
                      <span>Preview</span>
                    </button>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-green-400 font-semibold">$29/month</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Teams Tab */}
        {activeTab === 'teams' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Team Collaboration</h2>
              <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
                <Plus className="w-4 h-4" />
                <span>Create Team</span>
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-green-500/20 text-green-400 rounded-lg">
                      <Users className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Customer Success Team</h3>
                      <p className="text-sm text-gray-400">5 members • 8 agents</p>
                    </div>
                  </div>
                  <button className="p-1 hover:bg-white/10 rounded">
                    <Settings className="w-4 h-4 text-gray-400" />
                  </button>
                </div>

                <p className="text-gray-300 text-sm mb-4">Team responsible for customer support agents and satisfaction metrics.</p>

                <div className="space-y-3 mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-xs text-white">JD</div>
                      <span className="text-white text-sm">John Doe</span>
                    </div>
                    <span className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded">Owner</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-xs text-white">JS</div>
                      <span className="text-white text-sm">Jane Smith</span>
                    </div>
                    <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded">Admin</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-xs text-white">MW</div>
                      <span className="text-white text-sm">Mike Wilson</span>
                    </div>
                    <span className="text-xs px-2 py-1 bg-gray-500/20 text-gray-400 rounded">Developer</span>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <div className="flex items-center space-x-2">
                    <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                      <UserPlus className="w-3 h-3" />
                      <span>Invite</span>
                    </button>
                    <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                      <Eye className="w-3 h-3" />
                      <span>View Agents</span>
                    </button>
                  </div>
                  <div className="flex items-center space-x-1">
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Edit className="w-4 h-4 text-gray-400" />
                    </button>
                    <button className="p-1 hover:bg-white/10 rounded">
                      <Settings className="w-4 h-4 text-gray-400" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Performance Analytics</h2>
              <div className="flex items-center space-x-2">
                <select className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500">
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                  <option value="90d">Last 90 Days</option>
                </select>
                <button className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors">
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Total Agents</p>
                    <p className="text-2xl font-bold text-white">24</p>
                  </div>
                  <Brain className="w-8 h-8 text-blue-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+3 this week</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Active Experiments</p>
                    <p className="text-2xl font-bold text-white">7</p>
                  </div>
                  <TestTube className="w-8 h-8 text-purple-400" />
                </div>
                <p className="text-blue-400 text-sm mt-2">2 completing soon</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Avg Success Rate</p>
                    <p className="text-2xl font-bold text-white">92.4%</p>
                  </div>
                  <Target className="w-8 h-8 text-green-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+2.1% improvement</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Total Interactions</p>
                    <p className="text-2xl font-bold text-white">847K</p>
                  </div>
                  <Activity className="w-8 h-8 text-yellow-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+15% this month</p>
              </div>
            </div>

            {/* Performance Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <LineChart className="w-5 h-5 mr-2 text-blue-400" />
                  Agent Performance Trends
                </h3>
                <div className="h-64 flex items-center justify-center text-gray-400">
                  <p>Performance chart visualization would go here</p>
                </div>
              </div>

              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <PieChart className="w-5 h-5 mr-2 text-green-400" />
                  Agent Type Distribution
                </h3>
                <div className="h-64 flex items-center justify-center text-gray-400">
                  <p>Distribution chart visualization would go here</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white">Framework Settings</h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass p-6 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-4">General Settings</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Default Model</label>
                    <select className="w-full px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500">
                      <option value="gpt-4-turbo">GPT-4 Turbo</option>
                      <option value="gpt-4">GPT-4</option>
                      <option value="claude-3">Claude 3</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Auto-scaling Threshold</label>
                    <input
                      type="number"
                      defaultValue={80}
                      className="w-full px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Enable Monitoring</span>
                    <button className="relative inline-flex h-6 w-11 items-center rounded-full bg-blue-600">
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white transition translate-x-6"></span>
                    </button>
                  </div>
                </div>
              </div>

              <div className="glass p-6 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-4">Security & Permissions</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Require 2FA</span>
                    <button className="relative inline-flex h-6 w-11 items-center rounded-full bg-blue-600">
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white transition translate-x-6"></span>
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Audit Logging</span>
                    <button className="relative inline-flex h-6 w-11 items-center rounded-full bg-blue-600">
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white transition translate-x-6"></span>
                    </button>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Session Timeout (minutes)</label>
                    <input
                      type="number"
                      defaultValue={60}
                      className="w-full px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentManagementFramework;
