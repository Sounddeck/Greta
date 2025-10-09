import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Brain, 
  Activity, 
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
  Folder as FolderThreeQuarters,
  Users,
  Layers
} from 'lucide-react';
import { apiService } from '../services/api';

// Autonomous Operations Interfaces
interface AutonomousWorkflow {
  id: string;
  name: string;
  description: string;
  type: 'decision_tree' | 'rule_based' | 'ml_driven' | 'hybrid';
  status: 'active' | 'paused' | 'completed' | 'error' | 'draft';
  priority: 'low' | 'medium' | 'high' | 'critical';
  triggers: {
    type: 'schedule' | 'event' | 'condition' | 'manual';
    configuration: any;
  }[];
  steps: {
    id: string;
    name: string;
    type: 'action' | 'decision' | 'condition' | 'loop' | 'parallel';
    configuration: any;
    dependencies: string[];
    timeout: number;
    retry_count: number;
  }[];
  performance: {
    success_rate: number;
    avg_execution_time: number;
    total_executions: number;
    last_execution: string;
    error_count: number;
  };
  created_at: string;
  updated_at: string;
  created_by: string;
}

interface DecisionNode {
  id: string;
  name: string;
  type: 'condition' | 'action' | 'branch' | 'merge';
  conditions: {
    field: string;
    operator: 'equals' | 'not_equals' | 'greater_than' | 'less_than' | 'contains' | 'regex';
    value: any;
    logic: 'and' | 'or';
  }[];
  actions: {
    type: 'api_call' | 'agent_invoke' | 'data_transform' | 'notification' | 'delay';
    configuration: any;
  }[];
  children: string[];
  position: { x: number; y: number };
}

interface AutonomousTask {
  id: string;
  workflow_id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  started_at: string;
  completed_at?: string;
  execution_log: {
    timestamp: string;
    step_id: string;
    step_name: string;
    status: 'started' | 'completed' | 'failed' | 'skipped';
    duration: number;
    output: any;
    error?: string;
  }[];
  input_data: any;
  output_data: any;
  assigned_agents: string[];
}

interface SystemMetrics {
  total_workflows: number;
  active_workflows: number;
  total_executions_today: number;
  success_rate: number;
  avg_execution_time: number;
  resource_utilization: {
    cpu: number;
    memory: number;
    storage: number;
    network: number;
  };
  agent_utilization: {
    total_agents: number;
    active_agents: number;
    idle_agents: number;
    busy_agents: number;
  };
}

// Mock Data
const mockWorkflows: AutonomousWorkflow[] = [
  {
    id: 'workflow-1',
    name: 'Customer Onboarding Automation',
    description: 'Automated customer onboarding process with document verification and account setup',
    type: 'hybrid',
    status: 'active',
    priority: 'high',
    triggers: [
      {
        type: 'event',
        configuration: { event: 'new_customer_signup', source: 'crm' }
      }
    ],
    steps: [
      {
        id: 'step-1',
        name: 'Verify Customer Information',
        type: 'action',
        configuration: { agent: 'verification-agent', timeout: 300 },
        dependencies: [],
        timeout: 300,
        retry_count: 3
      },
      {
        id: 'step-2',
        name: 'Document Processing',
        type: 'parallel',
        configuration: { agents: ['document-agent', 'compliance-agent'] },
        dependencies: ['step-1'],
        timeout: 600,
        retry_count: 2
      }
    ],
    performance: {
      success_rate: 94.2,
      avg_execution_time: 847,
      total_executions: 1247,
      last_execution: '2024-01-20T15:30:00Z',
      error_count: 23
    },
    created_at: '2024-01-10T09:00:00Z',
    updated_at: '2024-01-20T14:00:00Z',
    created_by: 'john.doe'
  }
];

const AutonomousOperationsCenter: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'workflows' | 'tasks' | 'decision_trees' | 'monitoring' | 'settings'>('dashboard');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [showCreateWorkflow, setShowCreateWorkflow] = useState(false);

  const queryClient = useQueryClient();

  // Queries
  const { data: workflows = mockWorkflows, isLoading: workflowsLoading } = useQuery<AutonomousWorkflow[]>({
    queryKey: ['autonomous-workflows'],
    queryFn: async (): Promise<AutonomousWorkflow[]> => {
      try {
        const response = await apiService.request<AutonomousWorkflow[]>('/api/v1/autonomous/workflows');
        return response.data || mockWorkflows;
      } catch (error) {
        return mockWorkflows;
      }
    }
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-purple-600 rounded-xl">
              <Workflow className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Autonomous Operations Center</h1>
              <p className="text-gray-400">Self-directed workflows and intelligent automation systems</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowCreateWorkflow(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Create Workflow</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
              <Brain className="w-4 h-4" />
              <span>AI Assistant</span>
            </button>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-400">Autonomous Mode Active</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-blue-500/10 p-1 rounded-lg">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
            { id: 'workflows', label: 'Workflows', icon: Workflow },
            { id: 'tasks', label: 'Active Tasks', icon: Activity },
            { id: 'decision_trees', label: 'Decision Trees', icon: GitBranch },
            { id: 'monitoring', label: 'Monitoring', icon: Monitor },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-blue-500/15'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Active Workflows</p>
                    <p className="text-2xl font-bold text-white">12</p>
                  </div>
                  <Workflow className="w-8 h-8 text-purple-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+3 this week</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Tasks Today</p>
                    <p className="text-2xl font-bold text-white">847</p>
                  </div>
                  <Activity className="w-8 h-8 text-blue-400" />
                </div>
                <p className="text-blue-400 text-sm mt-2">234 completed</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Success Rate</p>
                    <p className="text-2xl font-bold text-white">94.2%</p>
                  </div>
                  <Target className="w-8 h-8 text-green-400" />
                </div>
                <div className="mt-4 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                  <div className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full" style={{ width: '94.2%' }}></div>
                </div>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Avg Execution</p>
                    <p className="text-2xl font-bold text-white">14.2s</p>
                  </div>
                  <Clock className="w-8 h-8 text-yellow-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">-2.1s improvement</p>
              </div>
            </div>

            {/* System Status and Recent Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <Monitor className="w-5 h-5 mr-2 text-blue-400" />
                  System Status
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Workflow Engine</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-green-400 text-sm">Operational</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Decision Engine</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-green-400 text-sm">Operational</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Agent Pool</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                      <span className="text-yellow-400 text-sm">High Load</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Data Pipeline</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-green-400 text-sm">Operational</span>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-gray-600">
                  <h4 className="text-lg font-medium text-white mb-3">Resource Utilization</h4>
                  <div className="space-y-3">
                    {[
                      { name: 'CPU', usage: 67, color: 'blue' },
                      { name: 'Memory', usage: 84, color: 'yellow' },
                      { name: 'Storage', usage: 45, color: 'green' },
                      { name: 'Network', usage: 23, color: 'purple' }
                    ].map((resource) => (
                      <div key={resource.name}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-400">{resource.name}</span>
                          <span className="text-white">{resource.usage}%</span>
                        </div>
                        <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full bg-${resource.color}-500`}
                            style={{ width: `${resource.usage}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <Activity className="w-5 h-5 mr-2 text-green-400" />
                  Recent Activity
                </h3>
                <div className="space-y-4">
                  {[
                    {
                      id: 1,
                      workflow: 'Customer Onboarding',
                      action: 'Completed successfully',
                      time: '2 minutes ago',
                      status: 'success'
                    },
                    {
                      id: 2,
                      workflow: 'Invoice Processing',
                      action: 'Started execution',
                      time: '5 minutes ago',
                      status: 'running'
                    },
                    {
                      id: 3,
                      workflow: 'Data Backup',
                      action: 'Failed at step 3',
                      time: '12 minutes ago',
                      status: 'error'
                    },
                    {
                      id: 4,
                      workflow: 'Report Generation',
                      action: 'Waiting for approval',
                      time: '18 minutes ago',
                      status: 'pending'
                    }
                  ].map((activity) => (
                    <div key={activity.id} className="flex items-start space-x-3 p-3 bg-blue-500/10 rounded-lg">
                      <div className={`w-2 h-2 rounded-full mt-2 ${
                        activity.status === 'success' ? 'bg-green-400' :
                        activity.status === 'running' ? 'bg-blue-400' :
                        activity.status === 'error' ? 'bg-red-400' :
                        'bg-yellow-400'
                      }`}></div>
                      <div className="flex-1">
                        <p className="text-white text-sm font-medium">{activity.workflow}</p>
                        <p className="text-gray-400 text-xs">{activity.action}</p>
                        <p className="text-gray-500 text-xs mt-1">{activity.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Workflows Tab */}
        {activeTab === 'workflows' && (
          <div className="space-y-6">
            {/* Filters and Search */}
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search workflows..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
                  />
                </div>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="paused">Paused</option>
                  <option value="completed">Completed</option>
                  <option value="error">Error</option>
                  <option value="draft">Draft</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['autonomous-workflows'] })}
                  className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Workflows Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {workflows.map((workflow) => (
                <div key={workflow.id} className="glass p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        workflow.status === 'active' ? 'bg-green-500/20 text-green-400' :
                        workflow.status === 'paused' ? 'bg-yellow-500/20 text-yellow-400' :
                        workflow.status === 'completed' ? 'bg-blue-500/20 text-blue-400' :
                        workflow.status === 'error' ? 'bg-red-500/20 text-red-400' :
                        'bg-blue-600/20 text-gray-400'
                      }`}>
                        <Workflow className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">{workflow.name}</h3>
                        <p className="text-sm text-gray-400">{workflow.type} • {workflow.priority} priority</p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      workflow.status === 'active' ? 'bg-green-500/20 text-green-400' :
                      workflow.status === 'paused' ? 'bg-yellow-500/20 text-yellow-400' :
                      workflow.status === 'completed' ? 'bg-blue-500/20 text-blue-400' :
                      workflow.status === 'error' ? 'bg-red-500/20 text-red-400' :
                      'bg-blue-600/20 text-gray-400'
                    }`}>
                      {workflow.status}
                    </span>
                  </div>

                  <p className="text-gray-300 text-sm mb-4 line-clamp-2">{workflow.description}</p>

                  {/* Performance Metrics */}
                  <div className="space-y-3 mb-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Success Rate</p>
                        <p className="text-white font-semibold">{workflow.performance.success_rate}%</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Executions</p>
                        <p className="text-white font-semibold">{workflow.performance.total_executions}</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Avg Time</p>
                        <p className="text-white font-semibold">{workflow.performance.avg_execution_time}s</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Steps</p>
                        <p className="text-white font-semibold">{workflow.steps.length}</p>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                    <div className="flex items-center space-x-2">
                      {workflow.status === 'paused' || workflow.status === 'draft' ? (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                          <Play className="w-3 h-3" />
                          <span>Start</span>
                        </button>
                      ) : workflow.status === 'active' ? (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs transition-colors">
                          <Pause className="w-3 h-3" />
                          <span>Pause</span>
                        </button>
                      ) : null}
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <Eye className="w-3 h-3" />
                        <span>Monitor</span>
                      </button>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => setSelectedWorkflow(workflow.id)}
                        className="p-1 hover:bg-blue-500/15 rounded"
                      >
                        <Edit className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-blue-500/15 rounded">
                        <Copy className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-blue-500/15 rounded">
                        <Settings className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AutonomousOperationsCenter;
