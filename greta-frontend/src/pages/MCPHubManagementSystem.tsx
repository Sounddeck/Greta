import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Server, 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  Settings, 
  Search,
  Globe,
  Database,
  Code,
  Brain,
  Zap,
  Shield,
  Eye,
  TrendingUp,
  FileText,
  Image,
  MessageSquare,
  Mail,
  Calendar,
  DollarSign,
  BarChart3,
  Users,
  Wrench,
  Package,
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
  Lightbulb as Candle,
  Lightbulb as Lamp,
  Lightbulb as Lantern,
  Lightbulb as Spotlight,
  Radio,
  Tv,
  Smartphone,
  Tablet,
  Laptop,
  Monitor as Desktop,
  Clock as Watch,
  Headphones,
  Speaker,
  Mic as Microphone,
  Music,
  Disc,
  Music as Cassette,
  Music as Vinyl,
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
  Ear,
  Ear as Hearing,
  EarOff as Deaf,
  Hand as SignLanguage,
  Circle as Braille,
  Accessibility,
  Circle as Wheelchair,
  Circle as WheelchairMove,
  Ruler as Cane,
  User as Guide,
  Heart as ServiceDog,
  Glasses,
  Glasses as Sunglasses,
  Circle as Monocle,
  Telescope,
  Microscope,
  Search as Magnifier,
  Search as MagnifyingGlass,
  Binoculars,
  Telescope as Periscope,
  Radar,
  Satellite,
  Radio as Antenna,
  Signal,
  Signal as SignalHigh,
  Signal as SignalLow,
  Signal as SignalMedium,
  SignalZero,
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
  FolderArchive,
  Folder as FolderRestore,
  Folder as FolderDelete,
  Folder as FolderEmpty,
  Folder as FolderFull,
  Folder as FolderHalf,
  Folder as FolderQuarter,
  Folder as FolderThreeQuarters,
  Film,
  Target,
  Clock,
  Filter,
  Layers,
  Network,
  HardDrive,
  Wifi,
  Cpu,
  Monitor,
  Terminal,
  ExternalLink,
  User,
  Copy,
  Play,
  Pause,
  Square,
  Workflow,
  TestTube
} from 'lucide-react';
import { apiService } from '../services/api';

// MCP Hub Management System Interfaces
interface MCPServer {
  id: string;
  name: string;
  description: string;
  version: string;
  status: 'active' | 'inactive' | 'error' | 'connecting' | 'deploying' | 'updating';
  capabilities: string[];
  provider: string;
  endpoint: string;
  category: 'search' | 'communication' | 'analytics' | 'development' | 'business' | 'ai' | 'security' | 'integration';
  deployment: {
    environment: 'development' | 'staging' | 'production';
    replicas: number;
    auto_scaling: boolean;
    health_checks: boolean;
    monitoring_enabled: boolean;
    last_deployed: string;
  };
  configuration: {
    api_key?: string;
    base_url?: string;
    timeout: number;
    retry_count: number;
    rate_limit: number;
    custom_headers: { [key: string]: string };
    environment_variables: { [key: string]: string };
  };
  metrics: {
    total_requests: number;
    successful_requests: number;
    failed_requests: number;
    avg_response_time: number;
    uptime_percentage: number;
    last_used: string;
    error_rate: number;
  };
  marketplace: {
    published: boolean;
    price: number;
    downloads: number;
    rating: number;
    reviews: number;
    author: string;
    license: string;
    documentation_url: string;
    repository_url: string;
  };
  created_at: string;
  updated_at: string;
}

interface MCPIntegration {
  id: string;
  name: string;
  description: string;
  server_ids: string[];
  workflow_template: string;
  configuration: any;
  status: 'active' | 'inactive' | 'error';
  usage_count: number;
  created_at: string;
}

interface MCPMarketplaceItem {
  id: string;
  name: string;
  description: string;
  category: string;
  author: string;
  version: string;
  price: number;
  is_free: boolean;
  downloads: number;
  rating: number;
  reviews: number;
  tags: string[];
  screenshots: string[];
  documentation_url: string;
  repository_url: string;
  demo_url: string;
  license: string;
  requirements: string[];
  compatibility: string[];
  last_updated: string;
  featured: boolean;
  verified: boolean;
}

// Mock Data
const mockServers: MCPServer[] = [
  {
    id: 'server-1',
    name: 'Exa Search',
    description: 'Advanced web search and content discovery with AI-powered relevance',
    version: '2.1.0',
    status: 'active',
    capabilities: ['web_search', 'content_discovery', 'semantic_search', 'real_time_data'],
    provider: 'Exa AI',
    endpoint: 'https://api.exa.ai/v1',
    category: 'search',
    deployment: {
      environment: 'production',
      replicas: 3,
      auto_scaling: true,
      health_checks: true,
      monitoring_enabled: true,
      last_deployed: '2024-01-20T10:00:00Z'
    },
    configuration: {
      api_key: '***hidden***',
      base_url: 'https://api.exa.ai/v1',
      timeout: 30000,
      retry_count: 3,
      rate_limit: 1000,
      custom_headers: {},
      environment_variables: {}
    },
    metrics: {
      total_requests: 15420,
      successful_requests: 14987,
      failed_requests: 433,
      avg_response_time: 1.2,
      uptime_percentage: 99.7,
      last_used: '2024-01-20T15:30:00Z',
      error_rate: 2.8
    },
    marketplace: {
      published: true,
      price: 0,
      downloads: 25000,
      rating: 4.8,
      reviews: 342,
      author: 'Exa AI',
      license: 'MIT',
      documentation_url: 'https://docs.exa.ai',
      repository_url: 'https://github.com/exa-labs/exa-mcp'
    },
    created_at: '2024-01-10T09:00:00Z',
    updated_at: '2024-01-20T14:00:00Z'
  }
];

const MCPHubManagementSystem: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'servers' | 'marketplace' | 'integrations' | 'monitoring' | 'deployment' | 'settings'>('servers');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterCategory, setFilterCategory] = useState('all');
  const [selectedServer, setSelectedServer] = useState<string | null>(null);
  const [showAddServer, setShowAddServer] = useState(false);

  const queryClient = useQueryClient();

  // Queries
  const { data: servers = mockServers, isLoading: serversLoading } = useQuery<MCPServer[]>({
    queryKey: ['mcp-servers'],
    queryFn: async (): Promise<MCPServer[]> => {
      try {
        const response = await apiService.request<MCPServer[]>('/api/v1/mcp/servers');
        return response.data || mockServers;
      } catch (error) {
        return mockServers;
      }
    }
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-orange-600 rounded-xl">
              <Server className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">MCP Hub Management System</h1>
              <p className="text-gray-400">Deploy, configure, monitor, and manage Model Context Protocol servers</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowAddServer(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Add Server</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
              <Package className="w-4 h-4" />
              <span>Marketplace</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors">
              <Workflow className="w-4 h-4" />
              <span>Integrations</span>
            </button>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-400">MCP Hub Active</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-white/5 p-1 rounded-lg">
          {[
            { id: 'servers', label: 'Servers', icon: Server },
            { id: 'marketplace', label: 'Marketplace', icon: Package },
            { id: 'integrations', label: 'Integrations', icon: Workflow },
            { id: 'monitoring', label: 'Monitoring', icon: Monitor },
            { id: 'deployment', label: 'Deployment', icon: Rocket },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-orange-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/10'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Servers Tab */}
        {activeTab === 'servers' && (
          <div className="space-y-6">
            {/* Filters and Search */}
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search servers..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-orange-500"
                  />
                </div>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-orange-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                  <option value="error">Error</option>
                  <option value="connecting">Connecting</option>
                  <option value="deploying">Deploying</option>
                  <option value="updating">Updating</option>
                </select>
                <select
                  value={filterCategory}
                  onChange={(e) => setFilterCategory(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-orange-500"
                >
                  <option value="all">All Categories</option>
                  <option value="search">Search</option>
                  <option value="communication">Communication</option>
                  <option value="analytics">Analytics</option>
                  <option value="development">Development</option>
                  <option value="business">Business</option>
                  <option value="ai">AI</option>
                  <option value="security">Security</option>
                  <option value="integration">Integration</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['mcp-servers'] })}
                  className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Servers Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {servers.map((server) => (
                <div key={server.id} className="glass p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        server.status === 'active' ? 'bg-green-500/20 text-green-400' :
                        server.status === 'inactive' ? 'bg-gray-500/20 text-gray-400' :
                        server.status === 'error' ? 'bg-red-500/20 text-red-400' :
                        server.status === 'connecting' ? 'bg-yellow-500/20 text-yellow-400' :
                        server.status === 'deploying' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-purple-500/20 text-purple-400'
                      }`}>
                        <Server className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">{server.name}</h3>
                        <p className="text-sm text-gray-400">{server.category} • v{server.version}</p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      server.status === 'active' ? 'bg-green-500/20 text-green-400' :
                      server.status === 'inactive' ? 'bg-gray-500/20 text-gray-400' :
                      server.status === 'error' ? 'bg-red-500/20 text-red-400' :
                      server.status === 'connecting' ? 'bg-yellow-500/20 text-yellow-400' :
                      server.status === 'deploying' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-purple-500/20 text-purple-400'
                    }`}>
                      {server.status}
                    </span>
                  </div>

                  <p className="text-gray-300 text-sm mb-4 line-clamp-2">{server.description}</p>

                  {/* Server Metrics */}
                  <div className="space-y-3 mb-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Uptime</p>
                        <p className="text-white font-semibold">{server.metrics.uptime_percentage}%</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Requests</p>
                        <p className="text-white font-semibold">{server.metrics.total_requests.toLocaleString()}</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Avg Response</p>
                        <p className="text-white font-semibold">{server.metrics.avg_response_time}s</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Error Rate</p>
                        <p className="text-white font-semibold">{server.metrics.error_rate}%</p>
                      </div>
                    </div>
                    <div className="text-sm">
                      <p className="text-gray-400">Capabilities</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {server.capabilities.slice(0, 3).map((capability) => (
                          <span key={capability} className="text-xs px-2 py-1 bg-orange-500/20 text-orange-400 rounded">
                            {capability}
                          </span>
                        ))}
                        {server.capabilities.length > 3 && (
                          <span className="text-xs px-2 py-1 bg-gray-500/20 text-gray-400 rounded">
                            +{server.capabilities.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                    <div className="flex items-center space-x-2">
                      {server.status === 'inactive' ? (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                          <Play className="w-3 h-3" />
                          <span>Start</span>
                        </button>
                      ) : server.status === 'active' ? (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs transition-colors">
                          <Pause className="w-3 h-3" />
                          <span>Stop</span>
                        </button>
                      ) : null}
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <TestTube className="w-3 h-3" />
                        <span>Test</span>
                      </button>
                      <button className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-xs transition-colors">
                        <Rocket className="w-3 h-3" />
                        <span>Deploy</span>
                      </button>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => setSelectedServer(server.id)}
                        className="p-1 hover:bg-white/10 rounded"
                      >
                        <Eye className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded">
                        <Edit className="w-4 h-4 text-gray-400" />
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

        {/* Marketplace Tab */}
        {activeTab === 'marketplace' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">MCP Server Marketplace</h2>
              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search marketplace..."
                    className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-orange-500"
                  />
                </div>
                <select className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-orange-500">
                  <option value="all">All Categories</option>
                  <option value="featured">Featured</option>
                  <option value="free">Free</option>
                  <option value="premium">Premium</option>
                  <option value="verified">Verified</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
                      <Search className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Advanced Web Search Pro</h3>
                      <p className="text-sm text-gray-400">by SearchCorp</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400 fill-current" />
                    <span className="text-sm text-white">4.9</span>
                  </div>
                </div>

                <p className="text-gray-300 text-sm mb-4">Enhanced web search with AI-powered relevance, real-time data, and advanced filtering capabilities.</p>

                <div className="flex flex-wrap gap-1 mb-4">
                  <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded">Search</span>
                  <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">Verified</span>
                  <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded">Premium</span>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                  <div>
                    <p className="text-gray-400">Downloads</p>
                    <p className="text-white">45.2K</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Version</p>
                    <p className="text-white">v3.1.2</p>
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
                    <p className="text-sm text-green-400 font-semibold">$19/month</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Monitoring Tab */}
        {activeTab === 'monitoring' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Server Monitoring</h2>
              <div className="flex items-center space-x-2">
                <select className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-orange-500">
                  <option value="1h">Last Hour</option>
                  <option value="24h">Last 24 Hours</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
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
                    <p className="text-gray-400 text-sm">Total Servers</p>
                    <p className="text-2xl font-bold text-white">24</p>
                  </div>
                  <Server className="w-8 h-8 text-orange-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+3 this week</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Active Servers</p>
                    <p className="text-2xl font-bold text-white">21</p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-green-400" />
                </div>
                <p className="text-blue-400 text-sm mt-2">87.5% uptime</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Total Requests</p>
                    <p className="text-2xl font-bold text-white">847K</p>
                  </div>
                  <Activity className="w-8 h-8 text-blue-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+15% this month</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Avg Response</p>
                    <p className="text-2xl font-bold text-white">1.2s</p>
                  </div>
                  <Clock className="w-8 h-8 text-yellow-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">-0.3s improvement</p>
              </div>
            </div>

            {/* Performance Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <LineChart className="w-5 h-5 mr-2 text-blue-400" />
                  Request Volume
                </h3>
                <div className="h-64 flex items-center justify-center text-gray-400">
                  <p>Request volume chart visualization would go here</p>
                </div>
              </div>

              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <PieChart className="w-5 h-5 mr-2 text-green-400" />
                  Server Status Distribution
                </h3>
                <div className="h-64 flex items-center justify-center text-gray-400">
                  <p>Status distribution chart visualization would go here</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MCPHubManagementSystem;
