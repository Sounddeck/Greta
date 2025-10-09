import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Database,
  Brain,
  Search,
  Filter,
  RefreshCw,
  Plus,
  Edit,
  Trash2,
  Eye,
  EyeOff,
  Settings,
  Download,
  Upload,
  Save,
  X,
  CheckCircle,
  AlertTriangle,
  Star,
  Award,
  Target,
  Activity,
  BarChart3,
  LineChart,
  PieChart,
  Layers,
  Network,
  Workflow,
  Zap,
  Globe,
  Shield,
  Lock,
  Unlock,
  Key,
  User,
  Users,
  UserCheck,
  UserX,
  UserPlus,
  UserMinus,
  Crown,
  Briefcase,
  Home,
  Building,
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
  Archive,
  Folder,
  FolderOpen,
  FileText,
  Image,
  Video,
  Music,
  Disc,
  Film,
  Radio,
  Tv,
  Smartphone,
  Tablet,
  Laptop,
  Monitor,
  Headphones,
  Speaker,
  Mic,
  Camera,
  Volume2,
  VolumeX,
  Play,
  Pause,
  Square,
  SkipForward,
  SkipBack,
  FastForward,
  Rewind,
  RotateCcw,
  Copy,
  Clipboard,
  Link,
  Unlink,
  ExternalLink,
  Share,
  Send,
  Inbox,
  Mail,
  MessageSquare,
  Phone,
  Calendar,
  Timer,
  Bell,
  BellOff,
  Flag,
  Bookmark,
  BookmarkCheck,
  Tag,
  Tags,
  Hash,
  AtSign,
  Percent,
  DollarSign,
  CreditCard,
  Wallet,
  Receipt,
  ShoppingCart,
  ShoppingBag,
  Package,
  Gift,
  Heart,
  HeartOff,
  ThumbsUp,
  ThumbsDown,
  Smile,
  Frown,
  Meh,
  Skull,
  Sword,
  Axe,
  Hammer,
  Wrench,
  Drill,
  Scissors,
  Ruler,
  Pencil,
  Pen,
  Paintbrush,
  Palette,
  Eraser,
  Paperclip,
  Pin,
  Calculator,
  Scale,
  Magnet,
  Microscope,
  Telescope,
  Binoculars,
  Glasses,
  Flashlight,
  Lightbulb,
  Sun,
  Moon,
  Cloud,
  CloudRain,
  CloudSnow,
  CloudLightning,
  Umbrella,
  Snowflake,
  Droplets,
  Flame,
  Bolt,
  Sparkles,
  Bomb,
  Rocket,
  Satellite,
  Drone,
  Feather,
  Leaf,
  Trees,
  TreePine,
  Flower,
  Cherry,
  Apple,
  Banana,
  Grape,
  Carrot,
  Fish,
  Coffee,
  Wheat,
  Nut,
  Circle,
  Triangle,
  Bot,
  Wand2,
  TrendingUp,
  Clock,
  Bug
} from 'lucide-react';
import { apiService } from '../services/api';

// Knowledge Base Interfaces
interface KnowledgeArticle {
  id: string;
  title: string;
  content: string;
  summary: string;
  category: string;
  tags: string[];
  author: string;
  created_at: string;
  updated_at: string;
  views: number;
  rating: number;
  status: 'draft' | 'published' | 'archived';
  metadata: {
    word_count: number;
    reading_time: number;
    difficulty_level: 'beginner' | 'intermediate' | 'advanced';
    source: string;
    verified: boolean;
  };
}

interface KnowledgeCategory {
  id: string;
  name: string;
  description: string;
  article_count: number;
  subcategories: string[];
  color: string;
}

interface KnowledgeMetrics {
  total_articles: number;
  total_categories: number;
  total_views: number;
  avg_rating: number;
  articles_this_month: number;
  popular_categories: { name: string; count: number }[];
  recent_activity: { type: string; title: string; timestamp: string }[];
}

const IntelligentKnowledgeBase: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  const [selectedArticle, setSelectedArticle] = useState<KnowledgeArticle | null>(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  // Mock data queries
  const { data: articles = [] } = useQuery({
    queryKey: ['knowledge-articles'],
    queryFn: async () => {
      try {
        const response = await apiService.request('/api/v1/knowledge/articles', { method: 'GET' });
        return response.data;
      } catch (error) {
        return [
          {
            id: '1',
            title: 'Getting Started with AI Agents',
            content: 'Comprehensive guide to building and deploying AI agents...',
            summary: 'Learn the fundamentals of AI agent development',
            category: 'AI Development',
            tags: ['ai', 'agents', 'tutorial'],
            author: 'AI Expert',
            created_at: '2024-01-15T10:00:00Z',
            updated_at: '2024-01-18T12:00:00Z',
            views: 1247,
            rating: 4.8,
            status: 'published',
            metadata: {
              word_count: 2500,
              reading_time: 12,
              difficulty_level: 'beginner',
              source: 'internal',
              verified: true
            }
          }
        ];
      }
    }
  });

  const { data: categories = [] } = useQuery({
    queryKey: ['knowledge-categories'],
    queryFn: async () => {
      try {
        const response = await apiService.request('/api/v1/knowledge/categories', { method: 'GET' });
        return response.data;
      } catch (error) {
        return [
          {
            id: '1',
            name: 'AI Development',
            description: 'Articles about AI and machine learning development',
            article_count: 45,
            subcategories: ['Neural Networks', 'Deep Learning', 'NLP'],
            color: 'blue'
          }
        ];
      }
    }
  });

  const { data: metrics } = useQuery<KnowledgeMetrics>({
    queryKey: ['knowledge-metrics'],
    queryFn: async (): Promise<KnowledgeMetrics> => {
      try {
        const response = await apiService.request<KnowledgeMetrics>('/api/v1/knowledge/metrics', { method: 'GET' });
        return response.data || {
          total_articles: 1247,
          total_categories: 28,
          total_views: 45891,
          avg_rating: 4.6,
          articles_this_month: 23,
          popular_categories: [
            { name: 'AI Development', count: 45 },
            { name: 'Machine Learning', count: 38 },
            { name: 'Data Science', count: 32 }
          ],
          recent_activity: [
            { type: 'created', title: 'New AI Agent Tutorial', timestamp: '2024-01-20T15:30:00Z' },
            { type: 'updated', title: 'ML Best Practices', timestamp: '2024-01-20T14:15:00Z' }
          ]
        };
      } catch (error) {
        return {
          total_articles: 1247,
          total_categories: 28,
          total_views: 45891,
          avg_rating: 4.6,
          articles_this_month: 23,
          popular_categories: [
            { name: 'AI Development', count: 45 },
            { name: 'Machine Learning', count: 38 },
            { name: 'Data Science', count: 32 }
          ],
          recent_activity: [
            { type: 'created', title: 'New AI Agent Tutorial', timestamp: '2024-01-20T15:30:00Z' },
            { type: 'updated', title: 'ML Best Practices', timestamp: '2024-01-20T14:15:00Z' }
          ]
        };
      }
    }
  });

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Articles</p>
              <p className="text-2xl font-bold text-white">{metrics?.total_articles || 0}</p>
            </div>
            <FileText className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Categories</p>
              <p className="text-2xl font-bold text-white">{metrics?.total_categories || 0}</p>
            </div>
            <Folder className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Views</p>
              <p className="text-2xl font-bold text-white">{metrics?.total_views?.toLocaleString() || 0}</p>
            </div>
            <Eye className="w-8 h-8 text-yellow-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Rating</p>
              <p className="text-2xl font-bold text-white">{metrics?.avg_rating || 0}/5</p>
            </div>
            <Star className="w-8 h-8 text-purple-400" />
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Intelligent Knowledge Base</h1>
            <p className="text-gray-300">Centralized knowledge management and intelligent content discovery</p>
          </div>
          <div className="flex space-x-4">
            <button className="glass px-4 py-2 rounded-lg text-white hover:bg-blue-500/15 transition-colors flex items-center">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </button>
            <button 
              onClick={() => setIsCreateModalOpen(true)}
              className="bg-gradient-to-r from-blue-500 to-purple-600 px-4 py-2 rounded-lg text-white hover:from-blue-600 hover:to-purple-700 transition-colors flex items-center"
            >
              <Plus className="w-4 h-4 mr-2" />
              Add Article
            </button>
          </div>
        </div>

        <div className="glass rounded-xl p-6 mb-6">
          <div className="flex space-x-1 mb-6">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'articles', label: 'Articles', icon: FileText },
              { id: 'categories', label: 'Categories', icon: Folder },
              { id: 'search', label: 'Search', icon: Search },
              { id: 'settings', label: 'Settings', icon: Settings }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg flex items-center transition-colors ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-blue-500/15'
                }`}
              >
                <tab.icon className="w-4 h-4 mr-2" />
                {tab.label}
              </button>
            ))}
          </div>

          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'articles' && (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-400">Articles management interface coming soon...</p>
            </div>
          )}
          {activeTab === 'categories' && (
            <div className="text-center py-12">
              <Folder className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-400">Categories management interface coming soon...</p>
            </div>
          )}
          {activeTab === 'search' && (
            <div className="text-center py-12">
              <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-400">Advanced search interface coming soon...</p>
            </div>
          )}
          {activeTab === 'settings' && (
            <div className="text-center py-12">
              <Settings className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-400">Knowledge base settings coming soon...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default IntelligentKnowledgeBase;
