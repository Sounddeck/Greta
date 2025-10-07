import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  GraduationCap, 
  BookOpen, 
  Target, 
  Brain, 
  Users, 
  Clock, 
  CheckCircle, 
  Play, 
  Pause, 
  RotateCcw, 
  Award, 
  TrendingUp, 
  BarChart3, 
  Settings, 
  Plus, 
  Search, 
  Filter, 
  Download, 
  Upload,
  Star,
  Calendar,
  User,
  Zap,
  Database,
  Code,
  MessageSquare,
  Eye,
  Shield,
  Globe,
  Layers,
  Network,
  Activity,
  FileText,
  Video,
  Headphones,
  Image,
  Edit,
  Trash2,
  Save,
  X,
  ChevronRight,
  ChevronDown,
  ArrowRight
} from 'lucide-react';
import { apiService } from '../services/api';

interface TrainingProgram {
  id: string;
  title: string;
  description: string;
  category: 'greta_mastery' | 'ai_fundamentals' | 'business_intelligence' | 'technical_skills' | 'custom';
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  duration: number; // in hours
  modules: TrainingModule[];
  prerequisites: string[];
  learning_objectives: string[];
  completion_rate: number;
  rating: number;
  enrolled_users: number;
  created_at: string;
  updated_at: string;
  instructor: string;
  certification: boolean;
}

interface TrainingModule {
  id: string;
  title: string;
  description: string;
  type: 'video' | 'interactive' | 'hands_on' | 'assessment' | 'reading';
  duration: number; // in minutes
  content_url?: string;
  completed: boolean;
  progress: number;
  order: number;
  resources: TrainingResource[];
}

interface TrainingResource {
  id: string;
  title: string;
  type: 'document' | 'video' | 'link' | 'code' | 'dataset';
  url: string;
  description: string;
}

interface UserProgress {
  program_id: string;
  user_id: string;
  progress: number;
  completed_modules: string[];
  current_module?: string;
  started_at: string;
  last_accessed: string;
  time_spent: number; // in minutes
  quiz_scores: { [moduleId: string]: number };
  certificates_earned: string[];
}

interface LearningPath {
  id: string;
  title: string;
  description: string;
  programs: string[];
  estimated_duration: number;
  difficulty_progression: string[];
  target_audience: string;
  outcomes: string[];
}

export const TrainingPlatform: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'programs' | 'paths' | 'progress' | 'create'>('dashboard');
  const [selectedProgram, setSelectedProgram] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateProgram, setShowCreateProgram] = useState(false);
  const [newProgram, setNewProgram] = useState({
    title: '',
    description: '',
    category: 'custom' as const,
    difficulty: 'beginner' as const,
    learning_objectives: [''],
    prerequisites: ['']
  });

  const queryClient = useQueryClient();

  const { data: programs, isLoading: programsLoading } = useQuery<TrainingProgram[]>({
    queryKey: ['training-programs'],
    queryFn: async () => {
      try {
        const response = await apiService.request<TrainingProgram[]>('/api/v1/training/programs');
        return response.data || getMockPrograms();
      } catch (error) {
        return getMockPrograms();
      }
    }
  });

  const { data: userProgress, isLoading: progressLoading } = useQuery<UserProgress[]>({
    queryKey: ['user-progress'],
    queryFn: async () => {
      try {
        const response = await apiService.request<UserProgress[]>('/api/v1/training/progress');
        return response.data || getMockProgress();
      } catch (error) {
        return getMockProgress();
      }
    }
  });

  const { data: learningPaths, isLoading: pathsLoading } = useQuery<LearningPath[]>({
    queryKey: ['learning-paths'],
    queryFn: async () => {
      try {
        const response = await apiService.request<LearningPath[]>('/api/v1/training/paths');
        return response.data || getMockLearningPaths();
      } catch (error) {
        return getMockLearningPaths();
      }
    }
  });

  const createProgramMutation = useMutation({
    mutationFn: async (programData: any) => {
      const response = await apiService.request('/api/v1/training/programs', {
        method: 'POST',
        body: JSON.stringify(programData)
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-programs'] });
      setShowCreateProgram(false);
      setNewProgram({
        title: '',
        description: '',
        category: 'custom',
        difficulty: 'beginner',
        learning_objectives: [''],
        prerequisites: ['']
      });
    }
  });

  const getMockPrograms = (): TrainingProgram[] => [
    {
      id: 'prog-greta-1',
      title: 'GRETA PAI System Mastery',
      description: 'Complete guide to mastering the GRETA Personal AI system, from basic operations to advanced agent orchestration',
      category: 'greta_mastery',
      difficulty: 'intermediate',
      duration: 12,
      modules: [
        {
          id: 'mod-1',
          title: 'GRETA System Overview',
          description: 'Understanding the architecture and capabilities of GRETA',
          type: 'video',
          duration: 45,
          completed: true,
          progress: 100,
          order: 1,
          resources: []
        },
        {
          id: 'mod-2',
          title: 'Master Agent Control',
          description: 'Learning to use the Master Agent for system orchestration',
          type: 'hands_on',
          duration: 90,
          completed: false,
          progress: 30,
          order: 2,
          resources: []
        },
        {
          id: 'mod-3',
          title: 'OSINT Agent Operations',
          description: 'Advanced OSINT techniques for lead generation and intelligence gathering',
          type: 'interactive',
          duration: 120,
          completed: false,
          progress: 0,
          order: 3,
          resources: []
        }
      ],
      prerequisites: ['Basic AI understanding'],
      learning_objectives: [
        'Master GRETA system navigation and core features',
        'Deploy and manage specialized agents effectively',
        'Create custom workflows for business intelligence',
        'Optimize system performance and resource allocation'
      ],
      completion_rate: 78,
      rating: 4.8,
      enrolled_users: 156,
      created_at: new Date(Date.now() - 2592000000).toISOString(),
      updated_at: new Date(Date.now() - 86400000).toISOString(),
      instructor: 'GRETA AI Training Team',
      certification: true
    },
    {
      id: 'prog-osint-1',
      title: 'Advanced OSINT for Business Intelligence',
      description: 'Professional-grade open source intelligence gathering techniques for competitive analysis and lead generation',
      category: 'business_intelligence',
      difficulty: 'advanced',
      duration: 16,
      modules: [
        {
          id: 'mod-osint-1',
          title: 'OSINT Fundamentals',
          description: 'Core principles and methodologies of open source intelligence',
          type: 'video',
          duration: 60,
          completed: true,
          progress: 100,
          order: 1,
          resources: []
        },
        {
          id: 'mod-osint-2',
          title: 'Social Media Intelligence',
          description: 'Advanced techniques for gathering intelligence from social platforms',
          type: 'hands_on',
          duration: 120,
          completed: false,
          progress: 45,
          order: 2,
          resources: []
        }
      ],
      prerequisites: ['Basic research skills', 'Understanding of business processes'],
      learning_objectives: [
        'Master advanced OSINT methodologies',
        'Develop effective lead generation strategies',
        'Create comprehensive intelligence reports',
        'Implement ethical intelligence gathering practices'
      ],
      completion_rate: 65,
      rating: 4.6,
      enrolled_users: 89,
      created_at: new Date(Date.now() - 1728000000).toISOString(),
      updated_at: new Date(Date.now() - 172800000).toISOString(),
      instructor: 'Intelligence Professionals Network',
      certification: true
    },
    {
      id: 'prog-ai-1',
      title: 'AI Fundamentals for Business Professionals',
      description: 'Essential AI concepts and applications for modern business environments',
      category: 'ai_fundamentals',
      difficulty: 'beginner',
      duration: 8,
      modules: [
        {
          id: 'mod-ai-1',
          title: 'Introduction to AI',
          description: 'Basic concepts and terminology in artificial intelligence',
          type: 'video',
          duration: 45,
          completed: false,
          progress: 0,
          order: 1,
          resources: []
        }
      ],
      prerequisites: [],
      learning_objectives: [
        'Understand core AI concepts and terminology',
        'Identify AI applications in business contexts',
        'Evaluate AI tools and solutions',
        'Develop AI implementation strategies'
      ],
      completion_rate: 82,
      rating: 4.4,
      enrolled_users: 234,
      created_at: new Date(Date.now() - 1209600000).toISOString(),
      updated_at: new Date(Date.now() - 259200000).toISOString(),
      instructor: 'AI Education Consortium',
      certification: false
    }
  ];

  const getMockProgress = (): UserProgress[] => [
    {
      program_id: 'prog-greta-1',
      user_id: 'user-1',
      progress: 43,
      completed_modules: ['mod-1'],
      current_module: 'mod-2',
      started_at: new Date(Date.now() - 1209600000).toISOString(),
      last_accessed: new Date(Date.now() - 86400000).toISOString(),
      time_spent: 180,
      quiz_scores: { 'mod-1': 95 },
      certificates_earned: []
    },
    {
      program_id: 'prog-osint-1',
      user_id: 'user-1',
      progress: 72,
      completed_modules: ['mod-osint-1'],
      current_module: 'mod-osint-2',
      started_at: new Date(Date.now() - 864000000).toISOString(),
      last_accessed: new Date(Date.now() - 172800000).toISOString(),
      time_spent: 240,
      quiz_scores: { 'mod-osint-1': 88 },
      certificates_earned: []
    }
  ];

  const getMockLearningPaths = (): LearningPath[] => [
    {
      id: 'path-1',
      title: 'GRETA Power User Certification',
      description: 'Complete learning path to become a certified GRETA power user, covering all system capabilities and advanced techniques',
      programs: ['prog-ai-1', 'prog-greta-1', 'prog-osint-1'],
      estimated_duration: 36,
      difficulty_progression: ['beginner', 'intermediate', 'advanced'],
      target_audience: 'Business professionals, analysts, and AI enthusiasts',
      outcomes: [
        'Certified GRETA Power User credential',
        'Advanced system administration capabilities',
        'Expert-level agent orchestration skills',
        'Professional intelligence gathering certification'
      ]
    },
    {
      id: 'path-2',
      title: 'Business Intelligence Specialist',
      description: 'Specialized track for professionals focused on business intelligence and competitive analysis',
      programs: ['prog-ai-1', 'prog-osint-1'],
      estimated_duration: 24,
      difficulty_progression: ['beginner', 'advanced'],
      target_audience: 'Business analysts, market researchers, and strategic planners',
      outcomes: [
        'Business Intelligence Specialist certification',
        'Advanced OSINT capabilities',
        'Competitive analysis expertise',
        'Strategic intelligence reporting skills'
      ]
    }
  ];

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'greta_mastery': return <Zap className="w-5 h-5" />;
      case 'ai_fundamentals': return <Brain className="w-5 h-5" />;
      case 'business_intelligence': return <BarChart3 className="w-5 h-5" />;
      case 'technical_skills': return <Code className="w-5 h-5" />;
      case 'custom': return <Settings className="w-5 h-5" />;
      default: return <BookOpen className="w-5 h-5" />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'greta_mastery': return 'text-blue-400 bg-blue-400/20';
      case 'ai_fundamentals': return 'text-purple-400 bg-purple-400/20';
      case 'business_intelligence': return 'text-green-400 bg-green-400/20';
      case 'technical_skills': return 'text-orange-400 bg-orange-400/20';
      case 'custom': return 'text-gray-400 bg-gray-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-400 bg-green-400/20';
      case 'intermediate': return 'text-yellow-400 bg-yellow-400/20';
      case 'advanced': return 'text-orange-400 bg-orange-400/20';
      case 'expert': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getModuleTypeIcon = (type: string) => {
    switch (type) {
      case 'video': return <Video className="w-4 h-4" />;
      case 'interactive': return <MessageSquare className="w-4 h-4" />;
      case 'hands_on': return <Target className="w-4 h-4" />;
      case 'assessment': return <CheckCircle className="w-4 h-4" />;
      case 'reading': return <FileText className="w-4 h-4" />;
      default: return <BookOpen className="w-4 h-4" />;
    }
  };

  const filteredPrograms = (programs || []).filter(program => {
    const matchesCategory = selectedCategory === 'all' || program.category === selectedCategory;
    const matchesSearch = program.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         program.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const handleCreateProgram = () => {
    if (newProgram.title && newProgram.description) {
      createProgramMutation.mutate(newProgram);
    }
  };

  if (programsLoading || progressLoading || pathsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <GraduationCap className="w-8 h-8 mr-3 text-blue-400" />
            Training Platform
          </h1>
          <p className="text-gray-400 mt-2">
            Comprehensive learning system for GRETA mastery and professional development
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowCreateProgram(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Create Program</span>
          </button>
        </div>
      </div>

      {/* Learning Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Programs Enrolled</p>
              <p className="text-2xl font-bold text-white">{(userProgress || []).length}</p>
            </div>
            <BookOpen className="w-8 h-8 text-blue-400" />
          </div>
        </div>
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg. Progress</p>
              <p className="text-2xl font-bold text-white">
                {userProgress && userProgress.length > 0
                  ? Math.round(userProgress.reduce((sum, p) => sum + p.progress, 0) / userProgress.length)
                  : 0}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-400" />
          </div>
        </div>
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Hours Studied</p>
              <p className="text-2xl font-bold text-white">
                {userProgress
                  ? Math.round(userProgress.reduce((sum, p) => sum + p.time_spent, 0) / 60)
                  : 0}h
              </p>
            </div>
            <Clock className="w-8 h-8 text-purple-400" />
          </div>
        </div>
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Certificates</p>
              <p className="text-2xl font-bold text-white">
                {userProgress
                  ? userProgress.reduce((sum, p) => sum + p.certificates_earned.length, 0)
                  : 0}
              </p>
            </div>
            <Award className="w-8 h-8 text-yellow-400" />
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-white/5 p-1 rounded-lg">
        {[
          { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
          { id: 'programs', label: 'Programs', icon: BookOpen },
          { id: 'paths', label: 'Learning Paths', icon: Target },
          { id: 'progress', label: 'My Progress', icon: TrendingUp },
          { id: 'create', label: 'Create Content', icon: Plus }
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

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-6">
          {/* Continue Learning */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Play className="w-5 h-5 mr-2 text-green-400" />
              Continue Learning
            </h3>
            <div className="space-y-4">
              {(userProgress || []).filter(p => p.progress < 100).slice(0, 3).map((progress) => {
                const program = (programs || []).find(p => p.id === progress.program_id);
                if (!program) return null;

                return (
                  <div key={progress.program_id} className="bg-white/5 p-4 rounded-lg hover:bg-white/10 transition-all cursor-pointer">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded-lg ${getCategoryColor(program.category)}`}>
                          {getCategoryIcon(program.category)}
                        </div>
                        <div>
                          <h4 className="text-white font-medium">{program.title}</h4>
                          <p className="text-gray-400 text-sm">
                            {progress.current_module ?
                              `Current: ${program.modules.find(m => m.id === progress.current_module)?.title || 'Unknown'}` :
                              'Ready to continue'
                            }
                          </p>
                        </div>
                      </div>
                      <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors">
                        Continue
                      </button>
                    </div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Progress</span>
                      <span className="text-white">{progress.progress}%</span>
                    </div>
                    <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress.progress}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recommended Programs */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Star className="w-5 h-5 mr-2 text-yellow-400" />
              Recommended for You
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {(programs || []).filter(p => !(userProgress || []).some(up => up.program_id === p.id)).slice(0, 3).map((program) => (
                <div key={program.id} className="bg-white/5 p-6 rounded-xl hover:bg-white/10 transition-all">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-2 rounded-lg ${getCategoryColor(program.category)}`}>
                      {getCategoryIcon(program.category)}
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${getDifficultyColor(program.difficulty)}`}>
                      {program.difficulty}
                    </span>
                  </div>

                  <h4 className="text-white font-semibold mb-2">{program.title}</h4>
                  <p className="text-gray-300 text-sm mb-4">{program.description}</p>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Duration</span>
                      <span className="text-white">{program.duration}h</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Rating</span>
                      <div className="flex items-center space-x-1">
                        <Star className="w-3 h-3 text-yellow-400 fill-current" />
                        <span className="text-white">{program.rating}</span>
                      </div>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Enrolled</span>
                      <span className="text-white">{program.enrolled_users}</span>
                    </div>
                  </div>

                  <button className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                    Enroll Now
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Programs Tab */}
      {activeTab === 'programs' && (
        <div className="space-y-6">
          {/* Search and Filter */}
          <div className="glass p-6 rounded-xl">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                    placeholder="Search programs..."
                  />
                </div>
              </div>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Categories</option>
                <option value="greta_mastery">GRETA Mastery</option>
                <option value="ai_fundamentals">AI Fundamentals</option>
                <option value="business_intelligence">Business Intelligence</option>
                <option value="technical_skills">Technical Skills</option>
                <option value="custom">Custom</option>
              </select>
            </div>
          </div>

          {/* Programs Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredPrograms.map((program) => {
              const userProg = (userProgress || []).find(p => p.program_id === program.id);

              return (
                <div key={program.id} className="glass p-6 rounded-xl hover:bg-white/10 transition-all">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-2 rounded-lg ${getCategoryColor(program.category)}`}>
                      {getCategoryIcon(program.category)}
                    </div>
                    <div className="flex flex-col items-end space-y-1">
                      <span className={`px-2 py-1 text-xs rounded-full ${getDifficultyColor(program.difficulty)}`}>
                        {program.difficulty}
                      </span>
                      {program.certification && (
                        <Award className="w-4 h-4 text-yellow-400" />
                      )}
                    </div>
                  </div>

                  <h4 className="text-white font-semibold mb-2">{program.title}</h4>
                  <p className="text-gray-300 text-sm mb-4 line-clamp-3">{program.description}</p>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Duration</span>
                      <span className="text-white">{program.duration}h</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Modules</span>
                      <span className="text-white">{program.modules.length}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Rating</span>
                      <div className="flex items-center space-x-1">
                        <Star className="w-3 h-3 text-yellow-400 fill-current" />
                        <span className="text-white">{program.rating}</span>
                      </div>
                    </div>
                  </div>

                  {userProg ? (
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Your Progress</span>
                        <span className="text-white">{userProg.progress}%</span>
                      </div>
                      <div className="w-full bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${userProg.progress}%` }}
                        ></div>
                      </div>
                      <button className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors">
                        Continue Learning
                      </button>
                    </div>
                  ) : (
                    <button className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                      Enroll Now
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Learning Paths Tab */}
      {activeTab === 'paths' && (
        <div className="space-y-6">
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4">Structured Learning Paths</h3>
            <div className="space-y-6">
              {(learningPaths || []).map((path) => (
                <div key={path.id} className="bg-white/5 p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h4 className="text-white font-semibold text-lg mb-2">{path.title}</h4>
                      <p className="text-gray-300 mb-4">{path.description}</p>
                      <p className="text-gray-400 text-sm mb-2">Target Audience: {path.target_audience}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-white font-medium">{path.estimated_duration}h</p>
                      <p className="text-gray-400 text-sm">Total Duration</p>
                    </div>
                  </div>

                  <div className="mb-4">
                    <h5 className="text-white font-medium mb-2">Learning Outcomes:</h5>
                    <ul className="space-y-1">
                      {path.outcomes.map((outcome, index) => (
                        <li key={index} className="text-gray-300 text-sm flex items-start">
                          <CheckCircle className="w-4 h-4 text-green-400 mr-2 mt-0.5 flex-shrink-0" />
                          {outcome}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="mb-4">
                    <h5 className="text-white font-medium mb-2">Program Sequence:</h5>
                    <div className="flex items-center space-x-2 overflow-x-auto">
                      {path.programs.map((programId, index) => {
                        const program = (programs || []).find(p => p.id === programId);
                        if (!program) return null;

                        return (
                          <React.Fragment key={programId}>
                            <div className="flex-shrink-0 bg-white/10 p-3 rounded-lg min-w-[200px]">
                              <div className="flex items-center space-x-2 mb-2">
                                <div className={`p-1 rounded ${getCategoryColor(program.category)}`}>
                                  {getCategoryIcon(program.category)}
                                </div>
                                <span className={`px-2 py-1 text-xs rounded-full ${getDifficultyColor(program.difficulty)}`}>
                                  {program.difficulty}
                                </span>
                              </div>
                              <h6 className="text-white font-medium text-sm">{program.title}</h6>
                              <p className="text-gray-400 text-xs">{program.duration}h</p>
                            </div>
                            {index < path.programs.length - 1 && (
                              <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                            )}
                          </React.Fragment>
                        );
                      })}
                    </div>
                  </div>

                  <button className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
                    Start Learning Path
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
