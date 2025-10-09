import React, { useState } from 'react';
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
  ArrowRight,
  RefreshCw,
  Copy,
  Share,
  ExternalLink,
  Bookmark,
  Tag,
  Hash,
  Mic,
  Camera,
  Monitor,
  Smartphone,
  Tablet,
  Laptop,
  Workflow,
  Package,
  Box,
  Archive,
  Folder,
  FolderOpen,
  File
} from 'lucide-react';
import { apiService } from '../services/api';

// Training Platform Management Interfaces
interface Course {
  id: string;
  title: string;
  description: string;
  subject: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  duration: number; // in hours
  status: 'draft' | 'published' | 'archived';
  instructor: {
    id: string;
    name: string;
    avatar: string;
    expertise: string[];
  };
  modules: Module[];
  assessments: Assessment[];
  enrollment: {
    total_students: number;
    active_students: number;
    completion_rate: number;
    average_rating: number;
    reviews_count: number;
  };
  content: {
    videos: number;
    documents: number;
    quizzes: number;
    assignments: number;
    interactive_elements: number;
  };
  metadata: {
    created_at: string;
    updated_at: string;
    published_at?: string;
    tags: string[];
    category: string;
    prerequisites: string[];
    learning_objectives: string[];
    skills_gained: string[];
  };
  pricing: {
    type: 'free' | 'paid' | 'subscription';
    price?: number;
    currency?: string;
    discount?: {
      percentage: number;
      valid_until: string;
    };
  };
  analytics: {
    views: number;
    enrollments: number;
    completions: number;
    average_time_spent: number;
    engagement_score: number;
    satisfaction_score: number;
  };
}

interface Module {
  id: string;
  title: string;
  description: string;
  order: number;
  duration: number; // in minutes
  type: 'video' | 'text' | 'interactive' | 'quiz' | 'assignment' | 'project';
  content: {
    video_url?: string;
    text_content?: string;
    interactive_elements?: any[];
    resources?: {
      type: 'pdf' | 'link' | 'download' | 'reference';
      title: string;
      url: string;
      description: string;
    }[];
  };
  completion_criteria: {
    type: 'time_based' | 'interaction_based' | 'assessment_based';
    threshold: number;
  };
  prerequisites: string[];
  learning_objectives: string[];
  status: 'draft' | 'published' | 'archived';
  analytics: {
    completion_rate: number;
    average_time_spent: number;
    engagement_score: number;
    difficulty_rating: number;
  };
}

interface Assessment {
  id: string;
  title: string;
  description: string;
  type: 'quiz' | 'assignment' | 'project' | 'exam' | 'peer_review';
  module_id?: string;
  questions: Question[];
  settings: {
    time_limit?: number; // in minutes
    attempts_allowed: number;
    passing_score: number;
    randomize_questions: boolean;
    show_correct_answers: boolean;
    immediate_feedback: boolean;
  };
  grading: {
    type: 'automatic' | 'manual' | 'peer' | 'ai_assisted';
    rubric?: {
      criteria: string;
      points: number;
      description: string;
    }[];
  };
  analytics: {
    attempts: number;
    average_score: number;
    pass_rate: number;
    average_time_taken: number;
    difficulty_analysis: {
      easy_questions: number;
      medium_questions: number;
      hard_questions: number;
    };
  };
}

interface Question {
  id: string;
  type: 'multiple_choice' | 'true_false' | 'short_answer' | 'essay' | 'code' | 'matching' | 'ordering';
  question: string;
  options?: string[];
  correct_answer: string | string[];
  explanation?: string;
  points: number;
  difficulty: 'easy' | 'medium' | 'hard';
  tags: string[];
  media?: {
    type: 'image' | 'video' | 'audio';
    url: string;
    alt_text?: string;
  };
}

interface StudentProgress {
  student_id: string;
  course_id: string;
  enrollment_date: string;
  progress: {
    modules_completed: string[];
    assessments_completed: string[];
    overall_progress: number;
    time_spent: number;
    last_activity: string;
  };
  performance: {
    average_score: number;
    best_score: number;
    attempts_made: number;
    strengths: string[];
    areas_for_improvement: string[];
  };
  engagement: {
    login_frequency: number;
    session_duration: number;
    interaction_rate: number;
    discussion_participation: number;
  };
  certifications: {
    id: string;
    name: string;
    issued_date: string;
    credential_url: string;
  }[];
}

interface LearningPath {
  id: string;
  title: string;
  description: string;
  courses: string[];
  estimated_duration: number;
  difficulty_progression: string[];
  prerequisites: string[];
  target_audience: string[];
  learning_outcomes: string[];
  industry_relevance: string[];
  career_paths: string[];
  certification: {
    available: boolean;
    name?: string;
    issuer?: string;
    validity_period?: number;
  };
}

// Mock Data
const mockCourses: Course[] = [
  {
    id: 'course-1',
    title: 'Complete Guide to GRETA PAI System',
    description: 'Master the GRETA Personal AI system from basic concepts to advanced agent management and deployment strategies.',
    subject: 'Artificial Intelligence',
    difficulty: 'intermediate',
    duration: 24,
    status: 'published',
    instructor: {
      id: 'instructor-1',
      name: 'Dr. Sarah Chen',
      avatar: '/avatars/sarah-chen.jpg',
      expertise: ['AI Systems', 'Machine Learning', 'PAI Architecture']
    },
    modules: [],
    assessments: [],
    enrollment: {
      total_students: 1247,
      active_students: 892,
      completion_rate: 78.5,
      average_rating: 4.7,
      reviews_count: 234
    },
    content: {
      videos: 45,
      documents: 23,
      quizzes: 12,
      assignments: 8,
      interactive_elements: 15
    },
    metadata: {
      created_at: '2024-01-10T09:00:00Z',
      updated_at: '2024-01-20T15:30:00Z',
      published_at: '2024-01-15T10:00:00Z',
      tags: ['AI', 'PAI', 'GRETA', 'Automation', 'Agents'],
      category: 'Technology',
      prerequisites: ['Basic Computer Skills', 'Understanding of AI Concepts'],
      learning_objectives: [
        'Understand GRETA PAI architecture',
        'Build and deploy AI agents',
        'Manage agent teams effectively',
        'Implement business automation'
      ],
      skills_gained: ['Agent Management', 'PAI Configuration', 'Workflow Automation', 'System Integration']
    },
    pricing: {
      type: 'paid',
      price: 299,
      currency: 'USD',
      discount: {
        percentage: 20,
        valid_until: '2024-02-01T00:00:00Z'
      }
    },
    analytics: {
      views: 5847,
      enrollments: 1247,
      completions: 979,
      average_time_spent: 18.5,
      engagement_score: 87.3,
      satisfaction_score: 94.2
    }
  }
];

const TrainingPlatformManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'courses' | 'builder' | 'students' | 'analytics' | 'certifications' | 'settings'>('courses');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSubject, setFilterSubject] = useState('all');
  const [filterDifficulty, setFilterDifficulty] = useState('all');
  const [selectedCourse, setSelectedCourse] = useState<string | null>(null);

  const queryClient = useQueryClient();

  // Queries
  const { data: courses = mockCourses, isLoading: coursesLoading } = useQuery<Course[]>({
    queryKey: ['training-courses'],
    queryFn: async (): Promise<Course[]> => {
      try {
        const response = await apiService.request('/api/v1/training/courses');
        return response.data as Course[];
      } catch (error) {
        return mockCourses;
      }
    }
  });

  // Mutations
  const createCourseMutation = useMutation({
    mutationFn: async (courseData: Partial<Course>) => {
      const response = await apiService.request('/api/v1/training/courses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(courseData)
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-courses'] });
    }
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-500/20 text-green-400';
      case 'intermediate': return 'bg-blue-500/20 text-blue-400';
      case 'advanced': return 'bg-orange-500/20 text-orange-400';
      case 'expert': return 'bg-red-500/20 text-red-400';
      default: return 'bg-blue-600/20 text-gray-400';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published': return 'bg-green-500/20 text-green-400';
      case 'draft': return 'bg-yellow-500/20 text-yellow-400';
      case 'archived': return 'bg-blue-600/20 text-gray-400';
      default: return 'bg-blue-600/20 text-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-blue-600 rounded-xl">
              <GraduationCap className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Training Platform Management</h1>
              <p className="text-gray-400">Course builder, content management, assessment tools, progress tracking, and certification systems</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
              <Plus className="w-4 h-4" />
              <span>New Course</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors">
              <Brain className="w-4 h-4" />
              <span>AI Assistant</span>
            </button>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-400">Training Platform Active</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-blue-500/10 p-1 rounded-lg">
          {[
            { id: 'courses', label: 'Courses', icon: BookOpen },
            { id: 'builder', label: 'Course Builder', icon: Code },
            { id: 'students', label: 'Students', icon: Users },
            { id: 'analytics', label: 'Analytics', icon: BarChart3 },
            { id: 'certifications', label: 'Certifications', icon: Award },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-blue-500/15'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Courses Tab */}
        {activeTab === 'courses' && (
          <div className="space-y-6">
            {/* Filters and Search */}
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search courses..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <select
                  value={filterSubject}
                  onChange={(e) => setFilterSubject(e.target.value)}
                  className="px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="all">All Subjects</option>
                  <option value="ai">Artificial Intelligence</option>
                  <option value="programming">Programming</option>
                  <option value="business">Business</option>
                  <option value="design">Design</option>
                  <option value="marketing">Marketing</option>
                </select>
                <select
                  value={filterDifficulty}
                  onChange={(e) => setFilterDifficulty(e.target.value)}
                  className="px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="all">All Levels</option>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                  <option value="expert">Expert</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['training-courses'] })}
                  className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Course Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Total Courses</p>
                    <p className="text-2xl font-bold text-white">47</p>
                  </div>
                  <BookOpen className="w-8 h-8 text-blue-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+3 this month</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Active Students</p>
                    <p className="text-2xl font-bold text-white">2,847</p>
                  </div>
                  <Users className="w-8 h-8 text-green-400" />
                </div>
                <p className="text-blue-400 text-sm mt-2">+247 this week</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Completion Rate</p>
                    <p className="text-2xl font-bold text-white">78.5%</p>
                  </div>
                  <Target className="w-8 h-8 text-purple-400" />
                </div>
                <p className="text-green-400 text-sm mt-2">+2.3% improvement</p>
              </div>

              <div className="glass p-6 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Avg Rating</p>
                    <p className="text-2xl font-bold text-white">4.7</p>
                  </div>
                  <Star className="w-8 h-8 text-yellow-400" />
                </div>
                <p className="text-yellow-400 text-sm mt-2">Excellent feedback</p>
              </div>
            </div>

            {/* Courses Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {courses.map((course) => (
                <div key={course.id} className="glass p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
                        <BookOpen className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white line-clamp-1">{course.title}</h3>
                        <p className="text-sm text-gray-400">{course.subject}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded-full ${getDifficultyColor(course.difficulty)}`}>
                        {course.difficulty}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full ${getStatusColor(course.status)}`}>
                        {course.status}
                      </span>
                    </div>
                  </div>

                  <p className="text-gray-300 text-sm mb-4 line-clamp-2">{course.description}</p>

                  {/* Course Stats */}
                  <div className="space-y-3 mb-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Students</p>
                        <p className="text-white font-semibold">{course.enrollment.total_students.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Duration</p>
                        <p className="text-white font-semibold">{course.duration}h</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Completion</p>
                        <div className="flex items-center space-x-2">
                          <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2 flex-1">
                            <div
                              className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                              style={{ width: `${course.enrollment.completion_rate}%` }}
                            ></div>
                          </div>
                          <span className="text-white text-xs">{course.enrollment.completion_rate}%</span>
                        </div>
                      </div>
                      <div>
                        <p className="text-gray-400">Rating</p>
                        <div className="flex items-center space-x-1">
                          <Star className="w-3 h-3 text-yellow-400 fill-current" />
                          <span className="text-white font-semibold">{course.enrollment.average_rating}</span>
                          <span className="text-gray-400 text-xs">({course.enrollment.reviews_count})</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Content Overview */}
                  <div className="mb-4">
                    <p className="text-gray-400 text-sm mb-2">Content</p>
                    <div className="flex flex-wrap gap-2 text-xs">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded">
                        {course.content.videos} videos
                      </span>
                      <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded">
                        {course.content.documents} docs
                      </span>
                      <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded">
                        {course.content.quizzes} quizzes
                      </span>
                      <span className="px-2 py-1 bg-orange-500/20 text-orange-400 rounded">
                        {course.content.assignments} assignments
                      </span>
                    </div>
                  </div>

                  {/* Instructor */}
                  <div className="mb-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                        <User className="w-3 h-3 text-white" />
                      </div>
                      <span className="text-sm text-gray-300">{course.instructor.name}</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                    <div className="flex items-center space-x-2">
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <Eye className="w-3 h-3" />
                        <span>View</span>
                      </button>
                      <button className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-xs transition-colors">
                        <Edit className="w-3 h-3" />
                        <span>Edit</span>
                      </button>
                      <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                        <Copy className="w-3 h-3" />
                        <span>Clone</span>
                      </button>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => setSelectedCourse(course.id)}
                        className="p-1 hover:bg-blue-500/15 rounded"
                      >
                        <BarChart3 className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-blue-500/15 rounded">
                        <Share className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-blue-500/15 rounded">
                        <Trash2 className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Course Builder Tab */}
        {activeTab === 'builder' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Course Builder</h2>
              <div className="flex items-center space-x-2">
                <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
                  <Plus className="w-4 h-4" />
                  <span>New Course</span>
                </button>
                <button className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors">
                  <Brain className="w-4 h-4" />
                  <span>AI Assistant</span>
                </button>
              </div>
            </div>

            {/* Course Builder Interface */}
            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4">Create New Course</h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Course Information */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Course Title</label>
                    <input
                      type="text"
                      placeholder="Enter course title..."
                      className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Description</label>
                    <textarea
                      rows={4}
                      placeholder="Describe your course..."
                      className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Subject</label>
                      <select className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500">
                        <option value="">Select subject...</option>
                        <option value="ai">Artificial Intelligence</option>
                        <option value="programming">Programming</option>
                        <option value="business">Business</option>
                        <option value="design">Design</option>
                        <option value="marketing">Marketing</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Difficulty</label>
                      <select className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500">
                        <option value="">Select difficulty...</option>
                        <option value="beginner">Beginner</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                        <option value="expert">Expert</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Learning Objectives</label>
                    <div className="space-y-2">
                      <input
                        type="text"
                        placeholder="Add learning objective..."
                        className="w-full px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                      />
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <Plus className="w-3 h-3" />
                        <span>Add Objective</span>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Course Structure */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white">Course Structure</h4>

                  <div className="space-y-3">
                    <div className="p-4 bg-blue-500/10 rounded-lg border-l-4 border-blue-500">
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="text-white font-medium">Module 1: Introduction</h5>
                        <div className="flex items-center space-x-2">
                          <button className="p-1 hover:bg-blue-500/15 rounded">
                            <Edit className="w-4 h-4 text-gray-400" />
                          </button>
                          <button className="p-1 hover:bg-blue-500/15 rounded">
                            <Trash2 className="w-4 h-4 text-gray-400" />
                          </button>
                        </div>
                      </div>
                      <p className="text-gray-400 text-sm mb-2">Basic concepts and overview</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-400">
                        <span>3 lessons</span>
                        <span>45 minutes</span>
                        <span>1 quiz</span>
                      </div>
                    </div>

                    <div className="p-4 bg-blue-500/10 rounded-lg border-l-4 border-green-500">
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="text-white font-medium">Module 2: Fundamentals</h5>
                        <div className="flex items-center space-x-2">
                          <button className="p-1 hover:bg-blue-500/15 rounded">
                            <Edit className="w-4 h-4 text-gray-400" />
                          </button>
                          <button className="p-1 hover:bg-blue-500/15 rounded">
                            <Trash2 className="w-4 h-4 text-gray-400" />
                          </button>
                        </div>
                      </div>
                      <p className="text-gray-400 text-sm mb-2">Core principles and techniques</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-400">
                        <span>5 lessons</span>
                        <span>90 minutes</span>
                        <span>2 quizzes</span>
                        <span>1 assignment</span>
                      </div>
                    </div>
                  </div>

                  <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors w-full justify-center">
                    <Plus className="w-4 h-4" />
                    <span>Add Module</span>
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between pt-6 border-t border-gray-600 mt-6">
                <div className="flex items-center space-x-2">
                  <button className="flex items-center space-x-2 px-4 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors">
                    <Save className="w-4 h-4" />
                    <span>Save Draft</span>
                  </button>
                  <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
                    <Eye className="w-4 h-4" />
                    <span>Preview</span>
                  </button>
                </div>
                <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
                  <CheckCircle className="w-4 h-4" />
                  <span>Publish Course</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingPlatformManagement;
