import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  GraduationCap, 
  Play, 
  BookOpen, 
  Trophy, 
  Target,
  Clock,
  CheckCircle,
  Star,
  TrendingUp,
  Users,
  Code,
  Brain,
  Zap,
  Award,
  BarChart3,
  ChevronRight
} from 'lucide-react';
import { apiService } from '../services/api';

interface Lesson {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: number;
  completed: boolean;
  score?: number;
  prerequisites: string[];
  topics: string[];
}

interface TrainingProgress {
  current_level: string;
  completed_lessons: number;
  total_lessons: number;
  learning_streak: number;
  total_score: number;
  achievements: string[];
  recommended_study_time: number;
}

interface Curriculum {
  current_level: string;
  available_lessons: Lesson[];
  priority_lessons: string[];
  completed_lessons: string[];
  learning_streak: number;
  advanced_unlocked: boolean;
  recommended_study_time: number;
}

export const Training: React.FC = () => {
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const [activeTab, setActiveTab] = useState<'curriculum' | 'progress' | 'leaderboard'>('curriculum');
  const [userId] = useState('demo_user'); // In real app, get from auth

  const queryClient = useQueryClient();

  const { data: curriculum, isLoading: curriculumLoading } = useQuery({
    queryKey: ['training-curriculum', userId],
    queryFn: async () => {
      try {
        const response = await apiService.request(`/api/training/curriculum/${userId}`);
        return response.data as Curriculum;
      } catch (error) {
        // Mock data for development
        return {
          current_level: 'intermediate',
          available_lessons: [
            {
              id: 'ai_basics',
              title: 'AI Agent Fundamentals',
              description: 'Learn the core concepts of AI agents and their capabilities',
              difficulty: 'beginner',
              duration: 45,
              completed: true,
              score: 92,
              prerequisites: [],
              topics: ['AI Concepts', 'Agent Architecture', 'Basic Reasoning']
            },
            {
              id: 'memory_systems',
              title: 'Memory and Context Management',
              description: 'Understanding how AI agents store and retrieve information',
              difficulty: 'intermediate',
              duration: 60,
              completed: true,
              score: 87,
              prerequisites: ['ai_basics'],
              topics: ['Memory Systems', 'Context Windows', 'Information Retrieval']
            },
            {
              id: 'advanced_reasoning',
              title: 'Hierarchical Reasoning Models',
              description: 'Deep dive into multi-level reasoning and decision making',
              difficulty: 'advanced',
              duration: 90,
              completed: false,
              prerequisites: ['memory_systems'],
              topics: ['HRM', 'Decision Trees', 'Strategic Planning']
            },
            {
              id: 'agent_coordination',
              title: 'Multi-Agent Coordination',
              description: 'Learn how multiple AI agents work together effectively',
              difficulty: 'advanced',
              duration: 75,
              completed: false,
              prerequisites: ['advanced_reasoning'],
              topics: ['Agent Communication', 'Task Distribution', 'Conflict Resolution']
            }
          ],
          priority_lessons: ['advanced_reasoning'],
          completed_lessons: ['ai_basics', 'memory_systems'],
          learning_streak: 7,
          advanced_unlocked: true,
          recommended_study_time: 60
        } as Curriculum;
      }
    }
  });

  const { data: leaderboard } = useQuery({
    queryKey: ['training-leaderboard'],
    queryFn: async () => {
      try {
        const response = await apiService.request('/api/training/leaderboard');
        return response.data;
      } catch (error) {
        // Mock data
        return [
          { user_id: 'user1', total_score: 2450, lessons_completed: 12, avg_score: 94 },
          { user_id: 'user2', total_score: 2380, lessons_completed: 11, avg_score: 91 },
          { user_id: 'demo_user', total_score: 1790, lessons_completed: 8, avg_score: 89 },
          { user_id: 'user3', total_score: 1650, lessons_completed: 7, avg_score: 87 }
        ];
      }
    }
  });

  const startLessonMutation = useMutation({
    mutationFn: async (lessonId: string) => {
      const response = await apiService.request('/api/training/lesson/start', {
        method: 'POST',
        body: JSON.stringify({ user_id: userId, lesson_id: lessonId })
      });
      return response.data;
    },
    onSuccess: (data: any) => {
      if (data.status === 'lesson_started') {
        // Navigate to lesson interface or show lesson content
        console.log('Lesson started:', data);
      }
    }
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-400 bg-green-400/20';
      case 'intermediate': return 'text-yellow-400 bg-yellow-400/20';
      case 'advanced': return 'text-red-400 bg-red-400/20';
      default: return 'text-blue-400 bg-blue-400/20';
    }
  };

  const getProgressPercentage = () => {
    if (!curriculum) return 0;
    return (curriculum.completed_lessons.length / curriculum.available_lessons.length) * 100;
  };

  if (curriculumLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <GraduationCap className="w-8 h-8 mr-3 text-purple-400" />
            AI Training Platform
          </h1>
          <p className="text-gray-400 mt-2">
            Master AI agent development through interactive lessons and hands-on practice
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="text-sm text-gray-400">Learning Streak</p>
            <p className="text-xl font-bold text-orange-400 flex items-center">
              <Star className="w-5 h-5 mr-1" />
              {curriculum?.learning_streak || 0} days
            </p>
          </div>
        </div>
      </div>

      {/* Progress Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Progress</p>
              <p className="text-2xl font-bold text-white">{Math.round(getProgressPercentage())}%</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-400" />
          </div>
          <div className="mt-4 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${getProgressPercentage()}%` }}
            ></div>
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Completed</p>
              <p className="text-2xl font-bold text-white">{curriculum?.completed_lessons.length || 0}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
          <p className="text-gray-400 text-sm mt-2">
            of {curriculum?.available_lessons.length || 0} lessons
          </p>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Current Level</p>
              <p className="text-2xl font-bold text-white capitalize">{curriculum?.current_level || 'Beginner'}</p>
            </div>
            <Trophy className="w-8 h-8 text-yellow-400" />
          </div>
        </div>

        <div className="glass p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Study Time</p>
              <p className="text-2xl font-bold text-white">{curriculum?.recommended_study_time || 0}m</p>
            </div>
            <Clock className="w-8 h-8 text-blue-400" />
          </div>
          <p className="text-gray-400 text-sm mt-2">recommended daily</p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-blue-500/10 p-1 rounded-lg">
        {[
          { id: 'curriculum', label: 'Curriculum', icon: BookOpen },
          { id: 'progress', label: 'Progress', icon: BarChart3 },
          { id: 'leaderboard', label: 'Leaderboard', icon: Trophy }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 flex items-center justify-center space-x-2 py-2 px-4 rounded-lg transition-all ${
              activeTab === tab.id 
                ? 'bg-purple-500 text-white' 
                : 'text-gray-400 hover:text-white hover:bg-blue-500/10'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'curriculum' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Lesson List */}
          <div className="lg:col-span-2 space-y-4">
            <h3 className="text-xl font-semibold text-white mb-4">Available Lessons</h3>
            {curriculum?.available_lessons.map((lesson) => (
              <div 
                key={lesson.id}
                className={`glass p-6 rounded-xl cursor-pointer transition-all hover:bg-blue-500/15 ${
                  selectedLesson?.id === lesson.id ? 'ring-2 ring-purple-500' : ''
                }`}
                onClick={() => setSelectedLesson(lesson)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h4 className="font-semibold text-white">{lesson.title}</h4>
                      <span className={`px-2 py-1 text-xs rounded-full ${getDifficultyColor(lesson.difficulty)}`}>
                        {lesson.difficulty}
                      </span>
                      {lesson.completed && (
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      )}
                    </div>
                    <p className="text-gray-400 text-sm mb-3">{lesson.description}</p>
                    <div className="flex items-center space-x-4 text-sm text-gray-400">
                      <div className="flex items-center space-x-1">
                        <Clock className="w-4 h-4" />
                        <span>{lesson.duration}m</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Brain className="w-4 h-4" />
                        <span>{lesson.topics.length} topics</span>
                      </div>
                      {lesson.score && (
                        <div className="flex items-center space-x-1">
                          <Star className="w-4 h-4 text-yellow-400" />
                          <span>{lesson.score}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {curriculum.priority_lessons.includes(lesson.id) && (
                      <span className="px-2 py-1 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                        Priority
                      </span>
                    )}
                    <ChevronRight className="w-5 h-5 text-gray-400" />
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {lesson.topics.map((topic, index) => (
                    <span key={index} className="px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded-full">
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Lesson Details */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4">Lesson Details</h3>
            {selectedLesson ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-white">{selectedLesson.title}</h4>
                  <p className="text-gray-400 text-sm mt-1">{selectedLesson.description}</p>
                </div>
                
                <div className="space-y-3">
                  <div>
                    <p className="text-sm text-gray-400">Difficulty</p>
                    <span className={`inline-block px-2 py-1 text-xs rounded-full capitalize ${getDifficultyColor(selectedLesson.difficulty)}`}>
                      {selectedLesson.difficulty}
                    </span>
                  </div>
                  
                  <div>
                    <p className="text-sm text-gray-400">Duration</p>
                    <p className="text-white">{selectedLesson.duration} minutes</p>
                  </div>
                  
                  {selectedLesson.prerequisites.length > 0 && (
                    <div>
                      <p className="text-sm text-gray-400 mb-2">Prerequisites</p>
                      <div className="space-y-1">
                        {selectedLesson.prerequisites.map((prereq, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <CheckCircle className="w-3 h-3 text-green-400" />
                            <span className="text-sm text-gray-300">{prereq}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div>
                    <p className="text-sm text-gray-400 mb-2">Topics Covered</p>
                    <div className="space-y-1">
                      {selectedLesson.topics.map((topic, index) => (
                        <div key={index} className="flex items-center space-x-2">
                          <Target className="w-3 h-3 text-blue-400" />
                          <span className="text-sm text-gray-300">{topic}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={() => startLessonMutation.mutate(selectedLesson.id)}
                  disabled={selectedLesson.completed || startLessonMutation.isPending}
                  className={`w-full py-3 px-4 rounded-lg font-semibold transition-all flex items-center justify-center space-x-2 ${
                    selectedLesson.completed 
                      ? 'bg-green-500/20 text-green-400 cursor-not-allowed'
                      : 'btn-primary hover:scale-105'
                  }`}
                >
                  {selectedLesson.completed ? (
                    <>
                      <CheckCircle className="w-4 h-4" />
                      <span>Completed</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>{startLessonMutation.isPending ? 'Starting...' : 'Start Lesson'}</span>
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="text-center text-gray-400 py-8">
                <BookOpen className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Select a lesson to view details</p>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'leaderboard' && (
        <div className="glass p-6 rounded-xl">
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <Trophy className="w-5 h-5 mr-2 text-yellow-400" />
            Training Leaderboard
          </h3>
          <div className="space-y-4">
            {(leaderboard as any[])?.map((entry: any, index: number) => (
              <div 
                key={entry.user_id}
                className={`flex items-center justify-between p-4 rounded-lg ${
                  entry.user_id === userId ? 'bg-purple-500/20 border border-purple-500' : 'bg-blue-500/10'
                }`}
              >
                <div className="flex items-center space-x-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                    index === 0 ? 'bg-yellow-500 text-black' :
                    index === 1 ? 'bg-blue-400 text-black' :
                    index === 2 ? 'bg-orange-600 text-white' :
                    'bg-blue-700 hover:bg-blue-800 text-white'
                  }`}>
                    {index + 1}
                  </div>
                  <div>
                    <p className="font-semibold text-white">
                      {entry.user_id === userId ? 'You' : `User ${entry.user_id.slice(-3)}`}
                    </p>
                    <p className="text-sm text-gray-400">
                      {entry.lessons_completed} lessons completed
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-bold text-white">{entry.total_score}</p>
                  <p className="text-sm text-gray-400">{entry.avg_score}% avg</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Training;
