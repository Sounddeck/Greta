import React, { useState, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Mic, 
  MicOff, 
  Camera, 
  CameraOff, 
  Video, 
  VideoOff, 
  Image, 
  FileText, 
  Upload, 
  Download, 
  Play, 
  Pause, 
  Square, 
  Volume2, 
  VolumeX, 
  Eye, 
  EyeOff, 
  Settings, 
  RefreshCw, 
  Zap, 
  Brain, 
  Layers, 
  Workflow, 
  BarChart3, 
  Clock, 
  CheckCircle, 
  AlertTriangle, 
  X, 
  Plus, 
  Edit, 
  Trash2, 
  Save,
  Search,
  Filter,
  Copy,
  Monitor, 
  Headphones, 
  Speaker, 
  Smartphone, 
  Tablet, 
  Laptop, 
  Tv, 
  Radio, 
  Film, 
  Music, 
  Disc, 
  Archive, 
  Folder, 
  FolderOpen, 
  Star, 
  Award, 
  Target, 
  TrendingUp, 
  Activity, 
  Database, 
  Server, 
  Cpu, 
  HardDrive, 
  Wifi, 
  Signal, 
  Bluetooth, 
  Usb, 
  Cable, 
  Router, 
  Network, 
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
  Bike
} from 'lucide-react';
import { apiService } from '../services/api';

// Multi-Modal Processing Interfaces
interface MediaFile {
  id: string;
  name: string;
  type: 'audio' | 'video' | 'image' | 'document' | 'text';
  format: string;
  size: number;
  duration?: number;
  dimensions?: { width: number; height: number };
  url: string;
  thumbnail?: string;
  metadata: {
    created_at: string;
    uploaded_at: string;
    processed_at?: string;
    source: 'upload' | 'recording' | 'generation' | 'capture';
    quality: 'low' | 'medium' | 'high' | 'ultra';
    encoding?: string;
    bitrate?: number;
    sample_rate?: number;
    channels?: number;
  };
  processing: {
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
    stage: string;
    estimated_time?: number;
    error_message?: string;
  };
  analysis: {
    transcription?: string;
    translation?: string;
    sentiment?: 'positive' | 'negative' | 'neutral';
    entities?: string[];
    keywords?: string[];
    summary?: string;
    confidence_score?: number;
    language?: string;
    topics?: string[];
    objects?: string[];
    faces?: number;
    text_content?: string;
    ocr_confidence?: number;
  };
  tags: string[];
  created_by: string;
}

interface ProcessingPipeline {
  id: string;
  name: string;
  description: string;
  input_types: string[];
  output_types: string[];
  stages: {
    id: string;
    name: string;
    type: 'analysis' | 'transformation' | 'generation' | 'enhancement';
    enabled: boolean;
    configuration: any;
    estimated_time: number;
  }[];
  status: 'active' | 'inactive' | 'error';
  usage_count: number;
  success_rate: number;
  avg_processing_time: number;
  created_at: string;
  updated_at: string;
}

interface GenerationRequest {
  id: string;
  type: 'text_to_speech' | 'text_to_image' | 'text_to_video' | 'image_to_text' | 'audio_to_text' | 'video_to_text';
  prompt: string;
  parameters: {
    style?: string;
    quality?: 'low' | 'medium' | 'high' | 'ultra';
    duration?: number;
    voice?: string;
    language?: string;
    format?: string;
    resolution?: string;
    fps?: number;
  };
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  result_url?: string;
  error_message?: string;
  created_at: string;
  estimated_completion: string;
}

interface RealTimeSession {
  id: string;
  type: 'voice_chat' | 'video_call' | 'screen_share' | 'live_transcription' | 'real_time_translation';
  status: 'active' | 'paused' | 'ended';
  participants: string[];
  duration: number;
  quality: 'low' | 'medium' | 'high';
  features: {
    recording: boolean;
    transcription: boolean;
    translation: boolean;
    noise_reduction: boolean;
    echo_cancellation: boolean;
    background_blur: boolean;
  };
  metrics: {
    latency: number;
    packet_loss: number;
    bandwidth_usage: number;
    cpu_usage: number;
    memory_usage: number;
  };
  created_at: string;
}

// Mock Data
const mockMediaFiles: MediaFile[] = [
  {
    id: 'media-1',
    name: 'Executive Meeting Recording.mp4',
    type: 'video',
    format: 'mp4',
    size: 524288000,
    duration: 3600,
    dimensions: { width: 1920, height: 1080 },
    url: '/media/executive-meeting.mp4',
    thumbnail: '/thumbnails/executive-meeting.jpg',
    metadata: {
      created_at: '2024-01-20T09:00:00Z',
      uploaded_at: '2024-01-20T09:05:00Z',
      processed_at: '2024-01-20T09:15:00Z',
      source: 'recording',
      quality: 'high',
      encoding: 'H.264',
      bitrate: 5000000,
      sample_rate: 48000,
      channels: 2
    },
    processing: {
      status: 'completed',
      progress: 100,
      stage: 'Analysis Complete'
    },
    analysis: {
      transcription: 'Welcome everyone to our quarterly executive meeting...',
      sentiment: 'positive',
      entities: ['Q4 Results', 'Revenue Growth', 'Market Expansion'],
      keywords: ['strategy', 'growth', 'innovation', 'market'],
      summary: 'Quarterly executive meeting discussing Q4 results and future strategy',
      confidence_score: 0.94,
      language: 'en',
      topics: ['Business Strategy', 'Financial Results', 'Market Analysis'],
      faces: 5
    },
    tags: ['meeting', 'executive', 'quarterly', 'strategy'],
    created_by: 'john.doe'
  }
];

const mockPipelines: ProcessingPipeline[] = [
  {
    id: 'pipeline-1',
    name: 'Video Analysis & Transcription',
    description: 'Complete video processing with transcription, sentiment analysis, and key moment detection',
    input_types: ['video/mp4', 'video/avi', 'video/mov'],
    output_types: ['text/plain', 'application/json', 'video/mp4'],
    stages: [
      {
        id: 'stage-1',
        name: 'Audio Extraction',
        type: 'transformation',
        enabled: true,
        configuration: { format: 'wav', quality: 'high' },
        estimated_time: 30
      },
      {
        id: 'stage-2',
        name: 'Speech Recognition',
        type: 'analysis',
        enabled: true,
        configuration: { language: 'auto', model: 'whisper-large' },
        estimated_time: 120
      },
      {
        id: 'stage-3',
        name: 'Sentiment Analysis',
        type: 'analysis',
        enabled: true,
        configuration: { model: 'bert-sentiment' },
        estimated_time: 60
      }
    ],
    status: 'active',
    usage_count: 247,
    success_rate: 96.8,
    avg_processing_time: 210,
    created_at: '2024-01-10T09:00:00Z',
    updated_at: '2024-01-20T14:00:00Z'
  }
];

const MultiModalProcessing: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'media' | 'processing' | 'generation' | 'realtime' | 'analytics' | 'settings'>('media');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [recordingType, setRecordingType] = useState<'audio' | 'video' | 'screen'>('audio');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  // Queries
  const { data: mediaFiles = mockMediaFiles, isLoading: mediaLoading } = useQuery<MediaFile[]>({
    queryKey: ['media-files'],
    queryFn: async (): Promise<MediaFile[]> => {
      try {
        const response = await apiService.request('/api/v1/multimodal/media');
        return response.data as MediaFile[];
      } catch (error) {
        return mockMediaFiles;
      }
    }
  });

  const { data: pipelines = mockPipelines, isLoading: pipelinesLoading } = useQuery<ProcessingPipeline[]>({
    queryKey: ['processing-pipelines'],
    queryFn: async (): Promise<ProcessingPipeline[]> => {
      try {
        const response = await apiService.request('/api/v1/multimodal/pipelines');
        return response.data as ProcessingPipeline[];
      } catch (error) {
        return mockPipelines;
      }
    }
  });

  // Mutations
  const uploadFileMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      const response = await apiService.request('/api/v1/multimodal/upload', {
        method: 'POST',
        body: formData
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['media-files'] });
    }
  });

  const processFileMutation = useMutation({
    mutationFn: async ({ fileId, pipelineId }: { fileId: string; pipelineId: string }) => {
      const response = await apiService.request(`/api/v1/multimodal/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: fileId, pipeline_id: pipelineId })
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['media-files'] });
    }
  });

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      Array.from(files).forEach(file => {
        uploadFileMutation.mutate(file);
      });
    }
  };

  const handleStartRecording = () => {
    setIsRecording(true);
    // Implement recording logic
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    // Implement stop recording logic
  };

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'audio': return <Music className="w-4 h-4" />;
      case 'video': return <Video className="w-4 h-4" />;
      case 'image': return <Image className="w-4 h-4" />;
      case 'document': return <FileText className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-purple-600 rounded-xl">
              <Layers className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Multi-Modal Processing</h1>
              <p className="text-gray-400">Voice, image, video, document processing with real-time analysis and generation</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="audio/*,video/*,image/*,.pdf,.doc,.docx,.txt"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>Upload</span>
            </button>
            <button
              onClick={isRecording ? handleStopRecording : handleStartRecording}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                isRecording 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isRecording ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              <span>{isRecording ? 'Stop' : 'Record'}</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors">
              <Brain className="w-4 h-4" />
              <span>Generate</span>
            </button>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-purple-400">Multi-Modal Active</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-white/5 p-1 rounded-lg">
          {[
            { id: 'media', label: 'Media Library', icon: Database },
            { id: 'processing', label: 'Processing', icon: Workflow },
            { id: 'generation', label: 'Generation', icon: Brain },
            { id: 'realtime', label: 'Real-time', icon: Activity },
            { id: 'analytics', label: 'Analytics', icon: BarChart3 },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/10'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Media Library Tab */}
        {activeTab === 'media' && (
          <div className="space-y-6">
            {/* Filters and Search */}
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search media files..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-white/10 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
                  />
                </div>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Types</option>
                  <option value="audio">Audio</option>
                  <option value="video">Video</option>
                  <option value="image">Image</option>
                  <option value="document">Document</option>
                </select>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Status</option>
                  <option value="pending">Pending</option>
                  <option value="processing">Processing</option>
                  <option value="completed">Completed</option>
                  <option value="failed">Failed</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['media-files'] })}
                  className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Media Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {mediaFiles.map((file) => (
                <div key={file.id} className="glass p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        file.type === 'audio' ? 'bg-green-500/20 text-green-400' :
                        file.type === 'video' ? 'bg-blue-500/20 text-blue-400' :
                        file.type === 'image' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {getFileIcon(file.type)}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white line-clamp-1">{file.name}</h3>
                        <p className="text-sm text-gray-400">{file.format.toUpperCase()} • {formatFileSize(file.size)}</p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      file.processing.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                      file.processing.status === 'processing' ? 'bg-blue-500/20 text-blue-400' :
                      file.processing.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {file.processing.status}
                    </span>
                  </div>

                  {/* File Preview */}
                  {file.thumbnail && (
                    <div className="mb-4">
                      <img
                        src={file.thumbnail}
                        alt={file.name}
                        className="w-full h-32 object-cover rounded-lg"
                      />
                    </div>
                  )}

                  {/* File Details */}
                  <div className="space-y-3 mb-4">
                    {file.duration && (
                      <div className="text-sm">
                        <p className="text-gray-400">Duration</p>
                        <p className="text-white">{formatDuration(file.duration)}</p>
                      </div>
                    )}
                    {file.dimensions && (
                      <div className="text-sm">
                        <p className="text-gray-400">Resolution</p>
                        <p className="text-white">{file.dimensions.width}x{file.dimensions.height}</p>
                      </div>
                    )}
                    {file.analysis.transcription && (
                      <div className="text-sm">
                        <p className="text-gray-400">Transcription Preview</p>
                        <p className="text-white line-clamp-2">{file.analysis.transcription}</p>
                      </div>
                    )}
                    {file.analysis.confidence_score && (
                      <div className="text-sm">
                        <p className="text-gray-400">Analysis Confidence</p>
                        <div className="flex items-center space-x-2">
                          <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2 flex-1">
                            <div
                              className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                              style={{ width: `${file.analysis.confidence_score * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-white text-xs">{Math.round(file.analysis.confidence_score * 100)}%</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Processing Progress */}
                  {file.processing.status === 'processing' && (
                    <div className="mb-4">
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-gray-400">{file.processing.stage}</span>
                        <span className="text-white">{file.processing.progress}%</span>
                      </div>
                      <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${file.processing.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {/* Tags */}
                  {file.tags.length > 0 && (
                    <div className="mb-4">
                      <div className="flex flex-wrap gap-1">
                        {file.tags.slice(0, 3).map((tag) => (
                          <span key={tag} className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded">
                            {tag}
                          </span>
                        ))}
                        {file.tags.length > 3 && (
                          <span className="text-xs px-2 py-1 bg-gray-500/20 text-gray-400 rounded">
                            +{file.tags.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                    <div className="flex items-center space-x-2">
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <Play className="w-3 h-3" />
                        <span>Play</span>
                      </button>
                      <button
                        onClick={() => processFileMutation.mutate({ fileId: file.id, pipelineId: 'pipeline-1' })}
                        className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-xs transition-colors"
                      >
                        <Workflow className="w-3 h-3" />
                        <span>Process</span>
                      </button>
                      <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                        <Download className="w-3 h-3" />
                        <span>Export</span>
                      </button>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => setSelectedFile(file.id)}
                        className="p-1 hover:bg-white/10 rounded"
                      >
                        <Eye className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded">
                        <Edit className="w-4 h-4 text-gray-400" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded">
                        <Trash2 className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Processing Tab */}
        {activeTab === 'processing' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Processing Pipelines</h2>
              <div className="flex items-center space-x-2">
                <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
                  <Plus className="w-4 h-4" />
                  <span>New Pipeline</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {pipelines.map((pipeline) => (
                <div key={pipeline.id} className="glass p-6 rounded-xl">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{pipeline.name}</h3>
                      <p className="text-sm text-gray-400">{pipeline.description}</p>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      pipeline.status === 'active' ? 'bg-green-500/20 text-green-400' :
                      pipeline.status === 'inactive' ? 'bg-gray-500/20 text-gray-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {pipeline.status}
                    </span>
                  </div>

                  <div className="space-y-3 mb-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Usage Count</p>
                        <p className="text-white font-semibold">{pipeline.usage_count}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Success Rate</p>
                        <p className="text-white font-semibold">{pipeline.success_rate}%</p>
                      </div>
                    </div>
                    <div className="text-sm">
                      <p className="text-gray-400">Avg Processing Time</p>
                      <p className="text-white font-semibold">{Math.round(pipeline.avg_processing_time / 60)} minutes</p>
                    </div>
                  </div>

                  <div className="mb-4">
                    <p className="text-gray-400 text-sm mb-2">Pipeline Stages</p>
                    <div className="space-y-2">
                      {pipeline.stages.map((stage, index) => (
                        <div key={stage.id} className="flex items-center justify-between p-2 bg-white/5 rounded">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-1 rounded">
                              {index + 1}
                            </span>
                            <span className="text-sm text-white">{stage.name}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-gray-400">{stage.estimated_time}s</span>
                            <div className={`w-2 h-2 rounded-full ${
                              stage.enabled ? 'bg-green-400' : 'bg-gray-400'
                            }`}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                    <div className="flex items-center space-x-2">
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                        <Play className="w-3 h-3" />
                        <span>Run</span>
                      </button>
                      <button className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-xs transition-colors">
                        <Copy className="w-3 h-3" />
                        <span>Clone</span>
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
              ))}
            </div>
          </div>
        )}

        {/* Real-time Tab */}
        {activeTab === 'realtime' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Real-time Processing</h2>
              <div className="flex items-center space-x-2">
                <select
                  value={recordingType}
                  onChange={(e) => setRecordingType(e.target.value as any)}
                  className="px-3 py-2 bg-white/10 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="audio">Audio Only</option>
                  <option value="video">Video Call</option>
                  <option value="screen">Screen Share</option>
                </select>
                <button
                  onClick={isStreaming ? () => setIsStreaming(false) : () => setIsStreaming(true)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                    isStreaming
                      ? 'bg-red-600 hover:bg-red-700'
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  {isStreaming ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  <span>{isStreaming ? 'Stop Stream' : 'Start Stream'}</span>
                </button>
              </div>
            </div>

            {/* Real-time Controls */}
            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4">Live Session Controls</h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Audio Controls */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white">Audio</h4>
                  <div className="space-y-3">
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Mic className="w-5 h-5 text-green-400" />
                        <span className="text-white">Microphone</span>
                      </div>
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    </button>
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Volume2 className="w-5 h-5 text-blue-400" />
                        <span className="text-white">Speaker</span>
                      </div>
                      <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    </button>
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Brain className="w-5 h-5 text-purple-400" />
                        <span className="text-white">Live Transcription</span>
                      </div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                    </button>
                  </div>
                </div>

                {/* Video Controls */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white">Video</h4>
                  <div className="space-y-3">
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Camera className="w-5 h-5 text-green-400" />
                        <span className="text-white">Camera</span>
                      </div>
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    </button>
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Monitor className="w-5 h-5 text-blue-400" />
                        <span className="text-white">Screen Share</span>
                      </div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                    </button>
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Eye className="w-5 h-5 text-purple-400" />
                        <span className="text-white">Object Detection</span>
                      </div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                    </button>
                  </div>
                </div>

                {/* Processing Controls */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white">Processing</h4>
                  <div className="space-y-3">
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Zap className="w-5 h-5 text-yellow-400" />
                        <span className="text-white">Real-time Analysis</span>
                      </div>
                      <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                    </button>
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Globe className="w-5 h-5 text-green-400" />
                        <span className="text-white">Translation</span>
                      </div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                    </button>
                    <button className="flex items-center justify-between w-full p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                      <div className="flex items-center space-x-3">
                        <Archive className="w-5 h-5 text-blue-400" />
                        <span className="text-white">Recording</span>
                      </div>
                      <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Live Transcription Display */}
            {isStreaming && (
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-white mb-4">Live Transcription</h3>
                <div className="bg-black/20 rounded-lg p-4 min-h-32">
                  <p className="text-gray-300 text-sm">
                    <span className="text-purple-400">[00:00:15]</span> Welcome to the live transcription demo. This text will update in real-time as speech is detected...
                  </p>
                  <div className="flex items-center space-x-2 mt-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-green-400 text-xs">Listening...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MultiModalProcessing;
