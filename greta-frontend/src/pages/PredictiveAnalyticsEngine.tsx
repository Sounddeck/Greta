import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  TrendingUp, 
  BarChart3, 
  LineChart, 
  PieChart, 
  Activity, 
  Database, 
  Brain, 
  Target, 
  Zap, 
  Settings, 
  Upload, 
  Download, 
  RefreshCw, 
  Play, 
  Pause, 
  Square, 
  Plus, 
  Edit, 
  Trash2, 
  Eye, 
  Search, 
  Filter, 
  Calendar, 
  Clock, 
  CheckCircle, 
  AlertTriangle, 
  ArrowUp, 
  ArrowDown, 
  TrendingDown,
  DollarSign,
  Users,
  ShoppingCart,
  Globe,
  Cpu,
  HardDrive,
  Network,
  Monitor,
  FileText,
  Save,
  X
} from 'lucide-react';
import { apiService } from '../services/api';

interface PredictionModel {
  id: string;
  name: string;
  type: 'regression' | 'classification' | 'time_series' | 'clustering' | 'anomaly_detection';
  status: 'training' | 'ready' | 'deployed' | 'error' | 'idle';
  accuracy: number;
  last_trained: string;
  data_sources: string[];
  predictions_made: number;
  confidence_score: number;
  target_variable: string;
  features: string[];
  training_data_size: number;
  model_version: string;
}

interface Prediction {
  id: string;
  model_id: string;
  model_name: string;
  prediction_type: string;
  input_data: any;
  predicted_value: any;
  confidence: number;
  created_at: string;
  actual_value?: any;
  accuracy?: number;
  business_impact: {
    category: 'revenue' | 'cost' | 'risk' | 'opportunity';
    estimated_value: number;
    currency: string;
  };
}

interface DataSource {
  id: string;
  name: string;
  type: 'database' | 'api' | 'file' | 'stream';
  status: 'connected' | 'disconnected' | 'error' | 'syncing';
  last_sync: string;
  records_count: number;
  schema: {
    [key: string]: 'string' | 'number' | 'boolean' | 'date';
  };
  connection_config: any;
}

interface AnalyticsDashboard {
  total_models: number;
  active_predictions: number;
  accuracy_avg: number;
  data_processed_gb: number;
  business_value_generated: number;
  predictions_today: number;
  model_performance: {
    model_name: string;
    accuracy: number;
    predictions: number;
    trend: 'up' | 'down' | 'stable';
  }[];
  revenue_forecast: {
    period: string;
    predicted: number;
    actual?: number;
    confidence: number;
  }[];
  risk_alerts: {
    id: string;
    type: 'high_risk' | 'anomaly' | 'data_quality' | 'model_drift';
    message: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    created_at: string;
  }[];
}

export const PredictiveAnalyticsEngine: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'models' | 'predictions' | 'data_sources' | 'training' | 'settings'>('dashboard');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [showCreateModel, setShowCreateModel] = useState(false);
  const [showDataIngestion, setShowDataIngestion] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [dateRange, setDateRange] = useState<'1d' | '7d' | '30d' | '90d'>('30d');

  const queryClient = useQueryClient();

  // Mock data for development
  const mockModels: PredictionModel[] = [
    {
      id: 'model-1',
      name: 'Sales Revenue Forecasting',
      type: 'time_series',
      status: 'deployed',
      accuracy: 94.2,
      last_trained: '2024-01-20T10:30:00Z',
      data_sources: ['crm_data', 'marketing_campaigns', 'economic_indicators'],
      predictions_made: 1247,
      confidence_score: 92.8,
      target_variable: 'monthly_revenue',
      features: ['lead_count', 'marketing_spend', 'seasonality', 'economic_index'],
      training_data_size: 50000,
      model_version: 'v2.1.3'
    },
    {
      id: 'model-2',
      name: 'Customer Churn Prediction',
      type: 'classification',
      status: 'ready',
      accuracy: 87.5,
      last_trained: '2024-01-19T14:22:00Z',
      data_sources: ['user_behavior', 'support_tickets', 'billing_data'],
      predictions_made: 892,
      confidence_score: 89.1,
      target_variable: 'will_churn',
      features: ['usage_frequency', 'support_tickets', 'payment_delays', 'feature_adoption'],
      training_data_size: 75000,
      model_version: 'v1.8.2'
    },
    {
      id: 'model-3',
      name: 'Market Demand Forecasting',
      type: 'regression',
      status: 'training',
      accuracy: 0,
      last_trained: '2024-01-20T15:45:00Z',
      data_sources: ['market_data', 'competitor_analysis', 'social_sentiment'],
      predictions_made: 0,
      confidence_score: 0,
      target_variable: 'demand_volume',
      features: ['market_trends', 'competitor_pricing', 'sentiment_score', 'seasonal_factors'],
      training_data_size: 30000,
      model_version: 'v1.0.0'
    },
    {
      id: 'model-4',
      name: 'Risk Assessment Engine',
      type: 'anomaly_detection',
      status: 'deployed',
      accuracy: 91.7,
      last_trained: '2024-01-18T09:15:00Z',
      data_sources: ['transaction_data', 'user_profiles', 'external_risk_feeds'],
      predictions_made: 2156,
      confidence_score: 94.3,
      target_variable: 'risk_score',
      features: ['transaction_patterns', 'user_behavior', 'external_risk_factors'],
      training_data_size: 100000,
      model_version: 'v3.2.1'
    }
  ];

  const mockPredictions: Prediction[] = [
    {
      id: 'pred-1',
      model_id: 'model-1',
      model_name: 'Sales Revenue Forecasting',
      prediction_type: 'Monthly Revenue',
      input_data: { month: '2024-02', lead_count: 450, marketing_spend: 25000 },
      predicted_value: 185000,
      confidence: 94.2,
      created_at: '2024-01-20T16:30:00Z',
      actual_value: 182000,
      accuracy: 98.4,
      business_impact: {
        category: 'revenue',
        estimated_value: 185000,
        currency: 'USD'
      }
    },
    {
      id: 'pred-2',
      model_id: 'model-2',
      model_name: 'Customer Churn Prediction',
      prediction_type: 'Churn Risk',
      input_data: { customer_id: 'cust_12345', usage_frequency: 0.3, support_tickets: 3 },
      predicted_value: 'high_risk',
      confidence: 87.5,
      created_at: '2024-01-20T15:45:00Z',
      business_impact: {
        category: 'risk',
        estimated_value: -15000,
        currency: 'USD'
      }
    },
    {
      id: 'pred-3',
      model_id: 'model-4',
      model_name: 'Risk Assessment Engine',
      prediction_type: 'Transaction Risk',
      input_data: { transaction_amount: 50000, user_id: 'user_67890' },
      predicted_value: 0.85,
      confidence: 91.7,
      created_at: '2024-01-20T14:20:00Z',
      business_impact: {
        category: 'risk',
        estimated_value: -50000,
        currency: 'USD'
      }
    }
  ];

  const mockDataSources: DataSource[] = [
    {
      id: 'ds-1',
      name: 'CRM Database',
      type: 'database',
      status: 'connected',
      last_sync: '2024-01-20T16:00:00Z',
      records_count: 125000,
      schema: {
        customer_id: 'string',
        revenue: 'number',
        signup_date: 'date',
        is_active: 'boolean'
      },
      connection_config: { host: 'crm.company.com', database: 'sales' }
    },
    {
      id: 'ds-2',
      name: 'Marketing Analytics API',
      type: 'api',
      status: 'connected',
      last_sync: '2024-01-20T15:30:00Z',
      records_count: 50000,
      schema: {
        campaign_id: 'string',
        spend: 'number',
        impressions: 'number',
        clicks: 'number'
      },
      connection_config: { endpoint: 'https://api.marketing.com/v1', auth_type: 'oauth2' }
    },
    {
      id: 'ds-3',
      name: 'Transaction Stream',
      type: 'stream',
      status: 'syncing',
      last_sync: '2024-01-20T16:35:00Z',
      records_count: 1000000,
      schema: {
        transaction_id: 'string',
        amount: 'number',
        timestamp: 'date',
        user_id: 'string'
      },
      connection_config: { stream_url: 'kafka://transactions.company.com:9092' }
    }
  ];

  const mockDashboard: AnalyticsDashboard = {
    total_models: 4,
    active_predictions: 156,
    accuracy_avg: 91.1,
    data_processed_gb: 2.4,
    business_value_generated: 450000,
    predictions_today: 89,
    model_performance: [
      { model_name: 'Sales Revenue Forecasting', accuracy: 94.2, predictions: 1247, trend: 'up' },
      { model_name: 'Customer Churn Prediction', accuracy: 87.5, predictions: 892, trend: 'stable' },
      { model_name: 'Risk Assessment Engine', accuracy: 91.7, predictions: 2156, trend: 'up' },
      { model_name: 'Market Demand Forecasting', accuracy: 0, predictions: 0, trend: 'stable' }
    ],
    revenue_forecast: [
      { period: '2024-01', predicted: 180000, actual: 182000, confidence: 94.2 },
      { period: '2024-02', predicted: 185000, confidence: 92.8 },
      { period: '2024-03', predicted: 192000, confidence: 89.5 },
      { period: '2024-04', predicted: 198000, confidence: 87.2 }
    ],
    risk_alerts: [
      {
        id: 'alert-1',
        type: 'model_drift',
        message: 'Customer Churn model accuracy decreased by 3.2%',
        severity: 'medium',
        created_at: '2024-01-20T14:30:00Z'
      },
      {
        id: 'alert-2',
        type: 'anomaly',
        message: 'Unusual transaction pattern detected in Risk Assessment',
        severity: 'high',
        created_at: '2024-01-20T13:45:00Z'
      }
    ]
  };

  const { data: models = mockModels, isLoading: modelsLoading } = useQuery<PredictionModel[]>({
    queryKey: ['prediction-models'],
    queryFn: async (): Promise<PredictionModel[]> => {
      try {
        const response = await apiService.request<PredictionModel[]>('/api/v1/analytics/models', { method: 'GET' });
        return response.data || mockModels;
      } catch (error) {
        return mockModels;
      }
    },
    refetchInterval: 10000
  });

  const { data: predictions = mockPredictions, isLoading: predictionsLoading } = useQuery<Prediction[]>({
    queryKey: ['predictions', dateRange],
    queryFn: async (): Promise<Prediction[]> => {
      try {
        const response = await apiService.request<Prediction[]>(`/api/v1/analytics/predictions?range=${dateRange}`, { method: 'GET' });
        return response.data || mockPredictions;
      } catch (error) {
        return mockPredictions;
      }
    },
    refetchInterval: 30000
  });

  const { data: dataSources = mockDataSources, isLoading: dataSourcesLoading } = useQuery<DataSource[]>({
    queryKey: ['data-sources'],
    queryFn: async (): Promise<DataSource[]> => {
      try {
        const response = await apiService.request<DataSource[]>('/api/v1/analytics/data-sources', { method: 'GET' });
        return response.data || mockDataSources;
      } catch (error) {
        return mockDataSources;
      }
    },
    refetchInterval: 15000
  });

  const { data: dashboard = mockDashboard, isLoading: dashboardLoading } = useQuery<AnalyticsDashboard>({
    queryKey: ['analytics-dashboard', dateRange],
    queryFn: async (): Promise<AnalyticsDashboard> => {
      try {
        const response = await apiService.request<AnalyticsDashboard>(`/api/v1/analytics/dashboard?range=${dateRange}`, { method: 'GET' });
        return response.data || mockDashboard;
      } catch (error) {
        return mockDashboard;
      }
    },
    refetchInterval: 30000
  });

  const trainModelMutation = useMutation({
    mutationFn: async (modelId: string) => {
      const response = await apiService.request(`/api/v1/analytics/models/${modelId}/train`, { method: 'POST' });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prediction-models'] });
    }
  });

  const deployModelMutation = useMutation({
    mutationFn: async (modelId: string) => {
      const response = await apiService.request(`/api/v1/analytics/models/${modelId}/deploy`, { method: 'POST' });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prediction-models'] });
    }
  });

  const createPredictionMutation = useMutation({
    mutationFn: async ({ modelId, inputData }: { modelId: string; inputData: any }) => {
      const response = await apiService.request(`/api/v1/analytics/models/${modelId}/predict`, { method: 'POST', body: JSON.stringify(inputData), headers: { 'Content-Type': 'application/json' } });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      queryClient.invalidateQueries({ queryKey: ['analytics-dashboard'] });
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'text-green-400';
      case 'ready': return 'text-blue-400';
      case 'training': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      case 'idle': return 'text-gray-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'deployed': return 'bg-green-500/20';
      case 'ready': return 'bg-blue-500/20';
      case 'training': return 'bg-yellow-500/20';
      case 'error': return 'bg-red-500/20';
      case 'idle': return 'bg-blue-600/20';
      default: return 'bg-blue-600/20';
    }
  };

  const getModelTypeIcon = (type: string) => {
    switch (type) {
      case 'regression': return <TrendingUp className="w-4 h-4" />;
      case 'classification': return <Target className="w-4 h-4" />;
      case 'time_series': return <LineChart className="w-4 h-4" />;
      case 'clustering': return <PieChart className="w-4 h-4" />;
      case 'anomaly_detection': return <AlertTriangle className="w-4 h-4" />;
      default: return <Brain className="w-4 h-4" />;
    }
  };

  const filteredModels = models.filter(model => {
    const matchesSearch = model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         model.type.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || model.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Brain className="w-8 h-8 mr-3 text-blue-400" />
            Predictive Analytics Engine
          </h1>
          <p className="text-gray-400 mt-2">
            Advanced machine learning models for business forecasting and decision support
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowCreateModel(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Create Model</span>
          </button>
          <button
            onClick={() => setShowDataIngestion(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span>Ingest Data</span>
          </button>
          <div className="flex items-center space-x-2 px-3 py-1 rounded-full glass">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-green-400">Models Active</span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-blue-500/10 p-1 rounded-lg">
        {[
          { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
          { id: 'models', label: 'Models', icon: Brain },
          { id: 'predictions', label: 'Predictions', icon: TrendingUp },
          { id: 'data_sources', label: 'Data Sources', icon: Database },
          { id: 'training', label: 'Training', icon: Zap },
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

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Models</p>
                  <p className="text-2xl font-bold text-white">{dashboard.total_models}</p>
                </div>
                <Brain className="w-8 h-8 text-blue-400" />
              </div>
              <p className="text-green-400 text-sm mt-2">+2 this month</p>
            </div>

            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Active Predictions</p>
                  <p className="text-2xl font-bold text-white">{dashboard.active_predictions}</p>
                </div>
                <Activity className="w-8 h-8 text-green-400" />
              </div>
              <p className="text-blue-400 text-sm mt-2">{dashboard.predictions_today} today</p>
            </div>

            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Avg Accuracy</p>
                  <p className="text-2xl font-bold text-white">{dashboard.accuracy_avg}%</p>
                </div>
                <Target className="w-8 h-8 text-yellow-400" />
              </div>
              <div className="mt-4 bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-yellow-500 to-green-500 h-2 rounded-full"
                  style={{ width: `${dashboard.accuracy_avg}%` }}
                ></div>
              </div>
            </div>

            <div className="glass p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Business Value</p>
                  <p className="text-2xl font-bold text-white">${(dashboard.business_value_generated / 1000).toFixed(0)}K</p>
                </div>
                <DollarSign className="w-8 h-8 text-green-400" />
              </div>
              <p className="text-green-400 text-sm mt-2">Generated this month</p>
            </div>
          </div>

          {/* Model Performance and Revenue Forecast */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                Model Performance
              </h3>
              <div className="space-y-4">
                {dashboard.model_performance.map((model, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-blue-500/10 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-2">
                        {model.trend === 'up' ? (
                          <ArrowUp className="w-4 h-4 text-green-400" />
                        ) : model.trend === 'down' ? (
                          <ArrowDown className="w-4 h-4 text-red-400" />
                        ) : (
                          <div className="w-4 h-4 bg-blue-400 rounded-full"></div>
                        )}
                        <span className="text-white text-sm">{model.model_name}</span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4 text-sm">
                      <span className="text-gray-400">{model.accuracy}%</span>
                      <span className="text-blue-400">{model.predictions} predictions</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-green-400" />
                Revenue Forecast
              </h3>
              <div className="space-y-4">
                {dashboard.revenue_forecast.map((forecast, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-blue-500/10 rounded-lg">
                    <div>
                      <p className="text-white font-medium">{forecast.period}</p>
                      <p className="text-gray-400 text-sm">Confidence: {forecast.confidence}%</p>
                    </div>
                    <div className="text-right">
                      <p className="text-white font-semibold">${(forecast.predicted / 1000).toFixed(0)}K</p>
                      {forecast.actual && (
                        <p className={`text-sm ${
                          forecast.actual >= forecast.predicted ? 'text-green-400' : 'text-red-400'
                        }`}>
                          Actual: ${(forecast.actual / 1000).toFixed(0)}K
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk Alerts */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2 text-yellow-400" />
              System Status & Alerts
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                {dashboard.risk_alerts.map((alert) => (
                  <div key={alert.id} className={`p-3 rounded-lg border ${
                    alert.severity === 'critical' ? 'bg-red-500/10 border-red-500/20' :
                    alert.severity === 'high' ? 'bg-orange-500/10 border-orange-500/20' :
                    alert.severity === 'medium' ? 'bg-yellow-500/10 border-yellow-500/20' :
                    'bg-blue-500/10 border-blue-500/20'
                  }`}>
                    <div className="flex items-start justify-between">
                      <div>
                        <p className={`text-sm font-medium ${
                          alert.severity === 'critical' ? 'text-red-400' :
                          alert.severity === 'high' ? 'text-orange-400' :
                          alert.severity === 'medium' ? 'text-yellow-400' :
                          'text-blue-400'
                        }`}>
                          {alert.type.replace('_', ' ').toUpperCase()}
                        </p>
                        <p className="text-white text-sm mt-1">{alert.message}</p>
                        <p className="text-gray-400 text-xs mt-2">
                          {new Date(alert.created_at).toLocaleString()}
                        </p>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        alert.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                        alert.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                        alert.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-blue-500/20 text-blue-400'
                      }`}>
                        {alert.severity}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Data Processed Today</span>
                  <span className="text-white font-semibold">{dashboard.data_processed_gb} GB</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Connected Sources</span>
                  <span className="text-green-400">{dataSources.filter(ds => ds.status === 'connected').length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Total Records</span>
                  <span className="text-blue-400">
                    {(dataSources.reduce((sum, ds) => sum + ds.records_count, 0) / 1000000).toFixed(1)}M
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Processing Speed</span>
                  <span className="text-yellow-400">1.2K records/sec</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <div className="space-y-6">
          {/* Filters and Search */}
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search models..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
              </div>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Status</option>
                <option value="deployed">Deployed</option>
                <option value="ready">Ready</option>
                <option value="training">Training</option>
                <option value="error">Error</option>
                <option value="idle">Idle</option>
              </select>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => queryClient.invalidateQueries({ queryKey: ['prediction-models'] })}
                className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Refresh</span>
              </button>
            </div>
          </div>

          {/* Models Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredModels.map((model) => (
              <div key={model.id} className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${getStatusBg(model.status)}`}>
                      {getModelTypeIcon(model.type)}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{model.name}</h3>
                      <p className="text-sm text-gray-400">{model.type} • {model.model_version}</p>
                    </div>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full ${getStatusBg(model.status)} ${getStatusColor(model.status)}`}>
                    {model.status}
                  </span>
                </div>

                <div className="space-y-3 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Accuracy</span>
                    <span className="text-white">{model.accuracy}%</span>
                  </div>
                  {model.accuracy > 0 && (
                    <div className="bg-blue-800 hover:bg-blue-900 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                        style={{ width: `${model.accuracy}%` }}
                      ></div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Predictions Made</p>
                      <p className="text-white">{model.predictions_made}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Confidence</p>
                      <p className="text-white">{model.confidence_score}%</p>
                    </div>
                  </div>

                  <div className="text-sm">
                    <p className="text-gray-400">Training Data</p>
                    <p className="text-white">{(model.training_data_size / 1000).toFixed(0)}K records</p>
                  </div>

                  <div className="text-sm">
                    <p className="text-gray-400">Target Variable</p>
                    <p className="text-white">{model.target_variable}</p>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <div className="flex items-center space-x-2">
                    {model.status === 'idle' || model.status === 'ready' ? (
                      <button
                        onClick={() => trainModelMutation.mutate(model.id)}
                        disabled={trainModelMutation.isPending}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-700 hover:bg-blue-800 disabled:cursor-not-allowed rounded text-xs transition-colors"
                      >
                        <Zap className="w-3 h-3" />
                        <span>Train</span>
                      </button>
                    ) : null}
                    {model.status === 'ready' ? (
                      <button
                        onClick={() => deployModelMutation.mutate(model.id)}
                        disabled={deployModelMutation.isPending}
                        className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 disabled:bg-blue-700 hover:bg-blue-800 disabled:cursor-not-allowed rounded text-xs transition-colors"
                      >
                        <Play className="w-3 h-3" />
                        <span>Deploy</span>
                      </button>
                    ) : null}
                    {model.status === 'deployed' ? (
                      <button
                        onClick={() => createPredictionMutation.mutate({
                          modelId: model.id,
                          inputData: { sample: 'data' }
                        })}
                        disabled={createPredictionMutation.isPending}
                        className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 disabled:bg-blue-700 hover:bg-blue-800 disabled:cursor-not-allowed rounded text-xs transition-colors"
                      >
                        <Target className="w-3 h-3" />
                        <span>Predict</span>
                      </button>
                    ) : null}
                  </div>
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={() => setSelectedModel(model.id)}
                      className="p-1 hover:bg-blue-500/15 rounded"
                    >
                      <Eye className="w-4 h-4 text-gray-400" />
                    </button>
                    <button className="p-1 hover:bg-blue-500/15 rounded">
                      <Edit className="w-4 h-4 text-gray-400" />
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

      {/* Predictions Tab */}
      {activeTab === 'predictions' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">Recent Predictions</h2>
            <div className="flex items-center space-x-2">
              <select
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value as any)}
                className="px-3 py-2 bg-blue-500/15 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="1d">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last 90 Days</option>
              </select>
              <button
                onClick={() => queryClient.invalidateQueries({ queryKey: ['predictions'] })}
                className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Refresh</span>
              </button>
            </div>
          </div>

          <div className="space-y-4">
            {predictions.map((prediction) => (
              <div key={prediction.id} className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{prediction.model_name}</h3>
                    <p className="text-sm text-gray-400">{prediction.prediction_type}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-400">Confidence</p>
                    <p className="text-lg font-semibold text-white">{prediction.confidence}%</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-400">Predicted Value</p>
                    <p className="text-white font-medium">
                      {typeof prediction.predicted_value === 'number'
                        ? prediction.predicted_value.toLocaleString()
                        : prediction.predicted_value}
                    </p>
                  </div>
                  {prediction.actual_value && (
                    <div>
                      <p className="text-sm text-gray-400">Actual Value</p>
                      <p className="text-white font-medium">
                        {typeof prediction.actual_value === 'number'
                          ? prediction.actual_value.toLocaleString()
                          : prediction.actual_value}
                      </p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm text-gray-400">Business Impact</p>
                    <p className={`font-medium ${
                      prediction.business_impact.estimated_value >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {prediction.business_impact.estimated_value >= 0 ? '+' : ''}
                      ${Math.abs(prediction.business_impact.estimated_value).toLocaleString()}
                      {prediction.business_impact.currency}
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <p className="text-sm text-gray-400">
                    Created {new Date(prediction.created_at).toLocaleString()}
                  </p>
                  {prediction.accuracy && (
                    <span className={`text-sm px-2 py-1 rounded-full ${
                      prediction.accuracy >= 95 ? 'bg-green-500/20 text-green-400' :
                      prediction.accuracy >= 85 ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {prediction.accuracy}% accurate
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Sources Tab */}
      {activeTab === 'data_sources' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">Data Sources</h2>
            <button
              onClick={() => setShowDataIngestion(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Add Source</span>
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {dataSources.map((source) => (
              <div key={source.id} className="glass p-6 rounded-xl">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      source.status === 'connected' ? 'bg-green-500/20 text-green-400' :
                      source.status === 'syncing' ? 'bg-yellow-500/20 text-yellow-400' :
                      source.status === 'error' ? 'bg-red-500/20 text-red-400' :
                      'bg-blue-600/20 text-gray-400'
                    }`}>
                      <Database className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{source.name}</h3>
                      <p className="text-sm text-gray-400">{source.type}</p>
                    </div>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    source.status === 'connected' ? 'bg-green-500/20 text-green-400' :
                    source.status === 'syncing' ? 'bg-yellow-500/20 text-yellow-400' :
                    source.status === 'error' ? 'bg-red-500/20 text-red-400' :
                    'bg-blue-600/20 text-gray-400'
                  }`}>
                    {source.status}
                  </span>
                </div>

                <div className="space-y-3 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Records</span>
                    <span className="text-white">{source.records_count.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Last Sync</span>
                    <span className="text-white">{new Date(source.last_sync).toLocaleString()}</span>
                  </div>
                  <div className="text-sm">
                    <p className="text-gray-400 mb-2">Schema Fields</p>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(source.schema).slice(0, 4).map(([field, type]) => (
                        <span key={field} className="text-xs px-2 py-1 bg-blue-500/15 rounded">
                          {field}: {type}
                        </span>
                      ))}
                      {Object.keys(source.schema).length > 4 && (
                        <span className="text-xs px-2 py-1 bg-blue-500/15 rounded">
                          +{Object.keys(source.schema).length - 4} more
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-gray-600">
                  <div className="flex items-center space-x-2">
                    <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors">
                      <RefreshCw className="w-3 h-3" />
                      <span>Sync</span>
                    </button>
                    <button className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors">
                      <Play className="w-3 h-3" />
                      <span>Test</span>
                    </button>
                  </div>
                  <div className="flex items-center space-x-1">
                    <button className="p-1 hover:bg-blue-500/15 rounded">
                      <Eye className="w-4 h-4 text-gray-400" />
                    </button>
                    <button className="p-1 hover:bg-blue-500/15 rounded">
                      <Edit className="w-4 h-4 text-gray-400" />
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
  );
};

export default PredictiveAnalyticsEngine;
