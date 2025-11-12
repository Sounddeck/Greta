/**
 * Training Page
 * Main page for LLM training automation with tabs for Datasets, Jobs, Models, and Stats
 */

import React, { useState, useEffect } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { Button } from '../components/ui/button';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { DatasetList } from '../components/training/DatasetList';
import { JobList } from '../components/training/JobList';
import { ModelList } from '../components/training/ModelList';
import { useTrainingStore } from '../stores/trainingStore';
import { toast } from 'sonner';

export default function Training() {
  const [activeTab, setActiveTab] = useState('datasets');
  const { stats, health, fetchStats, fetchHealth } = useTrainingStore();

  useEffect(() => {
    fetchStats();
    fetchHealth();
  }, [fetchStats, fetchHealth]);

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">🤖 LLM Training Automation</h1>
            <p className="text-gray-600 dark:text-gray-400">
              Manage datasets, training jobs, and model versions
            </p>
          </div>
          <div className="flex items-center gap-4">
            {health && (
              <div className="flex items-center gap-2">
                <Badge variant={health.status === 'healthy' ? 'outline' : 'destructive'}>
                  {health.status === 'healthy' ? '✅ Healthy' : '❌ Unhealthy'}
                </Badge>
                <span className="text-sm text-gray-500">v{health.version}</span>
              </div>
            )}
          </div>
        </div>

        {/* Quick Stats */}
        {stats && (
          <div className="grid grid-cols-4 gap-4 mt-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
              <p className="text-sm text-blue-600 dark:text-blue-400">Datasets</p>
              <p className="text-2xl font-bold">{stats.total_datasets}</p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
              <p className="text-sm text-green-600 dark:text-green-400">Training Jobs</p>
              <p className="text-2xl font-bold">{stats.total_jobs}</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
              <p className="text-sm text-purple-600 dark:text-purple-400">Models</p>
              <p className="text-2xl font-bold">{stats.total_models}</p>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded-lg">
              <p className="text-sm text-orange-600 dark:text-orange-400">Disk Usage</p>
              <p className="text-2xl font-bold">{(stats.disk_usage_mb / 1024).toFixed(1)} GB</p>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          {/* Tab Navigation */}
          <div className="bg-white dark:bg-gray-800 border-b px-6">
            <TabsList className="h-auto p-0 bg-transparent">
              <TabsTrigger
                value="datasets"
                className="px-4 py-3 data-[state=active]:border-b-2 data-[state=active]:border-blue-500 rounded-none"
              >
                📊 Datasets
              </TabsTrigger>
              <TabsTrigger
                value="jobs"
                className="px-4 py-3 data-[state=active]:border-b-2 data-[state=active]:border-blue-500 rounded-none"
              >
                🔄 Training Jobs
              </TabsTrigger>
              <TabsTrigger
                value="models"
                className="px-4 py-3 data-[state=active]:border-b-2 data-[state=active]:border-blue-500 rounded-none"
              >
                🤖 Models
              </TabsTrigger>
              <TabsTrigger
                value="stats"
                className="px-4 py-3 data-[state=active]:border-b-2 data-[state=active]:border-blue-500 rounded-none"
              >
                📈 Statistics
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-hidden">
            {/* Datasets Tab */}
            <TabsContent value="datasets" className="h-full m-0 p-6">
              <DatasetList />
            </TabsContent>

            {/* Training Jobs Tab */}
            <TabsContent value="jobs" className="h-full m-0 p-6">
              <JobList />
            </TabsContent>

            {/* Models Tab */}
            <TabsContent value="models" className="h-full m-0 p-6">
              <ModelList />
            </TabsContent>

            {/* Statistics Tab */}
            <TabsContent value="stats" className="h-full m-0 p-6 overflow-y-auto">
              <div className="space-y-6">
                <h2 className="text-xl font-semibold">Training Statistics</h2>

                {stats ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Dataset Stats */}
                    <Card>
                      <CardHeader>
                        <CardTitle>📊 Datasets</CardTitle>
                        <CardDescription>Training dataset statistics</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-500">Total Datasets</p>
                            <p className="text-3xl font-bold">{stats.total_datasets}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-500">Total Training Samples</p>
                            <p className="text-3xl font-bold">{stats.total_training_samples.toLocaleString()}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Job Stats */}
                    <Card>
                      <CardHeader>
                        <CardTitle>🔄 Training Jobs</CardTitle>
                        <CardDescription>Training job statistics</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-500">Total Jobs</p>
                            <p className="text-3xl font-bold">{stats.total_jobs}</p>
                          </div>
                          <div className="grid grid-cols-3 gap-2">
                            <div>
                              <p className="text-xs text-gray-500">Active</p>
                              <p className="text-lg font-bold text-blue-600">{stats.active_jobs}</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Completed</p>
                              <p className="text-lg font-bold text-green-600">{stats.completed_jobs}</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Failed</p>
                              <p className="text-lg font-bold text-red-600">{stats.failed_jobs}</p>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Model Stats */}
                    <Card>
                      <CardHeader>
                        <CardTitle>🤖 Models</CardTitle>
                        <CardDescription>Model version statistics</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-500">Total Models</p>
                            <p className="text-3xl font-bold">{stats.total_models}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-500">Active Model</p>
                            <p className="text-lg font-semibold">
                              {stats.active_model || 'None'}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* System Stats */}
                    <Card>
                      <CardHeader>
                        <CardTitle>💾 System</CardTitle>
                        <CardDescription>System resource usage</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-500">Disk Usage</p>
                            <p className="text-3xl font-bold">{(stats.disk_usage_mb / 1024).toFixed(2)} GB</p>
                          </div>
                          {health && (
                            <div>
                              <p className="text-sm text-gray-500 mb-2">Components</p>
                              <div className="space-y-1">
                                <div className="flex items-center justify-between text-sm">
                                  <span>Data Collection</span>
                                  <Badge variant={health.components.data_collection === 'active' ? 'outline' : 'secondary'}>
                                    {health.components.data_collection}
                                  </Badge>
                                </div>
                                <div className="flex items-center justify-between text-sm">
                                  <span>Training Pipeline</span>
                                  <Badge variant={health.components.training_pipeline === 'active' ? 'outline' : 'secondary'}>
                                    {health.components.training_pipeline}
                                  </Badge>
                                </div>
                                <div className="flex items-center justify-between text-sm">
                                  <span>Model Management</span>
                                  <Badge variant={health.components.model_management === 'active' ? 'outline' : 'secondary'}>
                                    {health.components.model_management}
                                  </Badge>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-64">
                    <div className="text-center">
                      <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                      <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading statistics...</p>
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </div>
  );
}

