import React, { useState, useEffect } from 'react';
import { cpas4Api, TaskExecuteRequest, PromptEnhanceRequest, KnowledgeQueryRequest } from '../services/cpas4-api';

const CPAS4: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'tasks' | 'prompts' | 'knowledge'>('tasks');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Task state
  const [taskDescription, setTaskDescription] = useState('');
  const [taskResult, setTaskResult] = useState<any>(null);
  const [taskHistory, setTaskHistory] = useState<any[]>([]);

  // Prompt state
  const [promptText, setPromptText] = useState('');
  const [enhancedPrompt, setEnhancedPrompt] = useState<any>(null);
  const [promptTemplates, setPromptTemplates] = useState<any[]>([]);

  // Knowledge state
  const [knowledgeQuery, setKnowledgeQuery] = useState('');
  const [knowledgeResults, setKnowledgeResults] = useState<any>(null);
  const [knowledgeContent, setKnowledgeContent] = useState('');

  // Load initial data
  useEffect(() => {
    loadTaskHistory();
    loadPromptTemplates();
  }, []);

  const loadTaskHistory = async () => {
    try {
      const response = await cpas4Api.getTaskHistory(5);
      if (response.data) {
        setTaskHistory(response.data);
      }
    } catch (err) {
      console.error('Failed to load task history:', err);
    }
  };

  const loadPromptTemplates = async () => {
    try {
      const response = await cpas4Api.getPromptTemplates();
      if (response.data) {
        setPromptTemplates(response.data);
      }
    } catch (err) {
      console.error('Failed to load prompt templates:', err);
    }
  };

  // Task handlers
  const handleExecuteTask = async () => {
    if (!taskDescription.trim()) {
      setError('Please enter a task description');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: TaskExecuteRequest = {
        description: taskDescription,
        agent_id: 'master_agent',
        max_iterations: 3,
      };

      const response = await cpas4Api.executeTask(request);
      
      if (response.error) {
        setError(response.error);
      } else {
        setTaskResult(response.data);
        loadTaskHistory();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to execute task');
    } finally {
      setLoading(false);
    }
  };

  // Prompt handlers
  const handleEnhancePrompt = async () => {
    if (!promptText.trim()) {
      setError('Please enter a prompt to enhance');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: PromptEnhanceRequest = {
        prompt: promptText,
        enhancement_type: 'all',
      };

      const response = await cpas4Api.enhancePrompt(request);
      
      if (response.error) {
        setError(response.error);
      } else {
        setEnhancedPrompt(response.data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to enhance prompt');
    } finally {
      setLoading(false);
    }
  };

  // Knowledge handlers
  const handleQueryKnowledge = async () => {
    if (!knowledgeQuery.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: KnowledgeQueryRequest = {
        query: knowledgeQuery,
        top_k: 5,
      };

      const response = await cpas4Api.queryKnowledge(request);
      
      if (response.error) {
        setError(response.error);
      } else {
        setKnowledgeResults(response.data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to query knowledge');
    } finally {
      setLoading(false);
    }
  };

  const handleAddKnowledge = async () => {
    if (!knowledgeContent.trim()) {
      setError('Please enter content to add');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await cpas4Api.addKnowledge({
        content: knowledgeContent,
        category: 'user_added',
      });
      
      if (response.error) {
        setError(response.error);
      } else {
        setKnowledgeContent('');
        alert('Knowledge added successfully!');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add knowledge');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🚀 CPAS4 Interface
          </h1>
          <p className="text-blue-200">
            Cognitive Processing & Agent System - Version 4
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setActiveTab('tasks')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'tasks'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-slate-800 text-gray-300 hover:bg-slate-700'
            }`}
          >
            📋 Tasks
          </button>
          <button
            onClick={() => setActiveTab('prompts')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'prompts'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-slate-800 text-gray-300 hover:bg-slate-700'
            }`}
          >
            ✨ Prompts
          </button>
          <button
            onClick={() => setActiveTab('knowledge')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'knowledge'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-slate-800 text-gray-300 hover:bg-slate-700'
            }`}
          >
            📚 Knowledge
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}

        {/* Tasks Tab */}
        {activeTab === 'tasks' && (
          <div className="space-y-6">
            <div className="glass p-6 rounded-xl">
              <h2 className="text-2xl font-bold text-white mb-4">Execute Task</h2>
              <textarea
                value={taskDescription}
                onChange={(e) => setTaskDescription(e.target.value)}
                placeholder="Describe the task you want CPAS4 to execute..."
                className="w-full h-32 px-4 py-3 bg-slate-800 text-white rounded-lg border border-blue-500/30 focus:border-blue-500 focus:outline-none resize-none"
              />
              <button
                onClick={handleExecuteTask}
                disabled={loading}
                className="mt-4 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Executing...' : 'Execute Task'}
              </button>
            </div>

            {taskResult && (
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-bold text-white mb-4">Task Result</h3>
                <div className="space-y-2">
                  <p className="text-gray-300"><span className="font-semibold">Status:</span> {taskResult.status}</p>
                  <p className="text-gray-300"><span className="font-semibold">ID:</span> {taskResult.id}</p>
                  {taskResult.result && (
                    <div className="mt-4 p-4 bg-slate-800 rounded-lg">
                      <p className="text-white whitespace-pre-wrap">{taskResult.result}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {taskHistory.length > 0 && (
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-bold text-white mb-4">Recent Tasks</h3>
                <div className="space-y-3">
                  {taskHistory.map((task, index) => (
                    <div key={index} className="p-4 bg-slate-800 rounded-lg">
                      <p className="text-white font-semibold">{task.id}</p>
                      <p className="text-gray-400 text-sm">{task.status} - {task.created_at}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Prompts Tab */}
        {activeTab === 'prompts' && (
          <div className="space-y-6">
            <div className="glass p-6 rounded-xl">
              <h2 className="text-2xl font-bold text-white mb-4">Enhance Prompt</h2>
              <textarea
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                placeholder="Enter your prompt to enhance..."
                className="w-full h-32 px-4 py-3 bg-slate-800 text-white rounded-lg border border-blue-500/30 focus:border-blue-500 focus:outline-none resize-none"
              />
              <button
                onClick={handleEnhancePrompt}
                disabled={loading}
                className="mt-4 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Enhancing...' : 'Enhance Prompt'}
              </button>
            </div>

            {enhancedPrompt && (
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-bold text-white mb-4">Enhanced Prompt</h3>
                <div className="space-y-4">
                  <div>
                    <p className="text-gray-400 text-sm mb-2">Original:</p>
                    <p className="text-gray-300 p-4 bg-slate-800 rounded-lg">{enhancedPrompt.original_prompt}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-2">Enhanced:</p>
                    <p className="text-white p-4 bg-blue-900/30 rounded-lg">{enhancedPrompt.enhanced_prompt}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-2">Improvements:</p>
                    <ul className="list-disc list-inside text-gray-300 space-y-1">
                      {enhancedPrompt.improvements?.map((imp: string, i: number) => (
                        <li key={i}>{imp}</li>
                      ))}
                    </ul>
                  </div>
                  <p className="text-gray-400">
                    Confidence Score: <span className="text-blue-400 font-semibold">{enhancedPrompt.confidence_score}%</span>
                  </p>
                </div>
              </div>
            )}

            {promptTemplates.length > 0 && (
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-bold text-white mb-4">Prompt Templates</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {promptTemplates.map((template, index) => (
                    <div key={index} className="p-4 bg-slate-800 rounded-lg">
                      <p className="text-white font-semibold">{template.name}</p>
                      <p className="text-gray-400 text-sm">{template.category}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Knowledge Tab */}
        {activeTab === 'knowledge' && (
          <div className="space-y-6">
            <div className="glass p-6 rounded-xl">
              <h2 className="text-2xl font-bold text-white mb-4">Query Knowledge Base</h2>
              <input
                type="text"
                value={knowledgeQuery}
                onChange={(e) => setKnowledgeQuery(e.target.value)}
                placeholder="Search the knowledge base..."
                className="w-full px-4 py-3 bg-slate-800 text-white rounded-lg border border-blue-500/30 focus:border-blue-500 focus:outline-none"
              />
              <button
                onClick={handleQueryKnowledge}
                disabled={loading}
                className="mt-4 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>

            {knowledgeResults && (
              <div className="glass p-6 rounded-xl">
                <h3 className="text-xl font-bold text-white mb-4">
                  Search Results ({knowledgeResults.total_results} found)
                </h3>
                <div className="space-y-3">
                  {knowledgeResults.results?.map((item: any, index: number) => (
                    <div key={index} className="p-4 bg-slate-800 rounded-lg">
                      <p className="text-white">{item.content}</p>
                      <p className="text-gray-400 text-sm mt-2">
                        Relevance: {(item.relevance_score * 100).toFixed(1)}%
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="glass p-6 rounded-xl">
              <h2 className="text-2xl font-bold text-white mb-4">Add Knowledge</h2>
              <textarea
                value={knowledgeContent}
                onChange={(e) => setKnowledgeContent(e.target.value)}
                placeholder="Enter content to add to the knowledge base..."
                className="w-full h-32 px-4 py-3 bg-slate-800 text-white rounded-lg border border-blue-500/30 focus:border-blue-500 focus:outline-none resize-none"
              />
              <button
                onClick={handleAddKnowledge}
                disabled={loading}
                className="mt-4 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Adding...' : 'Add Knowledge'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CPAS4;
