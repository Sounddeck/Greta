import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Send,
  Layers,
  Sparkles,
  BookOpen,
  CheckCircle,
  AlertCircle,
  Loader,
  Zap,
  Brain,
  Database
} from 'lucide-react';
import { cpas4Api, TaskExecuteRequest, PromptEnhanceRequest, KnowledgeQueryRequest } from '../services/cpas4-api';

interface Message {
  id: string;
  type: 'user' | 'system' | 'result';
  content: string;
  timestamp: Date;
  category?: 'task' | 'prompt' | 'knowledge';
  metadata?: any;
}

const CPAS4: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'system',
      content: 'Welcome to CPAS4 - Cognitive Processing & Agent System. I can help you execute tasks, enhance prompts, and manage knowledge. What would you like to do?',
      timestamp: new Date()
    }
  ]);
  
  const [inputMessage, setInputMessage] = useState('');
  const [activeMode, setActiveMode] = useState<'task' | 'prompt' | 'knowledge'>('task');
  const [isProcessing, setIsProcessing] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [inputMessage]);

  const addMessage = (type: 'user' | 'system' | 'result', content: string, category?: 'task' | 'prompt' | 'knowledge', metadata?: any) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      type,
      content,
      timestamp: new Date(),
      category,
      metadata
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleExecuteTask = async (description: string) => {
    setIsProcessing(true);
    addMessage('user', description, 'task');

    try {
      const request: TaskExecuteRequest = {
        description,
        agent_id: 'master_agent',
        max_iterations: 3,
      };

      const response = await cpas4Api.executeTask(request);
      
      if (response.error) {
        addMessage('system', `Error: ${response.error}`, 'task');
      } else if (response.data) {
        addMessage('result', response.data.result || response.data.output || 'Task completed successfully', 'task', response.data);
      }
    } catch (err) {
      addMessage('system', `Failed to execute task: ${err instanceof Error ? err.message : 'Unknown error'}`, 'task');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleEnhancePrompt = async (prompt: string) => {
    setIsProcessing(true);
    addMessage('user', prompt, 'prompt');

    try {
      const request: PromptEnhanceRequest = {
        prompt,
        enhancement_type: 'all',
      };

      const response = await cpas4Api.enhancePrompt(request);
      
      if (response.error) {
        addMessage('system', `Error: ${response.error}`, 'prompt');
      } else if (response.data) {
        const result = `Enhanced Prompt:\n${response.data.enhanced_prompt}\n\nImprovements:\n${response.data.improvements?.join('\n') || 'None'}\n\nConfidence: ${response.data.confidence_score}%`;
        addMessage('result', result, 'prompt', response.data);
      }
    } catch (err) {
      addMessage('system', `Failed to enhance prompt: ${err instanceof Error ? err.message : 'Unknown error'}`, 'prompt');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQueryKnowledge = async (query: string) => {
    setIsProcessing(true);
    addMessage('user', query, 'knowledge');

    try {
      const request: KnowledgeQueryRequest = {
        query,
        top_k: 5,
      };

      const response = await cpas4Api.queryKnowledge(request);
      
      if (response.error) {
        addMessage('system', `Error: ${response.error}`, 'knowledge');
      } else if (response.data) {
        const results = response.data.results || [];
        const resultText = results.length > 0
          ? `Found ${results.length} results:\n\n${results.map((r: any, i: number) => `${i + 1}. ${r.content} (${(r.relevance_score * 100).toFixed(1)}% relevant)`).join('\n\n')}`
          : 'No results found';
        addMessage('result', resultText, 'knowledge', response.data);
      }
    } catch (err) {
      addMessage('system', `Failed to query knowledge: ${err instanceof Error ? err.message : 'Unknown error'}`, 'knowledge');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendMessage = () => {
    if (!inputMessage.trim() || isProcessing) return;

    const message = inputMessage.trim();
    setInputMessage('');

    switch (activeMode) {
      case 'task':
        handleExecuteTask(message);
        break;
      case 'prompt':
        handleEnhancePrompt(message);
        break;
      case 'knowledge':
        handleQueryKnowledge(message);
        break;
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getModeIcon = (mode: 'task' | 'prompt' | 'knowledge') => {
    switch (mode) {
      case 'task': return <Zap className="w-5 h-5" />;
      case 'prompt': return <Sparkles className="w-5 h-5" />;
      case 'knowledge': return <BookOpen className="w-5 h-5" />;
    }
  };

  const getModeColor = (mode: 'task' | 'prompt' | 'knowledge') => {
    switch (mode) {
      case 'task': return '#3b82f6';
      case 'prompt': return '#8b5cf6';
      case 'knowledge': return '#10b981';
    }
  };

  return (
    <div className="cpas4-interface">
      {/* Header */}
      <div className="cpas4-header">
        <div className="flex items-center gap-3">
          <div className="cpas4-avatar">
            <Layers className="w-8 h-8 text-blue-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">CPAS4 Interface</h1>
            <p className="text-sm text-gray-400">
              Cognitive Processing & Agent System v4.0
            </p>
          </div>
        </div>
        
        <div className="mode-selector">
          <button
            onClick={() => setActiveMode('task')}
            className={`mode-btn ${activeMode === 'task' ? 'active' : ''}`}
            title="Task Execution Mode"
          >
            <Zap className="w-5 h-5" />
            <span>Tasks</span>
          </button>
          
          <button
            onClick={() => setActiveMode('prompt')}
            className={`mode-btn ${activeMode === 'prompt' ? 'active' : ''}`}
            title="Prompt Enhancement Mode"
          >
            <Sparkles className="w-5 h-5" />
            <span>Prompts</span>
          </button>
          
          <button
            onClick={() => setActiveMode('knowledge')}
            className={`mode-btn ${activeMode === 'knowledge' ? 'active' : ''}`}
            title="Knowledge Query Mode"
          >
            <BookOpen className="w-5 h-5" />
            <span>Knowledge</span>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="cpas4-messages-container">
        <div className="cpas4-messages-list">
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`cpas4-message ${message.type}`}
            >
              <div className="message-icon">
                {message.type === 'user' ? (
                  <div className="user-icon">U</div>
                ) : message.type === 'result' ? (
                  <CheckCircle className="w-6 h-6 text-green-400" />
                ) : (
                  <Brain className="w-6 h-6 text-blue-400" />
                )}
              </div>
              
              <div className="message-content-wrapper">
                <div className="message-header-info">
                  <span className="message-role-label">
                    {message.type === 'user' ? 'You' : message.type === 'result' ? 'Result' : 'CPAS4'}
                  </span>
                  <span className="message-timestamp">
                    {formatTime(message.timestamp)}
                  </span>
                  {message.category && (
                    <span className="message-category" style={{ color: getModeColor(message.category) }}>
                      {getModeIcon(message.category)}
                      {message.category}
                    </span>
                  )}
                </div>
                
                <div className={`message-text-content ${message.type}`}>
                  {message.content}
                </div>
              </div>
            </motion.div>
          ))}
          
          {isProcessing && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="cpas4-message system processing"
            >
              <div className="message-icon">
                <Loader className="w-6 h-6 text-blue-400 animate-spin" />
              </div>
              <div className="message-content-wrapper">
                <div className="processing-indicator">
                  <span>Processing your request...</span>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="cpas4-input-container">
        <div className="input-mode-indicator" style={{ borderColor: getModeColor(activeMode) }}>
          {getModeIcon(activeMode)}
          <span>
            {activeMode === 'task' && 'Describe the task you want to execute...'}
            {activeMode === 'prompt' && 'Enter a prompt to enhance...'}
            {activeMode === 'knowledge' && 'Search the knowledge base...'}
          </span>
        </div>
        
        <div className="input-wrapper-cpas4">
          <textarea
            ref={textareaRef}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              activeMode === 'task' ? 'Type your task description... (Shift+Enter for new line)' :
              activeMode === 'prompt' ? 'Type your prompt... (Shift+Enter for new line)' :
              'Type your search query... (Shift+Enter for new line)'
            }
            className="cpas4-input"
            disabled={isProcessing}
            rows={1}
          />
          
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isProcessing}
            className="cpas4-send-btn"
            title="Send message"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>

      <style>{`
        .cpas4-interface {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 25%, #2d3561 50%, #1e2749 75%, #0f1419 100%);
          color: white;
          position: relative;
          overflow: hidden;
        }

        .cpas4-interface::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background:
            radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
          pointer-events: none;
          z-index: 0;
        }

        .cpas4-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 24px 32px;
          border-bottom: 1px solid rgba(59, 130, 246, 0.2);
          background: rgba(15, 23, 42, 0.8);
          backdrop-filter: blur(24px);
          position: relative;
          z-index: 10;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .cpas4-header::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 1px;
          background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
        }

        .cpas4-avatar {
          width: 56px;
          height: 56px;
          background: linear-gradient(135deg, #3b82f6, #1e40af, #1d4ed8);
          border-radius: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow:
            0 0 20px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
          position: relative;
          overflow: hidden;
        }

        .cpas4-avatar::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
          animation: shimmer 3s infinite;
        }

        .mode-selector {
          display: flex;
          gap: 8px;
          background: rgba(0, 0, 0, 0.3);
          padding: 4px;
          border-radius: 12px;
        }

        .mode-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 16px;
          background: transparent;
          border: 1px solid transparent;
          border-radius: 8px;
          color: #9ca3af;
          transition: all 0.2s ease;
          cursor: pointer;
        }

        .mode-btn.active {
          background: linear-gradient(135deg, #3b82f6, #1d4ed8);
          border-color: rgba(59, 130, 246, 0.5);
          color: white;
          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        .mode-btn:hover:not(.active) {
          background: rgba(255, 255, 255, 0.1);
          color: white;
        }

        .cpas4-messages-container {
          flex: 1;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .cpas4-messages-list {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .cpas4-message {
          display: flex;
          gap: 12px;
          max-width: 85%;
        }

        .cpas4-message.user {
          align-self: flex-end;
          flex-direction: row-reverse;
        }

        .cpas4-message.system, .cpas4-message.result {
          align-self: flex-start;
        }

        .message-icon {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .cpas4-message.user .message-icon {
          background: linear-gradient(135deg, #10b981, #059669);
        }

        .cpas4-message.system .message-icon,
        .cpas4-message.result .message-icon {
          background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        }

        .user-icon {
          font-weight: bold;
          font-size: 18px;
        }

        .message-content-wrapper {
          flex: 1;
          min-width: 0;
        }

        .message-header-info {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 4px;
        }

        .message-role-label {
          font-weight: 600;
          font-size: 14px;
        }

        .message-timestamp {
          font-size: 12px;
          color: #9ca3af;
        }

        .message-category {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 11px;
          text-transform: uppercase;
          font-weight: 600;
        }

        .message-text-content {
          background: rgba(255, 255, 255, 0.1);
          padding: 12px 16px;
          border-radius: 12px;
          line-height: 1.6;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .cpas4-message.user .message-text-content {
          background: rgba(59, 130, 246, 0.2);
        }

        .cpas4-message.result .message-text-content {
          background: rgba(16, 185, 129, 0.15);
          border-left: 3px solid #10b981;
        }

        .processing-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          color: #9ca3af;
        }

        .cpas4-input-container {
          padding: 20px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(20px);
        }

        .input-mode-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          margin-bottom: 12px;
          background: rgba(255, 255, 255, 0.05);
          border-left: 3px solid;
          border-radius: 6px;
          font-size: 13px;
          color: #9ca3af;
        }

        .input-wrapper-cpas4 {
          display: flex;
          gap: 12px;
          align-items: flex-end;
        }

        .cpas4-input {
          flex: 1;
          min-height: 50px;
          max-height: 150px;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 12px;
          color: white;
          font-size: 16px;
          resize: none;
          transition: all 0.2s ease;
        }

        .cpas4-input:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        .cpas4-input::placeholder {
          color: #9ca3af;
        }

        .cpas4-send-btn {
          width: 50px;
          height: 50px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.2);
          background: linear-gradient(135deg, #3b82f6, #1d4ed8);
          color: white;
          display: flex;
          align-items: center;
          justify-center: center;
          transition: all 0.2s ease;
          cursor: pointer;
        }

        .cpas4-send-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        .cpas4-send-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        @keyframes shimmer {
          0% {
            left: -100%;
          }
          100% {
            left: 100%;
          }
        }

        /* Scrollbar styling */
        .cpas4-messages-list::-webkit-scrollbar {
          width: 6px;
        }

        .cpas4-messages-list::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
        }

        .cpas4-messages-list::-webkit-scrollbar-thumb {
          background: rgba(59, 130, 246, 0.5);
          border-radius: 3px;
        }

        .cpas4-messages-list::-webkit-scrollbar-thumb:hover {
          background: rgba(59, 130, 246, 0.7);
        }
      `}</style>
    </div>
  );
};

export default CPAS4;
