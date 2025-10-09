import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Send,
  Mic,
  MicOff,
  Paperclip,
  Bot,
  User,
  Loader2,
  Trash2
} from 'lucide-react';
import { apiService, handleApiResponse } from '../services/api';
import { useVoiceRecognition, useVoiceSynthesis } from '../services/voice';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  isTyping?: boolean;
}

interface ChatRequest {
  message: string;
  model?: string;
  context?: string;
}

interface Model {
  name: string;
  displayName: string;
  description: string;
}

// API functions
const sendMessage = async (request: ChatRequest): Promise<{ response: string }> => {
  try {
    const response = await apiService.sendChatMessage(request.message, request.model);
    const data = handleApiResponse(response) as { response: string };
    return data;
  } catch (error) {
    // Fallback to mock response if backend is not available
    await new Promise(resolve => setTimeout(resolve, 1500));
    return {
      response: `I understand you said: "${request.message}". As GRETA, your Personal AI Assistant, I'm here to help you with any questions or tasks. How can I assist you further?`
    };
  }
};

const fetchModels = async (): Promise<Model[]> => {
  try {
    const response = await apiService.getAvailableModels();
    const data = handleApiResponse(response) as { models: Model[] };
    return data.models || [];
  } catch (error) {
    // Fallback to mock data if backend is not available
    return [
      { name: 'greta-enhanced', displayName: 'GRETA Enhanced', description: 'Optimized for personal assistance' },
      { name: 'llama3-local', displayName: 'Llama 3 Local', description: 'Local processing, privacy-first' },
      { name: 'mock-model', displayName: 'Mock Model', description: 'For testing purposes' }
    ];
  }
};

const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.sender === 'user';
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div className={`flex items-start space-x-3 max-w-3xl ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-gradient-to-r from-blue-600 to-purple-600' 
            : 'bg-gradient-to-r from-purple-600 to-pink-600'
        }`}>
          {isUser ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-white" />}
        </div>
        
        <div className={`glass-strong p-4 rounded-2xl ${
          isUser 
            ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 border-blue-500/30' 
            : 'bg-blue-500/15 border-white/20'
        }`}>
          {message.isTyping ? (
            <div className="flex items-center space-x-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
              <span className="text-sm text-gray-400">GRETA is thinking...</span>
            </div>
          ) : (
            <p className="text-white whitespace-pre-wrap">{message.content}</p>
          )}
          
          <div className="mt-2 text-xs text-gray-400">
            {message.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m GRETA, your Personal AI Assistant. How can I help you today?',
      sender: 'assistant',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('greta-enhanced');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Voice functionality
  const { isListening, transcript, startListening, stopListening, isSupported: voiceSupported } = useVoiceRecognition();
  const { speak, isSpeaking } = useVoiceSynthesis();

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels
  });

  const sendMessageMutation = useMutation({
    mutationFn: sendMessage,
    onMutate: (variables) => {
      // Add user message immediately
      const userMessage: Message = {
        id: Date.now().toString(),
        content: variables.message,
        sender: 'user',
        timestamp: new Date()
      };

      // Add typing indicator
      const typingMessage: Message = {
        id: 'typing',
        content: '',
        sender: 'assistant',
        timestamp: new Date(),
        isTyping: true
      };

      setMessages(prev => [...prev, userMessage, typingMessage]);
      setInputMessage('');
    },
    onSuccess: (data) => {
      // Remove typing indicator and add response
      setMessages(prev => {
        const withoutTyping = prev.filter(msg => msg.id !== 'typing');
        const assistantMessage: Message = {
          id: Date.now().toString(),
          content: data.response,
          sender: 'assistant',
          timestamp: new Date()
        };

        // Speak the response if voice is enabled
        if (data.response) {
          speak(data.response);
        }

        return [...withoutTyping, assistantMessage];
      });
    },
    onError: () => {
      // Remove typing indicator and show error
      setMessages(prev => {
        const withoutTyping = prev.filter(msg => msg.id !== 'typing');
        const errorMessage: Message = {
          id: Date.now().toString(),
          content: 'Sorry, I encountered an error. Please try again.',
          sender: 'assistant',
          timestamp: new Date()
        };
        return [...withoutTyping, errorMessage];
      });
    }
  });

  const handleSendMessage = () => {
    if (!inputMessage.trim() || sendMessageMutation.isPending) return;
    
    sendMessageMutation.mutate({
      message: inputMessage,
      model: selectedModel
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: '1',
        content: 'Hello! I\'m GRETA, your Personal AI Assistant. How can I help you today?',
        sender: 'assistant',
        timestamp: new Date()
      }
    ]);
  };

  const toggleVoiceRecording = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  // Update input message when voice transcript changes
  useEffect(() => {
    if (transcript && !isListening) {
      setInputMessage(transcript);
    }
  }, [transcript, isListening]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between py-6 border-b border-white/10">
        <div>
          <h1 className="text-2xl font-bold text-white">🎭 Chat with GRETA</h1>
          <p className="text-gray-400">Your Personal AI Assistant</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="input-field text-sm"
          >
            {models?.map((model) => (
              <option key={model.name} value={model.name} className="bg-blue-900/40">
                {model.displayName}
              </option>
            ))}
          </select>
          
          <button
            onClick={clearChat}
            className="btn-secondary p-2"
            title="Clear chat"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-6">
        <AnimatePresence>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="py-6 border-t border-white/10">
        <div className="glass-strong p-4 rounded-2xl">
          <div className="flex items-end space-x-4">
            <button className="btn-secondary p-2 flex-shrink-0">
              <Paperclip className="w-5 h-5" />
            </button>
            
            <div className="flex-1">
              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="w-full bg-transparent text-white placeholder-gray-400 resize-none focus:outline-none"
                rows={1}
                style={{ minHeight: '24px', maxHeight: '120px' }}
              />
            </div>
            
            <button
              onClick={toggleVoiceRecording}
              disabled={!voiceSupported}
              className={`p-2 rounded-lg transition-colors ${
                isListening
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-blue-500/15 hover:bg-blue-500/20'
              } ${!voiceSupported ? 'opacity-50 cursor-not-allowed' : ''}`}
              title={voiceSupported ? (isListening ? 'Stop recording' : 'Start voice recording') : 'Voice not supported'}
            >
              {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </button>
            
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || sendMessageMutation.isPending}
              className="btn-primary p-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {sendMessageMutation.isPending ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
