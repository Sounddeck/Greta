import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, MicOff, Send, Zap, Brain, Settings, Activity,
  MessageSquare, Cpu, Database, Network, Shield
} from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../services/api';

interface CommandCenterProps {
  onCommandExecuted?: (result: any) => void;
}

interface CommandResult {
  success: boolean;
  command_id: string;
  routers_invoked: string[];
  results: Record<string, any>;
  execution_time: number;
  mode: string;
  status: string;
}

interface OrchestratorStatus {
  orchestrator_status: string;
  available_routers: string[];
  total_routers: number;
}

const CommandCenter: React.FC<CommandCenterProps> = ({ onCommandExecuted }) => {
  const [command, setCommand] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [mode, setMode] = useState<'text' | 'voice'>('text');
  const [priority, setPriority] = useState<'low' | 'normal' | 'high' | 'urgent'>('normal');
  const [isExpanded, setIsExpanded] = useState(false);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);
  const queryClient = useQueryClient();

  // Voice recognition setup
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setCommand(transcript);
        setMode('voice');
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
    }
  }, []);

  // Get orchestrator status
  const { data: orchestratorStatus } = useQuery({
    queryKey: ['orchestrator-status'],
    queryFn: async () => {
      const response = await apiService.request('/api/v1/orchestrator/status');
      return response.data;
    },
    refetchInterval: 5000
  });

  // Execute command mutation
  const executeCommandMutation = useMutation<CommandResult, Error, any>({
    mutationFn: async (commandData: any): Promise<CommandResult> => {
      const response = await apiService.request<CommandResult>('/api/v1/orchestrator/orchestrate', {
        method: 'POST',
        body: JSON.stringify(commandData)
      });
      return response.data || {} as CommandResult;
    },
    onSuccess: (data: CommandResult) => {
      // Add to command history
      if (command.trim()) {
        setCommandHistory(prev => [command, ...prev.slice(0, 9)]); // Keep last 10 commands
        setHistoryIndex(-1);
      }
      
      // Clear command input
      setCommand('');
      
      // Notify parent component
      if (onCommandExecuted) {
        onCommandExecuted(data);
      }
      
      // Refresh relevant queries
      queryClient.invalidateQueries({ queryKey: ['orchestrator-status'] });
    },
    onError: (error: any) => {
      console.error('Command execution failed:', error);
    }
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!command.trim()) return;

    executeCommandMutation.mutate({
      command: command.trim(),
      mode,
      priority,
      async_execution: false
    });
  };

  const toggleVoiceRecognition = () => {
    if (!recognitionRef.current) {
      alert('Speech recognition not supported in this browser');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
      setMode('voice');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setCommand(commandHistory[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setCommand(commandHistory[newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setCommand('');
      }
    }
  };

  const quickCommands = [
    { label: 'System Status', command: 'Show me the current system status and health metrics' },
    { label: 'Create Website', command: 'Create a professional business website with modern design' },
    { label: 'Research Market', command: 'Conduct comprehensive market research on AI industry trends' },
    { label: 'Generate Report', command: 'Generate a business intelligence report with current metrics' },
    { label: 'Deploy Agent', command: 'Create and deploy a new AI agent for customer support' },
    { label: 'Analyze Data', command: 'Analyze recent performance data and provide insights' }
  ];

  const priorityColors = {
    low: 'text-blue-400',
    normal: 'text-green-400',
    high: 'text-yellow-400',
    urgent: 'text-red-400'
  };

  return (
    <div className={`command-center ${isExpanded ? 'expanded' : ''}`}>
      {/* Header */}
      <div className="command-header">
        <div className="flex items-center gap-3">
          <div className="command-icon">
            <Zap className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">GRETA Command Center</h3>
            <p className="text-sm text-gray-400">
              {(orchestratorStatus as any)?.orchestrator_status === 'active' ? (
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  Online • {(orchestratorStatus as any)?.available_routers?.length || 0} routers active
                </span>
              ) : (
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                  Offline
                </span>
              )}
            </p>
          </div>
        </div>
        
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="expand-btn"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {/* Main Command Interface */}
      <form onSubmit={handleSubmit} className="command-form">
        <div className="command-input-container">
          <div className="input-wrapper">
            <input
              ref={inputRef}
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter command or speak naturally..."
              className="command-input"
              disabled={executeCommandMutation.isPending}
            />
            
            {/* Mode indicator */}
            <div className={`mode-indicator ${mode}`}>
              {mode === 'voice' ? (
                <Mic className="w-4 h-4" />
              ) : (
                <MessageSquare className="w-4 h-4" />
              )}
            </div>
          </div>

          <div className="command-controls">
            {/* Voice toggle */}
            <button
              type="button"
              onClick={toggleVoiceRecognition}
              className={`voice-btn ${isListening ? 'listening' : ''}`}
              disabled={executeCommandMutation.isPending}
            >
              {isListening ? (
                <MicOff className="w-5 h-5" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </button>

            {/* Priority selector */}
            <select
              value={priority}
              onChange={(e) => setPriority(e.target.value as any)}
              className="priority-select"
              disabled={executeCommandMutation.isPending}
            >
              <option value="low">Low</option>
              <option value="normal">Normal</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>

            {/* Submit button */}
            <button
              type="submit"
              disabled={!command.trim() || executeCommandMutation.isPending}
              className="submit-btn"
            >
              {executeCommandMutation.isPending ? (
                <div className="loading-spinner"></div>
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        {/* Execution status */}
        {executeCommandMutation.isPending && (
          <div className="execution-status">
            <div className="flex items-center gap-2">
              <div className="loading-spinner"></div>
              <span>Executing command...</span>
            </div>
          </div>
        )}

        {/* Last result */}
        {executeCommandMutation.data && (
          <div className="last-result">
            <div className="result-header">
              <span className={`status ${executeCommandMutation.data.success ? 'success' : 'error'}`}>
                {executeCommandMutation.data.success ? '✓' : '✗'}
              </span>
              <span>
                Executed in {executeCommandMutation.data.execution_time?.toFixed(2)}s
              </span>
              <span className="routers">
                {executeCommandMutation.data.routers_invoked?.join(', ')}
              </span>
            </div>
          </div>
        )}
      </form>

      {/* Expanded Controls */}
      {isExpanded && (
        <div className="expanded-controls">
          {/* Quick Commands */}
          <div className="quick-commands">
            <h4 className="text-sm font-medium text-gray-300 mb-2">Quick Commands</h4>
            <div className="quick-commands-grid">
              {quickCommands.map((cmd, index) => (
                <button
                  key={index}
                  onClick={() => setCommand(cmd.command)}
                  className="quick-command-btn"
                  disabled={executeCommandMutation.isPending}
                >
                  {cmd.label}
                </button>
              ))}
            </div>
          </div>

          {/* System Status */}
          <div className="system-status">
            <h4 className="text-sm font-medium text-gray-300 mb-2">System Status</h4>
            <div className="status-grid">
              <div className="status-item">
                <Brain className="w-4 h-4 text-purple-400" />
                <span>AI Core</span>
                <div className="status-indicator active"></div>
              </div>
              <div className="status-item">
                <Database className="w-4 h-4 text-blue-400" />
                <span>Memory</span>
                <div className="status-indicator active"></div>
              </div>
              <div className="status-item">
                <Network className="w-4 h-4 text-green-400" />
                <span>Network</span>
                <div className="status-indicator active"></div>
              </div>
              <div className="status-item">
                <Shield className="w-4 h-4 text-yellow-400" />
                <span>Security</span>
                <div className="status-indicator active"></div>
              </div>
            </div>
          </div>

          {/* Command History */}
          {commandHistory.length > 0 && (
            <div className="command-history">
              <h4 className="text-sm font-medium text-gray-300 mb-2">Recent Commands</h4>
              <div className="history-list">
                {commandHistory.slice(0, 5).map((cmd, index) => (
                  <button
                    key={index}
                    onClick={() => setCommand(cmd)}
                    className="history-item"
                    disabled={executeCommandMutation.isPending}
                  >
                    {cmd}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <style>{`
        .command-center {
          background: linear-gradient(135deg, rgba(139, 69, 19, 0.1), rgba(75, 0, 130, 0.1));
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          padding: 20px;
          margin-bottom: 20px;
          transition: all 0.3s ease;
        }

        .command-center.expanded {
          max-height: none;
        }

        .command-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .command-icon {
          width: 40px;
          height: 40px;
          background: linear-gradient(135deg, #8B4513, #4B0082);
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .expand-btn {
          padding: 8px;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 8px;
          color: white;
          transition: all 0.2s ease;
        }

        .expand-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .command-form {
          space-y: 12px;
        }

        .command-input-container {
          display: flex;
          gap: 12px;
          align-items: center;
        }

        .input-wrapper {
          flex: 1;
          position: relative;
        }

        .command-input {
          width: 100%;
          padding: 12px 16px 12px 40px;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 12px;
          color: white;
          font-size: 16px;
          transition: all 0.2s ease;
        }

        .command-input:focus {
          outline: none;
          border-color: #8B4513;
          box-shadow: 0 0 0 3px rgba(139, 69, 19, 0.2);
        }

        .mode-indicator {
          position: absolute;
          left: 12px;
          top: 50%;
          transform: translateY(-50%);
          color: #8B4513;
        }

        .mode-indicator.voice {
          color: #4B0082;
        }

        .command-controls {
          display: flex;
          gap: 8px;
          align-items: center;
        }

        .voice-btn {
          padding: 12px;
          background: rgba(139, 69, 19, 0.2);
          border: 1px solid rgba(139, 69, 19, 0.3);
          border-radius: 10px;
          color: #8B4513;
          transition: all 0.2s ease;
        }

        .voice-btn:hover {
          background: rgba(139, 69, 19, 0.3);
        }

        .voice-btn.listening {
          background: rgba(75, 0, 130, 0.3);
          border-color: rgba(75, 0, 130, 0.5);
          color: #4B0082;
          animation: pulse 2s infinite;
        }

        .priority-select {
          padding: 8px 12px;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 8px;
          color: white;
          font-size: 14px;
        }

        .submit-btn {
          padding: 12px;
          background: linear-gradient(135deg, #8B4513, #4B0082);
          border: none;
          border-radius: 10px;
          color: white;
          transition: all 0.2s ease;
        }

        .submit-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(139, 69, 19, 0.3);
        }

        .submit-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .loading-spinner {
          width: 20px;
          height: 20px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top: 2px solid white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        .execution-status {
          padding: 12px;
          background: rgba(75, 0, 130, 0.2);
          border: 1px solid rgba(75, 0, 130, 0.3);
          border-radius: 8px;
          color: #4B0082;
        }

        .last-result {
          padding: 12px;
          background: rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
        }

        .result-header {
          display: flex;
          gap: 12px;
          align-items: center;
          font-size: 14px;
          color: #ccc;
        }

        .status.success {
          color: #10b981;
        }

        .status.error {
          color: #ef4444;
        }

        .routers {
          color: #8B4513;
          font-weight: 500;
        }

        .expanded-controls {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          space-y: 16px;
        }

        .quick-commands-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 8px;
        }

        .quick-command-btn {
          padding: 8px 12px;
          background: rgba(139, 69, 19, 0.1);
          border: 1px solid rgba(139, 69, 19, 0.2);
          border-radius: 8px;
          color: #8B4513;
          font-size: 12px;
          text-align: left;
          transition: all 0.2s ease;
        }

        .quick-command-btn:hover:not(:disabled) {
          background: rgba(139, 69, 19, 0.2);
        }

        .status-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 12px;
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
          font-size: 14px;
          color: #ccc;
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-left: auto;
        }

        .status-indicator.active {
          background: #10b981;
          animation: pulse 2s infinite;
        }

        .history-list {
          space-y: 4px;
        }

        .history-item {
          width: 100%;
          padding: 8px;
          background: rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          color: #ccc;
          font-size: 12px;
          text-align: left;
          transition: all 0.2s ease;
        }

        .history-item:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.1);
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default CommandCenter;
