import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Send, Sparkles, Upload, Loader2, Lightbulb } from 'lucide-react';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import { useUIStore } from '../../stores/uiStore';
import { useChatStore } from '../../stores/chatStore';
import { cn } from '../../lib/utils';
import { taskAPI, promptAPI } from '../../lib/api';
import { getAgentConfig } from '../../config/agentConfig';
import type { Message, AgentType } from '../../types';

export function MasterAgentContext() {
  const { agentType } = useParams<{ agentType: string }>();
  const { backgroundImage, backgroundOpacity, setCurrentAgentType } = useUIStore();
  const { getMessages, addMessage } = useChatStore();

  // Get messages for current agent from store
  const messages = getMessages(agentType as AgentType);

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [_error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Get agent configuration
  const agentConfig = getAgentConfig(agentType as AgentType);

  // Update current agent type when URL changes
  useEffect(() => {
    if (agentType) {
      setCurrentAgentType(agentType as AgentType);
    } else {
      setCurrentAgentType(null);
    }
  }, [agentType, setCurrentAgentType]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const currentAgentType = (agentType as AgentType) || null;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    // Add user message to store
    addMessage(currentAgentType, userMessage);

    const userInput = input;
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      // Get current messages for context
      const currentMessages = getMessages(currentAgentType);

      // Call CPAS4 backend to execute task
      const response = await taskAPI.execute({
        agent_id: 'master_agent',
        description: userInput,
        input_data: {
          agent_type: agentType || 'general',
          conversation_history: currentMessages.slice(-5), // Last 5 messages for context
        },
        max_iterations: 3,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.result || response.data.output || 'Task completed successfully!',
        timestamp: new Date().toISOString(),
      };

      // Add assistant message to store
      addMessage(currentAgentType, assistantMessage);
    } catch (err: any) {
      console.error('Error executing task:', err);

      // Add error message to store
      // Error handling removed - errorMessage was unused
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEnhancePrompt = async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await promptAPI.enhance({
        prompt: input,
        agent_type: agentType || 'general',
      });

      if (response.data.enhanced_prompt) {
        setInput(response.data.enhanced_prompt);
      }
    } catch (err: any) {
      console.error('Error enhancing prompt:', err);
      setError('Failed to enhance prompt. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      Array.from(files).forEach((file) => {
        formData.append('files', file);
      });

// //       const response = await knowledgeAPI.upload(formData);

      // // setMessages((prev) => [...prev, successMessage]);
    } catch (err: any) {
      console.error('Error uploading files:', err);
      // // setMessages((prev) => [...prev, errorMessage]);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      Array.from(files).forEach((file) => {
        formData.append('files', file);
      });

// //       const response = await knowledgeAPI.upload(formData);

      // // setMessages((prev) => [...prev, successMessage]);
    } catch (err: any) {
      console.error('Error uploading files:', err);
      // // setMessages((prev) => [...prev, errorMessage]);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative h-full flex flex-col">
      {/* Background Image */}
      {backgroundImage && (
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: `url(${backgroundImage})`,
            opacity: backgroundOpacity,
          }}
        />
      )}

      {/* Default Gradient Background (when no image) */}
      {!backgroundImage && (
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-lime-50 dark:from-blue-950 dark:via-gray-900 dark:to-lime-950" />
      )}

      {/* Content Overlay */}
      <div className="relative z-10 h-full flex flex-col bg-background/80 backdrop-blur-sm rounded-lg border border-border shadow-lg m-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{agentConfig.icon}</span>
            <div>
              <h2 className="text-lg font-semibold" style={{ color: agentConfig.color }}>
                {agentConfig.name}
              </h2>
              <p className="text-xs text-muted-foreground">
                {agentConfig.description}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-xs text-muted-foreground">
                {messages.length} messages
              </div>
              {messages.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  Last: {new Date(messages[messages.length - 1].timestamp).toLocaleTimeString()}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div
          className={cn(
            'flex-1 overflow-y-auto p-4 space-y-4',
            isDragging && 'border-2 border-dashed border-primary bg-primary/5'
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {messages.length === 0 && (
            <div className="h-full flex items-center justify-center p-6">
              <div className="max-w-3xl w-full space-y-8">
                {/* Welcome Section */}
                <div className="text-center space-y-4">
                  <div className="text-6xl mb-4">{agentConfig.icon}</div>
                  <h3 className="text-2xl font-bold" style={{ color: agentConfig.color }}>
                    {agentConfig.name}
                  </h3>
                  <p className="text-base text-muted-foreground max-w-2xl mx-auto">
                    {agentConfig.welcomeMessage}
                  </p>
                </div>

                {/* Capabilities Section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <Lightbulb className="h-4 w-4" style={{ color: agentConfig.color }} />
                    <span>Capabilities</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {agentConfig.capabilities.map((capability, index) => (
                      <div
                        key={index}
                        className="flex items-start gap-2 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                      >
                        <span className="text-xs mt-0.5" style={{ color: agentConfig.color }}>✓</span>
                        <span className="text-sm text-foreground">{capability}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Suggested Prompts Section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <Sparkles className="h-4 w-4" style={{ color: agentConfig.color }} />
                    <span>Try asking me...</span>
                  </div>
                  <div className="grid gap-2">
                    {agentConfig.suggestedPrompts.map((prompt, index) => (
                      <button
                        key={index}
                        onClick={() => setInput(prompt)}
                        className="text-left p-3 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-all group"
                      >
                        <span className="text-sm text-foreground group-hover:text-primary">
                          {prompt}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={cn(
                  'max-w-[80%] rounded-lg px-4 py-2',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-foreground'
                )}
              >
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                <span className="text-xs opacity-70 mt-1 block">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-muted rounded-lg px-4 py-2">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-border">
          <div className="flex items-end gap-2">
            {/* Action Buttons */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleEnhancePrompt}
                    disabled={!input.trim() || isLoading}
                  >
                    <Sparkles className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Enhance Prompt with AI</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isLoading}
                  >
                    <Upload className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Upload Documents, Files, Images, or Links</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Hidden File Input */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.docx,.png,.jpg,.jpeg,.gif"
              onChange={handleFileUpload}
              className="hidden"
            />

            {/* Text Input */}
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
              disabled={isLoading}
              className="flex-1 min-h-[80px] max-h-[200px] resize-y"
              rows={3}
            />

            {/* Send Button */}
            <Button onClick={handleSend} disabled={!input.trim() || isLoading}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

