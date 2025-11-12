/**
 * Export Conversation Component
 * Quick Win #2: Export conversation to TXT, PDF, or JSON
 */

import { useState } from 'react';
import { Download, FileText, FileJson, FileType } from 'lucide-react';
import { Button } from '../ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { toast } from 'sonner';
import type { Message, AgentType } from '../../types';
import { getAgentConfig } from '../../config/agentConfig';

interface ExportConversationProps {
  messages: Message[];
  agentType: AgentType | null;
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'default' | 'sm' | 'lg' | 'icon';
}

export function ExportConversation({ messages, agentType, variant = 'outline', size = 'default' }: ExportConversationProps) {
  const [isExporting, setIsExporting] = useState(false);
  const agentConfig = getAgentConfig(agentType);

  const exportToTXT = () => {
    setIsExporting(true);
    try {
      const content = generateTextContent(messages, agentConfig.name);
      downloadFile(content, `${agentConfig.name.replace(/\s+/g, '_')}_conversation_${Date.now()}.txt`, 'text/plain');
      toast.success('Conversation exported as TXT');
    } catch (error) {
      toast.error('Failed to export conversation');
      console.error(error);
    } finally {
      setIsExporting(false);
    }
  };

  const exportToJSON = () => {
    setIsExporting(true);
    try {
      const data = {
        agentType,
        agentName: agentConfig.name,
        exportedAt: new Date().toISOString(),
        messageCount: messages.length,
        messages: messages.map(msg => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
        })),
      };
      const content = JSON.stringify(data, null, 2);
      downloadFile(content, `${agentConfig.name.replace(/\s+/g, '_')}_conversation_${Date.now()}.json`, 'application/json');
      toast.success('Conversation exported as JSON');
    } catch (error) {
      toast.error('Failed to export conversation');
      console.error(error);
    } finally {
      setIsExporting(false);
    }
  };

  const exportToMarkdown = () => {
    setIsExporting(true);
    try {
      const content = generateMarkdownContent(messages, agentConfig.name);
      downloadFile(content, `${agentConfig.name.replace(/\s+/g, '_')}_conversation_${Date.now()}.md`, 'text/markdown');
      toast.success('Conversation exported as Markdown');
    } catch (error) {
      toast.error('Failed to export conversation');
      console.error(error);
    } finally {
      setIsExporting(false);
    }
  };

  if (messages.length === 0) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant={variant} size={size} disabled={isExporting}>
          <Download className="h-4 w-4 mr-2" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={exportToTXT}>
          <FileText className="h-4 w-4 mr-2" />
          Export as TXT
        </DropdownMenuItem>
        <DropdownMenuItem onClick={exportToMarkdown}>
          <FileType className="h-4 w-4 mr-2" />
          Export as Markdown
        </DropdownMenuItem>
        <DropdownMenuItem onClick={exportToJSON}>
          <FileJson className="h-4 w-4 mr-2" />
          Export as JSON
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// Helper functions

function generateTextContent(messages: Message[], agentName: string): string {
  const header = `${agentName} - Conversation Export\n`;
  const separator = '='.repeat(60) + '\n';
  const timestamp = `Exported: ${new Date().toLocaleString()}\n`;
  const messageCount = `Total Messages: ${messages.length}\n`;
  
  let content = header + separator + timestamp + messageCount + separator + '\n';
  
  messages.forEach((msg, index) => {
    const role = msg.role === 'user' ? 'USER' : 'ASSISTANT';
    const time = new Date(msg.timestamp).toLocaleString();
    content += `[${index + 1}] ${role} - ${time}\n`;
    content += `${msg.content}\n\n`;
    content += '-'.repeat(60) + '\n\n';
  });
  
  return content;
}

function generateMarkdownContent(messages: Message[], agentName: string): string {
  let content = `# ${agentName} - Conversation Export\n\n`;
  content += `**Exported:** ${new Date().toLocaleString()}  \n`;
  content += `**Total Messages:** ${messages.length}\n\n`;
  content += `---\n\n`;
  
  messages.forEach((msg, index) => {
    const role = msg.role === 'user' ? '👤 **User**' : '🤖 **Assistant**';
    const time = new Date(msg.timestamp).toLocaleString();
    content += `## Message ${index + 1}\n\n`;
    content += `${role} - *${time}*\n\n`;
    content += `${msg.content}\n\n`;
    content += `---\n\n`;
  });
  
  return content;
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

