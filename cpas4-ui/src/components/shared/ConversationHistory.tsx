/**
 * Conversation History Component
 * Quick Win #4: Save and retrieve past conversations per agent type
 */

import { useState, useEffect } from 'react';
import { History, Trash2, Download, Clock, MessageSquare } from 'lucide-react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../ui/dialog';
import { toast } from 'sonner';
import type { Message, AgentType } from '../../types';
import { ProgressIndicator } from './ProgressIndicator';

interface Conversation {
  id: string;
  agent_type: string;
  title: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
  message_count: number;
}

interface ConversationHistoryProps {
  agentType: AgentType | null;
  onLoadConversation: (messages: Message[]) => void;
  variant?: 'button' | 'panel';
}

export function ConversationHistory({ agentType, onLoadConversation, variant = 'button' }: ConversationHistoryProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (open && agentType) {
      loadConversations();
    }
  }, [open, agentType]);

  const loadConversations = async () => {
    if (!agentType) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/v1/conversations/list/${agentType}`);
      if (!response.ok) throw new Error('Failed to load conversations');

      const data = await response.json();
      setConversations(data.conversations || []);
    } catch (error) {
      console.error('Error loading conversations:', error);
      toast.error('Failed to load conversation history');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadConversation = (conversation: Conversation) => {
    onLoadConversation(conversation.messages);
    setOpen(false);
    toast.success(`Loaded conversation: ${conversation.title}`);
  };

  const handleDeleteConversation = async (conversationId: string, event: React.MouseEvent) => {
    event.stopPropagation();

    if (!confirm('Are you sure you want to delete this conversation?')) return;

    try {
      const response = await fetch(`/api/v1/conversations/${conversationId}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete conversation');

      setConversations(prev => prev.filter(c => c.id !== conversationId));
      toast.success('Conversation deleted');
    } catch (error) {
      console.error('Error deleting conversation:', error);
      toast.error('Failed to delete conversation');
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  if (variant === 'panel') {
    return (
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <History className="h-4 w-4 text-primary" />
            Recent Conversations
          </h3>
          <Button variant="ghost" size="sm" onClick={loadConversations}>
            Refresh
          </Button>
        </div>

        {loading ? (
          <ProgressIndicator message="Loading conversations..." variant="minimal" />
        ) : conversations.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No saved conversations yet
          </p>
        ) : (
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className="p-3 rounded-lg border border-border hover:bg-muted/50 cursor-pointer transition-colors group"
                onClick={() => handleLoadConversation(conversation)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{conversation.title}</p>
                    <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <MessageSquare className="h-3 w-3" />
                        {conversation.message_count}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatDate(conversation.updated_at)}
                      </span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => handleDeleteConversation(conversation.id, e)}
                  >
                    <Trash2 className="h-3 w-3 text-destructive" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    );
  }

  // Button variant
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <History className="h-4 w-4 mr-2" />
          History
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Conversation History</DialogTitle>
          <DialogDescription>
            Load a previous conversation to continue where you left off
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="py-8">
              <ProgressIndicator message="Loading conversations..." variant="default" />
            </div>
          ) : conversations.length === 0 ? (
            <div className="text-center py-12">
              <History className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No saved conversations yet</p>
              <p className="text-sm text-muted-foreground mt-2">
                Your conversation history will appear here
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {conversations.map((conversation) => (
                <Card
                  key={conversation.id}
                  className="p-4 hover:bg-muted/50 cursor-pointer transition-colors group"
                  onClick={() => handleLoadConversation(conversation)}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium truncate">{conversation.title}</h4>
                      <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <MessageSquare className="h-4 w-4" />
                          {conversation.message_count} messages
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-4 w-4" />
                          {formatDate(conversation.updated_at)}
                        </span>
                      </div>
                      {conversation.messages.length > 0 && (
                        <p className="text-sm text-muted-foreground mt-2 line-clamp-2">
                          {conversation.messages[0].content}
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={(e) => handleDeleteConversation(conversation.id, e)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Save Conversation Button Component
 */
interface SaveConversationProps {
  messages: Message[];
  agentType: AgentType | null;
  onSaved?: () => void;
}

export function SaveConversationButton({ messages, agentType, onSaved }: SaveConversationProps) {
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (!agentType || messages.length === 0) {
      toast.error('No messages to save');
      return;
    }

    setSaving(true);
    try {
      const response = await fetch('/api/v1/conversations/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_type: agentType,
          messages,
        }),
      });

      if (!response.ok) throw new Error('Failed to save conversation');

      const data = await response.json();
      toast.success(`Conversation saved: ${data.title}`);
      onSaved?.();
    } catch (error) {
      console.error('Error saving conversation:', error);
      toast.error('Failed to save conversation');
    } finally {
      setSaving(false);
    }
  };

  if (messages.length === 0) return null;

  return (
    <Button variant="outline" size="sm" onClick={handleSave} disabled={saving}>
      <Download className="h-4 w-4 mr-2" />
      {saving ? 'Saving...' : 'Save'}
    </Button>
  );
}

