/**
 * Help Button Component
 * Provides contextual help for each step of the Agent Builder
 */

import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';
import { Button } from '../ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../ui/popover';

interface HelpContent {
  title: string;
  description: string;
  tips?: string[];
  learnMoreUrl?: string;
}

const helpContent: Record<string, HelpContent> = {
  'basic-info': {
    title: 'Basic Information',
    description: 'Define your agent\'s identity and purpose.',
    tips: [
      'Use descriptive names like "Customer Support Agent"',
      'Be specific in your description',
      'Choose the appropriate agent type',
      'Use tags for easy organization'
    ],
    learnMoreUrl: '#basic-information'
  },
  'llm-config': {
    title: 'LLM Configuration',
    description: 'Choose the AI model and configure its behavior.',
    tips: [
      'GPT-4 for complex tasks, GPT-3.5 for simple ones',
      'Lower temperature (0.0-0.3) for factual responses',
      'Higher temperature (0.8-1.5) for creative tasks',
      'Set max_tokens based on expected response length'
    ],
    learnMoreUrl: '#llm-configuration'
  },
  'tools': {
    title: 'Tools Selection',
    description: 'Give your agent access to external capabilities.',
    tips: [
      'Only select tools you actually need',
      'Use minimal permissions',
      'Test tools individually first',
      'Read = view only, Write = modify, Execute = full control'
    ],
    learnMoreUrl: '#tools-selection'
  },
  'mcp-tools': {
    title: 'MCP Tools',
    description: 'Connect to external MCP servers for specialized capabilities.',
    tips: [
      'MCP servers must be running and accessible',
      'Test connection before adding tools',
      'MCP tools are saved to MongoDB',
      'Use for database access, API integrations, custom logic'
    ],
    learnMoreUrl: '#mcp-tools'
  },
  'skills': {
    title: 'Skills Configuration',
    description: 'Define specialized capabilities and behaviors.',
    tips: [
      'Skills provide domain-specific knowledge',
      'Can combine multiple skills',
      'Each skill has specific configuration options',
      'Don\'t add too many skills - keep focused'
    ],
    learnMoreUrl: '#skills-configuration'
  },
  'memory': {
    title: 'Memory Configuration',
    description: 'Control how your agent remembers and learns.',
    tips: [
      'Short-term: Current conversation only',
      'Long-term: Permanent storage in vector database',
      'Enable long-term for customer support',
      'Disable for stateless agents'
    ],
    learnMoreUrl: '#memory-configuration'
  },
  'prompt': {
    title: 'Prompt Engineering',
    description: 'Define your agent\'s behavior through system prompts.',
    tips: [
      'Include role, capabilities, and constraints',
      'Be specific about what agent can/cannot do',
      'Provide examples of good responses',
      'Define escalation criteria',
      'Test prompts iteratively'
    ],
    learnMoreUrl: '#prompt-engineering'
  },
  'review': {
    title: 'Review & Deploy',
    description: 'Review all settings and deploy your agent.',
    tips: [
      'Check all settings carefully',
      'Review validation warnings',
      'Save as draft to test more',
      'Deploy when confident',
      'Can edit after deployment'
    ],
    learnMoreUrl: '#review-deploy'
  }
};

interface HelpButtonProps {
  topic: string;
  className?: string;
}

export function HelpButton({ topic, className = '' }: HelpButtonProps) {
  const [open, setOpen] = useState(false);
  const content = helpContent[topic];

  if (!content) return null;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className={`h-8 w-8 ${className}`}
          title="Get help with this step"
        >
          <HelpCircle className="h-4 w-4 text-muted-foreground hover:text-primary" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-3">
          <div>
            <h4 className="font-medium text-sm mb-1">{content.title}</h4>
            <p className="text-sm text-muted-foreground">{content.description}</p>
          </div>

          {content.tips && content.tips.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground">Quick Tips:</p>
              <ul className="space-y-1">
                {content.tips.map((tip, index) => (
                  <li key={index} className="text-xs text-muted-foreground flex items-start gap-1.5">
                    <span className="text-primary mt-0.5">•</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {content.learnMoreUrl && (
            <div className="pt-2 border-t">
              <a
                href={`/docs/agent-builder-guide${content.learnMoreUrl}`}
                className="text-xs text-primary hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                Learn more in the documentation →
              </a>
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

export default HelpButton;

