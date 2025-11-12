/**
 * Agent Builder Interactive Tutorial
 * Provides step-by-step guided tour of the Agent Builder features
 */

import React, { useState } from 'react';
import { X, ChevronLeft, ChevronRight, BookOpen, Lightbulb, AlertCircle } from 'lucide-react';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';

interface TutorialStep {
  title: string;
  description: string;
  tips?: string[];
  warnings?: string[];
  example?: string;
}

const tutorialSteps: Record<number, TutorialStep> = {
  1: {
    title: 'Basic Information',
    description: 'Define your agent\'s identity and purpose. This is the foundation of your agent.',
    tips: [
      'Use descriptive, action-oriented names (e.g., "Customer Support Agent")',
      'Be specific in your description about what the agent can and cannot do',
      'Choose the agent type that best matches your use case',
      'Use consistent tags for easy organization'
    ],
    warnings: [
      'Avoid generic names like "Agent 1" or "Test Agent"',
      'Don\'t leave the description vague - be specific!'
    ],
    example: `Name: "Product Support Agent"
Description: "Helps customers with product questions, troubleshooting, and returns"
Type: Assistant
Tags: ["support", "products", "customer-service"]`
  },
  2: {
    title: 'LLM Configuration',
    description: 'Choose the AI model and configure its behavior. This determines how your agent thinks and responds.',
    tips: [
      'GPT-4: Best for complex reasoning, slower, more expensive',
      'GPT-3.5-turbo: Fast and cheap, good for simple tasks',
      'Claude 3 Opus: Excellent for complex analysis and reasoning',
      'Temperature 0.0-0.3: Factual and consistent (support, data)',
      'Temperature 0.4-0.7: Balanced (general use)',
      'Temperature 0.8-1.5: Creative (content, brainstorming)'
    ],
    warnings: [
      'High temperature can lead to inconsistent responses',
      'Low max_tokens may cut off responses mid-sentence',
      'Expensive models can quickly increase costs'
    ],
    example: `Provider: OpenAI
Model: gpt-4
Temperature: 0.5 (balanced)
Max Tokens: 1500 (medium responses)`
  },
  3: {
    title: 'Tools Selection',
    description: 'Give your agent access to external capabilities. Tools extend what your agent can do beyond just text generation.',
    tips: [
      'Only select tools your agent actually needs',
      'Use minimal permissions (principle of least privilege)',
      'Test tools individually before combining them',
      'Read permission: View data only',
      'Write permission: Create and modify data',
      'Execute permission: Run operations (most powerful)'
    ],
    warnings: [
      'Don\'t select all tools "just in case" - this can confuse the agent',
      'Execute permission is powerful - use with caution',
      'Some tools require specific backend services to be running'
    ],
    example: `Selected Tools:
- web-search (read): Search the internet
- codebase-retrieval (read): Search codebase
- save-file (write): Create files`
  },
  4: {
    title: 'MCP Tools',
    description: 'Connect to external MCP servers for specialized capabilities. MCP (Model Context Protocol) allows your agent to use custom integrations.',
    tips: [
      'MCP servers must be running and accessible',
      'Test connection before adding tools',
      'MCP tools are saved to MongoDB (persistent)',
      'Use MCP for database access, API integrations, custom logic'
    ],
    warnings: [
      'If an MCP server goes offline, tools will fail',
      'MCP servers may disappear if localStorage is cleared - we\'ve fixed this!',
      'Always verify MCP server URL is correct'
    ],
    example: `MCP Server:
Name: "Database Server"
URL: "http://localhost:3000"
Tools: ["query_db", "insert_record"]`
  },
  5: {
    title: 'Skills Configuration',
    description: 'Define specialized capabilities and behaviors. Skills provide domain-specific knowledge.',
    tips: [
      'Skills can be combined for complex agents',
      'Each skill has specific configuration options',
      'Research skill: Web research and fact-checking',
      'Coding skill: Software development assistance',
      'Writing skill: Content creation and editing'
    ],
    warnings: [
      'Too many skills can make the agent unfocused',
      'Some skills may conflict with each other'
    ],
    example: `Selected Skills:
- Research (deep search, academic sources)
- Writing (blog posts, professional tone)`
  },
  6: {
    title: 'Memory Configuration',
    description: 'Control how your agent remembers and learns. Memory allows your agent to maintain context and learn from interactions.',
    tips: [
      'Short-term memory: Current conversation only',
      'Long-term memory: Permanent storage in vector database',
      'Enable long-term memory for customer support agents',
      'Use larger memory for complex reasoning tasks',
      'Disable memory for stateless agents (calculators, converters)'
    ],
    warnings: [
      'Large memory increases processing time',
      'Long-term memory requires ChromaDB to be running',
      'Memory can accumulate sensitive information - consider retention policies'
    ],
    example: `Memory Configuration:
Short-term: Enabled (4K tokens)
Long-term: Enabled (ChromaDB)
Retention: Keep important only`
  },
  7: {
    title: 'Prompt Engineering',
    description: 'Define your agent\'s behavior through system prompts. This is where you set personality, tone, and guidelines.',
    tips: [
      'Include role definition, capabilities, and constraints',
      'Be specific about what the agent can and cannot do',
      'Provide examples of good responses',
      'Define escalation criteria',
      'Set tone and style guidelines',
      'Test prompts iteratively'
    ],
    warnings: [
      'Vague prompts lead to unpredictable behavior',
      'Conflicting instructions confuse the agent',
      'Too long prompts may be truncated'
    ],
    example: `You are a customer support agent for Acme Corp.

Your role:
- Answer product questions
- Help troubleshoot issues
- Process returns under $500

Guidelines:
- Be friendly and empathetic
- Ask clarifying questions
- Provide step-by-step instructions

Constraints:
- Cannot process refunds over $500
- Cannot access payment information`
  },
  8: {
    title: 'Review & Deploy',
    description: 'Review all settings and deploy your agent. This is your final check before making the agent available.',
    tips: [
      'Review all settings carefully',
      'Check validation warnings and errors',
      'Save as draft if you need to test more',
      'Deploy when you\'re confident in the configuration',
      'You can edit the agent after deployment'
    ],
    warnings: [
      'Deployed agents are immediately available for use',
      'Changes to deployed agents may affect ongoing conversations',
      'Always test in a safe environment first'
    ],
    example: `Validation: 95% complete
Errors: 0
Warnings: 1 (Consider adding more tools)
Status: Ready to deploy`
  }
};

interface AgentBuilderTutorialProps {
  currentStep: number;
  onClose: () => void;
}

export function AgentBuilderTutorial({ currentStep, onClose }: AgentBuilderTutorialProps) {
  const [step, setStep] = useState(currentStep);
  const tutorial = tutorialSteps[step];

  if (!tutorial) return null;

  const handlePrevious = () => {
    if (step > 1) setStep(step - 1);
  };

  const handleNext = () => {
    if (step < 8) setStep(step + 1);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <CardHeader className="relative">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              <div>
                <CardTitle>Step {step}: {tutorial.title}</CardTitle>
                <CardDescription>{tutorial.description}</CardDescription>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="absolute top-4 right-4"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Tips Section */}
          {tutorial.tips && tutorial.tips.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium text-green-600 dark:text-green-400">
                <Lightbulb className="h-4 w-4" />
                <span>Tips & Best Practices</span>
              </div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                {tutorial.tips.map((tip, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-green-600 dark:text-green-400 mt-0.5">✓</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings Section */}
          {tutorial.warnings && tutorial.warnings.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium text-amber-600 dark:text-amber-400">
                <AlertCircle className="h-4 w-4" />
                <span>Common Pitfalls</span>
              </div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                {tutorial.warnings.map((warning, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-amber-600 dark:text-amber-400 mt-0.5">⚠</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Example Section */}
          {tutorial.example && (
            <div className="space-y-2">
              <div className="text-sm font-medium">Example Configuration</div>
              <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
                {tutorial.example}
              </pre>
            </div>
          )}

          {/* Navigation */}
          <div className="flex items-center justify-between pt-4 border-t">
            <Button
              variant="outline"
              onClick={handlePrevious}
              disabled={step === 1}
            >
              <ChevronLeft className="h-4 w-4 mr-2" />
              Previous
            </Button>

            <div className="text-sm text-muted-foreground">
              Step {step} of 8
            </div>

            <Button
              onClick={handleNext}
              disabled={step === 8}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
          </div>

          {/* Close Button */}
          <div className="flex justify-center pt-2">
            <Button variant="ghost" onClick={onClose} className="text-sm">
              Close Tutorial
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default AgentBuilderTutorial;

