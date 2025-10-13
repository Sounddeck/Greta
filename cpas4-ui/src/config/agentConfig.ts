import type { AgentType } from '../types';

export interface AgentConfig {
  name: string;
  icon: string;
  description: string;
  capabilities: string[];
  suggestedPrompts: string[];
  welcomeMessage: string;
  color: string;
}

export const AGENT_CONFIGS: Record<AgentType, AgentConfig> = {
  research: {
    name: 'Research Agent',
    icon: '🔍',
    description: 'Specialized in conducting comprehensive research, gathering information, and analyzing data from multiple sources.',
    capabilities: [
      'Web research and information gathering',
      'Academic paper analysis',
      'Market research and competitive analysis',
      'Data synthesis and summarization',
      'Fact-checking and verification',
      'Trend analysis and insights',
    ],
    suggestedPrompts: [
      'Research the latest trends in AI and machine learning',
      'Find and summarize academic papers on quantum computing',
      'Analyze competitors in the SaaS market',
      'What are the emerging technologies in 2025?',
      'Research best practices for remote team management',
    ],
    welcomeMessage: 'Hello! I\'m your Research Agent. I can help you gather information, analyze data, and provide comprehensive research on any topic. What would you like to research today?',
    color: '#3B82F6', // Blue
  },
  
  coding: {
    name: 'Coding Agent',
    icon: '💻',
    description: 'Expert in software development, code generation, debugging, and technical problem-solving.',
    capabilities: [
      'Code generation in multiple languages',
      'Debugging and error resolution',
      'Code review and optimization',
      'API integration and development',
      'Database design and queries',
      'Testing and documentation',
    ],
    suggestedPrompts: [
      'Write a Python function to process CSV files',
      'Help me debug this JavaScript error',
      'Create a REST API endpoint for user authentication',
      'Optimize this SQL query for better performance',
      'Generate unit tests for my React component',
    ],
    welcomeMessage: 'Hi! I\'m your Coding Agent. I can help you write, debug, and optimize code in any programming language. What are you building today?',
    color: '#10B981', // Green
  },
  
  data_analysis: {
    name: 'Data Analysis Agent',
    icon: '📊',
    description: 'Specialized in data processing, statistical analysis, visualization, and insights generation.',
    capabilities: [
      'Statistical analysis and modeling',
      'Data visualization and dashboards',
      'Predictive analytics',
      'Data cleaning and preprocessing',
      'Pattern recognition',
      'Business intelligence reporting',
    ],
    suggestedPrompts: [
      'Analyze this sales data and identify trends',
      'Create a visualization for customer demographics',
      'Predict next quarter\'s revenue based on historical data',
      'Find correlations in this dataset',
      'Generate a comprehensive analytics report',
    ],
    welcomeMessage: 'Welcome! I\'m your Data Analysis Agent. I can help you analyze data, create visualizations, and extract meaningful insights. What data would you like to explore?',
    color: '#8B5CF6', // Purple
  },
  
  content_creation: {
    name: 'Content Creation Agent',
    icon: '✍️',
    description: 'Expert in creating engaging content, copywriting, and creative writing across various formats.',
    capabilities: [
      'Blog posts and articles',
      'Marketing copy and ads',
      'Social media content',
      'Email campaigns',
      'Product descriptions',
      'Creative storytelling',
    ],
    suggestedPrompts: [
      'Write a blog post about sustainable technology',
      'Create engaging social media posts for product launch',
      'Draft an email campaign for our new service',
      'Write compelling product descriptions',
      'Generate creative headlines for our marketing campaign',
    ],
    welcomeMessage: 'Hello! I\'m your Content Creation Agent. I can help you create compelling content for any purpose. What would you like me to write today?',
    color: '#F59E0B', // Amber
  },
  
  task_automation: {
    name: 'Task Automation Agent',
    icon: '⚙️',
    description: 'Specialized in automating workflows, creating scripts, and optimizing business processes.',
    capabilities: [
      'Workflow automation',
      'Script generation',
      'Process optimization',
      'Integration setup',
      'Scheduled task management',
      'Efficiency analysis',
    ],
    suggestedPrompts: [
      'Automate my daily report generation',
      'Create a script to backup files automatically',
      'Set up automated email responses',
      'Optimize our customer onboarding workflow',
      'Build an automation for data synchronization',
    ],
    welcomeMessage: 'Hi! I\'m your Task Automation Agent. I can help you automate repetitive tasks and streamline your workflows. What would you like to automate?',
    color: '#6366F1', // Indigo
  },
  
  customer_support: {
    name: 'Customer Support Agent',
    icon: '💬',
    description: 'Expert in customer service, issue resolution, and providing helpful support responses.',
    capabilities: [
      'Customer inquiry handling',
      'Issue troubleshooting',
      'Knowledge base creation',
      'Support ticket management',
      'FAQ generation',
      'Customer satisfaction analysis',
    ],
    suggestedPrompts: [
      'Help me respond to this customer complaint',
      'Create a knowledge base article for common issues',
      'Draft a response to a refund request',
      'Generate FAQs for our new product',
      'Analyze customer feedback and suggest improvements',
    ],
    welcomeMessage: 'Welcome! I\'m your Customer Support Agent. I can help you provide excellent customer service and resolve issues efficiently. How can I assist you today?',
    color: '#EC4899', // Pink
  },
  
  sales: {
    name: 'Sales Agent',
    icon: '💰',
    description: 'Specialized in sales strategies, lead generation, and customer relationship management.',
    capabilities: [
      'Sales pitch creation',
      'Lead qualification',
      'Proposal writing',
      'CRM management',
      'Sales forecasting',
      'Objection handling',
    ],
    suggestedPrompts: [
      'Create a sales pitch for our new product',
      'Help me qualify these leads',
      'Write a proposal for this enterprise client',
      'Generate follow-up emails for prospects',
      'Analyze our sales pipeline and suggest improvements',
    ],
    welcomeMessage: 'Hello! I\'m your Sales Agent. I can help you close deals, manage leads, and grow your revenue. What sales challenge can I help you with?',
    color: '#10B981', // Emerald
  },
  
  marketing: {
    name: 'Marketing Agent',
    icon: '📢',
    description: 'Expert in marketing strategies, campaign planning, and brand development.',
    capabilities: [
      'Marketing strategy development',
      'Campaign planning and execution',
      'Brand positioning',
      'SEO and content marketing',
      'Social media strategy',
      'Marketing analytics',
    ],
    suggestedPrompts: [
      'Create a marketing strategy for Q1',
      'Plan a product launch campaign',
      'Develop our brand messaging',
      'Optimize our SEO strategy',
      'Analyze our marketing ROI',
    ],
    welcomeMessage: 'Hi! I\'m your Marketing Agent. I can help you create effective marketing strategies and campaigns. What marketing goals are you working on?',
    color: '#F59E0B', // Orange
  },
  
  finance: {
    name: 'Finance Agent',
    icon: '💵',
    description: 'Specialized in financial analysis, budgeting, forecasting, and financial planning.',
    capabilities: [
      'Financial analysis and reporting',
      'Budget planning and tracking',
      'Revenue forecasting',
      'Cost optimization',
      'Investment analysis',
      'Financial modeling',
    ],
    suggestedPrompts: [
      'Analyze our quarterly financial performance',
      'Create a budget plan for next year',
      'Forecast revenue for Q2',
      'Identify cost-saving opportunities',
      'Build a financial model for our new project',
    ],
    welcomeMessage: 'Welcome! I\'m your Finance Agent. I can help you with financial analysis, budgeting, and planning. What financial insights do you need?',
    color: '#059669', // Green
  },
  
  general: {
    name: 'General Agent',
    icon: '🤖',
    description: 'Versatile AI assistant capable of handling a wide range of tasks and queries.',
    capabilities: [
      'General question answering',
      'Task assistance',
      'Information lookup',
      'Problem-solving',
      'Brainstorming',
      'Multi-domain support',
    ],
    suggestedPrompts: [
      'Help me brainstorm ideas for my project',
      'Explain how blockchain technology works',
      'What are the best practices for time management?',
      'Summarize this article for me',
      'Give me advice on improving productivity',
    ],
    welcomeMessage: 'Hello! I\'m your General Agent. I can help you with a wide variety of tasks and questions. What can I do for you today?',
    color: '#6B7280', // Gray
  },
};

export function getAgentConfig(agentType: AgentType | null): AgentConfig {
  if (!agentType) {
    return AGENT_CONFIGS.general;
  }
  return AGENT_CONFIGS[agentType] || AGENT_CONFIGS.general;
}

