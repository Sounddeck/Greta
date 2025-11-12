/**
 * File Type Restrictions by Agent Type
 * Quick Win #1: Restrict file uploads based on agent specialization
 */

import type { AgentType } from '../types';

export interface FileTypeConfig {
  accept: string;
  extensions: string[];
  description: string;
  maxSize?: number; // in MB
}

export const FILE_TYPE_RESTRICTIONS: Record<AgentType, FileTypeConfig> = {
  data_analysis: {
    accept: '.csv,.xlsx,.xls,.json,.tsv,.parquet',
    extensions: ['csv', 'xlsx', 'xls', 'json', 'tsv', 'parquet'],
    description: 'Data files (CSV, Excel, JSON, TSV, Parquet)',
    maxSize: 50, // 50MB
  },
  
  coding: {
    accept: '.py,.js,.ts,.tsx,.jsx,.java,.cpp,.c,.h,.cs,.go,.rs,.rb,.php,.swift,.kt,.scala,.r,.sql,.sh,.bash,.yaml,.yml,.json,.xml,.html,.css,.scss,.sass,.md',
    extensions: ['py', 'js', 'ts', 'tsx', 'jsx', 'java', 'cpp', 'c', 'h', 'cs', 'go', 'rs', 'rb', 'php', 'swift', 'kt', 'scala', 'r', 'sql', 'sh', 'bash', 'yaml', 'yml', 'json', 'xml', 'html', 'css', 'scss', 'sass', 'md'],
    description: 'Code files (Python, JavaScript, TypeScript, Java, C++, etc.)',
    maxSize: 10, // 10MB
  },
  
  research: {
    accept: '.pdf,.txt,.docx,.doc,.md,.epub,.html,.htm',
    extensions: ['pdf', 'txt', 'docx', 'doc', 'md', 'epub', 'html', 'htm'],
    description: 'Documents (PDF, Word, Text, Markdown, HTML)',
    maxSize: 25, // 25MB
  },
  
  content_creation: {
    accept: '.txt,.docx,.doc,.md,.html,.htm,.pdf',
    extensions: ['txt', 'docx', 'doc', 'md', 'html', 'htm', 'pdf'],
    description: 'Text documents (Word, Markdown, HTML, PDF)',
    maxSize: 15, // 15MB
  },
  
  task_automation: {
    accept: '.json,.yaml,.yml,.xml,.csv,.txt,.sh,.bash,.py,.js',
    extensions: ['json', 'yaml', 'yml', 'xml', 'csv', 'txt', 'sh', 'bash', 'py', 'js'],
    description: 'Configuration and script files (JSON, YAML, XML, Scripts)',
    maxSize: 5, // 5MB
  },
  
  customer_support: {
    accept: '.txt,.pdf,.docx,.doc,.csv,.json,.html,.htm',
    extensions: ['txt', 'pdf', 'docx', 'doc', 'csv', 'json', 'html', 'htm'],
    description: 'Support documents (Text, PDF, Word, CSV, JSON)',
    maxSize: 10, // 10MB
  },
  
  sales: {
    accept: '.pdf,.docx,.doc,.xlsx,.xls,.csv,.pptx,.ppt,.txt',
    extensions: ['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'pptx', 'ppt', 'txt'],
    description: 'Sales materials (PDF, Word, Excel, PowerPoint)',
    maxSize: 20, // 20MB
  },
  
  marketing: {
    accept: '.pdf,.docx,.doc,.xlsx,.xls,.pptx,.ppt,.png,.jpg,.jpeg,.gif,.svg,.mp4,.mov,.txt,.csv',
    extensions: ['pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'png', 'jpg', 'jpeg', 'gif', 'svg', 'mp4', 'mov', 'txt', 'csv'],
    description: 'Marketing materials (Documents, Images, Videos)',
    maxSize: 100, // 100MB for videos
  },
  
  finance: {
    accept: '.xlsx,.xls,.csv,.pdf,.json,.txt',
    extensions: ['xlsx', 'xls', 'csv', 'pdf', 'json', 'txt'],
    description: 'Financial documents (Excel, CSV, PDF, JSON)',
    maxSize: 30, // 30MB
  },
  
  general: {
    accept: '.pdf,.txt,.docx,.doc,.xlsx,.xls,.csv,.png,.jpg,.jpeg,.gif,.json,.md,.html,.htm',
    extensions: ['pdf', 'txt', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'png', 'jpg', 'jpeg', 'gif', 'json', 'md', 'html', 'htm'],
    description: 'Common file types (Documents, Images, Data)',
    maxSize: 25, // 25MB
  },
};

/**
 * Get file type configuration for a specific agent type
 */
export function getFileTypeConfig(agentType: AgentType | null): FileTypeConfig {
  if (!agentType) {
    return FILE_TYPE_RESTRICTIONS.general;
  }
  return FILE_TYPE_RESTRICTIONS[agentType] || FILE_TYPE_RESTRICTIONS.general;
}

/**
 * Validate if a file is allowed for a specific agent type
 */
export function validateFile(file: File, agentType: AgentType | null): { valid: boolean; error?: string } {
  const config = getFileTypeConfig(agentType);
  
  // Check file extension
  const extension = file.name.split('.').pop()?.toLowerCase();
  if (!extension || !config.extensions.includes(extension)) {
    return {
      valid: false,
      error: `File type .${extension} is not supported for this agent. Supported types: ${config.description}`,
    };
  }
  
  // Check file size
  const fileSizeMB = file.size / (1024 * 1024);
  if (config.maxSize && fileSizeMB > config.maxSize) {
    return {
      valid: false,
      error: `File size (${fileSizeMB.toFixed(1)}MB) exceeds maximum allowed size of ${config.maxSize}MB`,
    };
  }
  
  return { valid: true };
}

/**
 * Validate multiple files
 */
export function validateFiles(files: File[], agentType: AgentType | null): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  for (const file of files) {
    const result = validateFile(file, agentType);
    if (!result.valid && result.error) {
      errors.push(`${file.name}: ${result.error}`);
    }
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

