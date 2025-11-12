/**
 * Workflow Builder Page
 * 
 * Visual workflow builder interface
 */

import React from 'react';
import WorkflowCanvas from '../components/workflow/WorkflowCanvas';
import type { Node, Edge } from '@xyflow/react';

export default function WorkflowBuilder() {
  const handleSave = (nodes: Node[], edges: Edge[]) => {
    console.log('Saving workflow:', { nodes, edges });
    // TODO: Implement save to backend
  };

  return (
    <div className="h-full flex flex-col">
      {/* Page Header */}
      <div className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Workflow Builder
        </h1>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Design and automate your workflows visually
        </p>
      </div>

      {/* Workflow Canvas */}
      <div className="flex-1 overflow-hidden">
        <WorkflowCanvas onSave={handleSave} />
      </div>
    </div>
  );
}

