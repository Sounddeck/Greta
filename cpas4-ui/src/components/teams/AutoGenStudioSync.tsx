/**
 * AutoGen Studio Sync Component
 * 
 * Import/export teams from/to AutoGen Studio
 * 
 * NOTE: This component has a TypeScript compilation issue that's being investigated.
 * For now, it's disabled to allow the UI to load.
 */

import React from 'react';

interface AutoGenStudioSyncProps {
  mode: 'import' | 'export';
  teamId?: string;
  onSuccess?: (team: any) => void;
  onCancel?: () => void;
}

/**
 * Placeholder component to prevent import errors
 * The original implementation had a TypeScript compilation error that needs investigation
 */
export default function AutoGenStudioSync({
  mode,
  teamId,
  onSuccess,
  onCancel,
}: AutoGenStudioSyncProps) {
  return (
    <div className="p-6 text-center">
      <p className="text-gray-500">AutoGen Studio Sync component is temporarily disabled</p>
    </div>
  );
}
