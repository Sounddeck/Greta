// @ts-nocheck
import React, { useState } from 'react';
import { AgentVersionHistory } from './AgentVersionHistory';
import { VersionComparison } from './VersionComparison';
import { GitBranch } from 'lucide-react';

interface AgentVersionManagerProps {
  agentId: string;
  currentVersion?: number;
  onVersionChange?: () => void;
}

export function AgentVersionManager({ agentId, currentVersion, onVersionChange }: AgentVersionManagerProps) {
  const [compareVersions, setCompareVersions] = useState<{ versionIdA: string; versionIdB: string } | null>(null);

  const handleRollback = () => {
    // Refresh agent data after rollback
    onVersionChange?.();
  };

  const handleCompare = (versionIdA: string, versionIdB: string) => {
    setCompareVersions({ versionIdA, versionIdB });
  };

  const handleCloseComparison = () => {
    setCompareVersions(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3 pb-4 border-b">
        <GitBranch className="w-6 h-6 text-blue-600" />
        <div>
          <h2 className="text-2xl font-bold">Version Control</h2>
          <p className="text-sm text-gray-600 mt-1">
            Track changes, compare versions, and rollback when needed
          </p>
        </div>
      </div>

      {/* Version History */}
      <AgentVersionHistory
        agentId={agentId}
        currentVersion={currentVersion}
        onRollback={handleRollback}
        onCompare={handleCompare}
      />

      {/* Version Comparison Modal */}
      {compareVersions && (
        <VersionComparison
          versionIdA={compareVersions.versionIdA}
          versionIdB={compareVersions.versionIdB}
          onClose={handleCloseComparison}
        />
      )}
    </div>
  );
}

