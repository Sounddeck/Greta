// @ts-nocheck
import React, { useState, useEffect } from 'react';
import { agentBuilderAPI } from '../../lib/api';
import { toast } from 'sonner';
import { Clock, GitBranch, RotateCcw, Eye, GitCompare, Tag, User, Calendar } from 'lucide-react';

interface AgentVersion {
  version_id: string;
  agent_id: string;
  version_number: number;
  changes: string;
  description?: string;
  tags: string[];
  configuration_snapshot: any;
  code_snapshot?: string;
  is_active: boolean;
  created_at: string;
  created_by: string;
}

interface AgentVersionHistoryProps {
  agentId: string;
  currentVersion?: number;
  onRollback?: (versionId: string) => void;
  onCompare?: (versionIdA: string, versionIdB: string) => void;
}

export function AgentVersionHistory({ agentId, currentVersion, onRollback, onCompare }: AgentVersionHistoryProps) {
  const [versions, setVersions] = useState<AgentVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const [showRollbackConfirm, setShowRollbackConfirm] = useState<string | null>(null);

  useEffect(() => {
    loadVersions();
  }, [agentId]);

  const loadVersions = async () => {
    setLoading(true);
    try {
      const response = await agentBuilderAPI.listVersions(agentId, 50);
      setVersions(response.data);
    } catch (error) {
      console.error('Failed to load versions:', error);
      toast.error('Failed to load version history');
    } finally {
      setLoading(false);
    }
  };

  const handleRollback = async (versionId: string) => {
    try {
      await agentBuilderAPI.rollbackVersion(agentId, versionId);
      toast.success('Successfully rolled back to version');
      setShowRollbackConfirm(null);
      loadVersions();
      onRollback?.(versionId);
    } catch (error) {
      console.error('Rollback failed:', error);
      toast.error('Failed to rollback version');
    }
  };

  const handleCompare = () => {
    if (selectedVersions.length === 2) {
      onCompare?.(selectedVersions[0], selectedVersions[1]);
    }
  };

  const toggleVersionSelection = (versionId: string) => {
    setSelectedVersions(prev => {
      if (prev.includes(versionId)) {
        return prev.filter(id => id !== versionId);
      } else if (prev.length < 2) {
        return [...prev, versionId];
      } else {
        // Replace oldest selection
        return [prev[1], versionId];
      }
    });
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-gray-600" />
          <h3 className="text-lg font-semibold">Version History</h3>
          <span className="text-sm text-gray-500">({versions.length} versions)</span>
        </div>
        
        {selectedVersions.length === 2 && (
          <button
            onClick={handleCompare}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <GitCompare className="w-4 h-4" />
            Compare Selected
          </button>
        )}
      </div>

      {/* Timeline */}
      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-200"></div>

        {/* Version items */}
        <div className="space-y-4">
          {versions.map((version, index) => {
            const isSelected = selectedVersions.includes(version.version_id);
            const isCurrent = version.version_number === currentVersion;

            return (
              <div
                key={version.version_id}
                className={`relative pl-14 pr-4 py-4 rounded-lg border transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50'
                    : isCurrent
                    ? 'border-green-500 bg-green-50'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                {/* Timeline dot */}
                <div
                  className={`absolute left-4 top-6 w-4 h-4 rounded-full border-2 ${
                    isCurrent
                      ? 'bg-green-500 border-green-600'
                      : isSelected
                      ? 'bg-blue-500 border-blue-600'
                      : 'bg-white border-gray-300'
                  }`}
                ></div>

                {/* Version content */}
                <div className="space-y-2">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <h4 className="font-semibold text-gray-900">
                          Version {version.version_number}
                        </h4>
                        {isCurrent && (
                          <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 rounded">
                            Current
                          </span>
                        )}
                        {version.tags.map(tag => (
                          <span
                            key={tag}
                            className="px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{version.changes}</p>
                      {version.description && (
                        <p className="text-sm text-gray-500 mt-1">{version.description}</p>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2 ml-4">
                      <button
                        onClick={() => toggleVersionSelection(version.version_id)}
                        className={`p-2 rounded-lg transition-colors ${
                          isSelected
                            ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                        title="Select for comparison"
                      >
                        <Eye className="w-4 h-4" />
                      </button>

                      {!isCurrent && (
                        <button
                          onClick={() => setShowRollbackConfirm(version.version_id)}
                          className="p-2 bg-orange-100 text-orange-700 rounded-lg hover:bg-orange-200 transition-colors"
                          title="Rollback to this version"
                        >
                          <RotateCcw className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Metadata */}
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      {formatDate(version.created_at)}
                    </div>
                    <div className="flex items-center gap-1">
                      <User className="w-3 h-3" />
                      {version.created_by}
                    </div>
                  </div>
                </div>

                {/* Rollback confirmation */}
                {showRollbackConfirm === version.version_id && (
                  <div className="mt-4 p-4 bg-orange-50 border border-orange-200 rounded-lg">
                    <p className="text-sm text-orange-900 mb-3">
                      Are you sure you want to rollback to version {version.version_number}?
                      This will create a backup of the current version.
                    </p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleRollback(version.version_id)}
                        className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors text-sm"
                      >
                        Confirm Rollback
                      </button>
                      <button
                        onClick={() => setShowRollbackConfirm(null)}
                        className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors text-sm"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {versions.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <GitBranch className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No version history available</p>
          <p className="text-sm mt-1">Versions will be created automatically when you update the agent</p>
        </div>
      )}
    </div>
  );
}

