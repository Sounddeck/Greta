// @ts-nocheck
import React, { useState, useEffect } from 'react';
import { agentBuilderAPI } from '../../lib/api';
import { toast } from 'sonner';
import { GitCompare, X, ArrowRight, Plus, Minus, Edit } from 'lucide-react';

interface VersionDiff {
  version_a_id: string;
  version_a_number: number;
  version_b_id: string;
  version_b_number: number;
  changes: Record<string, any>;
  summary: string;
}

interface VersionComparisonProps {
  versionIdA: string;
  versionIdB: string;
  onClose: () => void;
}

export function VersionComparison({ versionIdA, versionIdB, onClose }: VersionComparisonProps) {
  const [diff, setDiff] = useState<VersionDiff | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadComparison();
  }, [versionIdA, versionIdB]);

  const loadComparison = async () => {
    setLoading(true);
    try {
      const response = await agentBuilderAPI.compareVersions(versionIdA, versionIdB);
      setDiff(response.data);
    } catch (error) {
      console.error('Failed to load comparison:', error);
      toast.error('Failed to compare versions');
    } finally {
      setLoading(false);
    }
  };

  const renderChangeValue = (value: any) => {
    if (typeof value === 'object' && value !== null) {
      return <pre className="text-xs bg-gray-50 p-2 rounded overflow-x-auto">{JSON.stringify(value, null, 2)}</pre>;
    }
    return <span className="font-mono text-sm">{String(value)}</span>;
  };

  const renderChange = (key: string, change: any) => {
    // Handle added fields
    if ('added' in change && !('removed' in change)) {
      return (
        <div key={key} className="p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-start gap-2">
            <Plus className="w-4 h-4 text-green-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <div className="font-medium text-green-900 mb-1">{key}</div>
              <div className="text-green-700">
                {Array.isArray(change.added) ? (
                  <div className="space-y-1">
                    {change.added.map((item: any, idx: number) => (
                      <div key={idx} className="text-sm">+ {renderChangeValue(item)}</div>
                    ))}
                  </div>
                ) : (
                  renderChangeValue(change.added)
                )}
              </div>
            </div>
          </div>
        </div>
      );
    }

    // Handle removed fields
    if ('removed' in change && !('added' in change)) {
      return (
        <div key={key} className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start gap-2">
            <Minus className="w-4 h-4 text-red-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <div className="font-medium text-red-900 mb-1">{key}</div>
              <div className="text-red-700">
                {Array.isArray(change.removed) ? (
                  <div className="space-y-1">
                    {change.removed.map((item: any, idx: number) => (
                      <div key={idx} className="text-sm">- {renderChangeValue(item)}</div>
                    ))}
                  </div>
                ) : (
                  renderChangeValue(change.removed)
                )}
              </div>
            </div>
          </div>
        </div>
      );
    }

    // Handle modified fields (both added and removed)
    if ('added' in change && 'removed' in change) {
      return (
        <div key={key} className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-2">
            <Edit className="w-4 h-4 text-blue-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <div className="font-medium text-blue-900 mb-2">{key}</div>
              <div className="space-y-2">
                {Array.isArray(change.removed) && change.removed.length > 0 && (
                  <div className="text-red-700">
                    <div className="text-xs font-medium mb-1">Removed:</div>
                    {change.removed.map((item: any, idx: number) => (
                      <div key={idx} className="text-sm">- {renderChangeValue(item)}</div>
                    ))}
                  </div>
                )}
                {Array.isArray(change.added) && change.added.length > 0 && (
                  <div className="text-green-700">
                    <div className="text-xs font-medium mb-1">Added:</div>
                    {change.added.map((item: any, idx: number) => (
                      <div key={idx} className="text-sm">+ {renderChangeValue(item)}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      );
    }

    // Handle old/new value changes
    if ('old' in change && 'new' in change) {
      return (
        <div key={key} className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-start gap-2">
            <Edit className="w-4 h-4 text-yellow-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <div className="font-medium text-yellow-900 mb-2">{key}</div>
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <div className="text-xs text-gray-600 mb-1">Old:</div>
                  <div className="text-red-700">{renderChangeValue(change.old)}</div>
                </div>
                <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-xs text-gray-600 mb-1">New:</div>
                  <div className="text-green-700">{renderChangeValue(change.new)}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return null;
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Comparing versions...</p>
        </div>
      </div>
    );
  }

  if (!diff) {
    return null;
  }

  const changeCount = Object.keys(diff.changes).length;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            <GitCompare className="w-6 h-6 text-blue-600" />
            <div>
              <h2 className="text-xl font-semibold">Version Comparison</h2>
              <p className="text-sm text-gray-600 mt-1">
                Version {diff.version_a_number} → Version {diff.version_b_number}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Summary */}
        <div className="p-6 bg-gray-50 border-b">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-gray-700">Summary:</span>
            <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded">
              {changeCount} {changeCount === 1 ? 'change' : 'changes'}
            </span>
          </div>
          <p className="text-sm text-gray-600">{diff.summary || 'No changes detected'}</p>
        </div>

        {/* Changes */}
        <div className="flex-1 overflow-y-auto p-6">
          {changeCount > 0 ? (
            <div className="space-y-4">
              {Object.entries(diff.changes).map(([key, change]) => renderChange(key, change))}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <GitCompare className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No differences found between these versions</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

