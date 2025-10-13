/**
 * Lead Detail Modal
 * 
 * Modal for viewing and managing lead details.
 */

import React, { useEffect, useState } from 'react';
import {
  X,
  Building2,
  MapPin,
  Star,
  Users,
  TrendingUp,
  Search,
  Play,
  Pause,
  ExternalLink,
} from 'lucide-react';
import { useLeadGenerationStore, LeadStatus, ResearchDepth, MonitoringFrequency } from '../../stores/leadGenerationStore';
import { toast } from 'sonner';

interface LeadDetailModalProps {
  leadId: string;
  onClose: () => void;
}

export default function LeadDetailModal({ leadId, onClose }: LeadDetailModalProps) {
  const {
    selectedLead,
    loading,
    selectLead,
    updateLeadStatus,
    researchLead,
    startMonitoring,
    stopMonitoring,
  } = useLeadGenerationStore();

  const [researchDepth, setResearchDepth] = useState<ResearchDepth>('medium');
  const [monitoringFrequency, setMonitoringFrequency] = useState<MonitoringFrequency>('weekly');

  useEffect(() => {
    selectLead(leadId);
  }, [leadId, selectLead]);

  const handleStatusChange = async (status: LeadStatus) => {
    try {
      await updateLeadStatus(leadId, status);
      toast.success('Status updated');
    } catch (error) {
      toast.error('Failed to update status');
    }
  };

  const handleResearch = async () => {
    try {
      await researchLead(leadId, researchDepth);
      toast.success('Research completed');
    } catch (error) {
      toast.error('Research failed');
    }
  };

  const handleStartMonitoring = async () => {
    try {
      await startMonitoring(leadId, monitoringFrequency);
      toast.success('Monitoring started');
    } catch (error) {
      toast.error('Failed to start monitoring');
    }
  };

  const handleStopMonitoring = async () => {
    try {
      await stopMonitoring(leadId);
      toast.success('Monitoring stopped');
    } catch (error) {
      toast.error('Failed to stop monitoring');
    }
  };

  if (!selectedLead) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-8">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto" />
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center text-white font-bold text-xl">
              {selectedLead.company_name[0].toUpperCase()}
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {selectedLead.company_name}
              </h2>
              <div className="flex items-center gap-3 mt-1 text-sm text-gray-600 dark:text-gray-400">
                <span>{selectedLead.industry}</span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <MapPin className="w-4 h-4" />
                  {selectedLead.location}
                </span>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Lead Score & Status */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-750 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Star className="w-5 h-5 text-yellow-500" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Lead Score
                </span>
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                {selectedLead.lead_score.toFixed(0)}
              </div>
              <div className="mt-2 flex items-center gap-2">
                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-yellow-500 h-2 rounded-full transition-all"
                    style={{ width: `${selectedLead.lead_score}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 dark:text-gray-400">/100</span>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-750 rounded-lg p-4">
              <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Status
              </div>
              <select
                value={selectedLead.status}
                onChange={(e) => handleStatusChange(e.target.value as LeadStatus)}
                disabled={loading}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              >
                <option value="new">New</option>
                <option value="researching">Researching</option>
                <option value="qualified">Qualified</option>
                <option value="contacted">Contacted</option>
                <option value="converted">Converted</option>
                <option value="disqualified">Disqualified</option>
              </select>
            </div>
          </div>

          {/* Research Actions */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-3">
              Research Actions
            </h3>
            <div className="flex gap-3">
              <select
                value={researchDepth}
                onChange={(e) => setResearchDepth(e.target.value as ResearchDepth)}
                className="px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              >
                <option value="shallow">Shallow Research</option>
                <option value="medium">Medium Research</option>
                <option value="deep">Deep Research</option>
              </select>
              <button
                onClick={handleResearch}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                <Search className="w-4 h-4" />
                <span>Research Lead</span>
              </button>
            </div>
          </div>

          {/* Monitoring */}
          <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-purple-900 dark:text-purple-100 mb-3">
              Continuous Monitoring
            </h3>
            <div className="flex gap-3">
              {!selectedLead.monitoring_active ? (
                <>
                  <select
                    value={monitoringFrequency}
                    onChange={(e) => setMonitoringFrequency(e.target.value as MonitoringFrequency)}
                    className="px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
                  >
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                  </select>
                  <button
                    onClick={handleStartMonitoring}
                    disabled={loading}
                    className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    <span>Start Monitoring</span>
                  </button>
                </>
              ) : (
                <button
                  onClick={handleStopMonitoring}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
                >
                  <Pause className="w-4 h-4" />
                  <span>Stop Monitoring</span>
                </button>
              )}
            </div>
          </div>

          {/* Key People */}
          {selectedLead.key_people.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                <Users className="w-5 h-5" />
                Key People
              </h3>
              <div className="space-y-2">
                {selectedLead.key_people.map((person, index) => (
                  <div
                    key={index}
                    className="bg-gray-50 dark:bg-gray-750 rounded-lg p-3 flex items-center justify-between"
                  >
                    <div>
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {person.name}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {person.title}
                      </div>
                    </div>
                    {person.linkedin && (
                      <a
                        href={person.linkedin}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 dark:text-blue-400 hover:underline"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Opportunities */}
          {selectedLead.opportunities.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Opportunities
              </h3>
              <div className="space-y-2">
                {selectedLead.opportunities.map((opp, index) => (
                  <div
                    key={index}
                    className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-xs font-medium rounded-full">
                        {opp.opportunity_type.replace('_', ' ')}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {(opp.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {opp.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

