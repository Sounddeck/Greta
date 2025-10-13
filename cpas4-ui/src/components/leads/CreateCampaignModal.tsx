/**
 * Create Campaign Modal
 * 
 * Modal for creating a new lead generation campaign.
 */

import React, { useState } from 'react';
import { X } from 'lucide-react';
import { useLeadGenerationStore, CompanySize, ResearchDepth, MonitoringFrequency } from '../../stores/leadGenerationStore';
import { toast } from 'sonner';

interface CreateCampaignModalProps {
  userId: string;
  onClose: () => void;
}

export default function CreateCampaignModal({ userId, onClose }: CreateCampaignModalProps) {
  const { createCampaign } = useLeadGenerationStore();
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    campaign_name: '',
    target_industry: '',
    target_location: '',
    company_size: '51-200' as CompanySize,
    keywords: '',
    research_depth: 'medium' as ResearchDepth,
    monitoring_frequency: 'weekly' as MonitoringFrequency,
    max_leads: 100,
    auto_research: true,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const keywords = formData.keywords.split(',').map(k => k.trim()).filter(k => k);
      
      await createCampaign(userId, {
        ...formData,
        keywords,
      });

      toast.success('Campaign created successfully');
      onClose();
    } catch (error) {
      toast.error('Failed to create campaign');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Create Lead Generation Campaign
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Campaign Name *
            </label>
            <input
              type="text"
              required
              value={formData.campaign_name}
              onChange={(e) => setFormData({ ...formData, campaign_name: e.target.value })}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              placeholder="e.g., Tech Startups Q4 2025"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Target Industry *
              </label>
              <input
                type="text"
                required
                value={formData.target_industry}
                onChange={(e) => setFormData({ ...formData, target_industry: e.target.value })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
                placeholder="e.g., Technology"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Target Location *
              </label>
              <input
                type="text"
                required
                value={formData.target_location}
                onChange={(e) => setFormData({ ...formData, target_location: e.target.value })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
                placeholder="e.g., San Francisco, CA"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Company Size
            </label>
            <select
              value={formData.company_size}
              onChange={(e) => setFormData({ ...formData, company_size: e.target.value as CompanySize })}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
            >
              <option value="1-10">1-10 employees</option>
              <option value="11-50">11-50 employees</option>
              <option value="51-200">51-200 employees</option>
              <option value="201-1000">201-1000 employees</option>
              <option value="1000+">1000+ employees</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Keywords (comma-separated)
            </label>
            <input
              type="text"
              value={formData.keywords}
              onChange={(e) => setFormData({ ...formData, keywords: e.target.value })}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              placeholder="e.g., SaaS, B2B, AI"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Research Depth
              </label>
              <select
                value={formData.research_depth}
                onChange={(e) => setFormData({ ...formData, research_depth: e.target.value as ResearchDepth })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              >
                <option value="shallow">Shallow</option>
                <option value="medium">Medium</option>
                <option value="deep">Deep</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Monitoring Frequency
              </label>
              <select
                value={formData.monitoring_frequency}
                onChange={(e) => setFormData({ ...formData, monitoring_frequency: e.target.value as MonitoringFrequency })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Max Leads
            </label>
            <input
              type="number"
              min="1"
              max="1000"
              value={formData.max_leads}
              onChange={(e) => setFormData({ ...formData, max_leads: parseInt(e.target.value) })}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
            />
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="auto_research"
              checked={formData.auto_research}
              onChange={(e) => setFormData({ ...formData, auto_research: e.target.checked })}
              className="w-4 h-4 text-blue-600 rounded"
            />
            <label htmlFor="auto_research" className="text-sm text-gray-700 dark:text-gray-300">
              Automatically research new leads
            </label>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Creating...' : 'Create Campaign'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

