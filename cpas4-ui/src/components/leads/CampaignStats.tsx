/**
 * Campaign Stats Component
 * 
 * Displays campaign statistics and metrics.
 */

import React from 'react';
import {
  TrendingUp,
  Users,
  CheckCircle,
  MessageCircle,
  Star,
  Target,
} from 'lucide-react';
import { CampaignStats as CampaignStatsType } from '../../stores/leadGenerationStore';

interface CampaignStatsProps {
  stats: CampaignStatsType;
}

export default function CampaignStats({ stats }: CampaignStatsProps) {
  const conversionRate = stats.total_leads > 0
    ? ((stats.converted_leads / stats.total_leads) * 100).toFixed(1)
    : '0.0';

  const qualificationRate = stats.total_leads > 0
    ? ((stats.qualified_leads / stats.total_leads) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total Leads */}
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
            Total Leads
          </span>
          <Users className="w-5 h-5 text-blue-500" />
        </div>
        <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          {stats.total_leads}
        </div>
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
          {stats.qualified_leads} qualified ({qualificationRate}%)
        </div>
      </div>

      {/* Avg Lead Score */}
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
            Avg Lead Score
          </span>
          <Star className="w-5 h-5 text-yellow-500" />
        </div>
        <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          {stats.avg_lead_score.toFixed(0)}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="bg-yellow-500 h-2 rounded-full transition-all"
              style={{ width: `${stats.avg_lead_score}%` }}
            />
          </div>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            /100
          </span>
        </div>
      </div>

      {/* Contacted */}
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
            Contacted
          </span>
          <MessageCircle className="w-5 h-5 text-purple-500" />
        </div>
        <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          {stats.contacted_leads}
        </div>
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
          {stats.qualified_leads > 0
            ? `${((stats.contacted_leads / stats.qualified_leads) * 100).toFixed(0)}% of qualified`
            : 'No qualified leads yet'}
        </div>
      </div>

      {/* Converted */}
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
            Converted
          </span>
          <CheckCircle className="w-5 h-5 text-green-500" />
        </div>
        <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          {stats.converted_leads}
        </div>
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
          {conversionRate}% conversion rate
        </div>
      </div>

      {/* Top Opportunities */}
      {stats.top_opportunities.length > 0 && (
        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 md:col-span-2 lg:col-span-4">
          <div className="flex items-center gap-2 mb-4">
            <Target className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Top Opportunities
            </h3>
          </div>
          <div className="space-y-3">
            {stats.top_opportunities.slice(0, 3).map((opp, index) => (
              <div
                key={index}
                className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-750 rounded-lg"
              >
                <div className="flex-shrink-0 w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <span className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                    {index + 1}
                  </span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs font-medium rounded-full">
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
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

