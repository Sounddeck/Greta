/**
 * Lead Score Display Component
 * 
 * Displays lead score with visual indicators, factor breakdown, and recommendations.
 */

import React from 'react';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Building2,
  Target,
  Activity,
  DollarSign,
  User,
  MapPin,
  Info,
} from 'lucide-react';

export interface LeadScoreData {
  score: number;
  factors: {
    company_size: number;
    industry: number;
    engagement: number;
    budget: number;
    decision_maker: number;
    geographic: number;
  };
  confidence: number;
  recommendations: string[];
  calculated_at: string;
}

interface LeadScoreDisplayProps {
  scoreData: LeadScoreData | null;
  compact?: boolean;
  showFactors?: boolean;
  showRecommendations?: boolean;
}

export default function LeadScoreDisplay({
  scoreData,
  compact = false,
  showFactors = true,
  showRecommendations = true,
}: LeadScoreDisplayProps) {
  if (!scoreData) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400">
        No score available
      </div>
    );
  }

  const { score, factors, confidence, recommendations } = scoreData;

  // Determine score level and color
  const getScoreLevel = (score: number): { level: string; color: string; bgColor: string; icon: React.ReactNode } => {
    if (score >= 70) {
      return {
        level: 'High',
        color: 'text-green-600 dark:text-green-400',
        bgColor: 'bg-green-100 dark:bg-green-900/30',
        icon: <TrendingUp className="w-4 h-4" />,
      };
    } else if (score >= 30) {
      return {
        level: 'Medium',
        color: 'text-yellow-600 dark:text-yellow-400',
        bgColor: 'bg-yellow-100 dark:bg-yellow-900/30',
        icon: <Minus className="w-4 h-4" />,
      };
    } else {
      return {
        level: 'Low',
        color: 'text-red-600 dark:text-red-400',
        bgColor: 'bg-red-100 dark:bg-red-900/30',
        icon: <TrendingDown className="w-4 h-4" />,
      };
    }
  };

  const scoreLevel = getScoreLevel(score);

  // Factor icons and labels
  const factorConfig = {
    company_size: { icon: Building2, label: 'Company Size', weight: '25%' },
    industry: { icon: Target, label: 'Industry Match', weight: '20%' },
    engagement: { icon: Activity, label: 'Engagement', weight: '20%' },
    budget: { icon: DollarSign, label: 'Budget Indicators', weight: '15%' },
    decision_maker: { icon: User, label: 'Decision Maker', weight: '10%' },
    geographic: { icon: MapPin, label: 'Geographic Fit', weight: '10%' },
  };

  // Compact view (for list display)
  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div className={`flex items-center gap-1 px-2 py-1 rounded-full ${scoreLevel.bgColor} ${scoreLevel.color}`}>
          {scoreLevel.icon}
          <span className="text-sm font-semibold">{score.toFixed(0)}</span>
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          {scoreLevel.level} • {(confidence * 100).toFixed(0)}% confidence
        </div>
      </div>
    );
  }

  // Full view (for detail modal)
  return (
    <div className="space-y-4">
      {/* Score Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Lead Score
            </h3>
            <div className={`flex items-center gap-1 px-2 py-1 rounded-full ${scoreLevel.bgColor} ${scoreLevel.color}`}>
              {scoreLevel.icon}
              <span className="text-sm font-semibold">{scoreLevel.level}</span>
            </div>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Confidence: {(confidence * 100).toFixed(0)}%
          </p>
        </div>
        <div className="text-right">
          <div className={`text-4xl font-bold ${scoreLevel.color}`}>
            {score.toFixed(0)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">
            out of 100
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="relative w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`absolute top-0 left-0 h-full transition-all duration-500 ${
            score >= 70
              ? 'bg-gradient-to-r from-green-500 to-green-600'
              : score >= 30
              ? 'bg-gradient-to-r from-yellow-500 to-yellow-600'
              : 'bg-gradient-to-r from-red-500 to-red-600'
          }`}
          style={{ width: `${score}%` }}
        />
      </div>

      {/* Factor Breakdown */}
      {showFactors && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Info className="w-4 h-4" />
            Score Breakdown
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {Object.entries(factors).map(([key, value]) => {
              const config = factorConfig[key as keyof typeof factorConfig];
              const Icon = config.icon;
              const factorLevel = getScoreLevel(value);

              return (
                <div
                  key={key}
                  className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-750 rounded-lg"
                >
                  <div className={`flex-shrink-0 w-10 h-10 ${factorLevel.bgColor} rounded-lg flex items-center justify-center ${factorLevel.color}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {config.label}
                      </span>
                      <span className={`text-sm font-semibold ${factorLevel.color}`}>
                        {value.toFixed(0)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-300 ${
                            value >= 70
                              ? 'bg-green-500'
                              : value >= 30
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                          }`}
                          style={{ width: `${value}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {config.weight}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {showRecommendations && recommendations.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Recommendations
          </h4>
          <ul className="space-y-2">
            {recommendations.map((rec, index) => (
              <li
                key={index}
                className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300 p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg"
              >
                <span className="flex-shrink-0 w-5 h-5 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-semibold mt-0.5">
                  {index + 1}
                </span>
                <span className="flex-1">{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Timestamp */}
      <div className="text-xs text-gray-500 dark:text-gray-400 text-right">
        Last calculated: {new Date(scoreData.calculated_at).toLocaleString()}
      </div>
    </div>
  );
}

