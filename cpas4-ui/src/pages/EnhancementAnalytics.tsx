import { useEffect, useMemo } from 'react';
import { BarChart3, TrendingUp, Award, Zap } from 'lucide-react';
import { usePromptEnhancementStore } from '../stores/promptEnhancementStore';
import { Card } from '../components/ui/card';

export default function EnhancementAnalytics() {
  const { enhancements, statistics, fetchStatistics } = usePromptEnhancementStore();

  useEffect(() => {
    fetchStatistics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only once on mount

  const localStats = useMemo(() => {
    if (enhancements.length === 0) {
      return {
        total: 0,
        avgClarity: 0,
        avgSpecificity: 0,
        avgStructure: 0,
        engineCounts: { dspy: 0, template: 0, custom: 0 },
        avgRating: 0,
        helpfulPercentage: 0,
      };
    }

    const total = enhancements.length;
    const avgClarity = enhancements.reduce((sum, e) => sum + e.quality_metrics.clarity, 0) / total;
    const avgSpecificity = enhancements.reduce((sum, e) => sum + e.quality_metrics.specificity, 0) / total;
    const avgStructure = enhancements.reduce((sum, e) => sum + e.quality_metrics.structure, 0) / total;

    const engineCounts = enhancements.reduce(
      (acc, e) => {
        acc[e.engine] = (acc[e.engine] || 0) + 1;
        return acc;
      },
      { dspy: 0, template: 0, custom: 0 } as any
    );

    const withFeedback = enhancements.filter((e) => e.feedback);
    const avgRating = withFeedback.length > 0
      ? withFeedback.reduce((sum, e) => sum + (e.feedback?.rating || 0), 0) / withFeedback.length
      : 0;
    const helpfulPercentage = withFeedback.length > 0
      ? (withFeedback.filter((e) => e.feedback?.helpful).length / withFeedback.length) * 100
      : 0;

    return {
      total,
      avgClarity: Math.round(avgClarity),
      avgSpecificity: Math.round(avgSpecificity),
      avgStructure: Math.round(avgStructure),
      engineCounts,
      avgRating: avgRating.toFixed(1),
      helpfulPercentage: Math.round(helpfulPercentage),
    };
  }, [enhancements]);

  const topEnhancements = useMemo(() => {
    return [...enhancements]
      .sort((a, b) => {
        const avgA = (a.quality_metrics.clarity + a.quality_metrics.specificity + a.quality_metrics.structure) / 3;
        const avgB = (b.quality_metrics.clarity + b.quality_metrics.specificity + b.quality_metrics.structure) / 3;
        return avgB - avgA;
      })
      .slice(0, 5);
  }, [enhancements]);

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400';
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <BarChart3 className="h-8 w-8 text-primary" />
            Enhancement Analytics
          </h1>
          <p className="text-muted-foreground mt-2">
            Track your prompt enhancement performance and trends
          </p>
        </div>

        <div className="grid md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Enhancements</p>
                <p className="text-3xl font-bold mt-1">{localStats.total}</p>
              </div>
              <Zap className="h-8 w-8 text-primary" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg. Quality Score</p>
                <p className={`text-3xl font-bold mt-1 ${getQualityColor((localStats.avgClarity + localStats.avgSpecificity + localStats.avgStructure) / 3)}`}>
                  {Math.round((localStats.avgClarity + localStats.avgSpecificity + localStats.avgStructure) / 3)}%
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg. Rating</p>
                <p className="text-3xl font-bold mt-1">{localStats.avgRating} ⭐</p>
              </div>
              <Award className="h-8 w-8 text-yellow-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Helpful Rate</p>
                <p className="text-3xl font-bold mt-1">{localStats.helpfulPercentage}%</p>
              </div>
              <TrendingUp className="h-8 w-8 text-blue-500" />
            </div>
          </Card>
        </div>

        <Card className="p-6">
          <h2 className="text-xl font-bold mb-4">Average Quality Metrics</h2>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Clarity</span>
                <span className={`text-sm font-bold ${getQualityColor(localStats.avgClarity)}`}>
                  {localStats.avgClarity}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${localStats.avgClarity}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Specificity</span>
                <span className={`text-sm font-bold ${getQualityColor(localStats.avgSpecificity)}`}>
                  {localStats.avgSpecificity}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all"
                  style={{ width: `${localStats.avgSpecificity}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Structure</span>
                <span className={`text-sm font-bold ${getQualityColor(localStats.avgStructure)}`}>
                  {localStats.avgStructure}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all"
                  style={{ width: `${localStats.avgStructure}%` }}
                />
              </div>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <h2 className="text-xl font-bold mb-4">Engine Usage</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-sm text-muted-foreground">DSPy</p>
              <p className="text-3xl font-bold mt-1">⚡ {localStats.engineCounts.dspy}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {localStats.total > 0 ? Math.round((localStats.engineCounts.dspy / localStats.total) * 100) : 0}%
              </p>
            </div>

            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-muted-foreground">Template</p>
              <p className="text-3xl font-bold mt-1">📋 {localStats.engineCounts.template}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {localStats.total > 0 ? Math.round((localStats.engineCounts.template / localStats.total) * 100) : 0}%
              </p>
            </div>

            <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <p className="text-sm text-muted-foreground">Custom</p>
              <p className="text-3xl font-bold mt-1">🔧 {localStats.engineCounts.custom}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {localStats.total > 0 ? Math.round((localStats.engineCounts.custom / localStats.total) * 100) : 0}%
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <h2 className="text-xl font-bold mb-4">Top 5 Enhancements</h2>
          <div className="space-y-3">
            {topEnhancements.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No enhancements yet</p>
            ) : (
              topEnhancements.map((enhancement, index) => {
                const avgScore = Math.round(
                  (enhancement.quality_metrics.clarity +
                    enhancement.quality_metrics.specificity +
                    enhancement.quality_metrics.structure) / 3
                );
                return (
                  <div key={enhancement.id} className="p-3 bg-muted rounded-lg">
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                        {index + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{enhancement.enhanced_prompt}</p>
                        <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                          <span>Quality: <span className={`font-bold ${getQualityColor(avgScore)}`}>{avgScore}%</span></span>
                          <span>Engine: {enhancement.engine.toUpperCase()}</span>
                          <span>{new Date(enhancement.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
