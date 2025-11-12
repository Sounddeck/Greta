import { useState } from 'react';
import { Sparkles, Wand2, Copy, Check, ArrowRight, Info } from 'lucide-react';
import { usePromptEnhancementStore } from '../stores/promptEnhancementStore';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';

export default function PromptEnhancement() {
  const {
    currentEnhancement,
    isLoading,
    error,
    settings,
    enhancePrompt,
    updateSettings,
  } = usePromptEnhancementStore();

  const [inputPrompt, setInputPrompt] = useState('');
  const [agentType, setAgentType] = useState('general');
  const [copied, setCopied] = useState(false);

  const handleEnhance = async () => {
    if (!inputPrompt.trim()) {
      return;
    }
    await enhancePrompt(inputPrompt, agentType);
  };

  const handleCopy = () => {
    if (currentEnhancement?.enhanced_prompt) {
      navigator.clipboard.writeText(currentEnhancement.enhanced_prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400';
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Wand2 className="h-8 w-8 text-primary" />
            Prompt Enhancement
          </h1>
          <p className="text-muted-foreground mt-2">
            Improve your prompts with AI-powered enhancement
          </p>
        </div>

        {/* Settings Card */}
        <Card className="p-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium mb-2">Agent Type</label>
              <Select value={agentType} onValueChange={setAgentType}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select agent type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="general">General</SelectItem>
                  <SelectItem value="research">Research</SelectItem>
                  <SelectItem value="coding">Coding</SelectItem>
                  <SelectItem value="data_analysis">Data Analysis</SelectItem>
                  <SelectItem value="writing">Writing</SelectItem>
                  <SelectItem value="business">Business</SelectItem>
                  <SelectItem value="creative">Creative</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex-1">
              <label className="block text-sm font-medium mb-2">Enhancement Engine</label>
              <Select 
                value={settings.engine} 
                onValueChange={(value: 'dspy' | 'template' | 'custom') => 
                  updateSettings({ engine: value })
                }
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select engine" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="dspy">DSPy (AI-Powered)</SelectItem>
                  <SelectItem value="template">Template-Based</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </Card>

        {/* Input Section */}
        <Card className="p-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Original Prompt</h2>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="h-4 w-4" />
                <span>{inputPrompt.length} characters</span>
              </div>
            </div>
            
            <textarea
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              placeholder="Enter your prompt here... For example: 'Write a function to sort an array'"
              className="w-full h-48 px-4 py-3 bg-background border border-border rounded-lg text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary"
              disabled={isLoading}
            />

            <div className="flex items-center gap-3">
              <Button
                onClick={handleEnhance}
                disabled={!inputPrompt.trim() || isLoading}
                className="flex-1 gap-2"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Enhancing...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5" />
                    Enhance Prompt
                  </>
                )}
              </Button>
            </div>

            {error && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}
          </div>
        </Card>

        {/* Enhanced Output Section */}
        {currentEnhancement && (
          <Card className="p-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Enhanced Prompt
                </h2>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCopy}
                  className="gap-2"
                >
                  {copied ? (
                    <>
                      <Check className="h-4 w-4" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4" />
                      Copy
                    </>
                  )}
                </Button>
              </div>

              <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
                <pre className="text-sm text-foreground whitespace-pre-wrap font-sans">
                  {currentEnhancement.enhanced_prompt}
                </pre>
              </div>

              {/* Quality Metrics */}
              <div className="grid grid-cols-3 gap-4 pt-4 border-t border-border">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground mb-1">Clarity</p>
                  <p className={`text-2xl font-bold ${getQualityColor(currentEnhancement.quality_metrics.clarity)}`}>
                    {currentEnhancement.quality_metrics.clarity}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground mb-1">Specificity</p>
                  <p className={`text-2xl font-bold ${getQualityColor(currentEnhancement.quality_metrics.specificity)}`}>
                    {currentEnhancement.quality_metrics.specificity}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground mb-1">Structure</p>
                  <p className={`text-2xl font-bold ${getQualityColor(currentEnhancement.quality_metrics.structure)}`}>
                    {currentEnhancement.quality_metrics.structure}%
                  </p>
                </div>
              </div>

              {/* Comparison */}
              <div className="pt-4 border-t border-border">
                <h3 className="text-sm font-semibold mb-3">Before & After Comparison</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-2">Original</p>
                    <div className="bg-muted/50 rounded-lg p-3">
                      <p className="text-sm line-clamp-4">{currentEnhancement.original_prompt}</p>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {currentEnhancement.original_prompt.length} characters
                    </p>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1">
                      Enhanced
                      <ArrowRight className="h-3 w-3" />
                    </p>
                    <div className="bg-primary/10 rounded-lg p-3">
                      <p className="text-sm line-clamp-4">{currentEnhancement.enhanced_prompt}</p>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {currentEnhancement.enhanced_prompt.length} characters
                    </p>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-4">
                <Button variant="outline" className="flex-1">
                  Save to History
                </Button>
                <Button className="flex-1">
                  Use in Agent
                </Button>
              </div>
            </div>
          </Card>
        )}

        {/* Info Card */}
        {!currentEnhancement && !isLoading && (
          <Card className="p-6 bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
            <div className="flex gap-3">
              <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="space-y-2">
                <h3 className="font-semibold text-blue-900 dark:text-blue-100">
                  How Prompt Enhancement Works
                </h3>
                <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                  <li>• Analyzes your prompt for clarity, specificity, and structure</li>
                  <li>• Adds context and details to improve AI understanding</li>
                  <li>• Optimizes prompt format for better results</li>
                  <li>• Provides quality metrics to track improvements</li>
                </ul>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

