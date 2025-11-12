import { useState, useMemo } from 'react';
import { History, Search, Download, Filter, ChevronLeft, ChevronRight, ThumbsUp, ThumbsDown, Star } from 'lucide-react';
import { usePromptEnhancementStore } from '../stores/promptEnhancementStore';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../components/ui/dialog';

export default function EnhancementHistory() {
  const {
    enhancements,
    searchQuery,
    filterEngine,
    currentPage,
    pageSize,
    setSearchQuery,
    setFilterEngine,
    setPage,
    exportHistory,
    submitFeedback,
  } = usePromptEnhancementStore();

  const [localSearch, setLocalSearch] = useState(searchQuery);
  const [selectedEnhancement, setSelectedEnhancement] = useState<any>(null);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [feedbackData, setFeedbackData] = useState({
    rating: 5,
    helpful: true,
    comments: '',
    improvements: '',
  });

  const filteredEnhancements = useMemo(() => {
    return enhancements.filter((enhancement) => {
      const matchesSearch =
        enhancement.original_prompt.toLowerCase().includes(searchQuery.toLowerCase()) ||
        enhancement.enhanced_prompt.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesEngine = filterEngine === 'all' || enhancement.engine === filterEngine;
      return matchesSearch && matchesEngine;
    });
  }, [enhancements, searchQuery, filterEngine]);

  const paginatedEnhancements = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    const end = start + pageSize;
    return filteredEnhancements.slice(start, end);
  }, [filteredEnhancements, currentPage, pageSize]);

  const totalPages = Math.ceil(filteredEnhancements.length / pageSize);

  const handleSearch = () => {
    setSearchQuery(localSearch);
  };

  const handleViewDetails = (enhancement: any) => {
    setSelectedEnhancement(enhancement);
  };

  const handleProvideFeedback = (enhancement: any) => {
    setSelectedEnhancement(enhancement);
    setFeedbackData(enhancement.feedback || {
      rating: 5,
      helpful: true,
      comments: '',
      improvements: '',
    });
    setShowFeedbackModal(true);
  };

  const handleSubmitFeedback = async () => {
    if (selectedEnhancement) {
      await submitFeedback(selectedEnhancement.id, feedbackData);
      setShowFeedbackModal(false);
      setSelectedEnhancement(null);
    }
  };

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400';
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getEngineIcon = (engine: string) => {
    switch (engine) {
      case 'dspy':
        return '⚡';
      case 'template':
        return '📋';
      case 'custom':
        return '🔧';
      default:
        return '❓';
    }
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <History className="h-8 w-8 text-primary" />
            Enhancement History
          </h1>
          <p className="text-muted-foreground mt-2">
            View and manage your prompt enhancement history
          </p>
        </div>

        <Card className="p-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 flex gap-2">
              <input
                type="text"
                value={localSearch}
                onChange={(e) => setLocalSearch(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search prompts..."
                className="flex-1 px-3 py-2 bg-background border border-border rounded-md"
              />
              <Button onClick={handleSearch}>
                <Search className="h-4 w-4 mr-2" />
                Search
              </Button>
            </div>

            <div className="flex gap-2">
              <Select value={filterEngine} onValueChange={(value: any) => setFilterEngine(value)}>
                <SelectTrigger className="w-[180px]">
                  <Filter className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Filter by engine" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Engines</SelectItem>
                  <SelectItem value="dspy">DSPy</SelectItem>
                  <SelectItem value="template">Template</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>

              <Select onValueChange={(value: 'csv' | 'json') => exportHistory(value)}>
                <SelectTrigger className="w-[140px]">
                  <Download className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Export" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="csv">Export CSV</SelectItem>
                  <SelectItem value="json">Export JSON</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </Card>

        <div className="text-sm text-muted-foreground">
          Showing {paginatedEnhancements.length} of {filteredEnhancements.length} enhancements
        </div>

        <div className="space-y-4">
          {paginatedEnhancements.length === 0 ? (
            <Card className="p-8 text-center">
              <History className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No enhancements found</p>
              <p className="text-sm text-muted-foreground mt-2">
                Start enhancing prompts to see them here
              </p>
            </Card>
          ) : (
            paginatedEnhancements.map((enhancement) => (
              <Card key={enhancement.id} className="p-4">
                <div className="space-y-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{getEngineIcon(enhancement.engine)}</span>
                      <div>
                        <p className="text-sm font-medium">{enhancement.engine.toUpperCase()}</p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(enhancement.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" onClick={() => handleViewDetails(enhancement)}>
                        View Details
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => handleProvideFeedback(enhancement)}>
                        {enhancement.feedback ? <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" /> : <Star className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground mb-1">Original</p>
                      <p className="text-sm line-clamp-2">{enhancement.original_prompt}</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-muted-foreground mb-1">Enhanced</p>
                      <p className="text-sm line-clamp-2">{enhancement.enhanced_prompt}</p>
                    </div>
                  </div>

                  <div className="flex gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Clarity: </span>
                      <span className={`font-medium ${getQualityColor(enhancement.quality_metrics.clarity)}`}>
                        {enhancement.quality_metrics.clarity}%
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Specificity: </span>
                      <span className={`font-medium ${getQualityColor(enhancement.quality_metrics.specificity)}`}>
                        {enhancement.quality_metrics.specificity}%
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Structure: </span>
                      <span className={`font-medium ${getQualityColor(enhancement.quality_metrics.structure)}`}>
                        {enhancement.quality_metrics.structure}%
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            ))
          )}
        </div>

        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(currentPage - 1)}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm">
              Page {currentPage} of {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}

        {selectedEnhancement && !showFeedbackModal && (
          <Dialog open={!!selectedEnhancement} onOpenChange={() => setSelectedEnhancement(null)}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Enhancement Details</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium mb-2">Original Prompt</p>
                  <p className="text-sm bg-muted p-3 rounded-md">{selectedEnhancement.original_prompt}</p>
                </div>
                <div>
                  <p className="text-sm font-medium mb-2">Enhanced Prompt</p>
                  <p className="text-sm bg-muted p-3 rounded-md">{selectedEnhancement.enhanced_prompt}</p>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm font-medium">Clarity</p>
                    <p className={`text-2xl font-bold ${getQualityColor(selectedEnhancement.quality_metrics.clarity)}`}>
                      {selectedEnhancement.quality_metrics.clarity}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Specificity</p>
                    <p className={`text-2xl font-bold ${getQualityColor(selectedEnhancement.quality_metrics.specificity)}`}>
                      {selectedEnhancement.quality_metrics.specificity}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Structure</p>
                    <p className={`text-2xl font-bold ${getQualityColor(selectedEnhancement.quality_metrics.structure)}`}>
                      {selectedEnhancement.quality_metrics.structure}%
                    </p>
                  </div>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        )}

        {showFeedbackModal && selectedEnhancement && (
          <Dialog open={showFeedbackModal} onOpenChange={setShowFeedbackModal}>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Provide Feedback</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium mb-2">Was this enhancement helpful?</p>
                  <div className="flex gap-2">
                    <Button
                      variant={feedbackData.helpful ? 'default' : 'outline'}
                      onClick={() => setFeedbackData({ ...feedbackData, helpful: true })}
                      className="flex-1"
                    >
                      <ThumbsUp className="h-4 w-4 mr-2" />
                      Yes
                    </Button>
                    <Button
                      variant={!feedbackData.helpful ? 'default' : 'outline'}
                      onClick={() => setFeedbackData({ ...feedbackData, helpful: false })}
                      className="flex-1"
                    >
                      <ThumbsDown className="h-4 w-4 mr-2" />
                      No
                    </Button>
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium mb-2">Rate this enhancement (1-5 stars)</p>
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        onClick={() => setFeedbackData({ ...feedbackData, rating: star })}
                        className="p-1 hover:scale-110 transition-transform"
                      >
                        <Star
                          className={`h-6 w-6 ${
                            star <= feedbackData.rating
                              ? 'fill-yellow-400 text-yellow-400'
                              : 'text-gray-300'
                          }`}
                        />
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium mb-2">Comments (optional)</p>
                  <textarea
                    value={feedbackData.comments}
                    onChange={(e) => setFeedbackData({ ...feedbackData, comments: e.target.value })}
                    placeholder="Share your thoughts about this enhancement..."
                    className="w-full px-3 py-2 bg-background border border-border rounded-md min-h-[100px]"
                  />
                </div>

                <div>
                  <p className="text-sm font-medium mb-2">Suggestions for improvement (optional)</p>
                  <textarea
                    value={feedbackData.improvements}
                    onChange={(e) => setFeedbackData({ ...feedbackData, improvements: e.target.value })}
                    placeholder="How could this enhancement be better?"
                    className="w-full px-3 py-2 bg-background border border-border rounded-md min-h-[100px]"
                  />
                </div>

                <div className="flex gap-2">
                  <Button onClick={handleSubmitFeedback} className="flex-1">
                    Submit Feedback
                  </Button>
                  <Button variant="outline" onClick={() => setShowFeedbackModal(false)}>
                    Cancel
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>
    </div>
  );
}
