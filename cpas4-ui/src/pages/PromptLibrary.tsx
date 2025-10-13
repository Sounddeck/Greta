import React, { useEffect, useState } from 'react';
import { Search, Plus, Filter, BookTemplate, Star, TrendingUp, Sparkles } from 'lucide-react';
import { usePromptTemplateStore } from '../stores/promptTemplateStore';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import TemplateCard from '../components/prompts/TemplateCard';
import TemplateDetailModal from '../components/prompts/TemplateDetailModal';
import TemplateEditor from '../components/prompts/TemplateEditor';

const CATEGORIES = [
  { value: null, label: 'All Templates', icon: BookTemplate },
  { value: 'research', label: 'Research', icon: Search },
  { value: 'coding', label: 'Coding', icon: Sparkles },
  { value: 'data_analysis', label: 'Data Analysis', icon: TrendingUp },
  { value: 'writing', label: 'Writing', icon: BookTemplate },
  { value: 'business', label: 'Business', icon: TrendingUp },
  { value: 'creative', label: 'Creative', icon: Sparkles },
  { value: 'technical', label: 'Technical', icon: BookTemplate },
  { value: 'general', label: 'General', icon: Star },
];

export default function PromptLibrary() {
  const {
    templates,
    loading,
    error,
    searchQuery,
    selectedCategory,
    currentPage,
    pageSize,
    totalTemplates,
    fetchTemplates,
    searchTemplates,
    selectTemplate,
    setSearchQuery,
    setSelectedCategory,
    setPage,
    clearError,
    initializeTemplates,
  } = usePromptTemplateStore();

  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showEditorModal, setShowEditorModal] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<any>(null);
  const [localSearchQuery, setLocalSearchQuery] = useState(searchQuery);

  useEffect(() => {
    // Initialize templates on first load
    const init = async () => {
      if (templates.length === 0) {
        await initializeTemplates();
      } else {
        await fetchTemplates();
      }
    };
    init();
  }, [selectedCategory, currentPage]);

  const handleSearch = () => {
    if (localSearchQuery.trim()) {
      searchTemplates(localSearchQuery, selectedCategory || undefined);
    } else {
      fetchTemplates();
    }
  };

  const handleCategoryChange = (category: string | null) => {
    setSelectedCategory(category);
    setPage(1);
  };

  const handleTemplateClick = (template: any) => {
    selectTemplate(template);
    setShowDetailModal(true);
  };

  const handleCreateNew = () => {
    setEditingTemplate(null);
    setShowEditorModal(true);
  };

  const handleEdit = (template: any) => {
    setEditingTemplate(template);
    setShowEditorModal(true);
  };

  const handleCloseDetailModal = () => {
    setShowDetailModal(false);
    selectTemplate(null);
  };

  const handleCloseEditorModal = () => {
    setShowEditorModal(false);
    setEditingTemplate(null);
  };

  const totalPages = Math.ceil(totalTemplates / pageSize);

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
                <BookTemplate className="h-8 w-8 text-primary" />
                Prompt Library
              </h1>
              <p className="text-muted-foreground mt-1">
                Browse and use pre-built prompt templates for common tasks
              </p>
            </div>
            <Button onClick={handleCreateNew} className="gap-2">
              <Plus className="h-4 w-4" />
              Create Template
            </Button>
          </div>

          {/* Search Bar */}
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search templates by name, description, or tags..."
                value={localSearchQuery}
                onChange={(e) => setLocalSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="pl-10"
              />
            </div>
            <Button onClick={handleSearch} variant="secondary">
              Search
            </Button>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="px-6 pb-4">
          <div className="flex gap-2 overflow-x-auto pb-2">
            {CATEGORIES.map((cat) => {
              const Icon = cat.icon;
              const isActive = selectedCategory === cat.value;
              return (
                <button
                  key={cat.value || 'all'}
                  onClick={() => handleCategoryChange(cat.value)}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors
                    ${isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }
                  `}
                >
                  <Icon className="h-4 w-4" />
                  {cat.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mx-6 mt-4 p-4 bg-destructive/10 border border-destructive rounded-lg flex items-center justify-between">
          <p className="text-destructive text-sm">{error}</p>
          <Button variant="ghost" size="sm" onClick={clearError}>
            Dismiss
          </Button>
        </div>
      )}

      {/* Templates Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">Loading templates...</p>
            </div>
          </div>
        ) : templates.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <BookTemplate className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">No templates found</h3>
              <p className="text-muted-foreground mb-4">
                {searchQuery ? 'Try a different search query' : 'Create your first template to get started'}
              </p>
              <Button onClick={handleCreateNew}>
                <Plus className="h-4 w-4 mr-2" />
                Create Template
              </Button>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {templates.map((template) => (
                <TemplateCard
                  key={template.id}
                  template={template}
                  onClick={() => handleTemplateClick(template)}
                  onEdit={() => handleEdit(template)}
                />
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-6">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                  Page {currentPage} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                >
                  Next
                </Button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Modals */}
      {showDetailModal && (
        <TemplateDetailModal
          onClose={handleCloseDetailModal}
          onEdit={handleEdit}
        />
      )}

      {showEditorModal && (
        <TemplateEditor
          template={editingTemplate}
          onClose={handleCloseEditorModal}
        />
      )}
    </div>
  );
}

