// @ts-nocheck
import { useState, useEffect } from 'react';
import { Search, Plus, Trash2, Brain, AlertCircle, ArrowUp, ArrowDown } from 'lucide-react';
import { toast } from 'sonner';
import { api } from '../../../lib/api';

interface SkillsStepProps {
  draft: any;
  updateDraft: (updates: any) => void;
  validation?: any;
}

export function SkillsStep({ draft, updateDraft, validation }: SkillsStepProps) {
  const [availableSkills, setAvailableSkills] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const selectedSkills = draft.skills || [];

  // Load available skills
  useEffect(() => {
    loadSkills();
  }, []);

  const loadSkills = async () => {
    setLoading(true);
    try {
      const response = await api.get('/skills/');
      // Handle API response structure: {skills: [...], total: 5, categories: [...]}
      const skills = response.data?.skills || response.data || [];
      // Ensure skills is always an array
      setAvailableSkills(Array.isArray(skills) ? skills : []);
    } catch (error) {
      console.error('Failed to load skills:', error);
      toast.error('Failed to load skills');
      setAvailableSkills([]); // Set empty array on error
    } finally {
      setLoading(false);
    }
  };

  const addSkill = (skill: any) => {
    const isAlreadyAdded = selectedSkills.some(
      (s: any) => s.skill_name === skill.name
    );

    if (isAlreadyAdded) {
      toast.error('Skill already added');
      return;
    }

    const newSkill = {
      skill_name: skill.name, // API returns 'name', we store as 'skill_name'
      configuration: {},
      is_enabled: true,
      priority: selectedSkills.length, // Add at the end
    };

    updateDraft({
      skills: [...selectedSkills, newSkill],
    });

    toast.success(`Added ${skill.display_name}`);
  };

  const removeSkill = (skillName: string) => {
    updateDraft({
      skills: selectedSkills.filter((s: any) => s.skill_name !== skillName),
    });
    toast.success('Skill removed');
  };

  const toggleSkillEnabled = (skillName: string) => {
    updateDraft({
      skills: selectedSkills.map((s: any) =>
        s.skill_name === skillName ? { ...s, is_enabled: !s.is_enabled } : s
      ),
    });
  };

  const moveSkillUp = (index: number) => {
    if (index === 0) return;
    const newSkills = [...selectedSkills];
    [newSkills[index - 1], newSkills[index]] = [newSkills[index], newSkills[index - 1]];
    // Update priorities
    newSkills.forEach((skill, i) => {
      skill.priority = i;
    });
    updateDraft({ skills: newSkills });
  };

  const moveSkillDown = (index: number) => {
    if (index === selectedSkills.length - 1) return;
    const newSkills = [...selectedSkills];
    [newSkills[index], newSkills[index + 1]] = [newSkills[index + 1], newSkills[index]];
    // Update priorities
    newSkills.forEach((skill, i) => {
      skill.priority = i;
    });
    updateDraft({ skills: newSkills });
  };

  // Ensure availableSkills is an array before processing
  const safeSkills = Array.isArray(availableSkills) ? availableSkills : [];
  const categories = ['all', ...new Set(safeSkills.map((s) => s.category))];

  const filteredSkills = safeSkills.filter((skill) => {
    const matchesSearch =
      skill.display_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      skill.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || skill.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">Skills</h2>
        <p className="text-muted-foreground">
          Add specialized skills to your agent for advanced capabilities like SPARC methodology,
          pair programming, and swarm orchestration.
        </p>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search skills..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-3 py-2 border border-border rounded-md bg-background"
          />
        </div>
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="px-3 py-2 border border-border rounded-md bg-background"
        >
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat === 'all' ? 'All Categories' : cat}
            </option>
          ))}
        </select>
      </div>

      {/* Available Skills */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Available Skills</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-96 overflow-y-auto">
          {loading ? (
            <div className="col-span-2 text-center py-8 text-muted-foreground">
              Loading skills...
            </div>
          ) : filteredSkills.length === 0 ? (
            <div className="col-span-2 text-center py-8 text-muted-foreground">
              No skills found
            </div>
          ) : (
            filteredSkills.map((skill) => {
              const isAdded = selectedSkills.some(
                (s: any) => s.skill_name === skill.name
              );

              return (
                <div
                  key={skill.name}
                  className={`p-4 border rounded-lg ${
                    isAdded
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:border-primary/50'
                  } transition-colors`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Brain className="h-4 w-4 text-primary" />
                      <h4 className="font-medium">{skill.display_name}</h4>
                    </div>
                    <button
                      onClick={() => (isAdded ? removeSkill(skill.name) : addSkill(skill))}
                      className={`p-1 rounded ${
                        isAdded
                          ? 'text-destructive hover:bg-destructive/10'
                          : 'text-primary hover:bg-primary/10'
                      }`}
                    >
                      {isAdded ? <Trash2 className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
                    </button>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">{skill.description}</p>
                  <div className="flex items-center gap-2">
                    <span className="text-xs px-2 py-1 bg-secondary text-secondary-foreground rounded">
                      {skill.category}
                    </span>
                    {skill.confidence_threshold && (
                      <span className="text-xs text-muted-foreground">
                        Confidence: {(skill.confidence_threshold * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Selected Skills */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">
          Selected Skills ({selectedSkills.length})
        </h3>

        {selectedSkills.length === 0 ? (
          <div className="text-center py-8 border border-dashed border-border rounded-lg">
            <Brain className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
            <p className="text-muted-foreground">No skills selected</p>
            <p className="text-sm text-muted-foreground mt-1">
              Add skills to enhance your agent's capabilities
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {selectedSkills.map((skill: any, index: number) => {
              const safeAvailableSkills = Array.isArray(availableSkills) ? availableSkills : [];
              const skillInfo = safeAvailableSkills.find((s) => s.name === skill.skill_name);

              return (
                <div
                  key={skill.skill_name}
                  className="flex items-center justify-between p-3 border border-border rounded-lg"
                >
                  <div className="flex items-center gap-3 flex-1">
                    <input
                      type="checkbox"
                      checked={skill.is_enabled}
                      onChange={() => toggleSkillEnabled(skill.skill_name)}
                      className="h-4 w-4"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">
                          {skillInfo?.display_name || skill.skill_name}
                        </span>
                        <span className="text-xs px-2 py-0.5 bg-secondary text-secondary-foreground rounded">
                          Priority: {skill.priority}
                        </span>
                      </div>
                      {skillInfo?.description && (
                        <p className="text-sm text-muted-foreground">{skillInfo.description}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => moveSkillUp(index)}
                      disabled={index === 0}
                      className="p-2 text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <ArrowUp className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => moveSkillDown(index)}
                      disabled={index === selectedSkills.length - 1}
                      className="p-2 text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <ArrowDown className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => removeSkill(skill.skill_name)}
                      className="p-2 text-destructive hover:bg-destructive/10 rounded"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Info Box */}
      <div className="flex items-start gap-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5" />
        <div className="text-sm text-blue-600">
          <p className="font-medium mb-1">Skill Priority</p>
          <p>
            Skills are activated based on their priority order. Higher priority skills (at the top)
            are tried first when matching tasks.
          </p>
        </div>
      </div>

      {/* Validation Messages */}
      {validation?.warnings?.find((w: any) => w.field === 'skills') && (
        <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
          <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
          <p className="text-sm text-yellow-600">
            {validation.warnings.find((w: any) => w.field === 'skills').message}
          </p>
        </div>
      )}
    </div>
  );
}

export default SkillsStep;

