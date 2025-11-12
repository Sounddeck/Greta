/**
 * Team List Component
 * Display and manage multi-agent teams
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, Plus, Edit, Trash2, Play, Eye, Upload } from 'lucide-react';
import { toast } from 'sonner';
import { useTeamStore } from '../../stores/teamStore';
import type { Team } from '../../types/team';

interface TeamListProps {
  onExport?: (teamId: string) => void;
}

const PATTERN_LABELS: Record<string, string> = {
  sequential: 'Sequential',
  autonomous: 'Autonomous',
  round_robin: 'Round Robin',
  selector: 'Selector'
};

const PATTERN_COLORS: Record<string, string> = {
  sequential: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  autonomous: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  round_robin: 'bg-green-500/20 text-green-300 border-green-500/30',
  selector: 'bg-orange-500/20 text-orange-300 border-orange-500/30'
};

export function TeamList({ onExport }: TeamListProps = {}) {
  const navigate = useNavigate();
  const { teams, isLoading, fetchTeams, deleteTeam } = useTeamStore();
  const [filter, setFilter] = useState<'all' | 'active' | 'archived'>('active');

  useEffect(() => {
    loadTeams();
  }, [filter]);

  const loadTeams = async () => {
    await fetchTeams(filter !== 'all' ? filter : undefined);
  };

  const handleDelete = async (teamId: string, teamName: string) => {
    if (!confirm(`Are you sure you want to delete team "${teamName}"?`)) {
      return;
    }

    try {
      await deleteTeam(teamId);
    } catch (error) {
      // Error already handled in store
    }
  };

  const handleExecute = (teamId: string) => {
    navigate(`/teams/${teamId}/execute`);
  };

  const getPatternLabel = (pattern: string) => {
    return PATTERN_LABELS[pattern] || pattern;
  };

  const getPatternColor = (pattern: string) => {
    return PATTERN_COLORS[pattern] || 'bg-gray-500/20 text-gray-300 border-gray-500/30';
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Teams</h2>
          <p className="text-gray-400 mt-1">
            Manage multi-agent teams and workflows
          </p>
        </div>
        <button
          onClick={() => navigate('/teams/new')}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <Plus className="h-4 w-4" />
          Create Team
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setFilter('all')}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
            filter === 'all'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          All
        </button>
        <button
          onClick={() => setFilter('active')}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
            filter === 'active'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Active
        </button>
        <button
          onClick={() => setFilter('archived')}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
            filter === 'archived'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Archived
        </button>
      </div>

      {/* Teams Grid */}
      {isLoading ? (
        <div className="text-center py-12 text-gray-400">Loading teams...</div>
      ) : teams.length === 0 ? (
        <div className="text-center py-12">
          <Users className="h-12 w-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">No teams found</p>
          <button
            onClick={() => navigate('/teams/new')}
            className="mt-4 text-blue-400 hover:text-blue-300"
          >
            Create your first team
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {teams.map((team) => (
            <div
              key={team.team_id}
              className="bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-blue-600 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white">{team.name}</h3>
                  <p className="text-sm text-gray-400 mt-1 line-clamp-2">
                    {team.description}
                  </p>
                </div>
              </div>

              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Pattern:</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium border ${getPatternColor(team.workflow_pattern)}`}>
                    {getPatternLabel(team.workflow_pattern)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Agents:</span>
                  <span className="text-white">{team.agents.length}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Termination:</span>
                  <span className="text-white text-xs">
                    {team.termination.type === 'max_turns' ? `${team.termination.value} turns` : team.termination.type}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleExecute(team.team_id)}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
                >
                  <Play className="h-3 w-3" />
                  Execute
                </button>
                <button
                  onClick={() => navigate(`/teams/${team.team_id}`)}
                  className="p-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
                  title="View Details"
                >
                  <Eye className="h-4 w-4" />
                </button>
                <button
                  onClick={() => navigate(`/teams/${team.team_id}/edit`)}
                  className="p-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
                  title="Edit"
                >
                  <Edit className="h-4 w-4" />
                </button>
                {onExport && (
                  <button
                    onClick={() => onExport(team.team_id)}
                    className="p-2 bg-green-900/20 hover:bg-green-900/40 text-green-400 rounded transition-colors"
                    title="Export to AutoGen Studio"
                  >
                    <Upload className="h-4 w-4" />
                  </button>
                )}
                <button
                  onClick={() => handleDelete(team.team_id, team.name)}
                  className="p-2 bg-red-900/20 hover:bg-red-900/40 text-red-400 rounded transition-colors"
                  title="Delete"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

