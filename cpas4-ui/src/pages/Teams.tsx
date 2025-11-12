/**
 * Teams Page
 * 
 * Main page for team management with tabs for Canvas, List, and Templates
 */

import React, { useState } from 'react';
import { Tabs } from '../components/ui/tabs';
import { Button } from '../components/ui/button';
import { Dialog } from '../components/ui/dialog';
import TeamCanvas from '../components/teams/TeamCanvas';
import TeamList from '../components/agent-builder/TeamList';
import TeamTemplates from '../components/teams/TeamTemplates';
import AutoGenStudioSync from '../components/teams/AutoGenStudioSync';

export default function Teams() {
  const [activeTab, setActiveTab] = useState('canvas');
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleTeamSaved = (team: any) => {
    console.log('Team saved:', team);
    setRefreshKey(prev => prev + 1);
    setActiveTab('list');
  };

  const handleTemplateDeployed = (template: any, customizations: any) => {
    console.log('Template deployed:', template, customizations);
    setRefreshKey(prev => prev + 1);
    setActiveTab('list');
  };

  const handleImportSuccess = (team: any) => {
    console.log('Team imported:', team);
    setShowImportDialog(false);
    setRefreshKey(prev => prev + 1);
    setActiveTab('list');
  };

  const handleExportSuccess = (studioTeam: any) => {
    console.log('Team exported:', studioTeam);
    setShowExportDialog(false);
    alert('Team exported to AutoGen Studio successfully!');
  };

  const handleExportTeam = (teamId: string) => {
    setSelectedTeamId(teamId);
    setShowExportDialog(true);
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Agent Teams</h1>
            <p className="text-gray-600">Create and manage multi-agent teams</p>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={() => setShowImportDialog(true)}
              variant="outline"
              size="sm"
            >
              📥 Import from AutoGen Studio
            </Button>
            <Button
              onClick={() => setActiveTab('canvas')}
              variant="default"
              size="sm"
            >
              ➕ New Team
            </Button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full flex flex-col">
          {/* Tab Navigation */}
          <div className="bg-white border-b px-6">
            <div className="flex gap-4">
              <button
                onClick={() => setActiveTab('canvas')}
                className={`px-4 py-3 font-medium border-b-2 transition-colors ${
                  activeTab === 'canvas'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-600 hover:text-gray-900'
                }`}
              >
                🎨 Team Canvas
              </button>
              <button
                onClick={() => setActiveTab('list')}
                className={`px-4 py-3 font-medium border-b-2 transition-colors ${
                  activeTab === 'list'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-600 hover:text-gray-900'
                }`}
              >
                📋 My Teams
              </button>
              <button
                onClick={() => setActiveTab('templates')}
                className={`px-4 py-3 font-medium border-b-2 transition-colors ${
                  activeTab === 'templates'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-600 hover:text-gray-900'
                }`}
              >
                📚 Templates
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-auto">
            {activeTab === 'canvas' && (
              <TeamCanvas
                onSave={handleTeamSaved}
                onCancel={() => setActiveTab('list')}
                mode="create"
              />
            )}

            {activeTab === 'list' && (
              <div className="p-6">
                <TeamList
                  key={refreshKey}
                  onExport={handleExportTeam}
                />
              </div>
            )}

            {activeTab === 'templates' && (
              <TeamTemplates
                onDeploy={handleTemplateDeployed}
              />
            )}
          </div>
        </div>
      </div>

      {/* Import Dialog */}
      {showImportDialog && (
        <Dialog open={showImportDialog} onOpenChange={setShowImportDialog}>
          <AutoGenStudioSync
            mode="import"
            onSuccess={handleImportSuccess}
            onCancel={() => setShowImportDialog(false)}
          />
        </Dialog>
      )}

      {/* Export Dialog */}
      {showExportDialog && selectedTeamId && (
        <Dialog open={showExportDialog} onOpenChange={setShowExportDialog}>
          <AutoGenStudioSync
            mode="export"
            teamId={selectedTeamId}
            onSuccess={handleExportSuccess}
            onCancel={() => setShowExportDialog(false)}
          />
        </Dialog>
      )}
    </div>
  );
}

