import { useEffect, useState } from 'react';
import { Volume2, Mic, Globe, Save, Loader2, Plus, Trash2 } from 'lucide-react';
import { Button } from '../ui/button';
import { useVoiceStore } from '../../stores/voiceStore';

export function VoiceSettings() {
  // Use Zustand store
  const {
    profiles,
    languages,
    voices,
    isLoading,
    fetchProfiles,
    fetchLanguages,
    fetchVoices,
    createProfile,
    updateProfile,
    deleteProfile,
  } = useVoiceStore();

  // Local state for form inputs
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [selectedVoice, setSelectedVoice] = useState('default');
  const [profileName, setProfileName] = useState('Default Profile');
  const [speed, setSpeed] = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [volume, setVolume] = useState(1.0);

  const sttEngines = [
    { id: 'whisper', name: 'Whisper (Local)' },
    { id: 'google', name: 'Google Speech' },
    { id: 'azure', name: 'Azure Speech' },
    { id: 'mock', name: 'Mock (Testing)' },
  ];

  const ttsEngines = [
    { id: 'coqui', name: 'Coqui TTS (Local)' },
    { id: 'google', name: 'Google TTS' },
    { id: 'azure', name: 'Azure TTS' },
    { id: 'mock', name: 'Mock (Testing)' },
  ];

  // Load data on mount
  useEffect(() => {
    fetchProfiles();
    fetchLanguages();
    fetchVoices();
  }, [fetchProfiles, fetchLanguages, fetchVoices]);

  // Update voices when language changes
  useEffect(() => {
    if (selectedLanguage) {
      fetchVoices(selectedLanguage);
    }
  }, [selectedLanguage, fetchVoices]);

  const handleCreateProfile = async () => {
    await createProfile({
      name: profileName,
      language: selectedLanguage,
      voice: selectedVoice,
      speed,
      pitch,
      volume,
      isDefault: profiles.length === 0,
    });
  };

  const handleDeleteProfile = async (id: string) => {
    if (confirm('Are you sure you want to delete this profile?')) {
      await deleteProfile(id);
    }
  };

  if (isLoading && profiles.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading voice settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Voice Settings</h2>
        <p className="text-muted-foreground">
          Configure voice input and output preferences
        </p>
      </div>

      {/* Existing Profiles */}
      {profiles.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold">Saved Profiles</h3>
          <div className="grid gap-2">
            {profiles.map((profile) => (
              <div
                key={profile.id}
                className="flex items-center justify-between p-3 bg-muted rounded-md"
              >
                <div>
                  <p className="font-medium">{profile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {profile.language} • {profile.voice}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleDeleteProfile(profile.id)}
                  disabled={isLoading}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Create New Profile */}
      <div className="space-y-4 p-4 border border-border rounded-lg">
        <h3 className="font-semibold">Create New Profile</h3>

        {/* Profile Name */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Profile Name</label>
          <input
            type="text"
            value={profileName}
            onChange={(e) => setProfileName(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            placeholder="My Voice Profile"
            disabled={isLoading}
          />
        </div>

        {/* Language Selection */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Globe className="h-4 w-4 text-primary" />
            <label className="text-sm font-medium">Language</label>
          </div>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            disabled={isLoading}
          >
            {languages.length > 0 ? (
              languages.map((lang) => (
                <option key={lang} value={lang}>
                  {lang.toUpperCase()}
                </option>
              ))
            ) : (
              <option value="en">EN (English)</option>
            )}
          </select>
        </div>

        {/* Voice Selection */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Volume2 className="h-4 w-4 text-primary" />
            <label className="text-sm font-medium">Voice</label>
          </div>
          <select
            value={selectedVoice}
            onChange={(e) => setSelectedVoice(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            disabled={isLoading}
          >
            {voices.length > 0 ? (
              voices.map((voice) => (
                <option key={voice} value={voice}>
                  {voice}
                </option>
              ))
            ) : (
              <option value="default">Default Voice</option>
            )}
          </select>
        </div>

        {/* Voice Parameters */}
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Speed</label>
            <input
              type="number"
              min="0.5"
              max="2.0"
              step="0.1"
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-background border border-border rounded-md"
              disabled={isLoading}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Pitch</label>
            <input
              type="number"
              min="0.5"
              max="2.0"
              step="0.1"
              value={pitch}
              onChange={(e) => setPitch(parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-background border border-border rounded-md"
              disabled={isLoading}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Volume</label>
            <input
              type="number"
              min="0.0"
              max="1.0"
              step="0.1"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-background border border-border rounded-md"
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Create Button */}
        <Button
          onClick={handleCreateProfile}
          disabled={isLoading || !profileName}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Creating...
            </>
          ) : (
            <>
              <Plus className="h-4 w-4 mr-2" />
              Create Profile
            </>
          )}
        </Button>
      </div>

      {/* Engine Information */}
      <div className="space-y-3 p-4 bg-muted rounded-lg">
        <h3 className="font-semibold">Available Engines</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Mic className="h-4 w-4 text-primary" />
              <p className="text-sm font-medium">Speech-to-Text</p>
            </div>
            <ul className="text-sm text-muted-foreground space-y-1">
              {sttEngines.map((engine) => (
                <li key={engine.id}>• {engine.name}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Volume2 className="h-4 w-4 text-primary" />
              <p className="text-sm font-medium">Text-to-Speech</p>
            </div>
            <ul className="text-sm text-muted-foreground space-y-1">
              {ttsEngines.map((engine) => (
                <li key={engine.id}>• {engine.name}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
