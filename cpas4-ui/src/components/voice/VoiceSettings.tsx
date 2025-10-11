import { useEffect, useState } from 'react';
import { Volume2, Mic, Globe, Save, Loader2, Plus, Trash2 } from 'lucide-react';
import { Button } from '../ui/button';
import { useVoiceStore } from '../../stores/voiceStore';
import { VoiceProfileSkeletonList } from '../skeletons/VoiceProfileSkeleton';

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

  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Voice Settings</h2>
        <p className="text-muted-foreground">
          Configure voice input and output preferences
        </p>
      </div>

      {/* Existing Profiles - Show skeleton during initial load */}
      {isLoading && profiles.length === 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Saved Profiles</h3>
          <VoiceProfileSkeletonList count={3} />
        </div>
      ) : profiles.length > 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Saved Profiles</h3>
          <div className="grid gap-2">
            {profiles.map((profile) => (
              <div key={profile.id} className="p-3 border border-border rounded-md">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="font-medium">{profile.name}</p>
                      {profile.isDefault && (
                        <span className="text-xs px-2 py-0.5 bg-primary/20 text-primary rounded-full">
                          Default
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Globe className="h-3 w-3" />
                        {profile.language}
                      </span>
                      <span className="flex items-center gap-1">
                        <Volume2 className="h-3 w-3" />
                        {profile.voice}
                      </span>
                    </div>
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
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Create New Profile */}
      <div className="space-y-4 p-4 border border-border rounded-lg">
        <h3 className="font-semibold">Create New Profile</h3>

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

        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Language</label>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded-md"
              disabled={isLoading}
            >
              {languages.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Voice</label>
            <select
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded-md"
              disabled={isLoading}
            >
              {voices.map((voice) => (
                <option key={voice.id} value={voice.id}>
                  {voice.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Speed: {speed.toFixed(1)}</label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-full"
              disabled={isLoading}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Pitch: {pitch.toFixed(1)}</label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={pitch}
              onChange={(e) => setPitch(parseFloat(e.target.value))}
              className="w-full"
              disabled={isLoading}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Volume: {volume.toFixed(1)}</label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.1"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="w-full"
              disabled={isLoading}
            />
          </div>
        </div>

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
    </div>
  );
}
