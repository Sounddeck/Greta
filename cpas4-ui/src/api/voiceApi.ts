/**
 * Voice API Client for CPAS4/Greta
 * Connects to Voice Service backend (Week 21-23)
 */

const API_BASE = 'http://localhost:8000/api/v1/voice';

export interface STTResult {
  text: string;
  confidence: number;
  language: string;
  duration: number;
  engine: string;
  timestamp: string;
}

export interface TTSRequest {
  text: string;
  voice?: string;
  language?: string;
  engine?: string;
}

export interface VoiceProfile {
  profile_id: string;
  user_id: string;
  name: string;
  preferred_language: string;
  preferred_voice: string;
  settings?: Record<string, any>;
  created_at: string;
}

export interface VoiceStats {
  total_stt_requests: number;
  total_tts_requests: number;
  total_commands: number;
  total_conversations: number;
  avg_stt_time: number;
  avg_tts_time: number;
  languages_used: string[];
  commands_by_type: Record<string, number>;
}

export interface VoiceCommand {
  command: string;
  parameters: Record<string, any>;
  confidence: number;
  text: string;
  timestamp: string;
}

// ============================================================================
// Speech-to-Text API
// ============================================================================

export async function speechToText(
  audioFile: File,
  language?: string,
  engine?: string
): Promise<STTResult> {
  const formData = new FormData();
  formData.append('audio', audioFile);
  if (language) formData.append('language', language);
  if (engine) formData.append('engine', engine);

  const response = await fetch(`${API_BASE}/stt`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`STT failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Text-to-Speech API
// ============================================================================

export async function textToSpeech(request: TTSRequest): Promise<Blob> {
  const response = await fetch(`${API_BASE}/tts`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`TTS failed: ${response.statusText}`);
  }

  return response.blob();
}

// ============================================================================
// Voice Activity Detection API
// ============================================================================

export async function detectVoiceActivity(
  audioFile: File,
  threshold: number = 0.5
): Promise<{ has_voice: boolean; segments: Array<[number, number]>; threshold: number }> {
  const formData = new FormData();
  formData.append('audio', audioFile);
  formData.append('threshold', threshold.toString());

  const response = await fetch(`${API_BASE}/vad`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`VAD failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Voice Command API
// ============================================================================

export async function detectWakeWord(text: string): Promise<{ has_wake_word: boolean }> {
  const response = await fetch(`${API_BASE}/commands/detect-wake-word?text=${encodeURIComponent(text)}`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Wake word detection failed: ${response.statusText}`);
  }

  return response.json();
}

export async function parseVoiceCommand(text: string): Promise<VoiceCommand | null> {
  const response = await fetch(`${API_BASE}/commands/parse?text=${encodeURIComponent(text)}`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Command parsing failed: ${response.statusText}`);
  }

  const result = await response.json();
  return result.command ? result : null;
}

export async function executeVoiceCommand(text: string): Promise<any> {
  const response = await fetch(`${API_BASE}/commands/execute?text=${encodeURIComponent(text)}`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Command execution failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Voice Profile API
// ============================================================================

export async function createVoiceProfile(profile: {
  profile_id: string;
  user_id: string;
  name: string;
  preferred_language?: string;
  preferred_voice?: string;
}): Promise<VoiceProfile> {
  const response = await fetch(`${API_BASE}/profiles`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(profile),
  });

  if (!response.ok) {
    throw new Error(`Profile creation failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getVoiceProfile(profileId: string): Promise<VoiceProfile> {
  const response = await fetch(`${API_BASE}/profiles/${profileId}`);

  if (!response.ok) {
    throw new Error(`Profile retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function listVoiceProfiles(): Promise<VoiceProfile[]> {
  const response = await fetch(`${API_BASE}/profiles`);

  if (!response.ok) {
    throw new Error(`Profile listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function updateVoiceProfile(
  profileId: string,
  updates: {
    preferred_language?: string;
    preferred_voice?: string;
    settings?: Record<string, any>;
  }
): Promise<VoiceProfile> {
  const response = await fetch(`${API_BASE}/profiles/${profileId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    throw new Error(`Profile update failed: ${response.statusText}`);
  }

  return response.json();
}

export async function deleteVoiceProfile(profileId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/profiles/${profileId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Profile deletion failed: ${response.statusText}`);
  }
}

// ============================================================================
// Conversation Mode API
// ============================================================================

export async function startConversation(
  conversationId: string,
  profileId?: string
): Promise<{ started: boolean }> {
  const response = await fetch(`${API_BASE}/conversation/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ conversation_id: conversationId, profile_id: profileId }),
  });

  if (!response.ok) {
    throw new Error(`Conversation start failed: ${response.statusText}`);
  }

  return response.json();
}

export async function stopConversation(): Promise<any> {
  const response = await fetch(`${API_BASE}/conversation/stop`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Conversation stop failed: ${response.statusText}`);
  }

  return response.json();
}

export async function processVoiceInput(
  audioFile: File,
  profileId?: string
): Promise<any> {
  const formData = new FormData();
  formData.append('audio', audioFile);
  if (profileId) formData.append('profile_id', profileId);

  const response = await fetch(`${API_BASE}/conversation/process`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Voice input processing failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Statistics & Configuration API
// ============================================================================

export async function getVoiceStats(): Promise<VoiceStats> {
  const response = await fetch(`${API_BASE}/stats`);

  if (!response.ok) {
    throw new Error(`Stats retrieval failed: ${response.statusText}`);
  }

  return response.json();
}

export async function resetVoiceStats(): Promise<void> {
  const response = await fetch(`${API_BASE}/stats/reset`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Stats reset failed: ${response.statusText}`);
  }
}

export async function getSupportedLanguages(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/languages`);

  if (!response.ok) {
    throw new Error(`Language listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getSupportedVoices(engine?: string): Promise<string[]> {
  const url = engine ? `${API_BASE}/voices?engine=${engine}` : `${API_BASE}/voices`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Voice listing failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getAudioFormats(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/formats`);

  if (!response.ok) {
    throw new Error(`Format listing failed: ${response.statusText}`);
  }

  return response.json();
}

