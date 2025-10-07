import React from 'react';
import { apiService } from './api';

// Extend the Window interface to include speech recognition
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

export interface VoiceRecognitionResult {
  transcript: string;
  confidence: number;
  isFinal: boolean;
}

export interface VoiceSynthesisOptions {
  voice?: string;
  rate?: number;
  pitch?: number;
  volume?: number;
}

interface VoiceResponse {
  audio_url?: string;
}

class VoiceService {
  private recognition: any = null;
  private synthesis: SpeechSynthesis;
  private isListening = false;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor() {
    this.synthesis = window.speechSynthesis;
    this.initializeSpeechRecognition();
  }

  private initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
      this.recognition = new (window as any).webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
      this.recognition = new (window as any).SpeechRecognition();
    }

    if (this.recognition) {
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = 'en-US';

      this.recognition.onstart = () => {
        this.isListening = true;
        this.emit('start', {});
      };

      this.recognition.onend = () => {
        this.isListening = false;
        this.emit('end', {});
      };

      this.recognition.onerror = (event: any) => {
        this.emit('error', { error: event.error });
      };

      this.recognition.onresult = (event: any) => {
        const results = Array.from(event.results);
        const latest = results[results.length - 1] as any;

        const result: VoiceRecognitionResult = {
          transcript: latest[0].transcript,
          confidence: latest[0].confidence,
          isFinal: latest.isFinal
        };

        this.emit('result', result);
      };
    }
  }

  private emit(event: string, data: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in voice listener for ${event}:`, error);
        }
      });
    }
  }

  public on(event: string, listener: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);

    return () => {
      const eventListeners = this.listeners.get(event);
      if (eventListeners) {
        eventListeners.delete(listener);
        if (eventListeners.size === 0) {
          this.listeners.delete(event);
        }
      }
    };
  }

  public startListening(): boolean {
    if (!this.recognition) {
      console.warn('Speech recognition not supported');
      return false;
    }

    if (this.isListening) {
      console.warn('Already listening');
      return false;
    }

    try {
      this.recognition.start();
      return true;
    } catch (error) {
      console.error('Failed to start speech recognition:', error);
      return false;
    }
  }

  public stopListening(): boolean {
    if (!this.recognition || !this.isListening) {
      return false;
    }

    try {
      this.recognition.stop();
      return true;
    } catch (error) {
      console.error('Failed to stop speech recognition:', error);
      return false;
    }
  }

  public async speak(text: string, options: VoiceSynthesisOptions = {}): Promise<void> {
    try {
      // Try to use Greta's backend voice synthesis first
      const response = await apiService.synthesizeVoice(text, options.voice);
      
      if (response.status === 200 && (response.data as VoiceResponse)?.audio_url) {
        // Play the audio from the backend
        const audio = new Audio((response.data as VoiceResponse).audio_url!);
        return new Promise((resolve, reject) => {
          audio.onended = () => resolve();
          audio.onerror = () => reject(new Error('Failed to play audio'));
          audio.play();
        });
      }
    } catch (error) {
      console.warn('Backend voice synthesis failed, falling back to browser TTS:', error);
    }

    // Fallback to browser's built-in speech synthesis
    return this.speakWithBrowserTTS(text, options);
  }

  private speakWithBrowserTTS(text: string, options: VoiceSynthesisOptions = {}): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.synthesis) {
        reject(new Error('Speech synthesis not supported'));
        return;
      }

      const utterance = new SpeechSynthesisUtterance(text);
      
      // Configure voice options
      utterance.rate = options.rate || 1;
      utterance.pitch = options.pitch || 1;
      utterance.volume = options.volume || 1;

      // Try to find a female American English voice for Greta
      const voices = this.synthesis.getVoices();
      const femaleVoice = voices.find(voice =>
        voice.lang.startsWith('en-US') &&
        (voice.name.toLowerCase().includes('female') ||
         voice.name.toLowerCase().includes('samantha') ||
         voice.name.toLowerCase().includes('karen') ||
         voice.name.toLowerCase().includes('alex'))
      );

      if (femaleVoice) {
        utterance.voice = femaleVoice;
      }

      utterance.onend = () => resolve();
      utterance.onerror = (event) => reject(new Error(`Speech synthesis error: ${event.error}`));

      this.synthesis.speak(utterance);
    });
  }

  public stopSpeaking(): void {
    if (this.synthesis) {
      this.synthesis.cancel();
    }
  }

  public getAvailableVoices(): SpeechSynthesisVoice[] {
    return this.synthesis ? this.synthesis.getVoices() : [];
  }

  public isSpeechRecognitionSupported(): boolean {
    return this.recognition !== null;
  }

  public isSpeechSynthesisSupported(): boolean {
    return this.synthesis !== null;
  }

  public getListeningState(): boolean {
    return this.isListening;
  }
}

// Create and export singleton instance
export const voiceService = new VoiceService();

// React hook for using voice recognition
export const useVoiceRecognition = () => {
  const [isListening, setIsListening] = React.useState(false);
  const [transcript, setTranscript] = React.useState('');
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const unsubscribeStart = voiceService.on('start', () => {
      setIsListening(true);
      setError(null);
    });

    const unsubscribeEnd = voiceService.on('end', () => {
      setIsListening(false);
    });

    const unsubscribeResult = voiceService.on('result', (result: VoiceRecognitionResult) => {
      setTranscript(result.transcript);
    });

    const unsubscribeError = voiceService.on('error', (data: { error: string }) => {
      setError(data.error);
      setIsListening(false);
    });

    return () => {
      unsubscribeStart();
      unsubscribeEnd();
      unsubscribeResult();
      unsubscribeError();
    };
  }, []);

  const startListening = () => voiceService.startListening();
  const stopListening = () => voiceService.stopListening();

  return {
    isListening,
    transcript,
    error,
    startListening,
    stopListening,
    isSupported: voiceService.isSpeechRecognitionSupported(),
  };
};

// React hook for voice synthesis
export const useVoiceSynthesis = () => {
  const [isSpeaking, setIsSpeaking] = React.useState(false);

  const speak = async (text: string, options?: VoiceSynthesisOptions) => {
    setIsSpeaking(true);
    try {
      await voiceService.speak(text, options);
    } finally {
      setIsSpeaking(false);
    }
  };

  const stop = () => {
    voiceService.stopSpeaking();
    setIsSpeaking(false);
  };

  return {
    speak,
    stop,
    isSpeaking,
    isSupported: voiceService.isSpeechSynthesisSupported(),
    availableVoices: voiceService.getAvailableVoices(),
  };
};

export default voiceService;
