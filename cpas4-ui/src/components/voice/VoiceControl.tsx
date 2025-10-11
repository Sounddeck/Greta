import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, VolumeX, Settings, User } from 'lucide-react';
import { Button } from '../ui/button';
import { toast } from 'sonner';
import { useVoiceStore } from '../../stores/voiceStore';
import { cn } from '../../lib/utils';

interface VoiceControlProps {
  onVoiceInput?: (text: string) => void;
  className?: string;
}

export function VoiceControl({ onVoiceInput, className }: VoiceControlProps) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [conversationMode, setConversationMode] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [volume, setVolume] = useState(0);
  
  // Use Zustand store
  const { isProcessing, speechToText, textToSpeech } = useVoiceStore();
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number>();
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Initialize audio context for visualization
  useEffect(() => {
    if (typeof window !== 'undefined' && 'AudioContext' in window) {
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Visualize audio input
  const visualizeAudio = () => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);

    const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
    setVolume(average / 255);

    animationFrameRef.current = requestAnimationFrame(visualizeAudio);
  };

  // Start listening
  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      if (audioContextRef.current && analyserRef.current) {
        const source = audioContextRef.current.createMediaStreamSource(stream);
        source.connect(analyserRef.current);
        visualizeAudio();
      }

      setIsListening(true);
      toast.success('Voice input started - speak now');

      // Set up MediaRecorder for audio capture
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          // Create audio blob from chunks
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

          // Send to backend STT service using Zustand store
          toast.info('Processing speech...');
          const text = await speechToText(audioBlob, 'en');

          if (text) {
            setTranscript(text);
            toast.success('Speech recognized!');

            if (onVoiceInput) {
              onVoiceInput(text);
            }
          } else {
            toast.warning('No speech detected');
          }
        } catch (error) {
          // Error handling is done in the store
          console.error('STT error:', error);
        } finally {
          audioChunksRef.current = [];
        }
      };

      // Start recording
      mediaRecorder.start();

      // Auto-stop after 5 seconds (or implement manual stop)
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          stream.getTracks().forEach(track => track.stop());
          setIsListening(false);
        }
      }, 5000);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast.error('Failed to access microphone');
      setIsListening(false);
    }
  };

  // Stop listening
  const stopListening = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    setIsListening(false);
    setVolume(0);
    toast.info('Voice input stopped');
  };

  // Toggle conversation mode
  const toggleConversationMode = () => {
    setConversationMode(!conversationMode);
    if (!conversationMode) {
      toast.success('Conversation mode enabled');
      startListening();
    } else {
      toast.info('Conversation mode disabled');
      stopListening();
    }
  };

  // Speak text (TTS) using Zustand store
  const speak = async (text: string) => {
    if (!text) {
      toast.error('No text to speak');
      return;
    }

    setIsSpeaking(true);

    try {
      toast.info('Generating speech...');

      // Call backend TTS service using Zustand store
      const audioBlob = await textToSpeech(text, 'default');

      // Play the audio
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      audio.onplay = () => {
        toast.success('Speaking...');
      };

      audio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = () => {
        toast.error('Failed to play audio');
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();

    } catch (error) {
      // Error handling is done in the store
      console.error('TTS error:', error);
      setIsSpeaking(false);
    }
  };

  // Stop speaking
  const stopSpeaking = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  return (
    <div className={cn('flex flex-col gap-4 p-4 bg-card rounded-lg border border-border', className)}>
      {/* Voice Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Voice Control</h3>
          {conversationMode && (
            <span className="text-xs px-2 py-1 bg-primary/20 text-primary rounded-full">
              Conversation Mode
            </span>
          )}
          {isProcessing && (
            <span className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-600 rounded-full animate-pulse">
              Processing...
            </span>
          )}
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => toast.info('Voice settings coming soon')}
        >
          <Settings className="h-4 w-4" />
        </Button>
      </div>

      {/* Audio Visualization */}
      {isListening && (
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-100"
              style={{ width: `${volume * 100}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground">{Math.round(volume * 100)}%</span>
        </div>
      )}

      {/* Transcript */}
      {transcript && (
        <div className="p-3 bg-muted rounded-md">
          <p className="text-sm text-muted-foreground mb-1">Transcript:</p>
          <p className="text-sm">{transcript}</p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2">
        {/* Microphone Button */}
        <Button
          variant={isListening ? 'destructive' : 'default'}
          size="sm"
          onClick={isListening ? stopListening : startListening}
          className="flex-1"
          disabled={isProcessing}
        >
          {isListening ? (
            <>
              <MicOff className="h-4 w-4 mr-2" />
              Stop Listening
            </>
          ) : (
            <>
              <Mic className="h-4 w-4 mr-2" />
              Start Listening
            </>
          )}
        </Button>

        {/* Speaker Button */}
        <Button
          variant={isSpeaking ? 'destructive' : 'outline'}
          size="sm"
          onClick={isSpeaking ? stopSpeaking : () => speak('Hello, I am Greta')}
          className="flex-1"
          disabled={isProcessing}
        >
          {isSpeaking ? (
            <>
              <VolumeX className="h-4 w-4 mr-2" />
              Stop Speaking
            </>
          ) : (
            <>
              <Volume2 className="h-4 w-4 mr-2" />
              Test Voice
            </>
          )}
        </Button>
      </div>

      {/* Conversation Mode Toggle */}
      <Button
        variant={conversationMode ? 'default' : 'outline'}
        size="sm"
        onClick={toggleConversationMode}
        className="w-full"
        disabled={isProcessing}
      >
        <User className="h-4 w-4 mr-2" />
        {conversationMode ? 'Exit Conversation Mode' : 'Start Conversation Mode'}
      </Button>

      {/* Voice Commands Help */}
      <div className="text-xs text-muted-foreground space-y-1">
        <p className="font-semibold">Wake Words:</p>
        <ul className="list-disc list-inside space-y-0.5 ml-2">
          <li>"Hey Greta"</li>
          <li>"Greta"</li>
          <li>"OK Greta"</li>
        </ul>
        <p className="font-semibold mt-2">Commands:</p>
        <ul className="list-disc list-inside space-y-0.5 ml-2">
          <li>"Generate [content]"</li>
          <li>"Search for [query]"</li>
          <li>"Create [item]"</li>
          <li>"Export [data]"</li>
          <li>"Stop" / "Help"</li>
        </ul>
      </div>
    </div>
  );
}
