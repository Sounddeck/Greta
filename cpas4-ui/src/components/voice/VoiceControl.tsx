import { useState, useRef, useEffect } from 'react';
import { Mic, Volume2, Loader2, Square } from 'lucide-react';
import { Button } from '../ui/button';
import { toast } from 'sonner';
import { useVoiceStore } from '../../stores/voiceStore';

interface VoiceControlProps {
  onVoiceInput?: (text: string) => void;
  className?: string;
}

export function VoiceControl({ onVoiceInput, className }: VoiceControlProps) {
  // Use Zustand store
  const { isProcessing, speechToText, textToSpeech } = useVoiceStore();

  // Local state for UI
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [audioLevel, setAudioLevel] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Visualize audio level during recording
  const visualizeAudio = (stream: MediaStream) => {
    audioContextRef.current = new AudioContext();
    analyserRef.current = audioContextRef.current.createAnalyser();
    const source = audioContextRef.current.createMediaStreamSource(stream);
    source.connect(analyserRef.current);
    analyserRef.current.fftSize = 256;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const updateLevel = () => {
      if (analyserRef.current) {
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        setAudioLevel(average / 255); // Normalize to 0-1
        animationFrameRef.current = requestAnimationFrame(updateLevel);
      }
    };

    updateLevel();
  };

  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // Visualize audio
      visualizeAudio(stream);

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        // Stop visualization
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        if (audioContextRef.current) {
          audioContextRef.current.close();
        }
        setAudioLevel(0);

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());

        try {
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
          }
        } catch (error) {
          // Error handling is done in the store
          console.error('STT error:', error);
        }
      };

      mediaRecorder.start();
      setIsListening(true);
      toast.success('Listening...');
    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast.error('Failed to access microphone');
    }
  };

  const stopListening = () => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
    }
  };

  const speak = async (text: string) => {
    if (!text) {
      toast.error('No text to speak');
      return;
    }

    try {
      setIsSpeaking(true);
      toast.info('Generating speech...');

      // Use Zustand store for TTS
      const audioBlob = await textToSpeech(text);

      if (audioBlob) {
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        audio.onended = () => {
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
          toast.success('Speech complete');
        };

        audio.onerror = () => {
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
          toast.error('Failed to play audio');
        };

        await audio.play();
      }
    } catch (error) {
      setIsSpeaking(false);
      // Error handling is done in the store
      console.error('TTS error:', error);
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center gap-3">
        {/* Voice Input Button */}
        <Button
          variant={isListening ? 'destructive' : 'default'}
          size="lg"
          onClick={isListening ? stopListening : startListening}
          disabled={isProcessing || isSpeaking}
          className="flex-1"
        >
          {isListening ? (
            <>
              <Square className="h-5 w-5 mr-2" />
              Stop Listening
            </>
          ) : (
            <>
              <Mic className="h-5 w-5 mr-2" />
              Start Listening
            </>
          )}
        </Button>

        {/* Text to Speech Button */}
        <Button
          variant="outline"
          size="lg"
          onClick={() => speak(transcript || 'Hello, this is a test.')}
          disabled={isProcessing || isListening || isSpeaking}
          className="flex-1"
        >
          {isSpeaking || isProcessing ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              {isProcessing ? 'Generating...' : 'Speaking...'}
            </>
          ) : (
            <>
              <Volume2 className="h-5 w-5 mr-2" />
              Speak
            </>
          )}
        </Button>
      </div>

      {/* Audio Level Visualizer */}
      {isListening && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Mic className="h-4 w-4 text-primary animate-pulse" />
            <span className="text-sm text-muted-foreground">Recording...</span>
          </div>
          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-100"
              style={{ width: `${audioLevel * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Processing Indicator */}
      {isProcessing && (
        <div className="flex items-center gap-2 p-3 bg-muted rounded-md">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          <span className="text-sm">Processing audio...</span>
        </div>
      )}

      {/* Transcript Display */}
      {transcript && (
        <div className="p-4 bg-muted rounded-lg">
          <p className="text-sm font-medium mb-1">Transcript:</p>
          <p className="text-sm">{transcript}</p>
        </div>
      )}

      {/* Usage Instructions */}
      <div className="text-xs text-muted-foreground space-y-1">
        <p>• Click "Start Listening" to record your voice</p>
        <p>• Click "Stop Listening" when done speaking</p>
        <p>• Click "Speak" to hear the text read aloud</p>
      </div>
    </div>
  );
}
