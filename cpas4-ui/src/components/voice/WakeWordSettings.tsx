/**
 * Wake Word Settings Component
 * Allows users to configure and control wake word detection
 */
import { useState, useEffect } from 'react';
import { useVoiceStore } from '@/stores/voiceStore';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Mic, MicOff, Activity } from 'lucide-react';

export function WakeWordSettings() {
  const {
    wakeWordEnabled,
    isListeningForWakeWord,
    wakeWordSensitivities,
    wakeWordStats,
    isLoading,
    startWakeWordDetection,
    stopWakeWordDetection,
    getWakeWordStatus,
    updateWakeWordSettings,
  } = useVoiceStore();
  
  const [sensitivity, setSensitivity] = useState(wakeWordSensitivities[0] || 0.5);
  
  useEffect(() => {
    // Poll for status every 5 seconds when listening
    if (isListeningForWakeWord) {
      const interval = setInterval(() => {
        getWakeWordStatus();
      }, 5000);
      
      return () => clearInterval(interval);
    }
  }, [isListeningForWakeWord, getWakeWordStatus]);
  
  const handleToggle = async () => {
    if (isListeningForWakeWord) {
      await stopWakeWordDetection();
    } else {
      // Update settings before starting
      updateWakeWordSettings(['greta'], [sensitivity]);
      await startWakeWordDetection();
    }
  };
  
  const handleSensitivityChange = (value: number[]) => {
    const newSensitivity = value[0];
    setSensitivity(newSensitivity);
    
    // Update settings if not currently listening
    if (!isListeningForWakeWord) {
      updateWakeWordSettings(['greta'], [newSensitivity]);
    }
  };
  
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Wake Word Detection
            </CardTitle>
            <CardDescription>
              Activate Greta with voice commands
            </CardDescription>
          </div>
          
          <Button
            onClick={handleToggle}
            disabled={isLoading}
            variant={isListeningForWakeWord ? "destructive" : "default"}
            className="gap-2"
          >
            {isListeningForWakeWord ? (
              <>
                <MicOff className="h-4 w-4" />
                Stop Listening
              </>
            ) : (
              <>
                <Mic className="h-4 w-4" />
                Start Listening
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Listening Indicator */}
        {isListeningForWakeWord && (
          <div className="flex items-center justify-center p-8">
            <div className="relative">
              <div className="w-24 h-24 rounded-full bg-blue-500/20 animate-pulse" />
              <Mic className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-12 w-12 text-blue-500" />
            </div>
          </div>
        )}
        
        {/* Sensitivity Slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">
              Sensitivity
            </label>
            <span className="text-sm text-muted-foreground">
              {sensitivity.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[sensitivity]}
            onValueChange={handleSensitivityChange}
            min={0}
            max={1}
            step={0.05}
            disabled={isListeningForWakeWord}
            className="mt-2"
          />
          <p className="text-xs text-muted-foreground">
            {isListeningForWakeWord 
              ? 'Stop listening to adjust sensitivity' 
              : 'Higher values = more sensitive (may trigger false positives)'}
          </p>
        </div>
        
        {/* Wake Word Info */}
        <div className="p-4 bg-muted rounded-lg space-y-2">
          <h4 className="text-sm font-medium">Wake Word</h4>
          <p className="text-2xl font-bold">"Greta"</p>
          <p className="text-xs text-muted-foreground">
            Say this word to activate the voice assistant
          </p>
        </div>
        
        {/* Statistics */}
        {wakeWordStats && (
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Total Detections</p>
              <p className="text-2xl font-bold">{wakeWordStats.total_detections}</p>
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Accuracy</p>
              <p className="text-2xl font-bold">{wakeWordStats.accuracy}%</p>
            </div>
            <div className="col-span-2 p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Last Detection</p>
              <p className="text-sm font-medium">
                {wakeWordStats.last_detection 
                  ? new Date(wakeWordStats.last_detection).toLocaleString()
                  : 'Never'}
              </p>
            </div>
          </div>
        )}
        
        {/* Help Text */}
        <div className="p-4 border border-blue-500/20 bg-blue-500/5 rounded-lg">
          <h4 className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-2">
            How to use
          </h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Click "Start Listening" to activate wake word detection</li>
            <li>• Say "Greta" clearly to trigger the assistant</li>
            <li>• Adjust sensitivity if detection is too sensitive or not sensitive enough</li>
            <li>• The system runs in the background and listens continuously</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

