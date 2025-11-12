#!/usr/bin/env python3
"""
Record Wake Word Training Samples

This script helps you record audio samples for training a custom wake word model.
It will record multiple clips of you saying the wake word and save them as WAV files.

Usage:
    python record_wake_word_samples.py --wake-word "Hey Greta" --num-samples 50 --output-dir ./training_data

Requirements:
    pip install pyaudio wave

Author: GRETA PAI System
Date: October 20, 2025
"""

import pyaudio
import wave
import argparse
import os
from pathlib import Path
from datetime import datetime
import time


class WakeWordRecorder:
    """Record wake word samples for training"""
    
    SAMPLE_RATE = 16000  # 16kHz required by openWakeWord
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK_SIZE = 1024
    RECORD_SECONDS = 2  # Duration of each recording
    
    def __init__(self, wake_word: str, output_dir: str, num_samples: int = 50):
        """
        Initialize recorder
        
        Args:
            wake_word: The wake word to record (e.g., "Hey Greta")
            output_dir: Directory to save recordings
            num_samples: Number of samples to record
        """
        self.wake_word = wake_word
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        print(f"🎤 Wake Word Recorder")
        print(f"   Wake word: {self.wake_word}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Number of samples: {self.num_samples}")
        print(f"   Sample rate: {self.SAMPLE_RATE} Hz")
        print(f"   Recording duration: {self.RECORD_SECONDS} seconds")
        print()
    
    def record_sample(self, sample_number: int) -> str:
        """
        Record a single sample
        
        Args:
            sample_number: The sample number (1-indexed)
            
        Returns:
            Path to the saved WAV file
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.wake_word.lower().replace(' ', '_')}_{sample_number:03d}_{timestamp}.wav"
        filepath = self.output_dir / filename
        
        print(f"[{sample_number}/{self.num_samples}] Recording in 3 seconds...")
        print(f"   Say: '{self.wake_word}'")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("   🔴 RECORDING NOW!")
        
        # Open audio stream
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )
        
        # Record audio
        frames = []
        for _ in range(0, int(self.SAMPLE_RATE / self.CHUNK_SIZE * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        
        print("   ✅ Recording complete!")
        
        # Save to WAV file
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"   💾 Saved: {filepath}")
        print()
        
        return str(filepath)
    
    def record_all_samples(self):
        """Record all samples"""
        print("=" * 60)
        print("INSTRUCTIONS:")
        print("=" * 60)
        print(f"1. You will record {self.num_samples} samples")
        print(f"2. Each recording is {self.RECORD_SECONDS} seconds long")
        print(f"3. Say '{self.wake_word}' clearly when prompted")
        print("4. Try to vary:")
        print("   - Your tone (normal, excited, quiet)")
        print("   - Your speed (fast, slow, normal)")
        print("   - Your distance from the microphone")
        print("5. Press Ctrl+C to stop at any time")
        print("=" * 60)
        print()
        
        input("Press Enter to start recording...")
        print()
        
        recorded_files = []
        
        try:
            for i in range(1, self.num_samples + 1):
                filepath = self.record_sample(i)
                recorded_files.append(filepath)
                
                # Pause between recordings
                if i < self.num_samples:
                    print("   Pausing for 2 seconds...")
                    time.sleep(2)
                    print()
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Recording interrupted by user")
        
        finally:
            # Cleanup
            self.audio.terminate()
            
            # Summary
            print()
            print("=" * 60)
            print("RECORDING COMPLETE!")
            print("=" * 60)
            print(f"✅ Recorded {len(recorded_files)} samples")
            print(f"📁 Saved to: {self.output_dir}")
            print()
            print("Next steps:")
            print("1. Review the recordings to ensure quality")
            print("2. Delete any bad recordings")
            print("3. Upload to Google Colab for training:")
            print("   https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb")
            print("=" * 60)
    
    def test_microphone(self):
        """Test microphone and playback"""
        print("🎤 Testing microphone...")
        print("   Recording 3 seconds of audio...")
        
        # Open audio stream
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )
        
        # Record audio
        frames = []
        for _ in range(0, int(self.SAMPLE_RATE / self.CHUNK_SIZE * 3)):
            data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        
        print("   ✅ Recording successful!")
        print("   🔊 Playing back...")
        
        # Playback
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            output=True
        )
        
        for frame in frames:
            stream.write(frame)
        
        stream.stop_stream()
        stream.close()
        
        print("   ✅ Playback complete!")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Record wake word samples for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 50 samples of "Hey Greta"
  python record_wake_word_samples.py --wake-word "Hey Greta" --num-samples 50
  
  # Record 100 samples of "Greta" to a custom directory
  python record_wake_word_samples.py --wake-word "Greta" --num-samples 100 --output-dir ./my_training_data
  
  # Test microphone
  python record_wake_word_samples.py --test-mic
        """
    )
    
    parser.add_argument(
        '--wake-word',
        type=str,
        default='Hey Greta',
        help='Wake word to record (default: "Hey Greta")'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of samples to record (default: 50)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./wake_word_training_data',
        help='Output directory for recordings (default: ./wake_word_training_data)'
    )
    
    parser.add_argument(
        '--test-mic',
        action='store_true',
        help='Test microphone and exit'
    )
    
    args = parser.parse_args()
    
    # Create recorder
    recorder = WakeWordRecorder(
        wake_word=args.wake_word,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # Test microphone or record samples
    if args.test_mic:
        recorder.test_microphone()
    else:
        recorder.record_all_samples()


if __name__ == '__main__':
    main()

