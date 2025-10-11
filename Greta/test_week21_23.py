"""
Test Suite for Week 21-23: Voice Integration (FINAL FEATURE!)

Tests:
1. Speech-to-Text (STT) (transcribe audio, multiple engines)
2. Text-to-Speech (TTS) (synthesize speech, multiple engines)
3. Voice Activity Detection (VAD) (detect voice, speech segments)
4. Voice Commands (wake words, command parsing, execution)
5. Voice Profiles (create, get, update, delete)
6. Audio Processing (format conversion, enhancement, language detection)
7. Voice Conversation Mode (start, stop, process input)
8. Monitoring & Statistics (stats, supported features)

Target: 100% pass rate
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.voice_service import (
    VoiceService,
    AudioData,
    AudioFormat,
    STTEngine,
    TTSEngine,
    Language,
    VoiceCommand,
)

# Test results tracking
test_results = {"passed": 0, "failed": 0, "total": 0}


def print_test_header(component: str):
    """Print test component header"""
    print("\n" + "=" * 80)
    print(f"Component {component}")
    print("=" * 80 + "\n")


def print_test(test_num: int, total: int, description: str):
    """Print test description"""
    print(f"[Test {test_num}/{total}] {description}...")


def test_passed():
    """Mark test as passed"""
    test_results["passed"] += 1
    test_results["total"] += 1
    print("✅ PASSED\n")


def test_failed(error: str):
    """Mark test as failed"""
    test_results["failed"] += 1
    test_results["total"] += 1
    print(f"❌ FAILED: {error}\n")


# ============================================================================
# Component 1: Speech-to-Text (STT)
# ============================================================================


async def test_speech_to_text():
    """Test speech-to-text functionality"""
    print_test_header("1: Speech-to-Text (STT)")

    service = VoiceService()

    # Test 1: Basic STT
    print_test(1, 4, "Transcribe audio to text")
    try:
        audio = AudioData(
            data=b"mock audio data",
            format=AudioFormat.WAV,
            duration=2.0,
        )

        result = await service.speech_to_text(audio)
        assert result.text != ""
        assert result.confidence >= 0.0
        print(f"   Transcribed: '{result.text}'")
        print(f"   Confidence: {result.confidence}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: STT with language
    print_test(2, 4, "Transcribe with specific language")
    try:
        audio = AudioData(data=b"mock audio", format=AudioFormat.WAV)
        result = await service.speech_to_text(audio, language=Language.SPANISH)
        assert result.language == Language.SPANISH
        print(f"   Language: {result.language.value}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: STT with different engine
    print_test(3, 4, "Transcribe with different engine")
    try:
        audio = AudioData(data=b"mock audio", format=AudioFormat.WAV)
        result = await service.speech_to_text(audio, engine=STTEngine.WHISPER)
        assert result.engine == STTEngine.WHISPER
        print(f"   Engine: {result.engine.value}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: STT statistics
    print_test(4, 4, "Check STT statistics")
    try:
        stats = await service.get_stats()
        assert stats.total_stt_requests > 0
        print(f"   Total STT requests: {stats.total_stt_requests}")
        print(f"   Avg STT time: {stats.avg_stt_time:.3f}s")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 2: Text-to-Speech (TTS)
# ============================================================================


async def test_text_to_speech():
    """Test text-to-speech functionality"""
    print_test_header("2: Text-to-Speech (TTS)")

    service = VoiceService()

    # Test 1: Basic TTS
    print_test(1, 4, "Synthesize text to speech")
    try:
        result = await service.text_to_speech("Hello, this is a test")
        assert result.audio_data is not None
        assert len(result.audio_data.data) > 0
        print(f"   Synthesized: '{result.text}'")
        print(f"   Audio size: {len(result.audio_data.data)} bytes")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: TTS with voice
    print_test(2, 4, "Synthesize with specific voice")
    try:
        result = await service.text_to_speech("Test", voice="female")
        assert result.voice == "female"
        print(f"   Voice: {result.voice}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: TTS with language
    print_test(3, 4, "Synthesize with specific language")
    try:
        result = await service.text_to_speech("Test", language=Language.FRENCH)
        assert result.language == Language.FRENCH
        print(f"   Language: {result.language.value}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: TTS statistics
    print_test(4, 4, "Check TTS statistics")
    try:
        stats = await service.get_stats()
        assert stats.total_tts_requests > 0
        print(f"   Total TTS requests: {stats.total_tts_requests}")
        print(f"   Avg TTS time: {stats.avg_tts_time:.3f}s")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 3: Voice Activity Detection (VAD)
# ============================================================================


async def test_voice_activity_detection():
    """Test voice activity detection"""
    print_test_header("3: Voice Activity Detection (VAD)")

    service = VoiceService()

    # Test 1: Detect voice activity
    print_test(1, 2, "Detect voice in audio")
    try:
        audio = AudioData(data=b"audio with voice", format=AudioFormat.WAV)
        has_voice = await service.detect_voice_activity(audio)
        assert isinstance(has_voice, bool)
        print(f"   Voice detected: {has_voice}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Detect speech segments
    print_test(2, 2, "Detect speech segments")
    try:
        audio = AudioData(
            data=b"audio with speech",
            format=AudioFormat.WAV,
            duration=5.0,
        )
        segments = await service.detect_speech_segments(audio)
        assert isinstance(segments, list)
        print(f"   Found {len(segments)} speech segments")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 4: Voice Commands
# ============================================================================


async def test_voice_commands():
    """Test voice commands"""
    print_test_header("4: Voice Commands")

    service = VoiceService()

    # Test 1: Detect wake word
    print_test(1, 4, "Detect wake word")
    try:
        has_wake = await service.detect_wake_word("Hey Greta, how are you?")
        assert has_wake is True
        print(f"   Wake word detected: {has_wake}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Parse voice command
    print_test(2, 4, "Parse voice command")
    try:
        command = await service.parse_voice_command("Generate a story about cats")
        assert command is not None
        assert command.command == VoiceCommand.GENERATE
        print(f"   Command: {command.command.value}")
        print(f"   Confidence: {command.confidence}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Execute voice command
    print_test(3, 4, "Execute voice command")
    try:
        command = await service.parse_voice_command("Search for documents")
        assert command is not None
        result = await service.execute_voice_command(command)
        assert result["status"] == "success"
        print(f"   Execution result: {result['status']}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Command statistics
    print_test(4, 4, "Check command statistics")
    try:
        stats = await service.get_stats()
        assert stats.total_commands > 0
        print(f"   Total commands: {stats.total_commands}")
        print(f"   Commands by type: {stats.commands_by_type}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 5: Voice Profiles
# ============================================================================


async def test_voice_profiles():
    """Test voice profiles"""
    print_test_header("5: Voice Profiles")

    service = VoiceService()

    # Test 1: Create voice profile
    print_test(1, 5, "Create voice profile")
    try:
        profile = await service.create_voice_profile(
            profile_id="test_profile",
            user_id="user123",
            name="Test User",
            preferred_language=Language.ENGLISH,
        )
        assert profile.profile_id == "test_profile"
        print(f"   Created profile: {profile.name}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Get voice profile
    print_test(2, 5, "Get voice profile")
    try:
        profile = await service.get_voice_profile("test_profile")
        assert profile is not None
        assert profile.name == "Test User"
        print(f"   Retrieved profile: {profile.name}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: List voice profiles
    print_test(3, 5, "List voice profiles")
    try:
        profiles = await service.list_voice_profiles()
        assert len(profiles) >= 1
        print(f"   Found {len(profiles)} profiles")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Update voice profile
    print_test(4, 5, "Update voice profile")
    try:
        updated = await service.update_voice_profile(
            "test_profile",
            {"preferred_language": "es", "preferred_voice": "female"},
        )
        assert updated is not None
        assert updated.preferred_language == Language.SPANISH
        print(f"   Updated profile language: {updated.preferred_language.value}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Delete voice profile
    print_test(5, 5, "Delete voice profile")
    try:
        result = await service.delete_voice_profile("test_profile")
        assert result is True
        profile = await service.get_voice_profile("test_profile")
        assert profile is None
        print(f"   Deleted profile: test_profile")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 6: Audio Processing
# ============================================================================


async def test_audio_processing():
    """Test audio processing"""
    print_test_header("6: Audio Processing")

    service = VoiceService()

    # Test 1: Convert audio format
    print_test(1, 3, "Convert audio format")
    try:
        audio = AudioData(data=b"audio data", format=AudioFormat.WAV)
        converted = await service.convert_audio_format(audio, AudioFormat.MP3)
        assert converted.format == AudioFormat.MP3
        print(f"   Converted: {audio.format.value} -> {converted.format.value}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Enhance audio
    print_test(2, 3, "Enhance audio quality")
    try:
        audio = AudioData(data=b"noisy audio", format=AudioFormat.WAV)
        enhanced = await service.enhance_audio(audio, reduce_noise=True)
        assert enhanced.metadata.get("enhanced") is True
        print(f"   Enhanced audio: {enhanced.audio_id}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Detect language
    print_test(3, 3, "Detect language from audio")
    try:
        audio = AudioData(data=b"audio in language", format=AudioFormat.WAV)
        language = await service.detect_language(audio)
        assert isinstance(language, Language)
        print(f"   Detected language: {language.value}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 7: Voice Conversation Mode
# ============================================================================


async def test_voice_conversation():
    """Test voice conversation mode"""
    print_test_header("7: Voice Conversation Mode")

    service = VoiceService()

    # Test 1: Start conversation
    print_test(1, 4, "Start voice conversation")
    try:
        result = await service.start_conversation("conv_123")
        assert result is True
        assert service.conversation_mode is True
        print(f"   Conversation started: conv_123")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Process voice input
    print_test(2, 4, "Process voice input in conversation")
    try:
        audio = AudioData(
            data=b"hey greta generate a story",
            format=AudioFormat.WAV,
            duration=2.0,
        )
        result = await service.process_voice_input(audio)
        assert "transcription" in result
        print(f"   Processed voice input")
        print(f"   Transcription: {result.get('transcription')}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Conversation context
    print_test(3, 4, "Check conversation context")
    try:
        assert service.conversation_mode is True
        assert "conversation_id" in service.conversation_context
        print(f"   Conversation ID: {service.conversation_context['conversation_id']}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Stop conversation
    print_test(4, 4, "Stop voice conversation")
    try:
        context = await service.stop_conversation()
        assert service.conversation_mode is False
        assert "conversation_id" in context
        print(f"   Conversation stopped")
        print(f"   Messages: {len(context.get('messages', []))}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 8: Monitoring & Statistics
# ============================================================================


async def test_monitoring():
    """Test monitoring and statistics"""
    print_test_header("8: Monitoring & Statistics")

    service = VoiceService()

    # Test 1: Get statistics
    print_test(1, 5, "Get voice statistics")
    try:
        stats = await service.get_stats()
        assert stats.total_stt_requests >= 0
        assert stats.total_tts_requests >= 0
        print(f"   STT requests: {stats.total_stt_requests}")
        print(f"   TTS requests: {stats.total_tts_requests}")
        print(f"   Commands: {stats.total_commands}")
        print(f"   Conversations: {stats.total_conversations}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Get supported languages
    print_test(2, 5, "Get supported languages")
    try:
        languages = await service.get_supported_languages()
        assert len(languages) > 0
        print(f"   Supported languages: {len(languages)}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Get supported voices
    print_test(3, 5, "Get supported voices")
    try:
        voices = await service.get_supported_voices()
        assert len(voices) > 0
        print(f"   Supported voices: {len(voices)}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Get audio formats
    print_test(4, 5, "Get supported audio formats")
    try:
        formats = await service.get_audio_formats()
        assert len(formats) > 0
        print(f"   Supported formats: {formats}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Reset statistics
    print_test(5, 5, "Reset statistics")
    try:
        result = await service.reset_stats()
        assert result is True
        stats = await service.get_stats()
        assert stats.total_stt_requests == 0
        print(f"   Statistics reset")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Main Test Runner
# ============================================================================


async def run_all_tests():
    """Run all test components"""
    print("\n" + "=" * 80)
    print("WEEK 21-23 TEST SUITE: Voice Integration (FINAL FEATURE!)")
    print("=" * 80)
    print("Testing all voice integration components...\n")

    # Run test components
    await test_speech_to_text()
    await test_text_to_speech()
    await test_voice_activity_detection()
    await test_voice_commands()
    await test_voice_profiles()
    await test_audio_processing()
    await test_voice_conversation()
    await test_monitoring()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")

    if test_results["failed"] == 0:
        print("\n🎉 ALL TESTS PASSED! 100% Success Rate")
    else:
        success_rate = (
            test_results["passed"] / test_results["total"] * 100
            if test_results["total"] > 0
            else 0
        )
        print(f"\n⚠️  Success Rate: {success_rate:.1f}%")

    print("=" * 80 + "\n")

    return test_results["failed"] == 0


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)

