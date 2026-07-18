"""
AI Voiceover Integration - offline TTS via pyttsx3 (no API key, no network needed).

Swap this out for ElevenLabs / OpenAI TTS if you have API keys and want higher quality
voices for the demo - just replace synthesize_speech()'s body and keep the same signature.
"""
import pyttsx3


def synthesize_speech(text: str, output_path: str, rate: int = 175, volume: float = 1.0) -> str:
    """Render `text` to a WAV file at output_path using the system TTS engine.

    A fresh engine instance is created and explicitly torn down per call - reusing
    a single global pyttsx3 engine across calls is unreliable on Linux (espeak driver).
    """
    if not text or not text.strip():
        text = " "
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass
    return output_path
