"""Phase 14: Modular Voice Service for Speech-to-Text and Text-to-Speech."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import BinaryIO


class VoiceService:
    """Modular speech synthesis and audio processing."""

    def __init__(self, cache_dir: str | Path = "outputs/audio_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def text_to_speech_bytes(self, text: str, language: str = "english") -> tuple[bool, bytes, str]:
        """Convert text guidance to speech audio bytes (MP3 format).

        Returns (success, audio_bytes, mime_type_or_error)
        """
        clean_text = text.strip()
        if not clean_text:
            return False, b"", "Empty text provided."

        # Limit TTS length for fast synthesis
        if len(clean_text) > 800:
            clean_text = clean_text[:800] + "..."

        lang_code = "mr" if language.lower() in ["marathi", "mr", "मराठी"] else "en"

        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang=lang_code, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return True, buffer.read(), "audio/mpeg"
        except Exception as e:
            # Non-fatal error; system must remain operable
            return False, b"", f"TTS generation error: {str(e)}"

    def text_to_speech_file(self, text: str, output_filename: str, language: str = "english") -> Path | None:
        """Synthesize text and save to file in cache directory."""
        success, audio_bytes, _ = self.text_to_speech_bytes(text, language)
        if not success:
            return None
        out_path = self.cache_dir / output_filename
        out_path.write_bytes(audio_bytes)
        return out_path


_default_voice = None


def get_voice_service() -> VoiceService:
    global _default_voice
    if _default_voice is None:
        _default_voice = VoiceService()
    return _default_voice
