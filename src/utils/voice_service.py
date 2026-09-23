"""Phase 14: High-Fidelity Neural Voice Service with Human-Like Expressive Speech."""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
from pathlib import Path
from typing import BinaryIO

# Pre-selected state-of-the-art Azure Neural voices via edge-tts
NEURAL_VOICES = {
    "english": "en-IN-NeerjaExpressiveNeural",  # Warm, natural, expressive Indian English
    "english_alt": "en-IN-PrabhatNeural",       # Natural Indian English male
    "marathi": "mr-IN-AarohiNeural",           # Clear, fluent, human-like Marathi
    "marathi_alt": "mr-IN-ManoharNeural",       # Natural Marathi male
}


class VoiceService:
    """Modular speech synthesis using Microsoft Azure Neural TTS with gTTS fallback."""

    def __init__(self, cache_dir: str | Path = "outputs/audio_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_for_speech(text: str, is_marathi: bool) -> str:
        """Strip markdown markers and expand abbreviations for smooth human-like cadence."""
        t = text
        # Remove bold, italics, code backticks, headers, bullet symbols
        t = re.sub(r"[*#_`~>]", "", t)
        t = re.sub(r"[•\-–—]\s*", " ", t)
        t = re.sub(r"https?://\S+", "", t)

        if is_marathi:
            t = t.replace("°C", " अंश सेल्सिअस ")
            t = t.replace("g/L", " ग्रॅम प्रति लिटर ")
            t = t.replace("ml/L", " मिली प्रति लिटर ")
            t = t.replace("%", " टक्के ")
            t = t.replace("@", " प्रमाणे ")
            t = t.replace("ICAR", " आयसीएआर ")
            t = t.replace("KVK", " केव्हीके ")
        else:
            t = t.replace("°C", " degrees Celsius ")
            t = t.replace("g/L", " grams per liter ")
            t = t.replace("ml/L", " milliliters per liter ")
            t = t.replace("%", " percent ")
            t = t.replace("@", " at ")
            t = t.replace("ICAR", " I.C.A.R. ")
            t = t.replace("KVK", " K.V.K. ")

        # Normalize whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def text_to_speech_bytes(self, text: str, language: str = "english") -> tuple[bool, bytes, str]:
        """Convert text guidance to natural human-like speech audio bytes (MP3 format).

        Returns (success, audio_bytes, mime_type_or_error)
        """
        raw_text = text.strip()
        if not raw_text:
            return False, b"", "Empty text provided."

        lang = (language or "english").lower().strip()
        is_marathi = lang in ["marathi", "mr", "मराठी"]
        clean_text = self._clean_for_speech(raw_text, is_marathi)

        # Truncate overly long text safely at sentence boundary (up to 1200 characters)
        if len(clean_text) > 1200:
            cut = clean_text[:1200]
            last_period = max(cut.rfind("."), cut.rfind("।"), cut.rfind("?"))
            clean_text = cut[:last_period + 1] if last_period > 600 else cut + "..."

        # Check local disk cache
        cache_key = hashlib.md5(f"{clean_text}_{'mr' if is_marathi else 'en'}".encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        if cache_file.exists():
            try:
                audio_data = cache_file.read_bytes()
                if len(audio_data) > 500:
                    return True, audio_data, "audio/mpeg"
            except Exception:
                pass

        # 1. Primary Engine: Neural Voice via edge-tts
        voice_name = NEURAL_VOICES["marathi"] if is_marathi else NEURAL_VOICES["english"]
        try:
            import edge_tts

            async def _synthesize() -> bytes:
                # Slight rate adjustment (-4%) makes agricultural instructions sound calm and authoritative
                communicate = edge_tts.Communicate(clean_text, voice_name, rate="-4%", pitch="+0Hz")
                data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        data += chunk["data"]
                return data

            audio_bytes = asyncio.run(_synthesize())
            if audio_bytes and len(audio_bytes) > 500:
                # Save to cache
                try:
                    cache_file.write_bytes(audio_bytes)
                except Exception:
                    pass
                return True, audio_bytes, "audio/mpeg"
        except Exception as e:
            print(f"Edge Neural TTS failed ({e}). Falling back to secondary engine...")

        # 2. Secondary Engine: gTTS Fallback
        try:
            from gtts import gTTS
            lang_code = "mr" if is_marathi else "en"
            tts = gTTS(text=clean_text[:600], lang=lang_code, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            data = buffer.read()
            if data and len(data) > 500:
                try:
                    cache_file.write_bytes(data)
                except Exception:
                    pass
                return True, data, "audio/mpeg"
        except Exception as e:
            print(f"gTTS fallback failed ({e})")

        return False, b"", "All voice synthesis engines unavailable."

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
