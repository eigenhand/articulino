"""Articulino: small, fast German and English text to speech with zero-shot voice cloning.

>>> from articulino import TTS
>>> tts = TTS.from_pretrained("eigenhand/Articulino-DE-EN-Realtime-Stream-TTS")
>>> tts.set_voice("voice.wav")
>>> speech = tts.say("Guten Morgen! Der Termin ist um 15:30 Uhr.", path="morgen.wav")
"""
from .tts import DEFAULT_REPO, TTS, Speech, Voice

__version__ = "1.0.0"
__all__ = ["TTS", "Speech", "Voice", "DEFAULT_REPO", "__version__"]
