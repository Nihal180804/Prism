"""Optional media models: Kokoro-82M text-to-speech and Whisper speech-to-text.

Both are heavy, optional dependencies. Everything here lazy-imports them and
degrades gracefully — if a model or its library isn't installed, the feature is
simply unavailable and the interview continues text-only. Device placement comes
from :mod:`prism.devices` (audio on CPU by default; GPU only with ample VRAM).
"""
