"""Kokoro-82M text-to-speech.

Synthesises a question to a WAV byte string the browser can play. Kokoro and its
audio deps are optional: if they're missing or synthesis fails, ``synthesize``
returns ``None`` and the caller just shows the question as text. The pipeline is
built once and cached; its device follows the audio-placement policy (CPU unless
there's ample free VRAM), so it won't crowd the LLM off the GPU.
"""
import io
import logging

from prism.config import settings
from prism import devices

log = logging.getLogger("prism.media.tts")

_pipeline = None
_pipeline_device = None


def available() -> bool:
    """True if the Kokoro library can be imported."""
    try:
        import kokoro  # noqa: F401
        return True
    except Exception:
        return False


def _get_pipeline(device):
    global _pipeline, _pipeline_device
    if _pipeline is None or _pipeline_device != device:
        from kokoro import KPipeline
        try:
            _pipeline = KPipeline(lang_code=settings.tts_lang_code, device=device)
        except TypeError:                       # older Kokoro without a device kwarg
            _pipeline = KPipeline(lang_code=settings.tts_lang_code)
        _pipeline_device = device
        log.info("Kokoro TTS pipeline ready on %s", device)
    return _pipeline


def synthesize(text, voice=None, device=None):
    """Return WAV bytes for ``text`` (mono, ``settings.tts_sample_rate``), or
    ``None`` if TTS is unavailable or fails."""
    if not text or not text.strip():
        return None
    try:
        import numpy as np
        import soundfile as sf
        dev = device or devices.choose_device(
            settings.tts_device, kind="audio",
            min_free_vram_mb=settings.gpu_audio_min_free_vram_mb,
        )
        pipe = _get_pipeline(dev)
        chunks = [audio for _, _, audio in pipe(text, voice=voice or settings.tts_voice)]
        if not chunks:
            return None
        wav = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, wav, settings.tts_sample_rate, format="WAV")
        return buf.getvalue()
    except Exception as e:
        log.warning("TTS synthesis failed (%s) — continuing text-only", e)
        return None
