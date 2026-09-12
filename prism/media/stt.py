"""Speech-to-text on browser-recorded audio.

The candidate records in their browser and uploads the clip; the server
transcribes it here with faster-whisper — the same engine RealtimeSTT is built
on — running on the CPU by default so it doesn't take VRAM from the LLM. Optional
dependency: if faster-whisper isn't installed or transcription fails,
``transcribe`` returns ``None`` and the candidate just types instead.

(RealtimeSTT proper streams a *server-side* microphone; since capture here is in
the candidate's browser and uploaded, we drive its underlying faster-whisper
model directly on the uploaded audio.)
"""
import io
import logging

from prism.config import settings
from prism import devices

log = logging.getLogger("prism.media.stt")

_model = None
_model_device = None


def available() -> bool:
    """True if the faster-whisper library can be imported."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _get_model(device):
    global _model, _model_device
    if _model is None or _model_device != device:
        from faster_whisper import WhisperModel
        # int8 on CPU is fast and light; float16 is better on GPU.
        compute = settings.stt_compute_type if device == "cpu" else "float16"
        _model = WhisperModel(settings.stt_model, device=device, compute_type=compute)
        _model_device = device
        log.info("Whisper STT model '%s' ready on %s (%s)", settings.stt_model, device, compute)
    return _model


def transcribe(audio_bytes, device=None):
    """Transcribe an uploaded audio clip to text, or ``None`` on failure."""
    if not audio_bytes:
        return None
    try:
        dev = device or devices.choose_device(
            settings.stt_device, kind="audio",
            min_free_vram_mb=settings.gpu_audio_min_free_vram_mb,
        )
        model = _get_model(dev)
        segments, _info = model.transcribe(io.BytesIO(audio_bytes))
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log.warning("STT transcription failed (%s) — candidate can type instead", e)
        return None
