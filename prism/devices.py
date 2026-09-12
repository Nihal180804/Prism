"""Hardware capability detection and model placement.

Policy: the LLM runs on the GPU (through LM Studio); the audio models (STT/TTS)
default to the CPU so they don't compete for the scarce VRAM the LLM needs —
which is the common case on laptops (e.g. a 6 GB card). Only when there is ample
FREE VRAM does ``auto`` place an audio model on the GPU.

Detection prefers ``torch`` (gives live free-VRAM), falls back to ``nvidia-smi``,
and finally reports no GPU. Everything is pure/injectable so the placement policy
is unit-tested without any GPU present.
"""
import shutil
import logging
import subprocess

log = logging.getLogger("prism.devices")

NO_GPU = {"cuda": False, "name": None, "total_vram_mb": 0, "free_vram_mb": 0, "source": "none"}


def _via_torch():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()          # bytes, current device
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda": True, "name": props.name,
            "total_vram_mb": total // (1024 * 1024),
            "free_vram_mb": free // (1024 * 1024),
            "source": "torch",
        }
    except Exception:
        return None


def _via_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        name, total, free = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {
            "cuda": True, "name": name,
            "total_vram_mb": int(total), "free_vram_mb": int(free),
            "source": "nvidia-smi",
        }
    except Exception:
        return None


def detect_gpu():
    """Return a dict describing the GPU (or :data:`NO_GPU` when there is none)."""
    return _via_torch() or _via_nvidia_smi() or dict(NO_GPU)


def choose_device(pref, kind="audio", gpu=None, min_free_vram_mb=7000):
    """Resolve a device preference to a concrete 'cpu' or 'cuda'.

    ``pref`` is 'cpu' | 'cuda' | 'auto'. For ``kind='audio'`` (STT/TTS) 'auto'
    keeps the model on the CPU unless the GPU has at least ``min_free_vram_mb``
    free — leaving VRAM for the LLM. ``kind='llm'`` prefers the GPU whenever one
    exists.
    """
    gpu = gpu if gpu is not None else detect_gpu()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if gpu["cuda"] else "cpu"
    # auto
    if not gpu["cuda"]:
        return "cpu"
    if kind == "audio":
        return "cuda" if gpu["free_vram_mb"] >= min_free_vram_mb else "cpu"
    return "cuda"


def summary(settings=None):
    """Human-readable placement summary for logging / the recruiter view."""
    gpu = detect_gpu()
    if settings is not None:
        thr = settings.gpu_audio_min_free_vram_mb
        stt = choose_device(settings.stt_device, "audio", gpu, thr)
        tts = choose_device(settings.tts_device, "audio", gpu, thr)
    else:
        stt = tts = choose_device("auto", "audio", gpu)
    return {
        "gpu": gpu,
        "llm_device": "cuda (LM Studio)" if gpu["cuda"] else "cpu (LM Studio)",
        "stt_device": stt,
        "tts_device": tts,
    }
