"""Pre-download the optional media models into the local cache.

The model weights are NOT committed to the repo (Kokoro ~330 MB, Whisper
~75–500 MB — too big for git). Instead they download from Hugging Face on first
use and cache under ~/.cache/huggingface. Run this once after installing the
media extras so the first interview doesn't pause to download:

    pip install -r requirements-media.txt
    python scripts/prefetch_models.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.config import settings


def main():
    warnings = 0

    # ── TTS: Kokoro-82M ────────────────────────────────────────────────────────
    try:
        from prism.media import tts
        if tts.available():
            print("Prefetching Kokoro-82M TTS (first run downloads ~330 MB)…")
            if tts.synthesize("Warming up the text to speech model.") is not None:
                print("  ✓ Kokoro ready.")
            else:
                warnings += 1
                print("  ! Kokoro imported but synthesis failed — check the log above.")
        else:
            print("  – Kokoro not installed (pip install kokoro soundfile) — skipping TTS.")
    except Exception as e:
        warnings += 1
        print(f"  ! TTS prefetch failed: {e}")

    # ── STT: faster-whisper ────────────────────────────────────────────────────
    try:
        from prism.media import stt
        if stt.available():
            print(f"Prefetching Whisper STT model '{settings.stt_model}'…")
            stt._get_model("cpu")            # triggers the download + load
            print("  ✓ Whisper ready.")
        else:
            print("  – faster-whisper not installed (pip install faster-whisper) — skipping STT.")
    except Exception as e:
        warnings += 1
        print(f"  ! STT prefetch failed: {e}")

    print("Done." if not warnings else f"Done with {warnings} warning(s).")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
