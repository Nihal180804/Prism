"""Central configuration for Prism.

Every environment-driven setting is collected here so the rest of the code can
depend on a single ``settings`` object instead of scattered ``os.getenv()``
calls. Importing this module loads ``.env`` exactly once.

Step 1 migrates the server / session / camera / recruiter / dev settings that
``app.py`` uses. LLM and email settings (still read directly in ``backend.py``)
fold in during the LLMClient step.
"""
import os
import secrets
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

# Load .env once, at import time, before any Settings field is evaluated.
load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0") == "1"


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of Prism's configuration, read from the environment."""

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = os.getenv("HOST", "127.0.0.1")            # safe default: localhost only
    port: int = int(os.getenv("PORT", "5000"))
    cors_origins: str = os.getenv("CORS_ALLOWED_ORIGINS", "*")
    # A random key per start when unset — fine for local/dev, set it in .env to
    # keep sessions valid across restarts.
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY") or secrets.token_hex(32))

    # ── Recruiter auth ────────────────────────────────────────────────────────
    recruiter_user: str = os.getenv("RECRUITER_USER", "recruiter")
    recruiter_password: Optional[str] = os.getenv("RECRUITER_PASSWORD")   # unset ⇒ dashboard open (dev only)

    # ── Face monitor (browser-captured frames; see app.py /api/face) ────────────
    face_warn_seconds: float = float(os.getenv("FACE_WARN_SECONDS", "3"))
    face_exit_seconds: float = float(os.getenv("FACE_EXIT_SECONDS", "15"))
    face_auto_exit: bool = _get_bool("FACE_AUTO_EXIT", False)   # end interview when candidate leaves frame

    # ── Dev ─────────────────────────────────────────────────────────────────────
    dev_mode: bool = _get_bool("PRISM_DEV", False)             # enables the practice-session helper

    # ── LLM (OpenAI-compatible endpoint, e.g. LM Studio / Ollama / vLLM) ──────────
    # LLM_URL / LLM_MODEL are the canonical names; the older MISTRAL_* names are
    # still honoured so existing .env files keep working unchanged.
    llm_url: str = os.getenv("LLM_URL") or os.getenv("MISTRAL_URL", "http://127.0.0.1:1234/v1/chat/completions")
    llm_model: str = os.getenv("LLM_MODEL") or os.getenv("MISTRAL_MODEL", "mistral-7b-instruct-v0.3")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # ── Email (Gmail SMTP with an app password) ──────────────────────────────────
    email_sender: Optional[str] = os.getenv("EMAIL_SENDER")
    email_password: Optional[str] = os.getenv("EMAIL_PASSWORD")

    # ── Uploads ──────────────────────────────────────────────────────────────────
    # Hard cap on a single request body (resumes + JD together). Rejects oversized
    # uploads before they can exhaust memory or disk. Default 25 MB.
    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", str(25 * 1024 * 1024)))

    # ── Interviewer agent ────────────────────────────────────────────────────────
    # The interviewer runs as a tool-using agent: it asks the planned questions,
    # may ask up to `agent_max_followups` dynamic follow-ups in total, and records
    # a live assessment of each answer. Keep the budgets small so a weak local
    # model stays on-task; `agent_max_steps` is a hard safety cap on loop length.
    # ── Media features (per-interview) ───────────────────────────────────────────
    # These are the DEFAULTS the recruiter dashboard can override per session (the
    # chosen values are written into that session's meta.txt). When a feature is
    # off, the candidate UI hides its controls entirely — no dead space.
    tts_enabled_default: bool = _get_bool("TTS_ENABLED", False)        # Kokoro-82M reads questions aloud
    stt_enabled_default: bool = _get_bool("STT_ENABLED", False)        # browser mic → server Whisper
    camera_enabled_default: bool = _get_bool("CAMERA_ENABLED", False)  # webcam presence / face monitor
    clarify_enabled_default: bool = _get_bool("CLARIFY_ENABLED", True) # candidate may ask the AI to clarify
    clarify_max_per_question: int = int(os.getenv("CLARIFY_MAX_PER_QUESTION", "2"))

    # ── TTS (Kokoro-82M) ─────────────────────────────────────────────────────────
    tts_voice: str = os.getenv("TTS_VOICE", "af_heart")
    tts_lang_code: str = os.getenv("TTS_LANG_CODE", "a")   # kokoro: 'a' = American English
    tts_sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

    # ── STT (faster-whisper — the RealtimeSTT engine — on uploaded audio) ─────────
    stt_model: str = os.getenv("STT_MODEL", "base.en")
    stt_compute_type: str = os.getenv("STT_COMPUTE_TYPE", "int8")

    # ── Device placement ─────────────────────────────────────────────────────────
    # 'auto' | 'cpu' | 'cuda'. The LLM runs on the GPU (via LM Studio); audio models
    # default to CPU so scarce VRAM stays free for the LLM. 'auto' only puts an audio
    # model on the GPU when there is ample FREE VRAM (below).
    stt_device: str = os.getenv("STT_DEVICE", "auto")
    tts_device: str = os.getenv("TTS_DEVICE", "auto")
    gpu_audio_min_free_vram_mb: int = int(os.getenv("GPU_AUDIO_MIN_FREE_VRAM_MB", "7000"))

    agent_enabled: bool = _get_bool("AGENT_ENABLED", True)
    agent_max_followups: int = int(os.getenv("AGENT_MAX_FOLLOWUPS", "3"))
    # Extra CODING questions the agent may add when it isn't satisfied with a
    # candidate's coding answer (own budget so verbal probes can't use them up).
    agent_max_code_followups: int = int(os.getenv("AGENT_MAX_CODE_FOLLOWUPS", "2"))
    # A coding answer scoring at or below this correctness (0-100, from the live
    # review) is treated as "unsatisfactory" and nudges the agent to follow up
    # with another coding question while its budget lasts.
    agent_code_followup_threshold: int = int(os.getenv("AGENT_CODE_FOLLOWUP_THRESHOLD", "70"))
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "40"))

    # ── Interview rules ──────────────────────────────────────────────────────────
    # Shown as a pre-interview agreement gate the candidate must accept before
    # starting. Edit this list freely — order is preserved and each item renders
    # as one numbered rule.
    interview_rules: tuple = (
        "This is an individual assessment. Do not get help from other people, and do not use AI assistants, search engines, or other outside resources unless a question explicitly allows it.",
        "Stay in view of your camera for the whole interview. Leaving the frame, or another person appearing, may be flagged.",
        "Do not switch to other tabs, windows, or applications while the interview is running.",
        "Answer every question yourself — in your own words, and your own code.",
        "You get one submission per question. Once you submit an answer you cannot change it.",
        "Your answers and your camera presence are recorded and shared with the recruiter for evaluation.",
        "If you lose connection, rejoin with the same Session ID as soon as you can.",
    )

    # ── Data directories ────────────────────────────────────────────────────────
    sessions_dir: str = "Job/sessions"
    responses_dir: str = "Job/responses"
    done_dir: str = "Job/done"
    evaluations_dir: str = "Job/evaluations"
    jd_dir: str = "Job/Jd"
    resume_dir: str = "Job/resume"

    @property
    def data_dirs(self) -> List[str]:
        return [
            self.sessions_dir, self.responses_dir, self.done_dir,
            self.evaluations_dir, self.jd_dir, self.resume_dir,
        ]


# The single shared configuration instance.
settings = Settings()
