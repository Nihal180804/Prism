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

    # ── Camera / face monitor ──────────────────────────────────────────────────
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    face_warn_seconds: float = float(os.getenv("FACE_WARN_SECONDS", "3"))
    face_exit_seconds: float = float(os.getenv("FACE_EXIT_SECONDS", "15"))
    face_auto_exit: bool = _get_bool("FACE_AUTO_EXIT", False)   # end interview when candidate leaves frame

    # ── Dev ─────────────────────────────────────────────────────────────────────
    dev_mode: bool = _get_bool("PRISM_DEV", False)             # enables the practice-session helper

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
