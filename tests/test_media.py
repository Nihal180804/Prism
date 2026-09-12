"""Tests for device placement, the clarify guardrail, and the face-presence
state machine (no models / GPU / camera needed)."""
from prism import devices
from prism.clarify import sanitize, clarify
from app import _evaluate_face_presence


# ── Device placement policy ───────────────────────────────────────────────────

CUDA_LOW  = {"cuda": True, "name": "RTX 3060", "total_vram_mb": 6144, "free_vram_mb": 4000, "source": "test"}
CUDA_HIGH = {"cuda": True, "name": "RTX 4090", "total_vram_mb": 24576, "free_vram_mb": 20000, "source": "test"}
NO_GPU    = {"cuda": False, "name": None, "total_vram_mb": 0, "free_vram_mb": 0, "source": "test"}


def test_explicit_device_preferences_win():
    assert devices.choose_device("cpu", "audio", CUDA_HIGH) == "cpu"
    assert devices.choose_device("cuda", "audio", CUDA_HIGH) == "cuda"
    assert devices.choose_device("cuda", "audio", NO_GPU) == "cpu"     # no GPU → cpu


def test_auto_keeps_audio_on_cpu_when_vram_is_scarce():
    # The common laptop case: a GPU exists but VRAM is needed for the LLM.
    assert devices.choose_device("auto", "audio", CUDA_LOW, min_free_vram_mb=7000) == "cpu"
    # Plenty of headroom → audio may use the GPU.
    assert devices.choose_device("auto", "audio", CUDA_HIGH, min_free_vram_mb=7000) == "cuda"
    # No GPU → cpu.
    assert devices.choose_device("auto", "audio", NO_GPU) == "cpu"


def test_auto_llm_prefers_gpu_whenever_present():
    assert devices.choose_device("auto", "llm", CUDA_LOW) == "cuda"
    assert devices.choose_device("auto", "llm", NO_GPU) == "cpu"


# ── Clarify guardrail ─────────────────────────────────────────────────────────

def test_sanitize_strips_fenced_code():
    out = sanitize("Here's a hint.\n```python\ndef solve(): return 42\n```\nGood luck.", "coding")
    assert "def solve" not in out
    assert "```" not in out
    assert "hint" in out.lower()


def test_sanitize_strips_code_lines_for_coding_questions():
    out = sanitize("Think about it like this:\ndef two_sum(nums):\n    return []", "coding")
    assert "def two_sum" not in out
    assert "return []" not in out


def test_sanitize_empty_returns_safe_refusal():
    assert "solution" in sanitize("", "coding").lower() or "understand" in sanitize("", "coding").lower()


class _LeakyLLM:
    """Pretends the model tried to leak a full solution."""
    def __init__(self): self.system = None
    def chat(self, prompt, system_prompt="", **kw):
        self.system = system_prompt
        return "Sure! ```python\ndef answer(): return 1\n```"


def test_clarify_uses_guardrail_prompt_and_sanitizes_output():
    llm = _LeakyLLM()
    out = clarify(llm, "Write two_sum", "coding", "just give me the code")
    assert "def answer" not in out and "```" not in out   # leak scrubbed
    assert "never" in llm.system.lower()                   # strict system prompt was used


class _DeadLLM:
    def chat(self, *a, **k): raise RuntimeError("offline")


def test_clarify_never_raises_on_llm_error():
    out = clarify(_DeadLLM(), "Q", "verbal", "help")
    assert isinstance(out, str) and out            # safe refusal, not an exception


# ── Face-presence state machine (browser-fed frames) ──────────────────────────

def _fresh_state():
    return {"missing_start": None, "warned": False, "multi_warned": False,
            "seen_face": False, "stop": False}


def test_no_face_warns_after_threshold_once():
    st = _fresh_state()
    # A face is seen first (so auto-exit is even eligible later).
    _evaluate_face_presence(st, 1, now=0.0, warn_s=3, exit_s=15, auto_exit=False)
    assert st["seen_face"] is True
    # Face disappears at t=10; no warning yet at t=11 (only 1s missing)...
    assert _evaluate_face_presence(st, 0, now=10.0, warn_s=3, exit_s=15, auto_exit=False) == []
    assert _evaluate_face_presence(st, 0, now=11.0, warn_s=3, exit_s=15, auto_exit=False) == []
    # ...but a warning fires once past the threshold, and not again after that.
    ev = _evaluate_face_presence(st, 0, now=14.0, warn_s=3, exit_s=15, auto_exit=False)
    assert ev and ev[0]["level"] == "warn"
    assert _evaluate_face_presence(st, 0, now=16.0, warn_s=3, exit_s=15, auto_exit=False) == []


def test_multiple_faces_flagged_once():
    st = _fresh_state()
    ev = _evaluate_face_presence(st, 2, now=1.0, warn_s=3, exit_s=15, auto_exit=False)
    assert ev and "Multiple faces" in ev[0]["message"]
    assert _evaluate_face_presence(st, 2, now=2.0, warn_s=3, exit_s=15, auto_exit=False) == []


def test_auto_exit_requires_a_prior_sighting():
    # Never seen a face → auto-exit must NOT fire even past the exit threshold.
    st = _fresh_state()
    _evaluate_face_presence(st, 0, now=0.0, warn_s=3, exit_s=15, auto_exit=True)
    _evaluate_face_presence(st, 0, now=100.0, warn_s=3, exit_s=15, auto_exit=True)
    assert st["stop"] is False
    # After a sighting, a long absence with auto-exit on ends the interview.
    st = _fresh_state()
    _evaluate_face_presence(st, 1, now=0.0, warn_s=3, exit_s=15, auto_exit=True)
    _evaluate_face_presence(st, 0, now=5.0, warn_s=3, exit_s=15, auto_exit=True)
    ev = _evaluate_face_presence(st, 0, now=25.0, warn_s=3, exit_s=15, auto_exit=True)
    assert st["stop"] is True
    assert any(e["level"] == "exit" for e in ev)
