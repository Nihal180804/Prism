"""Minimal agent scaffolding.

Local models (Mistral 7B in LM Studio, etc.) don't expose native tool-calling,
so an agent here works the ReAct way: it is shown a list of tools and asked to
reply with a single JSON object naming the tool and its arguments. The parsing
must be forgiving — small models wrap JSON in prose or fences and drift from the
schema — so :func:`parse_action` extracts the first usable object and rejects
anything that isn't a known tool. Callers pair this with a deterministic
fallback so an unparseable reply never stalls the loop.
"""
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("prism.agents")


@dataclass
class Tool:
    """One action the agent may take. ``args`` documents the expected argument
    shape for the model; it is descriptive, not enforced."""
    name: str
    description: str
    args: str = ""


@dataclass
class Action:
    """A decoded tool choice: the tool ``name`` and its parsed ``args``."""
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def _extract_json(text: str) -> Optional[dict]:
    """Return the first top-level JSON object found in ``text`` (or ``None``).

    Tries a ```json fenced block first, then the first ``{...}`` span. Both are
    common ways a small model returns structured output.
    """
    if not text:
        return None
    candidates: List[str] = []
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r'\{.*\}', text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def parse_action(text: str, valid: List[str]) -> Optional[Action]:
    """Decode a model reply into an :class:`Action`, or ``None`` if it names no
    known tool. Accepts both nested (``{"action":..,"args":{..}}``) and
    flattened (``{"action":..,"question":".."}``) argument forms."""
    obj = _extract_json(text)
    if not obj:
        return None
    name = str(obj.get("action") or obj.get("tool") or "").strip().lower()
    if name not in valid:
        return None
    args = obj.get("args")
    if not isinstance(args, dict):
        args = {k: v for k, v in obj.items() if k not in ("action", "tool")}
    return Action(name=name, args=args, raw=text)


def tools_block(tools: List[Tool]) -> str:
    """Render a tool list for a prompt, one line each."""
    lines = ["Available tools (choose exactly ONE per step):"]
    for t in tools:
        suffix = f" Args: {t.args}" if t.args else " No args."
        lines.append(f"- {t.name}: {t.description}{suffix}")
    return "\n".join(lines)
