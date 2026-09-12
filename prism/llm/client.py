"""A small OpenAI-compatible chat client.

Prism talks to a locally hosted model (LM Studio, Ollama, vLLM, …) over the
OpenAI ``/v1/chat/completions`` API. This client centralises that call — the
endpoint, timeout, and retry-with-backoff — so nothing else in the codebase
touches ``requests`` or model details directly. It is deliberately model-
agnostic: the historical ``ask_mistral`` name in ``backend.py`` is kept only as
a thin backward-compatible shim over this.
"""
import time
import logging

import requests

log = logging.getLogger("prism.llm")


class LLMError(RuntimeError):
    """Raised when the LLM endpoint cannot return a usable completion."""


class LLMClient:
    """Minimal chat-completions client with timeout and bounded retries."""

    def __init__(self, url: str, model: str, timeout: float = 60.0, max_retries: int = 2):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.",
             temperature: float = 0.4, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self.url, json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:                      # network, HTTP, or shape error
                last_err = e
                if attempt < self.max_retries:
                    backoff = 1.5 ** attempt
                    log.warning("LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                                attempt + 1, self.max_retries + 1, e, backoff)
                    time.sleep(backoff)
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempt(s): {last_err}") from last_err


# ── Module-level default client, configured from settings ─────────────────────
from prism.config import settings

default_client = LLMClient(
    settings.llm_url, settings.llm_model,
    timeout=settings.llm_timeout, max_retries=settings.llm_max_retries,
)


def chat(prompt: str, system_prompt: str = "You are a helpful assistant.",
         temperature: float = 0.4, max_tokens: int = 512) -> str:
    """Convenience wrapper over :data:`default_client`."""
    return default_client.chat(prompt, system_prompt, temperature, max_tokens)
