"""LLM access for Prism — a single, model-agnostic client for the local
OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, …)."""
from prism.llm.client import LLMClient, LLMError, default_client, chat

__all__ = ["LLMClient", "LLMError", "default_client", "chat"]
