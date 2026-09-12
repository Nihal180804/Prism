"""Prism's agent layer.

The interview is driven by a tool-using agent rather than a fixed loop — see
:mod:`prism.agents.interviewer`. :mod:`prism.agents.base` holds the small,
model-agnostic scaffolding (a ``Tool``/``Action`` model and a robust JSON-action
decoder) that lets a local model that lacks native tool-calling still act as an
agent.
"""
from prism.agents.base import Tool, Action, parse_action
from prism.agents.interviewer import InterviewerAgent

__all__ = ["Tool", "Action", "parse_action", "InterviewerAgent"]
