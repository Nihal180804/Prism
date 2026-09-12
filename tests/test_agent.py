"""Tests for the interviewer agent and its tool-decoding scaffolding.

These exercise the guardrails that make a small, unreliable local model safe to
put in the loop: no network is used — a fake session captures what the agent
asks, and scripted / offline LLMs stand in for the model.
"""
from prism.agents.base import parse_action, Action
from prism.agents.interviewer import InterviewerAgent


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeSession:
    """Captures the agent's questions and feeds back canned answers."""
    def __init__(self, questions, answers):
        self.questions  = questions
        self.transcript = []
        self.stopping   = False
        self.emitted    = []
        self._answers   = iter(answers)

    def emit(self, event, data):
        self.emitted.append((event, data))

    def begin_await(self):
        pass

    def wait_for_answer(self):
        return next(self._answers, "[No answer submitted]")


class OfflineLLM:
    def chat(self, *a, **k):
        raise RuntimeError("model offline")


class ScriptedLLM:
    """Returns each queued reply once, then defaults to finish."""
    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, *a, **k):
        return self._replies.pop(0) if self._replies else '{"action":"finish"}'


class FakeReviewer:
    """Stand-in for evaluate.review_code — returns a fixed correctness."""
    def __init__(self, correctness):
        self.correctness = correctness

    def __call__(self, code, language="python"):
        return {"correctness": self.correctness, "efficiency": 50, "style": 50, "feedback": []}


def _questions(n):
    return [{"type": "verbal", "text": f"Q{i}"} for i in range(1, n + 1)]


# ── Action decoding ───────────────────────────────────────────────────────────

def test_parse_action_nested_and_flattened():
    valid = ["ask_question", "probe"]
    a = parse_action('{"action":"probe","args":{"question":"why?"}}', valid)
    assert a.name == "probe" and a.args["question"] == "why?"
    b = parse_action('sure! {"action":"probe","question":"how?"} ok', valid)   # flattened + prose
    assert b.name == "probe" and b.args["question"] == "how?"


def test_parse_action_rejects_unknown_tool():
    assert parse_action('{"action":"delete_everything"}', ["ask_question"]) is None
    assert parse_action("not json at all", ["ask_question"]) is None


# ── Agent behaviour ───────────────────────────────────────────────────────────

def test_offline_llm_degrades_to_linear_interview():
    # With no model, the agent must still ask every planned question, in order,
    # with no follow-ups — a clean linear interview.
    s = FakeSession(_questions(3), ["a1", "a2", "a3"])
    InterviewerAgent(OfflineLLM()).run(s)
    assert [t["question"] for t in s.transcript] == ["Q1", "Q2", "Q3"]
    assert all(t["is_followup"] is False for t in s.transcript)


def test_probe_adds_one_followup():
    s = FakeSession(_questions(1), ["base answer", "follow-up answer"])
    llm = ScriptedLLM([
        '{"action":"ask_question"}',
        '{"action":"probe","args":{"question":"Can you elaborate?"}}',
        '{"action":"finish"}',
    ])
    InterviewerAgent(llm, max_followups=3).run(s, jd_snippet="A backend role.")
    assert len(s.transcript) == 2
    assert s.transcript[0]["is_followup"] is False
    assert s.transcript[1]["is_followup"] is True
    assert s.transcript[1]["question"] == "Can you elaborate?"


def test_followup_budget_is_enforced():
    # The model keeps trying to probe, but only max_followups may land.
    s = FakeSession(_questions(1), ["a"] + ["fu"] * 10)
    llm = ScriptedLLM(['{"action":"ask_question"}'] + ['{"action":"probe","args":{"question":"more?"}}'] * 10)
    InterviewerAgent(llm, max_followups=2, max_steps=30).run(s)
    followups = [t for t in s.transcript if t["is_followup"]]
    assert len(followups) == 2


def test_illegal_early_finish_still_covers_all_questions():
    # A model that just says "finish" from the start must not end the interview
    # early — the guardrail forces every planned question to be asked.
    s = FakeSession(_questions(2), ["a1", "a2"])
    llm = ScriptedLLM(['{"action":"finish"}'] * 20)
    InterviewerAgent(llm).run(s)
    assert [t["question"] for t in s.transcript] == ["Q1", "Q2"]


def test_code_challenge_adds_a_coding_followup():
    # A weak coding answer should be followed by another CODING question (which
    # flips the editor back on in the UI), and the code review is attached.
    s = FakeSession([{"type": "coding", "text": "Write two_sum(nums, target)."}],
                    ["def f(): pass", "def f2(): pass"])
    llm = ScriptedLLM([
        '{"action":"ask_question"}',
        '{"action":"code_challenge","args":{"question":"Now do it in O(n) time."}}',
        '{"action":"finish"}',
    ])
    InterviewerAgent(llm, max_code_followups=2, code_reviewer=FakeReviewer(30)).run(s)
    assert len(s.transcript) == 2
    assert s.transcript[0]["type"] == "coding" and s.transcript[0]["is_followup"] is False
    assert s.transcript[1]["type"] == "coding" and s.transcript[1]["is_followup"] is True
    assert s.transcript[1]["question"] == "Now do it in O(n) time."
    # The first coding answer was scored so the agent had a satisfaction signal.
    assert s.transcript[0]["code_review"]["correctness"] == 30


def test_code_challenge_budget_is_enforced():
    s = FakeSession([{"type": "coding", "text": "Q"}], ["a"] + ["b"] * 10)
    llm = ScriptedLLM(['{"action":"ask_question"}'] +
                      ['{"action":"code_challenge","args":{"question":"again"}}'] * 10)
    InterviewerAgent(llm, max_code_followups=1, max_steps=30, code_reviewer=FakeReviewer(10)).run(s)
    coding_followups = [t for t in s.transcript if t["is_followup"] and t["type"] == "coding"]
    assert len(coding_followups) == 1


def test_assess_records_and_clamps_score():
    s = FakeSession(_questions(1), ["a1"])
    llm = ScriptedLLM([
        '{"action":"ask_question"}',
        '{"action":"assess","args":{"score":250,"comment":"excellent"}}',
        '{"action":"finish"}',
    ])
    InterviewerAgent(llm).run(s)
    assert s.transcript[0]["assessment"] == {"score": 100, "comment": "excellent"}  # clamped
