"""Unit tests for Prism's pure logic — parsing, extraction, and scoring.

These cover the deterministic pieces that don't need a webcam, microphone, or a
running LLM, so they run anywhere with `pytest`.
"""
import pytest

from backend import (
    parse_questions,
    extract_email_from_text,
    extract_name_from_text,
    build_question_prompt,
)
from evaluate import InterviewEvaluator, parse_code_review


# ── Question parsing ──────────────────────────────────────────────────────────

def test_parse_questions_tags_types():
    raw = "1. [VERBAL] What is a closure?\n2. [CODING] Reverse a linked list.\n3. Describe REST."
    qs = parse_questions(raw)
    assert qs[0] == {"type": "verbal", "text": "What is a closure?"}
    assert qs[1] == {"type": "coding", "text": "Reverse a linked list."}
    assert qs[2] == {"type": "verbal", "text": "Describe REST."}  # untagged → verbal


def test_parse_questions_falls_back_to_lines_when_unnumbered():
    raw = "[CODING] Reverse a linked list.\n[VERBAL] What is a closure?"
    qs = parse_questions(raw)
    assert [q["type"] for q in qs] == ["coding", "verbal"]
    assert qs[0]["text"] == "Reverse a linked list."


def test_parse_questions_ignores_blank_lines():
    raw = "1. First\n\n2. Second\n"
    qs = parse_questions(raw)
    assert [q["text"] for q in qs] == ["First", "Second"]
    assert all(q["type"] == "verbal" for q in qs)


# ── Résumé field extraction ───────────────────────────────────────────────────

def test_extract_email_found():
    assert extract_email_from_text("Reach me at jane.doe@example.com anytime") == "jane.doe@example.com"


def test_extract_email_missing_returns_none():
    assert extract_email_from_text("no address in this text") is None


def test_extract_name_uses_first_clean_line():
    text = "Jane Doe\njane@example.com\nSoftware Engineer"
    assert extract_name_from_text(text) == "Jane Doe"


def test_extract_name_skips_lines_with_symbols():
    text = "jane@example.com\nJane Doe"
    assert extract_name_from_text(text) == "Jane Doe"


def test_build_question_prompt_includes_both_inputs():
    prompt = build_question_prompt("RESUME_MARKER", "JD_MARKER")
    assert "RESUME_MARKER" in prompt
    assert "JD_MARKER" in prompt


# ── Evaluation parsing (the scoring-integrity fix) ────────────────────────────

@pytest.fixture
def evaluator():
    return InterviewEvaluator()


def test_parse_eval_well_formed(evaluator):
    resp = (
        "OVERALL_SCORE: 82\nTECHNICAL_SCORE: 80\nCOMMUNICATION_SCORE: 85\n\n"
        "INDIVIDUAL_SCORES:\nQ1: 90 - Strong grasp of fundamentals\nQ2: 70 - Somewhat vague\n\n"
        "STRENGTHS:\n- Clear communicator\n\nIMPROVEMENTS:\n- Deepen edge cases\n\n"
        "SUMMARY:\nA strong candidate overall."
    )
    ev = evaluator.parse_eval_response(resp)
    assert ev["status"] == "ok"
    assert ev["overall_score"] == 82
    assert ev["technical_score"] == 80
    assert ev["communication_score"] == 85
    assert ev["individual_scores"][0] == {
        "question": 1,
        "score": 90,
        "feedback": "Strong grasp of fundamentals",
    }
    assert ev["strengths"] == ["Clear communicator"]


def test_parse_eval_malformed_is_error_not_zero(evaluator):
    # A model that ignores the format must NOT look like a genuine 0/100.
    ev = evaluator.parse_eval_response("Sorry, I can't score this.")
    assert ev["status"] == "error"
    assert ev["overall_score"] is None
    assert ev["error_reason"]


def test_parse_eval_empty_is_error(evaluator):
    ev = evaluator.parse_eval_response(None)
    assert ev["status"] == "error"
    assert ev["overall_score"] is None


# ── Response-file round trip ──────────────────────────────────────────────────

def test_parse_response_file_roundtrip(evaluator, tmp_path):
    content = (
        "Session ID: abc12345\nCandidate Name: Jane Doe\n"
        "Date: 2026-01-01\nTime: 10:00:00\n\n"
        "Q1: What is X?\nA1: X is a thing.\n\n"
        "Q2: What is Y?\nA2: Y is another thing.\n\n"
    )
    f = tmp_path / "interview_responses_Jane_Doe_x_abc12345.txt"
    f.write_text(content, encoding="utf-8")

    parsed = evaluator.parse_response_file(str(f))
    assert parsed["session_id"] == "abc12345"
    assert parsed["candidate_name"] == "Jane Doe"
    assert len(parsed["qa_pairs"]) == 2
    assert parsed["qa_pairs"][0]["question"] == "What is X?"
    assert parsed["qa_pairs"][0]["answer"] == "X is a thing."


# ── Live code review parsing ──────────────────────────────────────────────────

def test_parse_code_review_well_formed():
    resp = (
        "CORRECTNESS: 70\nEFFICIENCY: 60\nSTYLE: 85\n"
        "FEEDBACK:\n- [good] Handles the base case correctly.\n"
        "- [warn] Nested loop is O(n^2).\n- [tip] Use a hash map.\n"
    )
    r = parse_code_review(resp)
    assert (r["correctness"], r["efficiency"], r["style"]) == (70, 60, 85)
    assert r["feedback"][0] == {"icon": "✅", "text": "Handles the base case correctly."}
    assert r["feedback"][1]["icon"] == "⚠️"
    assert r["feedback"][2]["icon"] == "💡"


def test_parse_code_review_clamps_and_defaults():
    r = parse_code_review("CORRECTNESS: 250\nnonsense output")
    assert r["correctness"] == 100          # clamped to 0–100
    assert r["efficiency"] == 0             # missing → default 0
    assert r["style"] == 0
    assert r["feedback"] == []


def test_parse_code_review_untagged_feedback_lines():
    r = parse_code_review("CORRECTNESS: 50\nEFFICIENCY: 50\nSTYLE: 50\nFEEDBACK:\n- Looks reasonable.")
    assert r["feedback"] == [{"icon": "•", "text": "Looks reasonable."}]
