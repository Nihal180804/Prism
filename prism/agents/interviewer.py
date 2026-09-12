"""The interviewer agent.

Instead of iterating a fixed list of questions, the interview is conducted by a
tool-using agent in a decide-act loop. The questions generated from the résumé
and JD are the *backbone* the agent must get through, but between answers the
agent decides — as an explicit tool call — whether to probe deeper with a
dynamic verbal follow-up, whether to hand the candidate another coding question
when it isn't satisfied with their code (guided by the live code review's
correctness score), and records a private assessment of each answer that later
informs evaluation.

Reliability first. A small local model is easily derailed, so the loop is
fenced on every side:

* it can never ``finish`` while planned questions remain,
* it can never exceed the follow-up budget,
* repeated no-op choices are broken by a deterministic policy, and
* any unparseable / errored decision falls back to "ask the next question".

The upshot: the interview always completes and always covers every planned
question — even if the LLM is offline (then it simply degrades to a clean linear
interview with no follow-ups). The agency is additive, never load-bearing.
"""
import logging
from typing import Dict, List, Optional

from prism.agents.base import Tool, Action, parse_action, tools_block

log = logging.getLogger("prism.agents.interviewer")

TOOLS: List[Tool] = [
    Tool("ask_question", "Ask the next planned interview question."),
    Tool("probe", "Ask ONE dynamic VERBAL follow-up to dig into the candidate's last answer.",
         '{"question": "<one or two sentences>"}'),
    Tool("code_challenge", "Give the candidate ANOTHER coding question to solve in the editor. "
         "Use this when their last coding answer was wrong, incomplete, or inefficient.",
         '{"question": "<a concrete coding task, one or two sentences>"}'),
    Tool("assess", "Record a private judgement of the candidate's last answer.",
         '{"score": <0-100>, "comment": "<one short sentence>"}'),
    Tool("finish", "End the interview. Valid ONLY once every planned question is asked."),
]

VALID = [t.name for t in TOOLS]

SYSTEM = (
    "You are a focused, fair senior technical interviewer conducting a live interview. "
    "You act by choosing exactly ONE tool per step and replying with a single JSON object "
    'of the form {"action": "<tool>", "args": {...}}. Output nothing except that JSON.'
)


def _clamp_score(v) -> Optional[int]:
    try:
        return max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return None


class InterviewerAgent:
    def __init__(self, llm, max_followups: int = 3, max_code_followups: int = 2,
                 max_steps: int = 40, code_reviewer=None, code_followup_threshold: int = 70,
                 speak=None):
        self.llm = llm
        self.max_followups = max_followups
        self.max_code_followups = max_code_followups
        self.max_steps = max_steps
        # Optional callable(session, text) that voices a question (Kokoro TTS).
        self.speak = speak
        # Optional callable(code, language) -> {'correctness', ...}. When given,
        # each coding answer is scored so the agent has a concrete "satisfaction"
        # signal instead of eyeballing raw code. Failures are swallowed.
        self.code_reviewer = code_reviewer
        self.code_followup_threshold = code_followup_threshold
        self.level = None     # role seniority; set in run(), calibrates follow-ups
        self._llm_ok = True   # breaker: once the model errors, stop calling it

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, session, jd_snippet: str = "", level: str = None) -> None:
        """Conduct the interview on ``session`` until every planned question is
        answered (or the session is stopped), appending each exchange to
        ``session.transcript``. ``level`` (role seniority) calibrates the
        difficulty of any dynamic follow-ups."""
        self.level = level
        plan: List[Dict] = session.questions
        total = len(plan)
        asked = 0            # planned questions asked so far
        followups = 0        # dynamic verbal probes used so far
        code_followups = 0   # dynamic coding challenges used so far
        last: Optional[Dict] = None   # most recent exchange (may still need assessing)
        noop_streak = 0
        steps = 0

        while not session.stopping and steps < self.max_steps:
            steps += 1
            action = self._decide(session, plan, asked, followups, code_followups, last, jd_snippet)

            if action.name == "ask_question":
                if asked >= total:
                    break
                q = plan[asked]
                asked += 1
                last = self._ask(session, q["text"], q.get("type", "verbal"),
                                 number=asked, total=total, asked_before=asked - 1,
                                 is_followup=False, difficulty=q.get("difficulty"))
                session.transcript.append(last)
                noop_streak = 0

            elif action.name == "probe":
                text = str(action.args.get("question") or "").strip()
                if last is None or followups >= self.max_followups or not text:
                    noop_streak += 1                     # illegal probe → re-decide
                else:
                    followups += 1
                    last = self._ask(session, text, "verbal",
                                     number=last["number"], total=total, asked_before=asked,
                                     is_followup=True, difficulty=None)
                    session.transcript.append(last)
                    noop_streak = 0

            elif action.name == "code_challenge":
                text = str(action.args.get("question") or "").strip()
                if last is None or code_followups >= self.max_code_followups or not text:
                    noop_streak += 1                     # illegal / budget spent → re-decide
                else:
                    code_followups += 1
                    last = self._ask(session, text, "coding",
                                     number=last["number"], total=total, asked_before=asked,
                                     is_followup=True, difficulty=self._followup_difficulty())
                    session.transcript.append(last)
                    noop_streak = 0

            elif action.name == "assess":
                if last is not None and last.get("assessment") is None:
                    last["assessment"] = {
                        "score":   _clamp_score(action.args.get("score")),
                        "comment": str(action.args.get("comment") or "").strip(),
                    }
                    noop_streak = 0
                else:
                    noop_streak += 1                     # nothing to assess → re-decide

            elif action.name == "finish":
                # Don't let the model wrap up while a weak coding answer still
                # deserves (and can afford) a follow-up challenge.
                if asked >= total and not self._weak_code_pending(last, code_followups):
                    break
                noop_streak += 1                         # early finish, or weak code unaddressed

            # Guard against a model that spins on no-op choices.
            if noop_streak >= 3:
                if asked < total:
                    q = plan[asked]
                    asked += 1
                    last = self._ask(session, q["text"], q.get("type", "verbal"),
                                     number=asked, total=total, asked_before=asked - 1,
                                     is_followup=False, difficulty=q.get("difficulty"))
                    session.transcript.append(last)
                    noop_streak = 0
                else:
                    break

        # Safety net: if the loop exited (max_steps) with questions unasked,
        # ask them deterministically so the interview is always complete.
        while not session.stopping and asked < total:
            q = plan[asked]
            asked += 1
            last = self._ask(session, q["text"], q.get("type", "verbal"),
                             number=asked, total=total, asked_before=asked - 1,
                             is_followup=False, difficulty=q.get("difficulty"))
            session.transcript.append(last)

    # ── Acting ────────────────────────────────────────────────────────────────

    def _followup_difficulty(self) -> str:
        """Difficulty for a dynamic coding challenge, calibrated to the role level."""
        return "medium" if self.level == "junior" else "hard"

    def _ask(self, session, text: str, qtype: str, number: int, total: int,
             asked_before: int, is_followup: bool, difficulty: str = None) -> Dict:
        """Emit a question to the candidate's tab, block for their answer, and
        return the exchange record. Progress reflects planned questions only, so
        a follow-up doesn't nudge the bar."""
        progress = round((asked_before / total) * 100) if total else 0
        session.current_question = {"number": number, "type": qtype, "text": text}
        session.emit("chat_message", {"sender": "🤖 Interviewer", "message": text})
        session.emit("question", {
            "number": number, "total": total, "text": text,
            "type": qtype, "difficulty": difficulty, "progress": progress,
        })
        session.emit("status", {"state": "coding" if qtype == "coding" else "answering"})
        if self.speak:
            self.speak(session, text)
        session.begin_await()
        answer = session.wait_for_answer()
        session.emit("chat_message", {"sender": "You", "message": answer})
        session.emit("status", {"state": "idle"})

        # For coding answers, score the code so the agent has a concrete
        # "satisfaction" signal when deciding whether to add another challenge.
        code_review = None
        if qtype == "coding" and self.code_reviewer and answer and not answer.startswith("["):
            try:
                code_review = self.code_reviewer(answer, "python")
            except Exception as e:
                log.info("code review during interview failed (%s) — ignoring", e)

        return {
            "number": number, "type": qtype, "question": text, "difficulty": difficulty,
            "answer": answer, "is_followup": is_followup,
            "assessment": None, "code_review": code_review,
        }

    # ── Deciding ──────────────────────────────────────────────────────────────

    def _weak_code_pending(self, last, code_followups) -> bool:
        """True when the last answer was coding, scored at/below the follow-up
        threshold, and a coding-challenge is still affordable."""
        if last is None or code_followups >= self.max_code_followups:
            return False
        if last.get("type") != "coding":
            return False
        cr = last.get("code_review")
        return bool(cr and cr.get("correctness") is not None
                    and cr["correctness"] <= self.code_followup_threshold)

    def _decide(self, session, plan, asked, followups, code_followups, last, jd_snippet) -> Action:
        """Ask the model for the next tool call, with a deterministic fallback
        that keeps the interview moving on any error or unparseable reply."""
        def fallback() -> Action:
            return Action("ask_question") if asked < len(plan) else Action("finish")

        # Breaker: after the model has failed once, don't keep retrying it on
        # every subsequent question — degrade straight to the linear policy so
        # an offline model doesn't stall each question with retry backoff.
        if not self._llm_ok:
            return fallback()

        try:
            raw = self.llm.chat(
                self._decision_prompt(plan, asked, followups, code_followups, last, jd_snippet),
                system_prompt=SYSTEM, temperature=0.3, max_tokens=220,
            )
        except Exception as e:
            log.info("interviewer decision LLM unavailable (%s) — degrading to linear", e)
            self._llm_ok = False
            return fallback()

        action = parse_action(raw, VALID)
        if action is None:
            log.info("interviewer decision unparseable — using fallback")
            return fallback()
        return action

    def _decision_prompt(self, plan, asked, followups, code_followups, last, jd_snippet) -> str:
        remaining = len(plan) - asked
        parts: List[str] = []
        if jd_snippet:
            parts.append(f"Role context:\n{jd_snippet}\n")
        if self.level:
            parts.append(f"Role level: {self.level}. Calibrate any follow-up or coding challenge "
                         f"to this seniority — harder and deeper for senior/staff, foundational for junior.\n")
        parts.append(tools_block(TOOLS))
        parts.append(
            f"\nProgress: {asked}/{len(plan)} planned questions asked ({remaining} remaining). "
            f"Verbal follow-ups used: {followups}/{self.max_followups}. "
            f"Coding challenges used: {code_followups}/{self.max_code_followups}."
        )

        last_is_weak_code = False
        if last is not None:
            parts.append(f"\nMost recent question ({last.get('type', 'verbal')}):\n{last['question']}")
            parts.append(f"Candidate's answer:\n{last['answer']}")
            review = last.get("code_review")
            if review and review.get("correctness") is not None:
                parts.append(
                    f"Automated review of that code — correctness {review['correctness']}/100, "
                    f"efficiency {review.get('efficiency')}/100, style {review.get('style')}/100."
                )
                last_is_weak_code = review["correctness"] <= self.code_followup_threshold
            if last.get("assessment") is not None:
                parts.append("(You have already assessed this answer.)")
        else:
            parts.append("\nThe interview has not started yet.")

        weak_pending = last_is_weak_code and code_followups < self.max_code_followups

        rules: List[str] = []
        # Coding follow-up leads when the last coding answer was weak.
        if last is not None and code_followups < self.max_code_followups:
            rules.append("- code_challenge gives ANOTHER coding question. Use it when the last "
                         "CODING answer was wrong, incomplete, or inefficient — give a targeted "
                         "retry or a simpler variant of the same skill.")
            if last_is_weak_code:
                rules.append("  IMPORTANT: the last coding answer scored low on the automated "
                             "review. Use code_challenge now — do NOT finish yet.")
        if remaining > 0:
            rules.append("- ask_question moves to the next planned question.")
        if last is not None and followups < self.max_followups:
            rules.append("- probe asks ONE verbal follow-up — only when a verbal answer was "
                         "strong-but-shallow or ambiguous. Prefer moving on.")
        if last is not None and last.get("assessment") is None:
            rules.append("- assess records a private score/comment for the last answer.")
        if weak_pending:
            rules.append("- finish is NOT allowed yet — address the weak coding answer first.")
        elif remaining > 0:
            rules.append("- finish is ONLY valid when no planned questions remain.")
        else:
            rules.append("- All planned questions are asked; use finish to end.")
        parts.append("\nRules:\n" + "\n".join(rules))
        parts.append(
            '\nReply with ONE JSON object. Examples: {"action":"ask_question"} · '
            '{"action":"probe","args":{"question":"..."}} · '
            '{"action":"code_challenge","args":{"question":"..."}} · '
            '{"action":"assess","args":{"score":70,"comment":"..."}} · '
            '{"action":"finish"}'
        )
        return "\n".join(parts)
