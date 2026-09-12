import os
import re
import logging
from datetime import datetime
from typing import Dict, List

from backend import ask_mistral, find_jd_file, extract_text_from_pdf
from prism import persistence
from prism.config import settings

log = logging.getLogger("prism.evaluate")

RESPONSES_DIR   = settings.responses_dir
EVALUATIONS_DIR = settings.evaluations_dir
DONE_DIR        = settings.done_dir
JD_DIR          = settings.jd_dir


def _load_jd_snippet() -> str:
    """Return the first 600 chars of the active JD, or a generic fallback."""
    try:
        jd_path = find_jd_file(JD_DIR)
        text = extract_text_from_pdf(jd_path)
        return text[:600].strip()
    except Exception:
        return ""


def _build_system_prompt(jd_snippet: str) -> str:
    role_context = (
        f"The candidate was interviewed for a role described as follows:\n\n{jd_snippet}\n\n"
        if jd_snippet else
        "The candidate was interviewed for a technical role.\n\n"
    )
    return (
        "You are a strict, impartial technical interviewer evaluating a candidate's interview responses.\n\n"
        + role_context +
        "Evaluation rules:\n"
        "- Any answer that says 'Next question', 'Skip', 'N/A', 'Not sure', or is clearly off-topic must score 0/100 "
        "with explicit feedback that the question was skipped or not answered.\n"
        "- Do NOT infer intent or guess meaning. Vague or insufficiently detailed answers lose points.\n"
        "- Only award points when the answer contains specific, relevant, technically sound information.\n"
        "- Award a communication score only when the candidate articulates clearly; a vague answer earns a low communication score too.\n\n"
        "Return your evaluation in EXACTLY this format — no extra text:\n"
        "OVERALL_SCORE: [0–100]\n"
        "TECHNICAL_SCORE: [0–100]\n"
        "COMMUNICATION_SCORE: [0–100]\n\n"
        "INDIVIDUAL_SCORES:\n"
        "Q1: [score] - [feedback]\n"
        "...\n\n"
        "STRENGTHS:\n- ...\n\n"
        "IMPROVEMENTS:\n- ...\n\n"
        "SUMMARY:\n[2–3 sentence overall assessment]"
    )


class InterviewEvaluator:
    def __init__(self):
        os.makedirs(EVALUATIONS_DIR, exist_ok=True)
        os.makedirs(DONE_DIR,        exist_ok=True)
        self.jd_snippet = _load_jd_snippet()

    # ── LLM ─────────────────────────────────────────────────────────────────

    def evaluate_with_llm(self, prompt: str) -> str:
        system = _build_system_prompt(self.jd_snippet)
        return ask_mistral(prompt, system_prompt=system, temperature=0.3, max_tokens=1500)

    # ── Parsing ──────────────────────────────────────────────────────────────

    def parse_response_file(self, filepath: str) -> Dict:
        """Read one interview record (JSON source-of-truth, or a legacy .txt).

        Delegates to prism.persistence so there is a single, well-tested reader;
        returns None on error to match the previous contract.
        """
        rec = persistence.read_interview(filepath)
        if rec is None:
            log.error("Error parsing %s", filepath)
        return rec

    def build_eval_prompt(self, qa_pairs: List[Dict], interviewer_notes: List[str] = None) -> str:
        lines = ["Evaluate each answer below and return scores in the required format.\n\nInterview Q&A:"]
        for qa in qa_pairs:
            lines.append(f"\nQ{qa['question_num']}: {qa['question']}")
            lines.append(f"A{qa['question_num']}: {qa['answer']}")
        # The interviewer agent records a private judgement of each answer as it
        # runs; surface those as context (not gospel) for the final scorer.
        if interviewer_notes:
            lines.append("\nInterviewer's live notes (for context only — judge the answers yourself):")
            lines.extend(f"- {n}" for n in interviewer_notes)
        return "\n".join(lines)

    @staticmethod
    def _interviewer_notes(candidate: Dict) -> List[str]:
        notes = []
        for t in candidate.get("transcript", []):
            a = t.get("assessment")
            if a and a.get("score") is not None:
                notes.append(f"Q{t.get('number')}: {a['score']}/100 — {a.get('comment', '')}".rstrip(" —"))
        return notes

    def parse_eval_response(self, response: str) -> Dict:
        if not response:
            return self._fallback("No response returned by the evaluation model")
        try:
            overall_m = re.search(r'OVERALL_SCORE:\s*(\d+)', response)
            # The model must at least emit OVERALL_SCORE in the required format.
            # If it's missing the output is unparseable — flag it as an error
            # rather than silently reporting a real-looking 0/100.
            if not overall_m:
                return self._fallback("Evaluation output did not match the required format")

            def _int(pattern, default):
                m = re.search(pattern, response)
                return int(m.group(1)) if m else default

            overall  = int(overall_m.group(1))
            tech     = _int(r'TECHNICAL_SCORE:\s*(\d+)',     overall)
            comm     = _int(r'COMMUNICATION_SCORE:\s*(\d+)', overall)

            individual = []
            for m in re.finditer(r'Q(\d+):\s*(\d+)\s*-\s*(.+?)(?=\nQ\d+:|\n\n|\nSTRENGTHS:|\Z)', response, re.DOTALL):
                individual.append({'question': int(m.group(1)), 'score': int(m.group(2)), 'feedback': m.group(3).strip()})

            def _bullets(section_re):
                m = re.search(section_re, response, re.DOTALL)
                if not m:
                    return []
                return [l.strip('- ').strip() for l in m.group(1).strip().splitlines() if l.strip()]

            strengths    = _bullets(r'STRENGTHS:\s*\n(.*?)(?=\n\n|\nIMPROVEMENTS:|\Z)')
            improvements = _bullets(r'IMPROVEMENTS:\s*\n(.*?)(?=\n\n|\nSUMMARY:|\Z)')
            summary_m    = re.search(r'SUMMARY:\s*\n(.*?)(?=\Z)', response, re.DOTALL)
            summary      = summary_m.group(1).strip() if summary_m else "Evaluation completed."

            return {
                'status':              'ok',
                'overall_score':       overall,
                'technical_score':     tech,
                'communication_score': comm,
                'individual_scores':   individual,
                'strengths':           strengths,
                'improvements':        improvements,
                'summary':             summary,
            }
        except Exception as e:
            log.error("Error parsing eval response: %s", e)
            return self._fallback(f"Error while parsing evaluation output: {e}")

    def _fallback(self, reason: str = "Unknown error") -> Dict:
        return {
            'status':              'error',
            'error_reason':        reason,
            'overall_score':       None,
            'technical_score':     None,
            'communication_score': None,
            'individual_scores':   [],
            'strengths':           [],
            'improvements':        [],
            'summary':             '',
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_evaluation(self, candidate: Dict, ev: Dict) -> str:
        """Persist the evaluation as JSON (source of truth) + a .txt report."""
        json_path, _txt_path = persistence.save_evaluation(EVALUATIONS_DIR, candidate, ev)
        return json_path

    def update_summary(self, candidate: Dict, ev: Dict):
        summary_file = os.path.join(EVALUATIONS_DIR, "evaluation_summary.txt")
        mode = 'a' if os.path.exists(summary_file) else 'w'
        with open(summary_file, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write("=== INTERVIEW EVALUATION SUMMARY ===\n\n")
            f.write(f"Candidate: {candidate['candidate_name']}\n")
            f.write(f"Session ID: {candidate['session_id']}\n")
            f.write(f"Overall Score: {ev['overall_score']}/100\n")
            f.write(f"Technical Score: {ev['technical_score']}/100\n")
            f.write(f"Communication Score: {ev['communication_score']}/100\n")
            f.write(f"Interview Date: {candidate['date']} {candidate['time']}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")

    # ── Main entry point ─────────────────────────────────────────────────────

    def process_all_responses(self):
        # list_interviews dedupes a session's JSON + .txt into one record and
        # exposes every backing file via `_paths` for archival.
        records = persistence.list_interviews([RESPONSES_DIR])
        if not records:
            log.info("No response files found.")
            return {'total': 0, 'evaluated': 0, 'failed': 0}

        log.info("Found %d interview record(s).", len(records))
        evaluated = failed = 0
        for candidate in records:
            log.info("Processing: %s", os.path.basename(candidate['filepath']))

            log.info("  Evaluating with the LLM…")
            prompt = self.build_eval_prompt(candidate['qa_pairs'], self._interviewer_notes(candidate))
            try:
                raw = self.evaluate_with_llm(prompt)
            except Exception as e:
                log.error("  LLM error: %s", e)
                raw = None

            ev = self.parse_eval_response(raw)

            if ev.get('status') != 'ok':
                # Evaluation could not be parsed. Do NOT save a report and do NOT
                # move the response to done/ — leave it so the next evaluation run
                # retries it, instead of leaving a misleading 0/100 report behind.
                log.warning("  Evaluation failed: %s", ev.get('error_reason'))
                log.warning("  Left in responses/ for retry on the next run.")
                failed += 1
                continue

            eval_path = self.save_evaluation(candidate, ev)
            log.info("  Report saved: %s", eval_path)
            self.update_summary(candidate, ev)
            log.info("  Overall score: %s/100", ev['overall_score'])
            persistence.move_to_done(candidate.get('_paths', [candidate['filepath']]), DONE_DIR)
            log.info("  Response moved to done/")
            evaluated += 1

        log.info("All done. Reports in '%s'.", EVALUATIONS_DIR)
        return {'total': len(records), 'evaluated': evaluated, 'failed': failed}


# ── Live code review (used by the candidate's coding panel) ───────────────────

_REVIEW_ICONS = {'good': '✅', 'warn': '⚠️', 'tip': '💡', 'bad': '❌'}


def parse_code_review(response: str) -> Dict:
    """Parse the LLM's code-review output into the shape the UI expects:
    {correctness, efficiency, style, feedback: [{icon, text}]}.
    Missing scores default to 0; malformed feedback lines still render.
    """
    def _score(label: str) -> int:
        m = re.search(rf'{label}:\s*(\d+)', response or '', re.I)
        return max(0, min(100, int(m.group(1)))) if m else 0

    feedback = []
    sec = re.search(r'FEEDBACK:\s*\n(.*)', response or '', re.S | re.I)
    if sec:
        for raw in sec.group(1).splitlines():
            line = raw.strip().lstrip('-').strip()
            if not line:
                continue
            tag = re.match(r'\[(\w+)\]\s*(.+)', line)
            if tag:
                feedback.append({'icon': _REVIEW_ICONS.get(tag.group(1).lower(), '•'), 'text': tag.group(2).strip()})
            else:
                feedback.append({'icon': '•', 'text': line})

    return {
        'correctness': _score('CORRECTNESS'),
        'efficiency':  _score('EFFICIENCY'),
        'style':       _score('STYLE'),
        'feedback':    feedback[:6],
    }


def review_code(code: str, language: str = 'python') -> Dict:
    """Ask the LLM for a quick, structured review of a code snippet."""
    system = (
        "You are a senior engineer reviewing a candidate's code during a live technical "
        "interview. Judge correctness, efficiency, and style. Be concise and specific."
    )
    prompt = (
        f"Language: {language}\n\nCandidate's code:\n```\n{code}\n```\n\n"
        "Return your review in EXACTLY this format and nothing else:\n"
        "CORRECTNESS: <0-100>\nEFFICIENCY: <0-100>\nSTYLE: <0-100>\n"
        "FEEDBACK:\n- [good] <what works>\n- [warn] <a bug, risk, or complexity concern>\n"
        "- [tip] <one concrete improvement>"
    )
    raw = ask_mistral(prompt, system_prompt=system, temperature=0.3, max_tokens=400)
    return parse_code_review(raw)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(RESPONSES_DIR,   exist_ok=True)
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)
    evaluator = InterviewEvaluator()
    evaluator.process_all_responses()


if __name__ == "__main__":
    main()
