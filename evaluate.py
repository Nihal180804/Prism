import os
import glob
import re
import shutil
from datetime import datetime
from typing import Dict, List

from backend import ask_mistral, find_jd_file, extract_text_from_pdf

RESPONSES_DIR   = "Job/responses"
EVALUATIONS_DIR = "Job/evaluations"
DONE_DIR        = "Job/done"
JD_DIR          = "Job/Jd"


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
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            sid   = re.search(r'Session ID: (.+)',     content)
            name  = re.search(r'Candidate Name: (.+)', content)
            date  = re.search(r'Date: (.+)',            content)
            time_ = re.search(r'Time: (.+)',            content)

            qa_pairs = []
            for m in re.finditer(r'Q(\d+): (.+?)\nA\1: (.+?)(?=\n\nQ\d+:|\Z)', content, re.DOTALL):
                qa_pairs.append({
                    'question_num': int(m.group(1)),
                    'question':     m.group(2).strip(),
                    'answer':       m.group(3).strip(),
                })

            return {
                'session_id':     sid.group(1).strip()  if sid   else 'Unknown',
                'candidate_name': name.group(1).strip() if name  else 'Unknown',
                'date':           date.group(1).strip() if date  else 'Unknown',
                'time':           time_.group(1).strip() if time_ else 'Unknown',
                'qa_pairs':       qa_pairs,
                'filepath':       filepath,
            }
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None

    def build_eval_prompt(self, qa_pairs: List[Dict]) -> str:
        lines = ["Evaluate each answer below and return scores in the required format.\n\nInterview Q&A:"]
        for qa in qa_pairs:
            lines.append(f"\nQ{qa['question_num']}: {qa['question']}")
            lines.append(f"A{qa['question_num']}: {qa['answer']}")
        return "\n".join(lines)

    def parse_eval_response(self, response: str) -> Dict:
        if not response:
            return self._fallback()
        try:
            def _int(pattern, default=0):
                m = re.search(pattern, response)
                return int(m.group(1)) if m else default

            overall  = _int(r'OVERALL_SCORE:\s*(\d+)')
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
                'overall_score':       overall,
                'technical_score':     tech,
                'communication_score': comm,
                'individual_scores':   individual,
                'strengths':           strengths,
                'improvements':        improvements,
                'summary':             summary,
            }
        except Exception as e:
            print(f"Error parsing eval response: {e}")
            return self._fallback()

    def _fallback(self) -> Dict:
        return {
            'overall_score': 0, 'technical_score': 0, 'communication_score': 0,
            'individual_scores': [],
            'strengths': ['Unable to evaluate — system error'],
            'improvements': ['Please retry evaluation'],
            'summary': 'Evaluation could not be completed due to a system error. Please try again.',
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_evaluation(self, candidate: Dict, ev: Dict) -> str:
        now      = datetime.now()
        filename = f"evaluation_{candidate['candidate_name'].replace(' ','_')}_{candidate['session_id']}_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(EVALUATIONS_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== INTERVIEW EVALUATION REPORT ===\n\n")
            f.write(f"Candidate Name: {candidate['candidate_name']}\n")
            f.write(f"Session ID: {candidate['session_id']}\n")
            f.write(f"Interview Date: {candidate['date']}\n")
            f.write(f"Interview Time: {candidate['time']}\n")
            f.write(f"Evaluation Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=== SCORES ===\n")
            f.write(f"Overall Score: {ev['overall_score']}/100\n")
            f.write(f"Technical Score: {ev['technical_score']}/100\n")
            f.write(f"Communication Score: {ev['communication_score']}/100\n\n")

            if ev['individual_scores']:
                f.write("=== INDIVIDUAL QUESTION SCORES ===\n")
                for s in ev['individual_scores']:
                    f.write(f"Question {s['question']}: {s['score']}/100\n")
                    f.write(f"Feedback: {s['feedback']}\n\n")

            f.write("=== STRENGTHS ===\n")
            for s in ev['strengths']:
                f.write(f"• {s}\n")
            f.write("\n=== AREAS FOR IMPROVEMENT ===\n")
            for s in ev['improvements']:
                f.write(f"• {s}\n")
            f.write(f"\n=== SUMMARY ===\n{ev['summary']}\n\n")

            f.write("=== ORIGINAL Q&A PAIRS ===\n")
            for qa in candidate['qa_pairs']:
                f.write(f"Q{qa['question_num']}: {qa['question']}\n")
                f.write(f"A{qa['question_num']}: {qa['answer']}\n\n")

        return filepath

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

    def move_to_done(self, filepath: str):
        filename = os.path.basename(filepath)
        dest     = os.path.join(DONE_DIR, filename)
        if os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(DONE_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
        shutil.move(filepath, dest)

    # ── Main entry point ─────────────────────────────────────────────────────

    def process_all_responses(self):
        files = glob.glob(os.path.join(RESPONSES_DIR, "interview_responses_*.txt"))
        if not files:
            print("No response files found.")
            return

        print(f"Found {len(files)} response file(s).")
        for filepath in files:
            print(f"\nProcessing: {os.path.basename(filepath)}")
            candidate = self.parse_response_file(filepath)
            if not candidate:
                print("  Skipped — parse failed.")
                continue

            print("  Evaluating with Mistral…")
            prompt = self.build_eval_prompt(candidate['qa_pairs'])
            try:
                raw = self.evaluate_with_llm(prompt)
            except Exception as e:
                print(f"  LLM error: {e}")
                raw = None

            ev = self.parse_eval_response(raw)

            eval_path = self.save_evaluation(candidate, ev)
            print(f"  Report saved: {eval_path}")
            self.update_summary(candidate, ev)
            print(f"  Overall score: {ev['overall_score']}/100")
            self.move_to_done(filepath)
            print(f"  Response moved to done/")

        print(f"\nAll done. Reports in '{EVALUATIONS_DIR}'.")


def main():
    os.makedirs(RESPONSES_DIR,   exist_ok=True)
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)
    evaluator = InterviewEvaluator()
    evaluator.process_all_responses()


if __name__ == "__main__":
    main()
