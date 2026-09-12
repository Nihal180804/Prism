"""JSON-first persistence for interviews and evaluations.

These records used to be free-text files re-parsed with regexes, which broke
whenever a candidate's answer happened to contain a line like ``Q2:`` or a
score-shaped string. Each record is now written as JSON — the source of truth
every reader uses — with a human-readable ``.txt`` rendering written alongside
for recruiters who open the folder directly. The ``.txt`` is never parsed for
new records, so answer content can no longer corrupt a record.

Legacy ``.txt``-only files (written before this change) are still read via the
tolerant parsers here, so nothing already on disk is lost.
"""
import os
import re
import json
import glob
import shutil
from datetime import datetime
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

INTERVIEW_PREFIX  = "interview_responses_"
EVALUATION_PREFIX = "evaluation_"


# ── Interviews ────────────────────────────────────────────────────────────────

def _transcript_to_qa(transcript: List[Dict]) -> List[Dict]:
    return [
        {"question_num": i, "question": t["question"], "answer": t["answer"]}
        for i, t in enumerate(transcript, 1)
    ]


def render_interview_txt(record: Dict) -> str:
    """Human-readable rendering of an interview record (write-only; never parsed
    back for records that also have JSON)."""
    lines = [
        f"Session ID: {record['session_id']}",
        f"Candidate Name: {record['candidate_name']}",
        f"Date: {record['date']}",
        f"Time: {record['time']}",
        f"Agreed to interview rules: yes ({record.get('agreed_at') or 'unknown'})",
        "",
    ]
    for i, t in enumerate(record["transcript"], 1):
        tag = " (follow-up)" if t.get("is_followup") else ""
        lines.append(f"Q{i}{tag}: {t['question']}")
        lines.append(f"A{i}: {t['answer']}")
        assessment = t.get("assessment")
        if assessment and assessment.get("score") is not None:
            note = assessment.get("comment") or ""
            lines.append(f"[Interviewer note] {assessment['score']}/100 — {note}".rstrip(" —"))
        lines.append("")
    return "\n".join(lines)


def save_interview(responses_dir: str, record: Dict) -> Tuple[str, str]:
    """Write an interview record as ``.json`` (source of truth) + ``.txt``.
    Returns ``(json_path, txt_path)``."""
    os.makedirs(responses_dir, exist_ok=True)
    now = datetime.now()
    safe = record["candidate_name"].replace(" ", "_")
    base = f"{INTERVIEW_PREFIX}{safe}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{record['session_id']}"

    transcript = record["transcript"]
    record = {
        **record,
        "questions": [t["question"] for t in transcript],
        "responses": [t["answer"] for t in transcript],
        "qa_pairs":  _transcript_to_qa(transcript),
    }

    json_path = os.path.join(responses_dir, base + ".json")
    txt_path  = os.path.join(responses_dir, base + ".txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_interview_txt(record))
    return json_path, txt_path


def parse_interview_txt(content: str) -> Dict:
    """Tolerant parser for legacy ``.txt`` interview files."""
    def _find(pattern, default):
        m = re.search(pattern, content)
        return m.group(1).strip() if m else default

    qa_pairs = []
    for m in re.finditer(r'Q(\d+)[^\n:]*: (.+?)\nA\1: (.+?)(?=\n\nQ\d+|\Z)', content, re.DOTALL):
        qa_pairs.append({
            "question_num": int(m.group(1)),
            "question":     m.group(2).strip(),
            "answer":       m.group(3).strip(),
        })
    return {
        "session_id":     _find(r'Session ID: (.+)', "Unknown"),
        "candidate_name": _find(r'Candidate Name: (.+)', "Unknown"),
        "date":           _find(r'Date: (.+)', "Unknown"),
        "time":           _find(r'Time: (.+)', "Unknown"),
        "qa_pairs":       qa_pairs,
    }


def read_interview(path: str) -> Optional[Dict]:
    """Read one interview record from ``.json`` or legacy ``.txt``. Always
    returns a dict carrying at least ``session_id``, ``candidate_name``,
    ``date``, ``time``, ``qa_pairs`` and ``filepath`` — or ``None`` on error."""
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            rec.setdefault("qa_pairs", _transcript_to_qa(rec.get("transcript", [])))
        else:
            with open(path, "r", encoding="utf-8") as f:
                rec = parse_interview_txt(f.read())
        rec["filepath"] = path
        rec["filename"] = os.path.basename(path)
        return rec
    except Exception:
        return None


def list_interviews(dirs: List[str]) -> List[Dict]:
    """Return interview records across ``dirs``, deduped by session id (JSON wins
    over a same-session ``.txt``), newest first. Each record carries a
    ``_paths`` list of every file backing that session (for archival moves)."""
    by_session: Dict[str, Dict] = {}
    files: List[str] = []
    for base in dirs:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base), reverse=True):
            if name.startswith(INTERVIEW_PREFIX) and name.endswith((".json", ".txt")):
                files.append(os.path.join(base, name))

    # JSON first so it wins the dedupe over any sibling .txt.
    for path in sorted(files, key=lambda p: (not p.endswith(".json"), os.path.basename(p)), reverse=False):
        rec = read_interview(path)
        if not rec:
            continue
        sid = rec.get("session_id", "Unknown")
        if sid in by_session:
            by_session[sid]["_paths"].append(path)
            continue
        rec["_paths"] = [path]
        by_session[sid] = rec

    return sorted(by_session.values(),
                  key=lambda r: os.path.basename(r["filepath"]), reverse=True)


# ── Evaluations ───────────────────────────────────────────────────────────────

def render_evaluation_txt(candidate: Dict, ev: Dict) -> str:
    now = datetime.now()
    out = ["=== INTERVIEW EVALUATION REPORT ===", ""]
    out += [
        f"Candidate Name: {candidate['candidate_name']}",
        f"Session ID: {candidate['session_id']}",
        f"Interview Date: {candidate['date']}",
        f"Interview Time: {candidate['time']}",
        f"Evaluation Date: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== SCORES ===",
        f"Overall Score: {ev['overall_score']}/100",
        f"Technical Score: {ev['technical_score']}/100",
        f"Communication Score: {ev['communication_score']}/100",
        "",
    ]
    if ev.get("individual_scores"):
        out.append("=== INDIVIDUAL QUESTION SCORES ===")
        for s in ev["individual_scores"]:
            out.append(f"Question {s['question']}: {s['score']}/100")
            out.append(f"Feedback: {s['feedback']}")
            out.append("")
    out.append("=== STRENGTHS ===")
    out += [f"• {s}" for s in ev.get("strengths", [])]
    out.append("")
    out.append("=== AREAS FOR IMPROVEMENT ===")
    out += [f"• {s}" for s in ev.get("improvements", [])]
    out.append("")
    out.append("=== SUMMARY ===")
    out.append(ev.get("summary", ""))
    out.append("")
    out.append("=== ORIGINAL Q&A PAIRS ===")
    for qa in candidate.get("qa_pairs", []):
        out.append(f"Q{qa['question_num']}: {qa['question']}")
        out.append(f"A{qa['question_num']}: {qa['answer']}")
        out.append("")
    return "\n".join(out)


def save_evaluation(evaluations_dir: str, candidate: Dict, ev: Dict) -> Tuple[str, str]:
    """Write an evaluation as ``.json`` (source of truth) + ``.txt`` report."""
    os.makedirs(evaluations_dir, exist_ok=True)
    now  = datetime.now()
    safe = candidate["candidate_name"].replace(" ", "_")
    base = f"{EVALUATION_PREFIX}{safe}_{candidate['session_id']}_{now.strftime('%Y%m%d_%H%M%S')}"

    record = {
        "session_id":          candidate["session_id"],
        "candidate_name":      candidate["candidate_name"],
        "date":                candidate.get("date", "-"),
        "time":                candidate.get("time", "-"),
        "overall_score":       ev["overall_score"],
        "technical_score":     ev["technical_score"],
        "communication_score": ev["communication_score"],
        "strengths":           ev.get("strengths", []),
        "improvements":        ev.get("improvements", []),
        "summary":             ev.get("summary", ""),
        "individual_scores":   ev.get("individual_scores", []),
    }
    json_path = os.path.join(evaluations_dir, base + ".json")
    txt_path  = os.path.join(evaluations_dir, base + ".txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_evaluation_txt(candidate, ev))
    return json_path, txt_path


def parse_evaluation_txt(content: str) -> Dict:
    """Tolerant parser for legacy ``.txt`` evaluation reports."""
    def _int(pattern):
        m = re.search(pattern, content)
        return int(m.group(1)) if m else 0

    def _section(header, nxt):
        m = re.search(rf'=== {header} ===\n(.*?)(?=\n=== {nxt}|\Z)', content, re.DOTALL)
        if not m:
            return []
        return [l.strip("• \n") for l in m.group(1).strip().splitlines() if l.strip()]

    sid  = re.search(r'Session ID: (.+)', content)
    name = re.search(r'Candidate Name: (.+)', content)
    summary_m = re.search(r'=== SUMMARY ===\n(.*?)(?=\n===|\Z)', content, re.DOTALL)

    individual = []
    for m in re.finditer(r'Question (\d+): (\d+)/100\nFeedback: (.+?)(?=\n\n|\nQuestion|\Z)', content, re.DOTALL):
        individual.append({"q": int(m.group(1)), "score": int(m.group(2)), "feedback": m.group(3).strip()})

    return {
        "session_id":          sid.group(1).strip()  if sid  else "unknown",
        "candidate_name":      name.group(1).strip() if name else "Unknown",
        "overall_score":       _int(r'Overall Score: (\d+)/100'),
        "technical_score":     _int(r'Technical Score: (\d+)/100'),
        "communication_score": _int(r'Communication Score: (\d+)/100'),
        "strengths":           _section("STRENGTHS", "AREAS"),
        "improvements":        _section("AREAS FOR IMPROVEMENT", "SUMMARY"),
        "summary":             summary_m.group(1).strip() if summary_m else "",
        "individual_scores":   individual,
    }


def read_evaluation(path: str) -> Optional[Dict]:
    """Read one evaluation from ``.json`` or legacy ``.txt``; normalise
    ``individual_scores`` to the recruiter-UI shape (``q`` key)."""
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            rec["individual_scores"] = [
                {"q": s.get("q", s.get("question")), "score": s["score"], "feedback": s["feedback"]}
                for s in rec.get("individual_scores", [])
            ]
        else:
            with open(path, "r", encoding="utf-8") as f:
                rec = parse_evaluation_txt(f.read())
        rec["filename"] = os.path.basename(path)
        return rec
    except Exception:
        return None


def find_evaluation(evaluations_dir: str, session_id: str) -> Optional[str]:
    """Newest evaluation file (JSON preferred) for ``session_id``, or ``None``."""
    for ext in (".json", ".txt"):
        matches = sorted(
            glob.glob(os.path.join(evaluations_dir, f"{EVALUATION_PREFIX}*_{session_id}_*{ext}")),
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


# ── Archival ──────────────────────────────────────────────────────────────────

def move_to_done(paths: List[str], done_dir: str) -> None:
    """Move each path into ``done_dir``, de-clashing names with a timestamp."""
    os.makedirs(done_dir, exist_ok=True)
    for path in paths:
        if not os.path.exists(path):
            continue
        name = os.path.basename(path)
        dest = os.path.join(done_dir, name)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(name)
            dest = os.path.join(done_dir, f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
        shutil.move(path, dest)
