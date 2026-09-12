import os
import uuid
import smtplib
import re
import logging
import fitz
import shutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from prism.config import settings
from prism.llm import chat as llm_chat, LLMError

log = logging.getLogger("prism.backend")

# All configuration now lives in prism/config.py. These module-level names are
# thin aliases kept so the rest of this file (and its tests) read unchanged.
SESSIONS_DIR   = settings.sessions_dir
JD_PATH        = settings.jd_dir
RESUME_PATH    = settings.resume_dir
DONE_PATH      = settings.done_dir
EMAIL_SENDER   = settings.email_sender
EMAIL_PASSWORD = settings.email_password


# ── LLM helper ──────────────────────────────────────────────────────────────

def ask_mistral(prompt, system_prompt="You are a helpful assistant.", temperature=0.4, max_tokens=512):
    """Backward-compatible shim over the model-agnostic LLM client
    (prism/llm/client.py). Kept so existing call sites and tests are untouched.
    """
    try:
        return llm_chat(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)
    except LLMError as e:
        raise RuntimeError(str(e)) from e


# ── PDF / text helpers ───────────────────────────────────────────────────────

def extract_text_from_pdf(path):
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_email_from_text(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    return emails[0] if emails else None


def extract_name_from_text(text):
    """Best-effort candidate name extraction from the first non-empty line."""
    for line in text.splitlines():
        line = line.strip()
        # Skip lines carrying contact-info punctuation (@ • / |) — those are
        # email/URL/separator lines, not a name. (Previously written [@|•|/],
        # which, inside a character class, matched a literal '|' rather than the
        # intended set — a real bug now fixed.)
        if line and not re.search(r'[@•/|]', line) and len(line.split()) <= 5:
            return line
    return None


# ── Role seniority ────────────────────────────────────────────────────────────
# The interview's difficulty is calibrated to the seniority the JD implies, so an
# SDE 1 posting gets fundamentals and an SDE 2 / Senior posting gets harder,
# deeper problems. Detection is a heuristic (keywords + years-of-experience) so
# it stays deterministic and unit-testable; the model then generates to the
# matching difficulty mix below.

LEVEL_PROFILES = {
    'junior': {
        'label': 'Junior / SDE 1 (entry-level, 0–2 years)',
        'guidance': (
            "Test fundamentals and basic implementation. Coding tasks are small, self-contained "
            "functions (arrays, strings, hash maps, simple recursion). Avoid system design and "
            "deep concurrency. Difficulty mix: about 3 EASY, 2 MEDIUM, 1 HARD."),
    },
    'mid': {
        'label': 'Mid-level / SDE 2 (2–5 years)',
        'guidance': (
            "Expect solid problem-solving with awareness of time/space complexity and edge cases. "
            "Coding tasks require picking an efficient approach and handling edge cases; include one "
            "applied API or data-modelling problem. Difficulty mix: about 1 EASY, 3 MEDIUM, 2 HARD."),
    },
    'senior': {
        'label': 'Senior / SDE 3 (5+ years)',
        'guidance': (
            "Emphasise depth, trade-offs, optimisation, concurrency, and reliability. Include at "
            "least one non-trivial algorithmic problem with a required time/space complexity, and one "
            "design question that forces explicit trade-offs. Difficulty mix: about 2 MEDIUM, 4 HARD."),
    },
    'staff': {
        'label': 'Staff / Principal (8+ years)',
        'guidance': (
            "Emphasise system architecture, scalability, ambiguity, and cross-cutting trade-offs. "
            "Include an open-ended system-design question and a hard algorithmic/optimisation problem "
            "with explicit constraints. Difficulty mix: about 1 MEDIUM, 5 HARD."),
    },
}

# Exact per-level slate of (type, difficulty) slots. We assign the type and
# difficulty of every question ourselves and let the model supply only the
# wording — so the calibration is GUARANTEED instead of depending on a small
# model honouring a requested distribution. Every slate is 6 slots with exactly
# 2 coding questions.
SLATES = {
    'junior': [('verbal', 'easy'), ('coding', 'easy'), ('verbal', 'easy'),
               ('verbal', 'medium'), ('coding', 'medium'), ('verbal', 'hard')],
    'mid':    [('verbal', 'easy'), ('verbal', 'medium'), ('coding', 'medium'),
               ('verbal', 'medium'), ('coding', 'hard'), ('verbal', 'hard')],
    'senior': [('verbal', 'medium'), ('coding', 'medium'), ('verbal', 'hard'),
               ('coding', 'hard'), ('verbal', 'hard'), ('verbal', 'hard')],
    'staff':  [('verbal', 'medium'), ('coding', 'hard'), ('verbal', 'hard'),
               ('coding', 'hard'), ('verbal', 'hard'), ('verbal', 'hard')],
}


def detect_seniority(jd_text):
    """Infer the role level from the JD. Returns 'junior' | 'mid' | 'senior' | 'staff'.
    Falls back to 'mid' when the JD gives no signal."""
    t = (jd_text or "").lower()

    def has(*patterns):
        return any(re.search(rf'\b{p}\b', t) for p in patterns)

    # Highest seniority wins (checked first) so "senior staff engineer" -> staff.
    if has(r'staff', r'principal', r'distinguished', r'architect', r'l[67]'):
        return 'staff'
    if has(r'senior', r'sr\.?', r'lead', r'sde\s*-?\s*(?:iii|3)', r'swe\s*-?\s*(?:iii|3)'):
        return 'senior'
    if has(r'sde\s*-?\s*(?:ii|2)', r'swe\s*-?\s*(?:ii|2)', r'mid[-\s]?level', r'intermediate'):
        return 'mid'
    if has(r'sde\s*-?\s*(?:i|1)', r'swe\s*-?\s*(?:i|1)', r'junior', r'jr\.?', r'entry[-\s]?level',
           r'new\s*grad(?:uate)?', r'intern(?:ship)?', r'associate', r'trainee'):
        return 'junior'
    # Years-of-experience fallback: use the lower bound of the first figure found.
    ym = re.search(r'(\d+)\s*(?:\+|-\s*\d+)?\s*years?', t)
    if ym:
        y = int(ym.group(1))
        return 'staff' if y >= 8 else 'senior' if y >= 5 else 'mid' if y >= 2 else 'junior'
    return 'mid'


# ── Question generation ──────────────────────────────────────────────────────

def build_question_prompt(resume_text, jd_text, level=None):
    level   = level or detect_seniority(jd_text)
    profile = LEVEL_PROFILES.get(level, LEVEL_PROFILES['mid'])
    slate   = SLATES.get(level, SLATES['mid'])

    slot_lines = "\n".join(
        f"  {i + 1}. [{t.upper()}] [{d.upper()}] — a {d} "
        f"{'coding task (candidate writes code)' if t == 'coding' else 'conceptual/design question'}"
        for i, (t, d) in enumerate(slate)
    )
    return (
        "You are a senior technical interviewer preparing a personalised, level-calibrated interview.\n\n"
        f"ROLE LEVEL: {profile['label']}\n{profile['guidance']}\n\n"
        "Write exactly ONE question for each numbered slot below, honouring that slot's required "
        "TYPE and DIFFICULTY:\n" + slot_lines + "\n\n"
        "Rules:\n"
        "  • Match each slot's difficulty: EASY = foundational; MEDIUM = applied, needs the right "
        "approach and edge cases; HARD = deep, with trade-offs / edge cases / a required time complexity.\n"
        "  • Each question must be a CONCRETE problem, scenario, or task — NOT 'describe your experience "
        "with X'.\n"
        "  • Ground each in BOTH the JD requirements AND something specific in the resume; keep all six on "
        "DISTINCT topics.\n"
        "  • For CODING slots, state the expected input/output, the constraints, and the target time complexity.\n"
        "  • Keep each question to 1–3 sentences.\n"
        "  • Return ONLY a numbered list 1–6, one question per line, in slot order. No tags, no commentary.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE RESUME:\n{resume_text}"
    )


def questions_from_slate(raw_response, level):
    """Turn the model's wording into the level's guaranteed slate.

    The model only supplies text; the type and difficulty of every question come
    from ``SLATES[level]``, so the difficulty calibration and the exact coding
    count hold no matter how (un)cooperative the model was. Always returns
    exactly len(slate) questions — short model output is padded, extra lines are
    ignored.
    """
    slate = SLATES.get(level, SLATES['mid'])
    # Strip any leading dash/colon the model echoed from the slot bullet.
    texts = [re.sub(r'^[\s\-–—:.]+', '', q['text']).strip()
             for q in parse_questions(raw_response) if q['text']]
    texts = [t for t in texts if t]
    out = []
    for i, (qtype, difficulty) in enumerate(slate):
        text = texts[i] if i < len(texts) else (
            f"({difficulty.capitalize()} {qtype} question — walk me through a relevant problem "
            "from your experience for this role.)"
        )
        out.append({'type': qtype, 'difficulty': difficulty, 'text': text})
    return out


_QTAG_RE = re.compile(r'^\s*\[?\s*(coding|verbal|easy|medium|hard)\s*\]?\s*[:.\-]?\s*', re.I)


def parse_questions(raw_response):
    """Parse the LLM's numbered list into
    [{'type': 'coding'|'verbal', 'difficulty': 'easy'|'medium'|'hard'|None, 'text': str}].
    Each line may carry leading [CODING]/[VERBAL] and [EASY]/[MEDIUM]/[HARD] tags in any
    order; untagged lines default to a verbal question with no difficulty.
    """
    lines = re.findall(r'^\s*\d+\.\s+(.+)', raw_response.strip(), re.MULTILINE)
    if not lines:
        lines = [q.strip() for q in raw_response.strip().splitlines() if q.strip()]

    questions = []
    for line in lines:
        text, qtype, difficulty = line.strip(), 'verbal', None
        for _ in range(2):                       # consume up to two leading tags
            m = _QTAG_RE.match(text)
            if not m:
                break
            tag = m.group(1).lower()
            if tag in ('coding', 'verbal'):
                qtype = tag
            else:
                difficulty = tag
            text = text[m.end():]
        questions.append({'type': qtype, 'difficulty': difficulty, 'text': text.strip()})
    return questions


# ── Email ────────────────────────────────────────────────────────────────────

def send_email(receiver_email, session_id, candidate_name=None):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        raise ValueError("EMAIL_SENDER and EMAIL_PASSWORD must be set in the .env file")

    msg = MIMEMultipart()
    msg['From']    = EMAIL_SENDER
    msg['To']      = receiver_email
    msg['Subject'] = 'Your AI Interview Session ID'

    greeting = f"Hi {candidate_name},\n\n" if candidate_name else ""
    body = (
        f"{greeting}"
        f"Your interview session has been created.\n\n"
        f"Session ID: {session_id}\n\n"
        f"Open the interview portal and enter this ID to begin your session.\n\n"
        f"Good luck!"
    )
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)


# ── File helpers ─────────────────────────────────────────────────────────────

def find_all_resume_files(resume_dir=RESUME_PATH):
    if not os.path.exists(resume_dir):
        raise FileNotFoundError(f"Resume directory '{resume_dir}' not found")
    pdf_files = [f for f in os.listdir(resume_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{resume_dir}'")
    return [os.path.join(resume_dir, f) for f in pdf_files]


def find_jd_file(jd_dir=JD_PATH):
    if not os.path.exists(jd_dir):
        raise FileNotFoundError(f"JD directory '{jd_dir}' not found")
    pdf_files = [f for f in os.listdir(jd_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{jd_dir}'")
    return os.path.join(jd_dir, pdf_files[0])


def move_processed_resume(resume_file_path, session_id):
    os.makedirs(DONE_PATH, exist_ok=True)
    resume_filename = os.path.basename(resume_file_path)
    dest = os.path.join(DONE_PATH, f"{session_id}_{resume_filename}")
    shutil.move(resume_file_path, dest)
    log.info("Resume archived: %s", dest)


# ── Core processing ──────────────────────────────────────────────────────────

def process_single_resume(resume_file_path, jd_text, progress_cb=None, features=None):
    """
    Process one resume end-to-end.
    progress_cb(step: str) is called at each stage so callers can stream updates.
    features: optional {'tts','stt','camera','clarify': bool} written into the
    session's meta so the recruiter's per-interview choices reach the candidate UI.
    Returns dict with session_id, email, candidate_name on success; raises on failure.
    """
    features = features or {}
    def cb(msg):
        log.info("%s", msg)
        if progress_cb:
            progress_cb(msg)

    cb(f"Reading resume: {os.path.basename(resume_file_path)}")
    resume_text = extract_text_from_pdf(resume_file_path)

    email = extract_email_from_text(resume_text)
    if not email:
        raise ValueError(f"No email found in {os.path.basename(resume_file_path)}")
    cb(f"Email found: {email}")

    candidate_name = extract_name_from_text(resume_text)

    level = detect_seniority(jd_text)
    cb(f"Detected role level: {LEVEL_PROFILES[level]['label']}")

    cb("Generating interview questions…")
    prompt    = build_question_prompt(resume_text, jd_text, level=level)
    raw       = ask_mistral(prompt, system_prompt="You are a senior technical interviewer.", temperature=0.5, max_tokens=600)
    # Type + difficulty come from the level's slate (guaranteed calibration); the
    # model supplies only the wording.
    questions = questions_from_slate(raw, level)
    cb(f"{len(questions)} questions generated")

    session_id   = str(uuid.uuid4())[:8]
    session_path = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)

    # questions.txt lines are "TYPE|difficulty: text" (difficulty omitted when
    # unknown, e.g. "CODING: text") — the reader tolerates both forms.
    with open(os.path.join(session_path, "questions.txt"), "w", encoding="utf-8") as f:
        for q in questions:
            diff = f"|{q['difficulty']}" if q.get('difficulty') else ""
            f.write(f"{q['type'].upper()}{diff}: {q['text']}\n")

    # Persist candidate metadata alongside questions
    with open(os.path.join(session_path, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"email={email}\n")
        f.write(f"candidate_name={candidate_name or ''}\n")
        f.write(f"level={level}\n")
        # Per-interview media features chosen by the recruiter.
        for feat in ("tts", "stt", "camera", "clarify"):
            f.write(f"{feat}={1 if features.get(feat) else 0}\n")

    cb("Sending session email…")
    send_email(email, session_id, candidate_name)

    cb("Archiving resume…")
    move_processed_resume(resume_file_path, session_id)

    cb(f"Done — session {session_id} created for {email}")
    return {"session_id": session_id, "email": email, "candidate_name": candidate_name, "level": level}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(RESUME_PATH,  exist_ok=True)
    os.makedirs(JD_PATH,      exist_ok=True)
    os.makedirs(DONE_PATH,    exist_ok=True)

    try:
        resume_paths = find_all_resume_files()
        jd_path      = find_jd_file()
        jd_text      = extract_text_from_pdf(jd_path)

        log.info("%d resume(s) | JD: %s", len(resume_paths), jd_path)

        ok = err = 0
        for rp in resume_paths:
            try:
                process_single_resume(rp, jd_text)
                ok += 1
            except Exception as e:
                log.error("Failed (%s): %s", os.path.basename(rp), e)
                err += 1

        log.info("Complete — %d succeeded, %d failed", ok, err)

    except FileNotFoundError as e:
        log.error("%s", e)
    except Exception as e:
        log.error("Unexpected error: %s", e)


if __name__ == '__main__':
    main()
