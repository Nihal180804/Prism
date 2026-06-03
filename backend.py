import os
import uuid
import smtplib
import re
import fitz
import soundfile as sf
import shutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders
from dotenv import load_dotenv
import requests

load_dotenv()

MISTRAL_URL   = os.getenv("MISTRAL_URL", "http://127.0.0.1:1234/v1/chat/completions")
VOICE         = 'af_heart'
SESSIONS_DIR  = "Job/sessions"
JD_PATH       = "Job/Jd"
RESUME_PATH   = "Job/resume"
DONE_PATH     = "Job/done"
EMAIL_SENDER  = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Lazy-loaded TTS pipeline — only initialised when first needed so importing
# this module at Flask startup doesn't trigger a full model download/load.
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    return _pipeline


# ── LLM helper ──────────────────────────────────────────────────────────────

def ask_mistral(prompt, system_prompt="You are a helpful assistant.", temperature=0.4, max_tokens=512):
    headers  = {"Content-Type": "application/json"}
    payload  = {
        "model": "mistral-7b-instruct-v0.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }
    try:
        response = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Mistral request failed: {e}") from e


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
        if line and not re.search(r'[@|•|/]', line) and len(line.split()) <= 5:
            return line
    return None


# ── Question generation ──────────────────────────────────────────────────────

def build_question_prompt(resume_text, jd_text):
    return (
        "You are a senior technical interviewer preparing a personalised interview.\n\n"
        "Using the JOB DESCRIPTION and CANDIDATE RESUME below, write exactly 6 interview questions "
        "tailored to THIS specific candidate.\n\n"
        "Difficulty ladder:\n"
        "  Q1–Q2  Easy   — foundational concepts directly mentioned in the JD\n"
        "  Q3–Q4  Medium — applied / practical problems using the candidate's stated experience\n"
        "  Q5–Q6  Hard   — advanced scenarios or edge-cases that probe depth and seniority\n\n"
        "Rules:\n"
        "  • Every question must relate to both the JD requirements AND something visible in the resume.\n"
        "  • Include at least one behavioural question (situation/task/action/result format).\n"
        "  • Keep each question to 1-2 sentences — crisp and unambiguous.\n"
        "  • Return ONLY the numbered list (1. … 2. … etc.). No headings, no commentary.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE RESUME:\n{resume_text}"
    )


def parse_questions(raw_response):
    questions = re.findall(r'^\s*\d+\.\s+(.+)', raw_response.strip(), re.MULTILINE)
    if not questions:
        questions = [q.strip() for q in raw_response.strip().splitlines() if q.strip()]
    return questions


# ── TTS ──────────────────────────────────────────────────────────────────────

def generate_tts_files(questions, output_dir):
    pipeline = get_pipeline()
    for i, q in enumerate(questions, 1):
        audio_gen = pipeline(q, voice=VOICE)
        for _, _, audio in audio_gen:
            out_path = os.path.join(output_dir, f"q{i}.wav")
            sf.write(out_path, audio, 24000)
            break


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
    print(f"[+] Resume archived: {dest}")


# ── Core processing ──────────────────────────────────────────────────────────

def process_single_resume(resume_file_path, jd_text, progress_cb=None):
    """
    Process one resume end-to-end.
    progress_cb(step: str) is called at each stage so callers can stream updates.
    Returns dict with session_id, email, candidate_name on success; raises on failure.
    """
    def cb(msg):
        print(f"[+] {msg}")
        if progress_cb:
            progress_cb(msg)

    cb(f"Reading resume: {os.path.basename(resume_file_path)}")
    resume_text = extract_text_from_pdf(resume_file_path)

    email = extract_email_from_text(resume_text)
    if not email:
        raise ValueError(f"No email found in {os.path.basename(resume_file_path)}")
    cb(f"Email found: {email}")

    candidate_name = extract_name_from_text(resume_text)

    cb("Generating interview questions…")
    prompt    = build_question_prompt(resume_text, jd_text)
    raw       = ask_mistral(prompt, system_prompt="You are a senior technical interviewer.", temperature=0.5, max_tokens=600)
    questions = parse_questions(raw)
    cb(f"{len(questions)} questions generated")

    session_id   = str(uuid.uuid4())[:8]
    session_path = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)

    with open(os.path.join(session_path, "questions.txt"), "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q + "\n")

    # Persist candidate metadata alongside questions
    with open(os.path.join(session_path, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"email={email}\n")
        f.write(f"candidate_name={candidate_name or ''}\n")

    cb("Generating audio files…")
    generate_tts_files(questions, session_path)

    cb("Sending session email…")
    send_email(email, session_id, candidate_name)

    cb("Archiving resume…")
    move_processed_resume(resume_file_path, session_id)

    cb(f"Done — session {session_id} created for {email}")
    return {"session_id": session_id, "email": email, "candidate_name": candidate_name}


def main():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(RESUME_PATH,  exist_ok=True)
    os.makedirs(JD_PATH,      exist_ok=True)
    os.makedirs(DONE_PATH,    exist_ok=True)

    try:
        resume_paths = find_all_resume_files()
        jd_path      = find_jd_file()
        jd_text      = extract_text_from_pdf(jd_path)

        print(f"[+] {len(resume_paths)} resume(s) | JD: {jd_path}")

        ok = err = 0
        for rp in resume_paths:
            try:
                process_single_resume(rp, jd_text)
                ok += 1
            except Exception as e:
                print(f"[!] Failed ({os.path.basename(rp)}): {e}")
                err += 1

        print(f"\n[✔] Complete — {ok} succeeded, {err} failed")

    except FileNotFoundError as e:
        print(f"[!] {e}")
    except Exception as e:
        print(f"[!] Unexpected error: {e}")


if __name__ == '__main__':
    main()
