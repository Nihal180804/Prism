# Prism — AI-Powered Interview Platform

> See every candidate clearly.

Prism is a fully local, end-to-end automated hiring platform. It reads resumes, generates personalised interview questions using a local LLM, conducts written technical interviews with a live coding workspace, monitors candidate presence via webcam, and produces detailed scored evaluation reports — all from a clean web interface.

---

## How It Works

```
Recruiter uploads JD + Resumes
         │
         ▼
  Mistral 7B generates 6 personalised questions (Easy → Medium → Hard)
         │
         ▼
  Session ID emailed to candidate
         │
         ▼
  Candidate opens localhost:5000, enters Session ID
         │
         ▼
  Written interview — questions shown on screen; verbal answers typed, coding answers written in a live editor
  Face monitor runs in parallel (warns/exits if candidate leaves frame)
         │
         ▼
  Recruiter runs evaluation → Mistral scores answers 0–100
         │
         ▼
  Recruiter dashboard shows scores, strengths, improvements, full report
```

---

## Features

**Recruiter Dashboard** (`/recruiter`)
- Upload Job Description + multiple resumes in one go
- Live processing log as sessions are generated
- Candidate table with status (Pending / Completed / Evaluated)
- Expandable evaluation reports with per-question scores, strengths, and areas for improvement
- One-click evaluation trigger for all completed interviews

**Candidate Portal** (`/`)
- Session ID login — no account needed
- Live webcam feed with face-detection overlay and in-browser camera toggle (switch between built-in and external webcam)
- Initials avatar fallback when camera is unavailable
- Questions shown on screen; verbal answers typed, coding answers written in a live CodeMirror editor with AI review
- Difficulty badge per question (Easy / Medium / Hard)
- Full Q&A summary shown on completion

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-SocketIO |
| LLM | Mistral 7B Instruct (via LM Studio) |
| Vision | OpenCV (Haar cascade face detection) |
| Frontend | Vanilla HTML/CSS/JS + CodeMirror (no framework) |
| Email | Gmail SMTP with app password |

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 12 GB | 16–32 GB |
| GPU VRAM | None (CPU works) | 8 GB+ NVIDIA CUDA |
| Storage | 12 GB free | 20 GB free |
| Python | 3.9 | 3.10 / 3.11 |
| OS | Windows 10/11 | Windows 11 |
| Webcam | Optional | 720p+ |

---

## Prerequisites

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)
2. **LM Studio** — [lmstudio.ai](https://lmstudio.ai/)
   - Download **Mistral 7B Instruct v0.3 — Q4_K_M** (GGUF)
   - Start the local server on port `1234`
3. **Gmail app password** — [Create one here](https://myaccount.google.com/apppasswords) (2FA must be enabled)

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/prism.git
cd prism

# 2. Create and activate a virtual environment
python -m venv kokoro_env
kokoro_env\Scripts\activate        # Windows
# source kokoro_env/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
```

Edit `.env`:
```env
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
MISTRAL_URL=http://127.0.0.1:1234/v1/chat/completions
```

```bash
# 5. Start LM Studio and load Mistral 7B — then run:
python app.py
```

Open your browser:
- **Candidate portal** → `http://localhost:5000`
- **Recruiter dashboard** → `http://localhost:5000/recruiter`

---

## Usage

### As a Recruiter

1. Go to `http://localhost:5000/recruiter`
2. Navigate to **Upload & Process**
3. Upload the Job Description PDF and one or more candidate resume PDFs
4. Click **Generate Sessions** — questions are generated, audio synthesised, and Session IDs emailed automatically
5. Once candidates have completed their interviews, click **Run Evaluations** to score all responses
6. View results in the **Evaluations** tab

### As a Candidate

1. Check your email for your Session ID
2. Go to `http://localhost:5000`
3. Enter your Session ID and full name
4. Allow camera access when prompted (optional — used for presence monitoring)
5. Answer each question in the workspace — type verbal answers, write code for coding questions — then click **Submit Answer**
6. Your responses are saved automatically when the interview ends

---

## Project Structure

```
Prism/
├── app.py                ← Flask server, SocketIO events, interview flow
├── backend.py            ← resume processing, question generation, TTS, email
├── evaluate.py           ← LLM-based scoring and report generation
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example          ← copy to .env and fill in your credentials
├── prism/                ← refactored package (config, interview session/registry)
└── templates/
    ├── index.html        ← candidate portal
    └── recruiter.html    ← recruiter dashboard
```

> **Notes:**
> - `.env` is never committed — it holds your real credentials
> - `kokoro_env/` (virtual environment) is gitignored — recreate with `python -m venv kokoro_env`
> - `Job/` is created automatically at runtime and gitignored — it holds session data and candidate PII

---

## Configuration

All configuration is handled through `.env` in the project root (see `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `EMAIL_SENDER` | Gmail address used to send Session IDs | — |
| `EMAIL_PASSWORD` | Gmail app password (not your regular password) | — |
| `MISTRAL_URL` | LM Studio API endpoint | `http://127.0.0.1:1234/v1/chat/completions` |
| `MISTRAL_MODEL` | Model name sent to LM Studio | `mistral-7b-instruct-v0.3` |
| `HOST` | Interface to bind (use `0.0.0.0` to expose on the network) | `127.0.0.1` |
| `PORT` | Server port | `5000` |
| `SECRET_KEY` | Flask session signing key (random per start if unset) | random |
| `CORS_ALLOWED_ORIGINS` | Allowed Socket.IO origins | `*` |
| `RECRUITER_USER` | Username for the recruiter dashboard | `recruiter` |
| `RECRUITER_PASSWORD` | Password for the recruiter dashboard — **if unset the dashboard is open** | — |
| `CAMERA_INDEX` | Webcam device index (0 = built-in) | `0` |
| `FACE_WARN_SECONDS` | Seconds with no face before a warning | `3` |
| `FACE_EXIT_SECONDS` | Seconds with no face before ending the interview | `7` |

### Security

- The recruiter dashboard (`/recruiter`) and all `/api` routes are protected by HTTP
  Basic Auth when `RECRUITER_PASSWORD` is set. Leave it unset only for local development.
- `HOST` defaults to `127.0.0.1`, so the server is not reachable from the network
  unless you explicitly set `HOST=0.0.0.0` — do that only with a password set.

---

## Testing

Pure logic (question parsing, résumé field extraction, evaluation scoring) is covered
by a `pytest` suite that needs no webcam, microphone, or running LLM:

```bash
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built by [Nihal J](https://github.com/your-username)
