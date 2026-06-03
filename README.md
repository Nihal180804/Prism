# Prism — AI-Powered Interview Platform

> See every candidate clearly.

Prism is a fully local, end-to-end automated hiring platform. It reads resumes, generates personalised interview questions using a local LLM, conducts voice interviews with real-time speech recognition, monitors candidate presence via webcam, and produces detailed scored evaluation reports — all from a clean web interface.

---

## How It Works

```
Recruiter uploads JD + Resumes
         │
         ▼
  Mistral 7B generates 6 personalised questions (Easy → Medium → Hard)
         │
         ▼
  Kokoro TTS synthesises question audio
         │
         ▼
  Session ID emailed to candidate
         │
         ▼
  Candidate opens localhost:5000, enters Session ID
         │
         ▼
  Voice interview — questions played aloud, answers captured via STT
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
- Live webcam feed with face-detection overlay
- Initials avatar fallback when camera is unavailable
- Questions read aloud via TTS; answers captured by microphone
- Say **"Can you repeat"** to replay any question
- Difficulty badge per question (Easy / Medium / Hard)
- Full Q&A summary shown on completion

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-SocketIO |
| LLM | Mistral 7B Instruct (via LM Studio) |
| TTS | Kokoro-82M (`hexgrad/Kokoro-82M`) |
| STT | RealtimeSTT |
| Vision | OpenCV (Haar cascade face detection) |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
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
| Microphone | Required | USB / headset mic |
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
4. Allow microphone and camera access when prompted
5. Answer each question aloud — say *"Can you repeat"* to replay a question
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
├── speech/
│   ├── intro.wav
│   ├── warning1.wav
│   ├── warning2.wav
│   └── Exiting.wav
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

All configuration is handled through `.env` in the project root:

| Variable | Description |
|---|---|
| `EMAIL_SENDER` | Gmail address used to send Session IDs |
| `EMAIL_PASSWORD` | Gmail app password (not your regular password) |
| `MISTRAL_URL` | LM Studio API endpoint (default: `http://127.0.0.1:1234/v1/chat/completions`) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
