# Prism — AI-Powered Interview Platform

> See every candidate clearly.

Prism is a **fully local**, end-to-end automated hiring platform. It reads resumes, generates personalised interview questions with a local LLM whose **difficulty is calibrated to the role's seniority**, conducts technical interviews **driven by a tool-using interviewer agent** (live coding workspace, optional text-to-speech and voice answers, browser-side webcam proctoring), and produces detailed scored evaluation reports — all from a clean web interface, with **no data or API calls leaving your machine**.

---

## How It Works

```
Recruiter uploads JD + Resumes
         │
         ▼
  Role level detected from the JD (SDE 1 / SDE 2 / Senior / Staff)
         │
         ▼
  Local LLM generates 6 personalised questions, difficulty calibrated to that level
         │
         ▼
  Session ID emailed to candidate
         │
         ▼
  Candidate opens localhost:5000, enters Session ID
         │
         ▼
  Reviews and agrees to the interview rules
         │
         ▼
  Interviewer AGENT conducts the interview — asks each planned question, and
  decides (as explicit tool calls) when to probe deeper with a dynamic
  follow-up, recording a private assessment of each answer
  Presence is checked from the candidate's OWN browser webcam (warns / can
  auto-exit on no-face or multiple faces)
         │
         ▼
  Recruiter runs evaluation → the LLM scores answers 0–100
  (the agent's live notes are fed in as context)
         │
         ▼
  Recruiter dashboard shows scores, strengths, improvements, full report
```

### The interviewer agent

Rather than iterating a fixed list, the interview is run by an agent (`prism/agents/interviewer.py`) in a decide-act loop with five tools — `ask_question`, `probe` (verbal follow-up), `code_challenge` (another coding question), `assess`, and `finish`. The planned questions are a backbone it must complete, but between answers it dynamically decides whether to dig deeper. When a candidate's coding answer is weak — judged by the live code review's correctness score — the agent hands them another coding problem instead of moving on (and won't finish while a weak answer is still worth a follow-up). Because a small local model is easily derailed, the loop is fenced on every side: it can never finish early, never exceed the follow-up budget, breaks out of no-op spins, and falls back to "ask the next question" on any unparseable or errored decision. The result: the interview **always** completes and covers every planned question — and if the LLM is offline it simply degrades to a clean linear interview.

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
- Rules-agreement gate — the candidate must accept the interview rules before starting; acceptance is recorded with their responses (rules are editable in `prism/config.py`)
- Live presence from the candidate's **own** browser webcam (with a multi-camera switch and an initials-avatar fallback); frames are checked server-side for no-face / multiple-face integrity flags — so proctoring works for remote candidates too
- Optional **read-aloud questions** (Kokoro-82M TTS) and **voice answers** (browser mic → server Whisper)
- Questions shown on screen; verbal answers typed, coding answers written in a live CodeMirror editor with AI review
- **Level-calibrated difficulty** — the role seniority is inferred from the JD (SDE 1 vs SDE 2 vs Senior vs Staff) and each question's type + difficulty is fixed to a per-level slate, so a junior posting gets fundamentals and a senior/staff one gets harder, deeper problems
- **Agentic interviewer** — dynamically asks follow-up questions to probe shallow or ambiguous answers, and adds more coding questions (calibrated to the level) when a coding answer is weak
- **Company-controlled media features** (toggled per interview on the recruiter dashboard) — read questions aloud (Kokoro-82M TTS), voice answers (browser mic → server Whisper STT), and webcam presence/face monitoring. Anything left off is hidden in the candidate UI — no dead space.
- **Ask-the-AI clarify** — the candidate can ask the interviewer to rephrase or explain a question (capped per question); a guardrail ensures it never reveals the solution
- **Hardware-aware placement** — the LLM runs on the GPU while STT/TTS run on CPU/RAM by default, so low-VRAM machines (6–8 GB) aren't starved; audio only moves to the GPU when there's ample free VRAM
- Difficulty badge per question (Easy / Medium / Hard)
- Full Q&A summary shown on completion

---

## Architecture

Prism has **one host machine** (the company's — it holds the models and does the
compute) and **thin browser clients** (recruiter and candidates — no install).
Everything the interview does in real time flows over Socket.IO; file-shaped work
(uploads, audio clips, webcam frames) goes over plain HTTP endpoints.

```
        RECRUITER (browser)                         CANDIDATE (browser, any device)
   upload JD+resumes · run eval                 question text ▲   ▲ TTS audio
   view dashboard  (/recruiter)                 answers/clarify │   │ mic clip · webcam frames
            │  HTTP + Socket.IO                       Socket.IO │   │ HTTP (/api/stt, /api/face)
            ▼                                                   ▼   ▼
 ┌──────────────────────────── Prism server · Flask + Socket.IO ────────────────────────────┐
 │                                                                                            │
 │  InterviewerAgent  ── decide-act loop ─→ ask_question · probe · code_challenge · assess ·  │
 │   (prism/agents)                          finish   (JSON tool-calls, guardrailed)          │
 │        │              ┌───────────────┬──────────────┬────────────────┬────────────────┐  │
 │        ▼              ▼               ▼              ▼                ▼                ▼  │
 │   LLMClient      clarify (guard)  code review   persistence      face check       TTS/STT │
 │  (prism/llm)     prism/clarify    evaluate.py   prism/           OpenCV Haar      prism/  │
 │        │                                        persistence      (browser frames) media/  │
 │        │ HTTP /v1/chat/completions                (JSON + .txt)                            │
 │        ▼                                                                                   │
 │   ┌─────────────┐        device placement (prism/devices.py):                             │
 │   │  LM Studio  │  ◄───  LLM → GPU · Whisper STT + Kokoro TTS + Haar → CPU                 │
 │   │ Mistral 7B  │        (audio moves to GPU only when there's ample free VRAM)           │
 │   └─────────────┘                                                                          │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key design points**

- **Capture lives in the browser.** The candidate's mic and webcam are captured client-side (`getUserMedia`); audio clips and downscaled webcam frames are POSTed to the server. So proctoring and voice work for **remote** candidates — and because `getUserMedia` needs a secure origin, remote use goes through the HTTPS tunnel (see *Going live*).
- **The interviewer is an agent, not a script.** A ReAct-style loop asks the planned questions but decides between them whether to probe, add a coding challenge, or move on. It's fenced so it always completes and degrades to a clean linear interview if the LLM is offline.
- **Difficulty is guaranteed, not hoped for.** The role level is inferred from the JD; a per-level *slate* fixes each question's type and difficulty and the model only writes the wording — so an SDE 1 posting can't accidentally get staff-level questions.
- **Records can't be corrupted by answer text.** Interviews and evaluations are stored as JSON (source of truth) with human-readable `.txt` alongside; nothing is re-parsed with regexes.

## Models used

Everything runs locally. Weights are **not** committed (too big for git) — they
download from Hugging Face on first use and cache under `~/.cache/huggingface`.

| Role | Model (as tested) | ~Size | Runs on | Notes |
|---|---|---|---|---|
| Reasoning / questions / scoring | **Mistral 7B Instruct** (Q4, via LM Studio) | ~4.4 GB | **GPU** | Any OpenAI-compatible endpoint works (Ollama, vLLM). Tested with v0.1 & v0.3 Q4 — LM Studio uses whatever model is loaded. |
| Text-to-speech | **Kokoro-82M** (`hexgrad/Kokoro-82M`, voice `af_heart`) | ~330 MB | **CPU** | Reads questions aloud; 24 kHz. Pulls spaCy `en_core_web_sm` (g2p) on first run. |
| Speech-to-text | **faster-whisper `base.en`** (int8) | ~75 MB | **CPU** | The RealtimeSTT engine; transcribes uploaded voice answers. |
| Presence / proctoring | **OpenCV Haar cascade** (frontal face) | negligible | **CPU** | Runs on the webcam frames the browser uploads. |

## Hardware requirements & how the work is divided

The whole point of the device policy (`prism/devices.py`) is to run well on a
**typical 6–8 GB laptop GPU**: give the scarce VRAM to the LLM (which needs it
most) and keep the small audio/vision models on the CPU, where they're plenty
fast. `auto` only moves an audio model onto the GPU when there's ample free VRAM.

| Component | Placed on | Why |
|---|---|---|
| Mistral 7B (LLM) | **GPU** (via LM Studio) | By far the heaviest; benefits most from VRAM. |
| Whisper STT | **CPU** (int8) | Tiny; near-instant on CPU, no need to fight the LLM for VRAM. |
| Kokoro-82M TTS | **CPU** | Small (~82 M params); runs comfortably on CPU. |
| Face detection | **CPU** | Haar cascade is trivially cheap. |
| Mic / webcam capture | **Candidate's browser** | Offloaded to the client entirely — the server never opens a camera or mic. |

| | Minimum (text-only) | Recommended (all features) |
|---|---|---|
| RAM | 12 GB | 16–32 GB |
| GPU | None — Mistral runs on CPU (slower) | NVIDIA **6 GB+** (e.g. RTX 3060) for the LLM |
| Storage | ~8 GB free | ~20 GB free (models + Python/torch) |
| Python | 3.10 | 3.11 |
| OS | Windows 10/11, macOS, Linux | Windows 11 (primary test target) |
| Webcam / mic | not needed | any — only if the recruiter enables those features |

> Reference machine this was built and tested on: **ASUS ROG Zephyrus G14, RTX 3060 (6 GB)** — Mistral 7B on the GPU, Whisper + Kokoro + face detection on the CPU.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-SocketIO (threading async mode) |
| Agent | Tool-using interviewer loop (ReAct-style JSON tool-calls) |
| LLM | Any OpenAI-compatible endpoint — Mistral 7B via LM Studio by default |
| TTS / STT | Kokoro-82M · faster-whisper (both optional, per-interview) |
| Vision | OpenCV Haar cascade (on browser-uploaded frames) |
| Device mgmt | Custom placement policy (torch / nvidia-smi detection) |
| Persistence | JSON records (source of truth) + human-readable `.txt` |
| Frontend | Vanilla HTML/CSS/JS + CodeMirror (no framework) |
| Email | Gmail SMTP with app password |
| Tunnel | Cloudflare Tunnel (free HTTPS) for remote access |
| CI / tests | pytest + GitHub Actions |

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
LLM_URL=http://127.0.0.1:1234/v1/chat/completions
LLM_MODEL=mistral-7b-instruct-v0.3
```

```bash
# 5. Start LM Studio and load Mistral 7B — then run:
python app.py
```

### Optional: voice features (TTS + STT)

The base app runs text-only. To enable **read-aloud questions (Kokoro-82M)** and
**voice answers (Whisper)**, install the media extras and warm the model cache:

```bash
pip install -r requirements-media.txt
python scripts/prefetch_models.py
```

The model weights are **not** bundled in the repo — Kokoro (~330 MB) and Whisper
(~75–500 MB) are too big for git, so they download from Hugging Face on first use
and cache under `~/.cache/huggingface`. `prefetch_models.py` just does that once
up front so the first interview doesn't pause to download. The recruiter then
turns TTS/STT on per interview from the dashboard.

> If Kokoro fails to import with `cannot import name 'runtime_version' from google.protobuf`,
> upgrade protobuf: `pip install "protobuf>=5.26"` (an old protobuf left by
> TensorFlow is the usual cause).

Open your browser:
- **Candidate portal** → `http://localhost:5000`
- **Recruiter dashboard** → `http://localhost:5000/recruiter`

### Going live: share a public link (remote candidates)

Prism is a web app — you run it on the machine with LM Studio + the models, and
everyone else just needs a browser. To let **remote** candidates join (and for
their **mic/camera to work**, which browsers only allow over HTTPS or localhost),
serve it behind a free Cloudflare tunnel:

```bash
# one-time: install the tunnel tool
winget install --id Cloudflare.cloudflared        # Windows (or see cloudflare.com)

# set a dashboard password in .env (required — you're going public)
#   RECRUITER_PASSWORD=something-strong

# start the app + tunnel and print the shareable links
python scripts/serve_public.py
```

It prints a public HTTPS URL, e.g.:

```
Candidate portal :  https://<random>.trycloudflare.com/
Recruiter board  :  https://<random>.trycloudflare.com/recruiter
```

Compute stays on your machine (no cloud bill); only traffic is tunnelled. The
launcher **refuses to start without `RECRUITER_PASSWORD`** so you can't expose an
open dashboard by accident. Stop with Ctrl+C (closes the tunnel and the link).
For same-Wi-Fi-only demos you can instead set `HOST=0.0.0.0` and share your LAN
IP — but note mic and camera won't work over plain `http://` (they need the
tunnel's HTTPS).

---

## Usage

### As a Recruiter

1. Go to `http://localhost:5000/recruiter`
2. Navigate to **Upload & Process**
3. Upload the Job Description PDF and one or more candidate resume PDFs
4. Click **Generate Sessions** — questions are generated and Session IDs emailed automatically
5. Once candidates have completed their interviews, click **Run Evaluations** to score all responses
6. View results in the **Evaluations** tab

### As a Candidate

1. Check your email for your Session ID
2. Go to `http://localhost:5000`
3. Enter your Session ID and full name
4. If the recruiter enabled camera monitoring, allow access to **your own** webcam when prompted (mic too, if voice answers are on)
5. Answer each question in the workspace — type verbal answers, write code for coding questions (or speak, if voice is enabled) — then click **Submit Answer**. You can ask the AI to clarify a question (it won't give the answer).
6. Your responses are saved automatically when the interview ends

---

## Project Structure

```
Prism/
├── app.py                    ← Flask server, Socket.IO events, interview flow wiring,
│                                /api/stt · /api/face · /api/tts, recruiter API
├── backend.py                ← resume parsing, seniority detection, question generation, email
├── evaluate.py               ← LLM-based scoring + report generation, live code review
├── requirements.txt          ← base deps (text-only app)
├── requirements-media.txt    ← optional TTS/STT extras (Kokoro, faster-whisper)
├── .env.example              ← copy to .env and fill in your settings
├── prism/                    ← the application package
│   ├── config.py             ← single source of truth for every setting
│   ├── devices.py            ← GPU/VRAM detection + LLM-GPU / audio-CPU placement policy
│   ├── persistence.py        ← JSON-first storage for interviews & evaluations
│   ├── clarify.py            ← hint-only "ask the AI" guardrail
│   ├── llm/                  ← model-agnostic LLM client (timeout + retry)
│   ├── agents/               ← the interviewer agent + tool scaffolding
│   ├── media/                ← Kokoro TTS + Whisper STT wrappers (lazy, graceful)
│   └── interview/            ← InterviewSession + thread-safe SessionRegistry
├── scripts/
│   ├── prefetch_models.py    ← one-time model cache warm-up
│   └── serve_public.py       ← run app + Cloudflare tunnel, print shareable links
├── tests/                    ← pytest suite (logic, agent, persistence, media/devices/clarify)
├── .github/workflows/        ← CI (runs pytest on push / PR)
└── templates/
    ├── index.html            ← candidate portal
    └── recruiter.html        ← recruiter dashboard
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
| `LLM_URL` | OpenAI-compatible endpoint (LM Studio / Ollama / vLLM). `MISTRAL_URL` still honoured | `http://127.0.0.1:1234/v1/chat/completions` |
| `LLM_MODEL` | Model name sent to the endpoint. `MISTRAL_MODEL` still honoured | `mistral-7b-instruct-v0.3` |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | Per-request timeout (s) and retry count | `60` / `2` |
| `AGENT_ENABLED` | `1` = agentic interview; `0` = deterministic linear interview | `1` |
| `AGENT_MAX_FOLLOWUPS` | Max dynamic verbal follow-ups the agent may ask per interview | `3` |
| `AGENT_MAX_CODE_FOLLOWUPS` | Max extra coding questions the agent adds when unsatisfied with a coding answer | `2` |
| `AGENT_CODE_FOLLOWUP_THRESHOLD` | Live-review correctness (0–100) at/below which a coding answer is "weak" | `70` |
| `MAX_CONTENT_LENGTH` | Max upload size in bytes (resumes + JD) | `26214400` (25 MB) |
| `HOST` | Interface to bind (use `0.0.0.0` to expose on the network) | `127.0.0.1` |
| `PORT` | Server port | `5000` |
| `SECRET_KEY` | Flask session signing key (random per start if unset) | random |
| `CORS_ALLOWED_ORIGINS` | Allowed Socket.IO origins | `*` |
| `RECRUITER_USER` | Username for the recruiter dashboard | `recruiter` |
| `RECRUITER_PASSWORD` | Password for the recruiter dashboard — **if unset the dashboard is open** | — |
| `TTS_ENABLED` / `STT_ENABLED` / `CAMERA_ENABLED` / `CLARIFY_ENABLED` | Default media features (recruiter overrides per interview on the dashboard) | `0`/`0`/`0`/`1` |
| `CLARIFY_MAX_PER_QUESTION` | How many clarifications a candidate may request per question | `2` |
| `STT_DEVICE` / `TTS_DEVICE` | Audio model placement: `auto`/`cpu`/`cuda` (`auto` = CPU unless ample VRAM) | `auto` |
| `GPU_AUDIO_MIN_FREE_VRAM_MB` | Free VRAM required before `auto` puts an audio model on the GPU | `7000` |
| `FACE_WARN_SECONDS` | Seconds with no face before a warning | `3` |
| `FACE_EXIT_SECONDS` | Seconds with no face before ending (only if `FACE_AUTO_EXIT=1`) | `15` |

### Security

- The recruiter dashboard (`/recruiter`) and all `/api` routes are protected by HTTP
  Basic Auth when `RECRUITER_PASSWORD` is set. Leave it unset only for local development.
- `HOST` defaults to `127.0.0.1`, so the server is not reachable from the network
  unless you explicitly set `HOST=0.0.0.0` — do that only with a password set.

---

## Testing

A `pytest` suite covers the deterministic pieces — question parsing and seniority
detection, the per-level difficulty slate, résumé field extraction, evaluation
scoring, the interviewer agent's guardrails (with fake/offline models), the clarify
guardrail, device-placement policy, the face-presence state machine, and JSON
persistence — all with **no webcam, microphone, GPU, or running LLM required**. It
runs on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`):

```bash
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built by [Nihal J](https://github.com/your-username)
