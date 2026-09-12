import os
import sys
import re
import glob
import time
import uuid
import hmac
import logging
import threading
import shutil
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prism import persistence
from prism.config import settings
from prism.interview.session import InterviewSession, SessionRegistry
from prism.agents import InterviewerAgent
from prism.llm import default_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prism")

# ── Configuration ──────────────────────────────────────────────────────────────
# All settings now live in prism/config.py. These module-level names are thin
# aliases onto that single source of truth, kept so the rest of this file reads
# unchanged; new code should prefer `settings.<name>` directly.
HOST               = settings.host
PORT               = settings.port
CORS_ORIGINS       = settings.cors_origins
FACE_WARN_SECONDS  = settings.face_warn_seconds
FACE_EXIT_SECONDS  = settings.face_exit_seconds
FACE_AUTO_EXIT     = settings.face_auto_exit
RECRUITER_USER     = settings.recruiter_user
RECRUITER_PASSWORD = settings.recruiter_password
DEV_MODE           = settings.dev_mode

app = Flask(__name__)
app.config['SECRET_KEY'] = settings.secret_key
# Reject oversized request bodies (resume/JD uploads) before they can exhaust
# memory or disk. Werkzeug raises 413 once the body passes this limit.
app.config['MAX_CONTENT_LENGTH'] = settings.max_content_length


@app.errorhandler(413)
def too_large(_e):
    limit_mb = settings.max_content_length // (1024 * 1024)
    return jsonify({'error': f'Upload too large (limit {limit_mb} MB).'}), 413
# Use plain threading (NOT gevent): the interview does blocking work — audio
# playback, camera reads, Whisper STT, LLM calls — which would stall a gevent
# event loop and drop the websocket. Threading runs those on real threads.
socketio = SocketIO(app, cors_allowed_origins=CORS_ORIGINS, async_mode='threading')

# Tracks active interview sessions by websocket sid (replaces the old
# `active_session` global — see prism/interview/session.py).
registry = SessionRegistry(socketio, settings.sessions_dir)

if not RECRUITER_PASSWORD:
    log.warning("RECRUITER_PASSWORD is not set — the recruiter dashboard is UNPROTECTED. "
                "Set it in .env before exposing this server to anyone else.")


# ── Recruiter auth ────────────────────────────────────────────────────────────

def require_recruiter_auth(f):
    """HTTP Basic Auth guard for recruiter-only routes.

    If RECRUITER_PASSWORD is unset the guard is a no-op (convenient for local
    development); set it in .env to require a login. Comparisons are
    constant-time to avoid leaking credentials via timing.
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        if RECRUITER_PASSWORD:
            auth = request.authorization
            ok = (
                auth is not None
                and hmac.compare_digest(auth.username or "", RECRUITER_USER)
                and hmac.compare_digest(auth.password or "", RECRUITER_PASSWORD)
            )
            if not ok:
                return Response(
                    "Authentication required.", 401,
                    {"WWW-Authenticate": 'Basic realm="Prism Recruiter Dashboard"'},
                )
        return f(*args, **kwargs)
    return wrapped

# ── Directory constants ──────────────────────────────────────────────────────
# Aliases onto prism/config.py (single source of truth).
SESSIONS_DIR    = settings.sessions_dir
RESPONSES_DIR   = settings.responses_dir
DONE_DIR        = settings.done_dir
EVALUATIONS_DIR = settings.evaluations_dir
JD_DIR          = settings.jd_dir
RESUME_DIR      = settings.resume_dir

for d in settings.data_dirs:
    os.makedirs(d, exist_ok=True)

# ── Face monitor (browser-side capture) ───────────────────────────────────────
# The candidate's OWN webcam is captured in their browser (getUserMedia) and a
# small downscaled frame is POSTed to /api/face every couple of seconds. The
# server runs the same lightweight OpenCV Haar-cascade presence check on those
# frames — so proctoring works for REMOTE candidates over the tunnel, and no
# server-side camera is opened. Detection is cheap and stays on the CPU.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def _evaluate_face_presence(state, num_faces, now, warn_s, exit_s, auto_exit):
    """Pure presence/integrity state machine (no OpenCV) — returns the list of
    warning events to emit and mutates ``state`` in place. Unit-tested directly."""
    events = []
    if num_faces == 0:
        state['multi_warned'] = False
        if state['missing_start'] is None:
            state['missing_start'] = now
        elapsed = now - state['missing_start']
        if elapsed > warn_s and not state['warned']:
            events.append({'level': 'warn', 'message': 'No face detected — please stay in frame.'})
            state['warned'] = True
        # Auto-exit is opt-in and only once the candidate has been seen, so a
        # camera warm-up / missed detection can't end the interview prematurely.
        if auto_exit and state['seen_face'] and elapsed > exit_s:
            events.append({'level': 'exit', 'message': 'Candidate not detected. Ending interview.'})
            state['stop'] = True
    elif num_faces > 1:
        state['missing_start'] = None
        state['warned'] = False
        if not state['multi_warned']:
            events.append({'level': 'warn', 'message': 'Multiple faces detected — only the candidate should be visible.'})
            state['multi_warned'] = True
    else:
        state['seen_face'] = True
        state['missing_start'] = None
        state['warned'] = False
        state['multi_warned'] = False
    return events


def process_face_frame(session: InterviewSession, frame_bgr):
    """Run the Haar cascade on one browser frame and emit any presence warnings."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    events = _evaluate_face_presence(
        session.face_state, len(faces), time.time(),
        FACE_WARN_SECONDS, FACE_EXIT_SECONDS, FACE_AUTO_EXIT,
    )
    for ev in events:
        session.emit('warning', ev)
    if session.face_state.get('stop'):
        session.stopping = True


@app.route('/api/face', methods=['POST'])
def api_face():
    """Receive a webcam frame from the candidate's browser for the presence check.
    Bound to the caller's session by socket id; ignored unless camera is on."""
    sid = request.form.get('sid') or ''
    s = registry.get(sid)
    if not s or not s.camera_enabled:
        return ('', 204)
    f = request.files.get('frame')
    if not f:
        return ('', 204)
    frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        process_face_frame(s, frame)
    return ('', 204)


# ── Interview session ────────────────────────────────────────────────────────
# InterviewSession and SessionRegistry now live in prism/interview/session.py.


# ── Interview flow ────────────────────────────────────────────────────────────
# The interview is fully text-based: questions are shown on screen (no TTS) and
# every answer — typed prose for verbal questions, editor contents for coding
# ones — arrives via the `submit_answer` socket event. There is no audio or STT.
#
# The interview is conducted by an agent (prism/agents/interviewer.py): it works
# through the planned questions but decides — as explicit tool calls — when to
# probe deeper and records a private assessment of each answer. The agent is
# fenced so it always completes and always covers every planned question; if the
# LLM is offline it degrades to a clean linear interview (see _run_linear).


def _meta_flag(meta, key, default):
    """Read a truthy per-session flag from meta.txt, falling back to a default."""
    v = meta.get(key)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _jd_snippet() -> str:
    """Best-effort role context for the interviewer agent (never fatal)."""
    try:
        from evaluate import _load_jd_snippet
        return _load_jd_snippet()
    except Exception:
        return ""


def _speak(session: InterviewSession, text: str):
    """Voice a question with Kokoro TTS (off the interview thread) and stream the
    audio to the candidate's tab. No-op when TTS is disabled or unavailable."""
    if not session.tts_enabled:
        return

    def worker():
        import base64
        from prism.media import tts
        audio = tts.synthesize(text)
        if audio:
            session.emit('question_audio', {
                'b64': base64.b64encode(audio).decode('ascii'), 'mime': 'audio/wav',
            })

    threading.Thread(target=worker, daemon=True).start()


def run_interview(session: InterviewSession):
    if not session.questions:
        _finish_interview(session)
        return

    if settings.agent_enabled:
        # Give the agent the live code reviewer so it can tell when a coding
        # answer is weak and follow up with another coding challenge.
        from evaluate import review_code
        agent = InterviewerAgent(
            default_client,
            max_followups=settings.agent_max_followups,
            max_code_followups=settings.agent_max_code_followups,
            max_steps=settings.agent_max_steps,
            code_reviewer=review_code,
            code_followup_threshold=settings.agent_code_followup_threshold,
            speak=_speak,
        )
        agent.run(session, jd_snippet=_jd_snippet(), level=session.level)
    else:
        _run_linear(session)

    if not session.stopping:
        _finish_interview(session)


def _run_linear(session: InterviewSession):
    """Deterministic fallback (AGENT_ENABLED=0): ask each planned question in
    order with no LLM decisions. This is also what the agent degrades to when
    the model is unavailable."""
    total = len(session.questions)
    for i, q in enumerate(session.questions):
        if session.stopping:
            break
        qtype = q.get('type', 'verbal')
        qtext = q['text']
        qdiff = q.get('difficulty')
        session.current_question = {'number': i + 1, 'type': qtype, 'text': qtext}
        session.emit('chat_message', {'sender': '🤖 Interviewer', 'message': qtext})
        session.emit('question', {
            'number': i + 1, 'total': total, 'text': qtext,
            'type': qtype, 'difficulty': qdiff, 'progress': round((i / total) * 100) if total else 0,
        })
        session.emit('status', {'state': 'coding' if qtype == 'coding' else 'answering'})
        _speak(session, qtext)
        session.begin_await()
        answer = session.wait_for_answer()
        session.emit('chat_message', {'sender': 'You', 'message': answer})
        session.emit('status', {'state': 'idle'})
        session.transcript.append({
            'number': i + 1, 'type': qtype, 'question': qtext, 'difficulty': qdiff,
            'answer': answer, 'is_followup': False, 'assessment': None,
        })


def _finish_interview(session: InterviewSession):
    session.emit('status', {'state': 'saving'})

    now = datetime.now()
    record = {
        'session_id':     session.session_id,
        'candidate_name': session.candidate_name,
        'date':           now.strftime('%Y-%m-%d'),
        'time':           now.strftime('%H:%M:%S'),
        'agreed_at':      session.agreed_at,
        'transcript':     session.transcript,   # planned Q&A + follow-ups + notes
    }

    try:
        persistence.save_interview(RESPONSES_DIR, record)
    except Exception as e:
        log.error("Failed to save responses: %s", e)

    try:
        dest = os.path.join(DONE_DIR, session.session_id)
        if os.path.exists(dest):
            dest = f"{dest}_{now.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(session.session_folder, dest)
    except Exception as e:
        log.error("Failed to move session folder: %s", e)

    # Parallel arrays for the results screen — every asked question (including
    # follow-ups), in order, alongside its answer.
    session.emit('interview_complete', {
        'candidate':  session.candidate_name,
        'questions':  [t['question'] for t in session.transcript],
        'responses':  [t['answer']   for t in session.transcript],
    })
    session.stopping = True


# ── SocketIO — candidate ──────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    log.info("WebSocket connected: %s", request.sid)


@socketio.on('disconnect')
def on_disconnect():
    session = registry.remove(request.sid)
    log.info("WS disconnect %s (had session=%s)", request.sid, bool(session))
    if session:
        session.stopping = True


@socketio.on('start_interview')
def on_start_interview(data):
    session_id     = (data.get('session_id') or '').strip()
    candidate_name = (data.get('candidate_name') or '').strip()

    if not session_id or not candidate_name:
        emit('error', {'message': 'Session ID and name are required.'})
        return

    # The candidate must accept the interview rules before starting. The gate is
    # enforced in the UI; this is the server-side backstop so a session can't be
    # started without it.
    if not data.get('agreed'):
        emit('error', {'message': 'You must agree to the interview rules to begin.'})
        return

    # Reject anything that isn't a bare identifier — blocks path traversal
    # (e.g. "../../..") through the session_id into arbitrary folders.
    if session_id != secure_filename(session_id) or not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
        emit('error', {'message': f'Invalid session ID: {session_id}'})
        return

    session_folder = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_folder):
        emit('error', {'message': f'No session found for ID: {session_id}'})
        return

    session = registry.create(session_id, candidate_name, request.sid)
    try:
        session.load_questions()
    except Exception as e:
        registry.remove(request.sid)
        emit('error', {'message': f'Could not load questions: {e}'})
        return

    meta = session.load_meta()
    session.level = meta.get('level')   # role seniority → calibrates dynamic follow-ups
    session.agreed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Per-interview media features: the recruiter's per-session choice (in meta)
    # overrides the global default. When off, the candidate UI hides the control.
    session.tts_enabled     = _meta_flag(meta, 'tts',     settings.tts_enabled_default)
    session.stt_enabled     = _meta_flag(meta, 'stt',     settings.stt_enabled_default)
    session.camera_enabled  = _meta_flag(meta, 'camera',  settings.camera_enabled_default)
    session.clarify_enabled = _meta_flag(meta, 'clarify', settings.clarify_enabled_default)
    session.clarify_max     = settings.clarify_max_per_question

    emit('interview_started', {
        'total_questions': len(session.questions),
        'candidate':       candidate_name,
        'meta':            meta,
        # The candidate UI switches features on/off from this — no dead space when off.
        'capabilities': {
            'tts':         session.tts_enabled,
            'stt':         session.stt_enabled,
            'camera':      session.camera_enabled,
            'clarify':     session.clarify_enabled,
            'clarify_max': session.clarify_max,
        },
    })

    # Face monitoring (when enabled) is driven by frames the candidate's browser
    # POSTs to /api/face — no server-side camera thread to start here.
    threading.Thread(target=run_interview, args=(session,), daemon=True).start()


@socketio.on('exit_interview')
def on_exit_interview():
    session = registry.remove(request.sid)
    if session:
        session.stopping = True
    emit('force_exit', {})


# ── Live code review ──────────────────────────────────────────────────────────

_review_lock = threading.Lock()


def _run_code_review(sid, code, language, live):
    def worker():
        # For live (debounced) ticks, skip if a review is already running so we
        # don't stack LLM calls; explicit submits always wait their turn.
        if not _review_lock.acquire(blocking=not live):
            return
        try:
            from evaluate import review_code
            review = review_code(code, language)
        except Exception as e:
            log.warning("Code review failed: %s", e)
            review = {'error': str(e), 'correctness': None, 'efficiency': None, 'style': None,
                      'feedback': [{'icon': '⚠️', 'text': f'Live review unavailable: {e}'}]}
        finally:
            _review_lock.release()
        socketio.emit('code_review', review, to=sid)
    threading.Thread(target=worker, daemon=True).start()


@socketio.on('code_update')
def on_code_update(data):
    code = (data.get('code') or '').strip()
    if code:
        _run_code_review(request.sid, code, data.get('language') or 'python', live=True)


@socketio.on('submit_code')
def on_submit_code(data):
    code = (data.get('code') or '').strip()
    if code:
        _run_code_review(request.sid, code, data.get('language') or 'python', live=False)


@socketio.on('submit_answer')
def on_submit_answer(data):
    # Deliver the candidate's answer (typed prose for a verbal question, or
    # editor contents for a coding one) to the current question, unblocking the
    # interview loop blocked in session.wait_for_answer().
    s = registry.get(request.sid)
    answer = data.get('answer')
    if answer is None:                      # tolerate older clients that sent {code: ...}
        answer = data.get('code') or ''
    # session.submit() is atomic (see InterviewSession): it accepts the answer
    # only if the loop is awaiting one, so a double-click can't advance twice.
    accepted = s.submit(answer) if s else False
    log.info("submit_answer received: session=%s accepted=%s", bool(s), accepted)


@socketio.on('clarify_request')
def on_clarify_request(data):
    """Candidate asks the AI to clarify the current question. Hint-only and
    capped per question; the guardrail (prism/clarify.py) never reveals the
    answer."""
    s = registry.get(request.sid)
    if not s or not s.clarify_enabled:
        emit('clarify_response', {'error': 'Clarifications are not available in this interview.'})
        return
    q = s.current_question
    if not q:
        emit('clarify_response', {'error': 'No question is active right now.'})
        return

    used = s.clarify_used.get(q['number'], 0)
    if used >= s.clarify_max:
        emit('clarify_response', {
            'message': "You've used all your clarifications for this question — answer as best you can.",
            'remaining': 0,
        })
        return

    message = (data.get('message') or '').strip()
    from prism.clarify import clarify
    reply = clarify(default_client, q['text'], q['type'], message)
    s.clarify_used[q['number']] = used + 1
    remaining = s.clarify_max - (used + 1)

    # Echo into the chat transcript so the exchange is visible.
    if message:
        s.emit('chat_message', {'sender': 'You', 'message': f'💬 {message}'})
    s.emit('chat_message', {'sender': '🤖 Interviewer', 'message': reply})
    emit('clarify_response', {'message': reply, 'remaining': remaining})


@app.route('/api/stt', methods=['POST'])
def api_stt():
    """Transcribe a browser-recorded audio clip for a verbal answer. Bound to the
    caller's live session (by socket id) so it only works when STT is enabled."""
    sid = request.form.get('sid') or ''
    s = registry.get(sid)
    if not s or not s.stt_enabled:
        return jsonify({'error': 'STT not enabled for this session'}), 403
    audio = request.files.get('audio')
    if not audio:
        return jsonify({'error': 'No audio uploaded'}), 400
    from prism.media import stt
    text = stt.transcribe(audio.read())
    if text is None:
        return jsonify({'error': 'Transcription unavailable'}), 503
    return jsonify({'text': text})


# ── Data helpers (recruiter API) ─────────────────────────────────────────────
# All reads go through prism.persistence, which prefers the JSON source of truth
# and falls back to legacy .txt files. No regex parsing of answer content here.


def collect_candidates():
    candidates     = []
    completed_sids = set()

    # Completed interviews: records start in RESPONSES_DIR and move to DONE_DIR
    # once evaluated, so scan BOTH — otherwise evaluated candidates would vanish
    # from the tables. list_interviews dedupes by session, newest first.
    for rec in persistence.list_interviews([RESPONSES_DIR, DONE_DIR]):
        sid = rec.get('session_id', 'unknown')
        if sid in completed_sids:
            continue
        completed_sids.add(sid)

        score     = None
        eval_path = persistence.find_evaluation(EVALUATIONS_DIR, sid)
        if eval_path:
            ev = persistence.read_evaluation(eval_path)
            if ev:
                score = ev['overall_score']

        candidates.append({
            'session_id':      sid,
            'candidate_name':  rec.get('candidate_name', 'Unknown'),
            'date':            rec.get('date', '-'),
            'time':            rec.get('time', '-'),
            'questions_count': len(rec.get('qa_pairs', [])),
            'filepath':        rec.get('filepath'),
            'filename':        rec.get('filename'),
            'status':          'evaluated' if score is not None else 'completed',
            'score':           score,
        })

    # Pending sessions (not yet interviewed)
    if os.path.exists(SESSIONS_DIR):
        for sid in os.listdir(SESSIONS_DIR):
            if sid in completed_sids:
                continue
            folder = os.path.join(SESSIONS_DIR, sid)
            if not os.path.isdir(folder):
                continue
            q_path  = os.path.join(folder, "questions.txt")
            q_count = 0
            if os.path.exists(q_path):
                with open(q_path, encoding='utf-8') as qf:
                    q_count = sum(1 for l in qf if l.strip())
            meta = {}
            mp = os.path.join(folder, "meta.txt")
            if os.path.exists(mp):
                with open(mp, encoding='utf-8') as mf:
                    for line in mf:
                        if '=' in line:
                            k, v = line.strip().split('=', 1)
                            meta[k] = v
            candidates.append({
                'session_id':      sid,
                'candidate_name':  meta.get('candidate_name') or 'Awaiting candidate',
                'date':            '-',
                'time':            '-',
                'questions_count': q_count,
                'status':          'pending',
                'score':           None,
                'filename':        None,
            })

    return candidates


# ── Recruiter API routes ──────────────────────────────────────────────────────

@app.route('/api/stats')
@require_recruiter_auth
def api_stats():
    candidates  = collect_candidates()
    total       = len(candidates)
    pending     = sum(1 for c in candidates if c['status'] == 'pending')
    completed   = sum(1 for c in candidates if c['status'] == 'completed')
    evaluated   = sum(1 for c in candidates if c['status'] == 'evaluated')
    avg_score   = None
    scores      = [c['score'] for c in candidates if c['score'] is not None]
    if scores:
        avg_score = round(sum(scores) / len(scores))
    return jsonify({'total': total, 'pending': pending, 'completed': completed, 'evaluated': evaluated, 'avg_score': avg_score})


@app.route('/api/candidates')
@require_recruiter_auth
def api_candidates():
    return jsonify(collect_candidates())


@app.route('/api/evaluations')
@require_recruiter_auth
def api_evaluations():
    # Enumerate JSON (source of truth) first, then legacy .txt; dedupe by
    # basename stem so a JSON record wins over its own .txt rendering.
    by_stem = {}
    for fpath in sorted(glob.glob(os.path.join(EVALUATIONS_DIR, "evaluation_*.json")), reverse=True):
        by_stem[os.path.splitext(os.path.basename(fpath))[0]] = fpath
    for fpath in sorted(glob.glob(os.path.join(EVALUATIONS_DIR, "evaluation_*.txt")), reverse=True):
        if os.path.basename(fpath) == "evaluation_summary.txt":
            continue
        by_stem.setdefault(os.path.splitext(os.path.basename(fpath))[0], fpath)

    results = []
    for fpath in by_stem.values():
        data = persistence.read_evaluation(fpath)
        if data:
            results.append(data)
    return jsonify(results)


@app.route('/api/evaluation/<session_id>')
@require_recruiter_auth
def api_evaluation_detail(session_id):
    path = persistence.find_evaluation(EVALUATIONS_DIR, session_id)
    if not path:
        return jsonify({'error': 'Not found'}), 404
    data = persistence.read_evaluation(path)
    return jsonify(data) if data else (jsonify({'error': 'Parse failed'}), 500)


# Processing job registry for upload progress
_processing_jobs = {}


@app.route('/api/upload', methods=['POST'])
@require_recruiter_auth
def api_upload():
    jd_file      = request.files.get('jd_file')
    resume_files = request.files.getlist('resume_files')

    if not jd_file:
        return jsonify({'error': 'JD file required'}), 400
    if not resume_files:
        return jsonify({'error': 'At least one resume required'}), 400

    # Clear old JD and save new one
    for f in os.listdir(JD_DIR):
        os.remove(os.path.join(JD_DIR, f))
    jd_path = os.path.join(JD_DIR, secure_filename(jd_file.filename))
    jd_file.save(jd_path)

    # Save resumes
    saved_resumes = []
    for rf in resume_files:
        if rf.filename.lower().endswith('.pdf'):
            dest = os.path.join(RESUME_DIR, secure_filename(rf.filename))
            rf.save(dest)
            saved_resumes.append(dest)

    if not saved_resumes:
        return jsonify({'error': 'No valid PDF resumes uploaded'}), 400

    # Per-interview media features the recruiter ticked (default to the global
    # config default when the field is absent, e.g. an older client).
    def _form_flag(name, default):
        v = request.form.get(name)
        return default if v is None else str(v).strip().lower() in ('1', 'true', 'yes', 'on')
    features = {
        'tts':     _form_flag('tts',     settings.tts_enabled_default),
        'stt':     _form_flag('stt',     settings.stt_enabled_default),
        'camera':  _form_flag('camera',  settings.camera_enabled_default),
        'clarify': _form_flag('clarify', settings.clarify_enabled_default),
    }

    job_id = datetime.now().strftime('%Y%m%d%H%M%S')
    _processing_jobs[job_id] = {'status': 'running', 'log': [], 'results': []}
    # Bound the in-memory job registry — keep only the most recent 20 so a
    # long-running server doesn't accumulate finished jobs forever.
    if len(_processing_jobs) > 20:
        for stale in sorted(_processing_jobs)[:-20]:
            _processing_jobs.pop(stale, None)

    def run_processing():
        from backend import extract_text_from_pdf, process_single_resume
        try:
            jd_text = extract_text_from_pdf(jd_path)
            for rpath in saved_resumes:
                def cb(msg, rp=rpath):
                    _processing_jobs[job_id]['log'].append(msg)
                    socketio.emit('upload_progress', {'job_id': job_id, 'message': msg})
                try:
                    result = process_single_resume(rpath, jd_text, progress_cb=cb, features=features)
                    _processing_jobs[job_id]['results'].append({'ok': True, **result})
                    socketio.emit('upload_progress', {'job_id': job_id, 'message': f"✅ Session {result['session_id']} created", 'done_one': True})
                except Exception as e:
                    msg = f"❌ Failed ({os.path.basename(rpath)}): {e}"
                    _processing_jobs[job_id]['log'].append(msg)
                    _processing_jobs[job_id]['results'].append({'ok': False, 'error': str(e), 'file': os.path.basename(rpath)})
                    socketio.emit('upload_progress', {'job_id': job_id, 'message': msg})
        except Exception as e:
            socketio.emit('upload_progress', {'job_id': job_id, 'message': f"❌ Fatal: {e}"})
        finally:
            _processing_jobs[job_id]['status'] = 'done'
            socketio.emit('upload_done', {'job_id': job_id, 'results': _processing_jobs[job_id]['results']})

    threading.Thread(target=run_processing, daemon=True).start()
    return jsonify({'job_id': job_id, 'resumes': len(saved_resumes)})


@app.route('/api/evaluate', methods=['POST'])
@require_recruiter_auth
def api_run_evaluation():
    def run_eval():
        from evaluate import InterviewEvaluator
        socketio.emit('eval_progress', {'message': 'Starting evaluation…'})
        try:
            ev = InterviewEvaluator()
            result    = ev.process_all_responses() or {}
            evaluated = result.get('evaluated', 0)
            failed    = result.get('failed', 0)
            if failed:
                msg = f'Evaluation complete — {evaluated} scored, {failed} could not be parsed (left for retry).'
            else:
                msg = f'Evaluation complete — {evaluated} scored.'
            socketio.emit('eval_done', {'message': msg})
        except Exception as e:
            socketio.emit('eval_done', {'message': f'Evaluation failed: {e}'})
    threading.Thread(target=run_eval, daemon=True).start()
    return jsonify({'status': 'started'})


# ── Page routes ───────────────────────────────────────────────────────────────

# Fixed mixed coding/verbal set for the dev practice session (no LLM/email needed)
PRACTICE_QUESTIONS = [
    ('coding', 'easy',   'Write a function two_sum(nums, target) that returns the indices of the two numbers that add up to target.'),
    ('coding', 'medium', 'Write a function is_palindrome(s) that returns True if the string is a palindrome, ignoring case and non-alphanumeric characters.'),
    ('verbal', 'medium', 'To finish, briefly tell me about a project you are proud of and your role in it.'),
]


@app.route('/api/dev/practice_session', methods=['POST'])
def api_dev_practice_session():
    """Create a ready-to-run coding+verbal session for local testing. Dev-only."""
    if not DEV_MODE:
        return jsonify({'error': 'Not found'}), 404
    sid    = 'dev' + uuid.uuid4().hex[:5]
    folder = os.path.join(SESSIONS_DIR, sid)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'questions.txt'), 'w', encoding='utf-8') as f:
        for qtype, diff, text in PRACTICE_QUESTIONS:
            f.write(f"{qtype.upper()}|{diff}: {text}\n")
    with open(os.path.join(folder, 'meta.txt'), 'w', encoding='utf-8') as f:
        f.write("email=practice@example.com\ncandidate_name=Practice Candidate\nlevel=mid\n")
        # Practice sessions enable clarify (testable with no extra models) and
        # leave TTS/STT/camera off by default.
        f.write("clarify=1\ntts=0\nstt=0\ncamera=0\n")
    log.info("Created dev practice session %s", sid)
    return jsonify({'session_id': sid})


@app.route('/')
def index():
    return render_template('index.html', dev_mode=DEV_MODE, rules=settings.interview_rules)


@app.route('/recruiter')
@require_recruiter_auth
def recruiter():
    return render_template('recruiter.html')


if __name__ == '__main__':
    log.info("Candidate portal:    http://%s:%s/", HOST, PORT)
    log.info("Recruiter dashboard: http://%s:%s/recruiter", HOST, PORT)
    try:
        socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
    except TypeError:
        # Older Flask-SocketIO without the allow_unsafe_werkzeug kwarg
        socketio.run(app, host=HOST, port=PORT, debug=False)
