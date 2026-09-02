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
from itertools import zip_longest
from functools import wraps

import cv2
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prism.config import settings
from prism.interview.session import InterviewSession, SessionRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prism")

# ── Configuration ──────────────────────────────────────────────────────────────
# All settings now live in prism/config.py. These module-level names are thin
# aliases onto that single source of truth, kept so the rest of this file reads
# unchanged; new code should prefer `settings.<name>` directly.
HOST               = settings.host
PORT               = settings.port
CORS_ORIGINS       = settings.cors_origins
CAMERA_INDEX       = settings.camera_index
FACE_WARN_SECONDS  = settings.face_warn_seconds
FACE_EXIT_SECONDS  = settings.face_exit_seconds
FACE_AUTO_EXIT     = settings.face_auto_exit
RECRUITER_USER     = settings.recruiter_user
RECRUITER_PASSWORD = settings.recruiter_password
DEV_MODE           = settings.dev_mode

app = Flask(__name__)
app.config['SECRET_KEY'] = settings.secret_key
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

# ── Shared camera ────────────────────────────────────────────────────────────
_camera       = None
_camera_lock  = threading.Lock()
_camera_index = CAMERA_INDEX
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        # Try DirectShow first (better USB webcam support on Windows),
        # fall back to default backend if it fails
        _camera = cv2.VideoCapture(_camera_index, cv2.CAP_DSHOW)
        if not _camera.isOpened():
            _camera = cv2.VideoCapture(_camera_index)
    return _camera


def release_camera():
    global _camera
    if _camera and _camera.isOpened():
        _camera.release()
        _camera = None


@socketio.on('switch_camera')
def on_switch_camera(data):
    global _camera, _camera_index
    requested = int(data.get('index', 0))
    with _camera_lock:
        if _camera and _camera.isOpened():
            _camera.release()
        _camera = cv2.VideoCapture(requested, cv2.CAP_DSHOW)
        if not _camera.isOpened():
            _camera = cv2.VideoCapture(requested)
        if _camera.isOpened():
            _camera_index = requested
            emit('camera_switched', {'index': requested, 'ok': True})
        else:
            _camera = cv2.VideoCapture(_camera_index, cv2.CAP_DSHOW)
            if not _camera.isOpened():
                _camera = cv2.VideoCapture(_camera_index)
            emit('camera_switched', {'index': requested, 'ok': False})


# ── MJPEG video feed ─────────────────────────────────────────────────────────

def gen_frames():
    while True:
        with _camera_lock:
            cam = get_camera()
            ret, frame = cam.read()
        if not ret:
            time.sleep(0.05)
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (99, 102, 241), 2)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Interview session ────────────────────────────────────────────────────────
# InterviewSession and SessionRegistry now live in prism/interview/session.py.


# ── Face monitor ─────────────────────────────────────────────────────────────

def face_monitor_web(session: InterviewSession):
    missing_start = None
    warned        = False
    multi_warned  = False
    seen_face     = False   # never auto-exit until the candidate has been seen once

    while not session.stopping:
        with _camera_lock:
            cam = get_camera()
            ret, frame = cam.read()
        if not ret:
            time.sleep(1)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            multi_warned = False
            if missing_start is None:
                missing_start = time.time()
            elapsed = time.time() - missing_start

            if elapsed > FACE_WARN_SECONDS and not warned:
                session.emit('warning', {'level': 'warn', 'message': 'No face detected — please stay in frame.'})
                warned = True

            # Auto-exit is opt-in AND only fires once we've actually detected the
            # candidate — so camera warm-up / a missed detection can't kill the
            # interview before it has even begun.
            if FACE_AUTO_EXIT and seen_face and elapsed > FACE_EXIT_SECONDS:
                session.emit('warning', {'level': 'exit', 'message': 'Candidate not detected. Ending interview.'})
                session.stopping = True
                break
        elif len(faces) > 1:
            # More than one person in frame — flag as a possible integrity issue.
            missing_start = None
            warned        = False
            if not multi_warned:
                session.emit('warning', {'level': 'warn', 'message': 'Multiple faces detected — only the candidate should be visible.'})
                multi_warned = True
        else:
            seen_face     = True
            missing_start = None
            warned        = False
            multi_warned  = False

        time.sleep(1)


# ── Interview flow ────────────────────────────────────────────────────────────
# The interview is fully text-based: questions are shown on screen (no TTS) and
# every answer — typed prose for verbal questions, editor contents for coding
# ones — arrives via the `submit_answer` socket event. There is no audio or STT.

def run_interview(session: InterviewSession):
    total = len(session.questions)
    for i, q in enumerate(session.questions):
        if session.stopping:
            break

        q_num    = i + 1
        qtype    = q.get('type', 'verbal') if isinstance(q, dict) else 'verbal'
        qtext    = q.get('text', q)        if isinstance(q, dict) else q
        progress = round((i / total) * 100)

        session.emit('question', {
            'number':   q_num,
            'total':    total,
            'text':     qtext,
            'type':     qtype,
            'progress': progress,
        })

        if session.stopping:
            break

        answer = _await_answer(session, qtype)

        if session.stopping:
            break

        session.emit('chat_message', {'sender': 'You', 'message': answer})
        session.emit('status', {'state': 'idle'})
        session.responses.append(answer)

    if not session.stopping:
        _finish_interview(session)


def _await_answer(session: InterviewSession, qtype: str):
    """Block the interview thread until the candidate submits an answer.

    Both question kinds are answered the same way — via `submit_answer` — so
    this one waiter serves coding (editor contents) and verbal (typed prose).
    The only difference is the status hint the candidate sees while answering.
    """
    session.pending_answer  = None
    session.answer_event.clear()
    session.emit('status', {'state': 'coding' if qtype == 'coding' else 'answering'})
    session.awaiting_answer = True   # accept a submission only once the event is cleared

    while not session.answer_event.wait(timeout=1):
        if session.stopping:
            session.awaiting_answer = False
            return "[No answer submitted]"

    session.awaiting_answer = False
    answer = (session.pending_answer or "").strip()
    return answer if answer else "[No answer submitted]"


def _finish_interview(session: InterviewSession):
    session.emit('status', {'state': 'saving'})

    now           = datetime.now()
    safe_name     = session.candidate_name.replace(" ", "_")
    filename      = f"interview_responses_{safe_name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{session.session_id}.txt"
    filepath      = os.path.join(RESPONSES_DIR, filename)

    # Questions are stored as {'type','text'} dicts — flatten to plain text.
    question_texts = [q['text'] if isinstance(q, dict) else q for q in session.questions]

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Session ID: {session.session_id}\n")
            f.write(f"Candidate Name: {session.candidate_name}\n")
            f.write(f"Date: {now.strftime('%Y-%m-%d')}\n")
            f.write(f"Time: {now.strftime('%H:%M:%S')}\n")
            f.write(f"Agreed to interview rules: yes ({session.agreed_at or 'unknown'})\n\n")
            # zip_longest (not zip) so an interview that ended early still
            # records every question — unanswered ones get an explicit marker
            # instead of being silently dropped.
            for i, (q, a) in enumerate(zip_longest(question_texts, session.responses), 1):
                if q is None:
                    q = "[Unknown question]"
                if a is None:
                    a = "[Not reached — interview ended before this question]"
                f.write(f"Q{i}: {q}\nA{i}: {a}\n\n")
    except Exception as e:
        log.error("Failed to save responses: %s", e)

    try:
        dest = os.path.join(DONE_DIR, session.session_id)
        if os.path.exists(dest):
            dest = f"{dest}_{now.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(session.session_folder, dest)
    except Exception as e:
        log.error("Failed to move session folder: %s", e)

    session.emit('interview_complete', {
        'candidate':  session.candidate_name,
        'questions':  question_texts,
        'responses':  session.responses,
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
    session.agreed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    emit('interview_started', {
        'total_questions': len(session.questions),
        'candidate':       candidate_name,
        'meta':            meta,
    })

    threading.Thread(target=face_monitor_web, args=(session,), daemon=True).start()
    threading.Thread(target=run_interview,    args=(session,), daemon=True).start()


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
    # interview loop waiting in _await_answer.
    s = registry.get(request.sid)
    answer = data.get('answer')
    if answer is None:                      # tolerate older clients that sent {code: ...}
        answer = data.get('code') or ''
    log.info("submit_answer received: session=%s awaiting=%s",
             bool(s), getattr(s, 'awaiting_answer', None))
    if s and getattr(s, 'awaiting_answer', False):
        s.pending_answer  = answer
        s.awaiting_answer = False
        s.answer_event.set()
        log.info("submit_answer accepted — advancing interview")


# ── Data helpers (recruiter API) ─────────────────────────────────────────────

def _read_response_file(fpath):
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        sid   = re.search(r'Session ID: (.+)',     content)
        name  = re.search(r'Candidate Name: (.+)', content)
        date  = re.search(r'Date: (.+)',            content)
        time_ = re.search(r'Time: (.+)',            content)
        qa    = re.findall(r'^Q\d+:', content, re.MULTILINE)
        return {
            'session_id':     (sid.group(1).strip()  if sid  else 'unknown'),
            'candidate_name': (name.group(1).strip() if name else 'Unknown'),
            'date':           (date.group(1).strip() if date else '-'),
            'time':           (time_.group(1).strip() if time_ else '-'),
            'questions_count': len(qa),
            'filepath':       fpath,
            'filename':       os.path.basename(fpath),
        }
    except Exception:
        return None


def _read_eval_file(fpath):
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        overall = re.search(r'Overall Score: (\d+)/100',       content)
        tech    = re.search(r'Technical Score: (\d+)/100',     content)
        comm    = re.search(r'Communication Score: (\d+)/100', content)
        sid     = re.search(r'Session ID: (.+)',                content)
        name    = re.search(r'Candidate Name: (.+)',            content)

        strengths = []
        s_section = re.search(r'=== STRENGTHS ===\n(.*?)(?===)', content, re.DOTALL)
        if s_section:
            strengths = [l.strip('• \n') for l in s_section.group(1).strip().splitlines() if l.strip()]

        improvements = []
        i_section = re.search(r'=== AREAS FOR IMPROVEMENT ===\n(.*?)(?===)', content, re.DOTALL)
        if i_section:
            improvements = [l.strip('• \n') for l in i_section.group(1).strip().splitlines() if l.strip()]

        summary_m = re.search(r'=== SUMMARY ===\n(.*?)(?===|\Z)', content, re.DOTALL)
        summary   = summary_m.group(1).strip() if summary_m else ''

        individual = []
        for m in re.finditer(r'Question (\d+): (\d+)/100\nFeedback: (.+?)(?=\n\n|\nQuestion|\Z)', content, re.DOTALL):
            individual.append({'q': int(m.group(1)), 'score': int(m.group(2)), 'feedback': m.group(3).strip()})

        return {
            'session_id':         (sid.group(1).strip()  if sid     else 'unknown'),
            'candidate_name':     (name.group(1).strip() if name    else 'Unknown'),
            'overall_score':      (int(overall.group(1)) if overall else 0),
            'technical_score':    (int(tech.group(1))    if tech    else 0),
            'communication_score':(int(comm.group(1))    if comm    else 0),
            'strengths':          strengths,
            'improvements':       improvements,
            'summary':            summary,
            'individual_scores':  individual,
            'filename':           os.path.basename(fpath),
        }
    except Exception:
        return None


def collect_candidates():
    completed_sids = {}
    candidates     = []

    # Completed interviews: response files start in RESPONSES_DIR and are moved
    # to DONE_DIR once evaluated, so scan BOTH — otherwise evaluated candidates
    # would vanish from the tables. Dedupe by session, newest first.
    response_files = []
    for base in (RESPONSES_DIR, DONE_DIR):
        if os.path.isdir(base):
            for fname in os.listdir(base):
                if fname.startswith('interview_responses_') and fname.endswith('.txt'):
                    response_files.append(os.path.join(base, fname))
    response_files.sort(key=os.path.basename, reverse=True)

    for fpath in response_files:
        data = _read_response_file(fpath)
        if not data or data['session_id'] in completed_sids:
            continue
        completed_sids[data['session_id']] = True
        # Check for evaluation
        eval_pattern = os.path.join(EVALUATIONS_DIR, f"evaluation_*_{data['session_id']}_*.txt")
        eval_files   = sorted(glob.glob(eval_pattern), reverse=True)
        score        = None
        if eval_files:
            ev = _read_eval_file(eval_files[0])
            if ev:
                score = ev['overall_score']
        candidates.append({**data, 'status': 'evaluated' if score is not None else 'completed', 'score': score})

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
    results = []
    for fpath in sorted(glob.glob(os.path.join(EVALUATIONS_DIR, "evaluation_*.txt")), reverse=True):
        if os.path.basename(fpath) == "evaluation_summary.txt":
            continue
        data = _read_eval_file(fpath)
        if data:
            results.append(data)
    return jsonify(results)


@app.route('/api/evaluation/<session_id>')
@require_recruiter_auth
def api_evaluation_detail(session_id):
    pattern = os.path.join(EVALUATIONS_DIR, f"evaluation_*_{session_id}_*.txt")
    files   = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return jsonify({'error': 'Not found'}), 404
    data = _read_eval_file(files[0])
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

    job_id = datetime.now().strftime('%Y%m%d%H%M%S')
    _processing_jobs[job_id] = {'status': 'running', 'log': [], 'results': []}

    def run_processing():
        from backend import extract_text_from_pdf, process_single_resume
        try:
            jd_text = extract_text_from_pdf(jd_path)
            for rpath in saved_resumes:
                def cb(msg, rp=rpath):
                    _processing_jobs[job_id]['log'].append(msg)
                    socketio.emit('upload_progress', {'job_id': job_id, 'message': msg})
                try:
                    result = process_single_resume(rpath, jd_text, progress_cb=cb)
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
    ('coding', 'Write a function two_sum(nums, target) that returns the indices of the two numbers that add up to target.'),
    ('coding', 'Write a function is_palindrome(s) that returns True if the string is a palindrome, ignoring case and non-alphanumeric characters.'),
    ('verbal', 'To finish, briefly tell me about a project you are proud of and your role in it.'),
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
        for qtype, text in PRACTICE_QUESTIONS:
            f.write(f"{qtype.upper()}: {text}\n")
    with open(os.path.join(folder, 'meta.txt'), 'w', encoding='utf-8') as f:
        f.write("email=practice@example.com\ncandidate_name=Practice Candidate\n")
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
