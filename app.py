import os
import sys
import re
import glob
import time
import threading
import shutil
from datetime import datetime

import cv2
import simpleaudio as sa
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from RealtimeSTT import AudioToTextRecorder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'interview-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Directory constants ──────────────────────────────────────────────────────
SESSIONS_DIR    = "Job/sessions"
RESPONSES_DIR   = "Job/responses"
DONE_DIR        = "Job/done"
EVALUATIONS_DIR = "Job/evaluations"
JD_DIR          = "Job/Jd"
RESUME_DIR      = "Job/resume"

for d in [SESSIONS_DIR, RESPONSES_DIR, DONE_DIR, EVALUATIONS_DIR, JD_DIR, RESUME_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Shared camera ────────────────────────────────────────────────────────────
_camera      = None
_camera_lock = threading.Lock()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
    return _camera


def release_camera():
    global _camera
    if _camera and _camera.isOpened():
        _camera.release()
        _camera = None


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

class InterviewSession:
    def __init__(self, session_id, candidate_name, sid):
        self.session_id     = session_id
        self.candidate_name = candidate_name
        self.sid            = sid
        self.questions      = []
        self.responses      = []
        self.current_q      = 0
        self.stopping       = False
        self.session_folder = os.path.join(SESSIONS_DIR, session_id)

    def emit(self, event, data):
        socketio.emit(event, data, to=self.sid)

    # Compatibility shim used by face_monitor
    def insert_chat(self, sender, message):
        self.emit('chat_message', {'sender': sender, 'message': message})

    def exit_app(self):
        self.stopping = True
        self.emit('force_exit', {})

    def load_questions(self):
        q_path = os.path.join(self.session_folder, "questions.txt")
        with open(q_path, "r", encoding="utf-8") as f:
            self.questions = [l.strip() for l in f if l.strip()]

    def load_meta(self):
        meta = {}
        meta_path = os.path.join(self.session_folder, "meta.txt")
        if os.path.exists(meta_path):
            for line in open(meta_path, encoding='utf-8'):
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    meta[k] = v
        return meta


active_session: InterviewSession = None
recorder = None


# ── Face monitor ─────────────────────────────────────────────────────────────

def face_monitor_web(session: InterviewSession):
    missing_start = None
    warned        = False

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
            if missing_start is None:
                missing_start = time.time()
            elapsed = time.time() - missing_start

            if elapsed > 3 and not warned:
                session.emit('warning', {'level': 'warn', 'message': 'No face detected — please stay in frame.'})
                warned = True

            if elapsed > 7:
                session.emit('warning', {'level': 'exit', 'message': 'Candidate not detected. Ending interview.'})
                session.stopping = True
                break
        else:
            missing_start = None
            warned        = False

        time.sleep(1)


# ── Interview flow ────────────────────────────────────────────────────────────

def play_wav(path):
    try:
        wo = sa.WaveObject.from_wave_file(path)
        wo.play().wait_done()
    except Exception as e:
        print(f"[Audio] {path}: {e}")


def run_interview(session: InterviewSession):
    global recorder

    intro = "speech/intro.wav"
    if os.path.exists(intro):
        play_wav(intro)

    for i, question in enumerate(session.questions):
        if session.stopping:
            break

        q_num    = i + 1
        progress = round((i / len(session.questions)) * 100)

        session.emit('question', {
            'number':   q_num,
            'total':    len(session.questions),
            'text':     question,
            'progress': progress,
        })

        wav_path = os.path.join(session.session_folder, f"q{q_num}.wav")
        if os.path.exists(wav_path):
            play_wav(wav_path)

        if session.stopping:
            break

        session.emit('status', {'state': 'listening'})
        try:
            answer = recorder.text() or ""
        except Exception as e:
            answer = ""
            print(f"[STT] {e}")

        if session.stopping:
            break

        if not answer.strip():
            answer = "[No response detected]"

        if "can you repeat" in answer.strip().lower():
            session.emit('chat_message', {'sender': '🔁 System', 'message': 'Repeating the question…'})
            if os.path.exists(wav_path):
                play_wav(wav_path)
            session.emit('status', {'state': 'listening'})
            try:
                answer = recorder.text() or "[No response detected]"
            except Exception:
                answer = "[No response detected]"

        session.emit('chat_message', {'sender': 'You', 'message': answer})
        session.emit('status', {'state': 'idle'})
        session.responses.append(answer)

    if not session.stopping:
        _finish_interview(session)


def _finish_interview(session: InterviewSession):
    session.emit('status', {'state': 'saving'})

    now           = datetime.now()
    safe_name     = session.candidate_name.replace(" ", "_")
    filename      = f"interview_responses_{safe_name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{session.session_id}.txt"
    filepath      = os.path.join(RESPONSES_DIR, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Session ID: {session.session_id}\n")
            f.write(f"Candidate Name: {session.candidate_name}\n")
            f.write(f"Date: {now.strftime('%Y-%m-%d')}\n")
            f.write(f"Time: {now.strftime('%H:%M:%S')}\n\n")
            for i, (q, a) in enumerate(zip(session.questions, session.responses), 1):
                f.write(f"Q{i}: {q}\nA{i}: {a}\n\n")
    except Exception as e:
        print(f"[Save] {e}")

    try:
        dest = os.path.join(DONE_DIR, session.session_id)
        if os.path.exists(dest):
            dest = f"{dest}_{now.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(session.session_folder, dest)
    except Exception as e:
        print(f"[Move] {e}")

    session.emit('interview_complete', {
        'candidate':  session.candidate_name,
        'questions':  session.questions,
        'responses':  session.responses,
    })
    session.stopping = True


# ── SocketIO — candidate ──────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    print(f"[WS] connect {request.sid}")


@socketio.on('disconnect')
def on_disconnect():
    global active_session
    if active_session and active_session.sid == request.sid:
        active_session.stopping = True
        active_session = None


@socketio.on('start_interview')
def on_start_interview(data):
    global active_session, recorder

    session_id     = (data.get('session_id') or '').strip()
    candidate_name = (data.get('candidate_name') or '').strip()

    if not session_id or not candidate_name:
        emit('error', {'message': 'Session ID and name are required.'})
        return

    session_folder = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_folder):
        emit('error', {'message': f'No session found for ID: {session_id}'})
        return

    session = InterviewSession(session_id, candidate_name, request.sid)
    try:
        session.load_questions()
    except Exception as e:
        emit('error', {'message': f'Could not load questions: {e}'})
        return

    meta = session.load_meta()
    active_session = session

    if recorder is None:
        try:
            recorder = AudioToTextRecorder(language="en")
        except Exception as e:
            emit('error', {'message': f'Could not start audio recorder: {e}'})
            return

    emit('interview_started', {
        'total_questions': len(session.questions),
        'candidate':       candidate_name,
        'meta':            meta,
    })

    threading.Thread(target=face_monitor_web, args=(session,), daemon=True).start()
    threading.Thread(target=run_interview,    args=(session,), daemon=True).start()


@socketio.on('exit_interview')
def on_exit_interview():
    global active_session
    if active_session:
        active_session.stopping = True
        active_session = None
    exiting_wav = "speech/Exiting.wav"
    if os.path.exists(exiting_wav):
        threading.Thread(target=play_wav, args=(exiting_wav,), daemon=True).start()
    emit('force_exit', {})


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

    # Completed interviews (have response files)
    for fname in sorted(os.listdir(RESPONSES_DIR), reverse=True):
        if not fname.endswith('.txt'):
            continue
        data = _read_response_file(os.path.join(RESPONSES_DIR, fname))
        if data:
            completed_sids[data['session_id']] = True
            # Check for evaluation
            eval_pattern = os.path.join(EVALUATIONS_DIR, f"evaluation_*_{data['session_id']}_*.txt")
            eval_files   = glob.glob(eval_pattern)
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
            q_count = sum(1 for l in open(q_path) if l.strip()) if os.path.exists(q_path) else 0
            meta = {}
            mp = os.path.join(folder, "meta.txt")
            if os.path.exists(mp):
                for line in open(mp):
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
def api_candidates():
    return jsonify(collect_candidates())


@app.route('/api/evaluations')
def api_evaluations():
    results = []
    for fpath in sorted(glob.glob(os.path.join(EVALUATIONS_DIR, "evaluation_*.txt")), reverse=True):
        data = _read_eval_file(fpath)
        if data:
            results.append(data)
    return jsonify(results)


@app.route('/api/evaluation/<session_id>')
def api_evaluation_detail(session_id):
    pattern = os.path.join(EVALUATIONS_DIR, f"evaluation_*_{session_id}_*.txt")
    files   = glob.glob(pattern)
    if not files:
        return jsonify({'error': 'Not found'}), 404
    data = _read_eval_file(files[0])
    return jsonify(data) if data else (jsonify({'error': 'Parse failed'}), 500)


# Processing job registry for upload progress
_processing_jobs = {}


@app.route('/api/upload', methods=['POST'])
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
def api_run_evaluation():
    def run_eval():
        from evaluate import InterviewEvaluator
        socketio.emit('eval_progress', {'message': 'Starting evaluation…'})
        try:
            ev = InterviewEvaluator()
            ev.process_all_responses()
            socketio.emit('eval_done', {'message': 'Evaluation complete.'})
        except Exception as e:
            socketio.emit('eval_done', {'message': f'Evaluation failed: {e}'})
    threading.Thread(target=run_eval, daemon=True).start()
    return jsonify({'status': 'started'})


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recruiter')
def recruiter():
    return render_template('recruiter.html')


if __name__ == '__main__':
    print("[*] http://localhost:5000  — candidate interface")
    print("[*] http://localhost:5000/recruiter  — recruiter dashboard")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
