"""Interview session state and a thread-safe registry of active sessions.

``SessionRegistry`` replaces the former module-level ``active_session`` global
in ``app.py``. Keying sessions by websocket ``sid`` lets multiple candidates
interview concurrently and removes the single-session bottleneck (roadmap P0).

Note: the shared camera in ``app.py`` is still a process-wide singleton —
genuine multi-candidate concurrency also needs browser-side camera capture
(the remaining half of the P0 rework). The registry unblocks the session
side of it; the interview itself is now fully text-based (no audio/STT).
"""
import os
import re
import threading


class InterviewSession:
    """State for one candidate's interview, plus a channel to emit to their tab.

    ``socketio`` is injected so this class stays decoupled from ``app.py`` (no
    import cycle). ``emit`` targets only this session's own websocket (``sid``).
    """

    def __init__(self, session_id, candidate_name, sid, socketio, sessions_dir):
        self.session_id     = session_id
        self.candidate_name = candidate_name
        self.sid            = sid
        self._socketio      = socketio
        self.questions      = []   # planned backbone: [{'type': 'coding'|'verbal', 'text': str}]
        self.responses      = []
        # Full ordered record of the interview as conducted by the agent —
        # planned questions AND any dynamic follow-ups, each with the answer and
        # the interviewer's private assessment. See prism/agents/interviewer.py.
        self.transcript     = []   # [{number,type,question,answer,is_followup,assessment}]
        self.current_q      = 0
        self.stopping       = False
        self.level          = None   # role seniority (from meta.txt) — calibrates difficulty
        self.agreed_at      = None   # timestamp the candidate accepted the rules

        # ── Per-interview media features (set from meta on start) ──────────────
        self.tts_enabled     = False
        self.stt_enabled     = False
        self.camera_enabled  = False
        self.clarify_enabled = True
        self.clarify_max     = 2
        # The question currently on screen (so the clarify handler has context),
        # and how many clarifications have been spent per question number.
        self.current_question = None   # {'number','type','text'}
        self.clarify_used     = {}     # {question_number: count}
        # Presence/integrity state for the browser-fed face monitor (see app.py).
        self.face_state = {'missing_start': None, 'warned': False,
                           'multi_warned': False, 'seen_face': False, 'stop': False}
        self.session_folder = os.path.join(sessions_dir, session_id)
        # Answer handoff between the socket handler and the interview loop.
        # Used for BOTH typed verbal answers and submitted code — every
        # question is answered by a `submit_answer` event now that the
        # interview is fully text-based (no microphone/STT). The lock makes the
        # check-and-consume atomic across the socket and interview threads.
        self.pending_answer  = None
        self.awaiting_answer = False
        self.answer_event    = threading.Event()
        self._answer_lock    = threading.Lock()

    def emit(self, event, data):
        self._socketio.emit(event, data, to=self.sid)

    # ── Answer handoff ────────────────────────────────────────────────────────
    # The interview loop calls begin_await() then wait_for_answer(); the socket
    # handler calls submit(). All three touch the same fields, so the lock keeps
    # "am I awaiting? then consume" atomic and free of lost/duplicate answers.

    def begin_await(self):
        """Arm the session to accept exactly one answer for the current question."""
        with self._answer_lock:
            self.pending_answer  = None
            self.answer_event.clear()
            self.awaiting_answer = True

    def submit(self, answer) -> bool:
        """Deliver a candidate answer. Returns True if it was accepted (i.e. the
        loop was awaiting one), False if there was nothing to answer — so a
        double-click or stray event can never advance two questions."""
        with self._answer_lock:
            if not self.awaiting_answer:
                return False
            self.pending_answer  = answer
            self.awaiting_answer = False
            self.answer_event.set()
            return True

    def wait_for_answer(self, poll: float = 1.0) -> str:
        """Block the interview thread until an answer arrives or the session is
        stopped, polling so `stopping` is honoured promptly."""
        while not self.answer_event.wait(timeout=poll):
            if self.stopping:
                with self._answer_lock:
                    self.awaiting_answer = False
                return "[No answer submitted]"
        with self._answer_lock:
            self.awaiting_answer = False
        answer = (self.pending_answer or "").strip()
        return answer if answer else "[No answer submitted]"

    # Compatibility shim used by face_monitor
    def insert_chat(self, sender, message):
        self.emit('chat_message', {'sender': sender, 'message': message})

    def exit_app(self):
        self.stopping = True
        self.emit('force_exit', {})

    def load_questions(self):
        q_path = os.path.join(self.session_folder, "questions.txt")
        self.questions = []
        with open(q_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # "TYPE: text" or "TYPE|difficulty: text" (difficulty optional).
                m = re.match(r'(CODING|VERBAL)\s*(?:\|\s*(easy|medium|hard))?\s*:\s*(.+)', line, re.I)
                if m:
                    self.questions.append({
                        'type': m.group(1).lower(),
                        'difficulty': m.group(2).lower() if m.group(2) else None,
                        'text': m.group(3).strip(),
                    })
                else:
                    self.questions.append({'type': 'verbal', 'difficulty': None, 'text': line})

    def load_meta(self):
        meta = {}
        meta_path = os.path.join(self.session_folder, "meta.txt")
        if os.path.exists(meta_path):
            for line in open(meta_path, encoding='utf-8'):
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    meta[k] = v
        return meta


class SessionRegistry:
    """Thread-safe map of websocket ``sid`` → :class:`InterviewSession`.

    Sessions are created, looked up, and removed from both SocketIO handler
    threads and the background interview/face-monitor worker threads, so every
    access is guarded by a lock.
    """

    def __init__(self, socketio, sessions_dir):
        self._socketio     = socketio
        self._sessions_dir = sessions_dir
        self._by_sid       = {}
        self._lock         = threading.Lock()

    def create(self, session_id, candidate_name, sid) -> InterviewSession:
        session = InterviewSession(session_id, candidate_name, sid, self._socketio, self._sessions_dir)
        with self._lock:
            self._by_sid[sid] = session
        return session

    def get(self, sid) -> InterviewSession:
        with self._lock:
            return self._by_sid.get(sid)

    def remove(self, sid) -> InterviewSession:
        """Remove and return the session for ``sid`` (or ``None`` if absent)."""
        with self._lock:
            return self._by_sid.pop(sid, None)

    def all(self):
        with self._lock:
            return list(self._by_sid.values())
