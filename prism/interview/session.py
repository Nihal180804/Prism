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
        self.questions      = []   # list of {'type': 'coding'|'verbal', 'text': str}
        self.responses      = []
        self.current_q      = 0
        self.stopping       = False
        self.session_folder = os.path.join(sessions_dir, session_id)
        # Answer handoff between the socket handler and the interview loop.
        # Used for BOTH typed verbal answers and submitted code — every
        # question is answered by a `submit_answer` event now that the
        # interview is fully text-based (no microphone/STT).
        self.pending_answer  = None
        self.awaiting_answer = False
        self.answer_event    = threading.Event()

    def emit(self, event, data):
        self._socketio.emit(event, data, to=self.sid)

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
                m = re.match(r'(CODING|VERBAL)\s*:\s*(.+)', line, re.I)
                if m:
                    self.questions.append({'type': m.group(1).lower(), 'text': m.group(2).strip()})
                else:
                    self.questions.append({'type': 'verbal', 'text': line})

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
