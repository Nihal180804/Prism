"""Serve Prism behind a free HTTPS tunnel and print the shareable links.

Runs the Flask app locally and opens a public HTTPS URL that points at it, so
remote candidates can join — and, crucially, so the browser lets them use their
microphone and camera (getUserMedia needs a secure origin: https:// or localhost).
Compute stays on THIS machine (LM Studio + models); only the traffic is tunnelled.

Prefers Cloudflare Tunnel (no account needed); falls back to ngrok (needs a free
authtoken configured once via `ngrok config add-authtoken ...`).

    python scripts/serve_public.py

Safety: refuses to expose a dashboard with no password. Set RECRUITER_PASSWORD in
.env first (or pass --allow-open-dashboard to override, e.g. for a quick test).
Stop with Ctrl+C — both the app and the tunnel are shut down.
"""
import os
import re
import sys
import time
import shutil
import signal
import argparse
import threading
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.config import settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CF_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

_procs = []


def _shutdown(*_a):
    for p in _procs:
        try:
            p.terminate()
        except Exception:
            pass
    # Escalate to a hard kill for anything that ignored terminate (the SocketIO
    # dev server can be stubborn on Windows) so nothing is left holding the port.
    time.sleep(1)
    for p in _procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    print("\nStopped. The tunnel is closed and the public link no longer works.")
    sys.exit(0)


def _print_links(public_url):
    bar = "=" * 62
    print(f"\n{bar}", flush=True)
    print("  Prism is live. Share these links:", flush=True)
    print(f"    Candidate portal :  {public_url}/", flush=True)
    print(f"    Recruiter board  :  {public_url}/recruiter", flush=True)
    print(bar, flush=True)
    print("  Mic & camera now work for candidates (HTTPS).  Ctrl+C to stop.\n", flush=True)


def _watch_cloudflared(proc):
    """Stream cloudflared output, print the public URL once it appears."""
    seen = False
    for line in iter(proc.stderr.readline, ""):
        if not seen:
            m = CF_URL_RE.search(line)
            if m:
                seen = True
                _print_links(m.group(0))
    if not seen:
        print("cloudflared exited before a URL appeared — check it's installed and reachable.")


def _find_cloudflared():
    """Locate cloudflared on PATH, or at its common Windows install paths (the
    MSI adds it to PATH only for shells opened after install)."""
    exe = shutil.which("cloudflared")
    if exe:
        return exe
    for cand in (r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                 r"C:\Program Files\cloudflared\cloudflared.exe"):
        if os.path.isfile(cand):
            return cand
    return None


def _run_cloudflared(port):
    exe = _find_cloudflared()
    if not exe:
        return False
    print("Starting Cloudflare tunnel (no account needed)…")
    proc = subprocess.Popen(
        [exe, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    _procs.append(proc)
    threading.Thread(target=_watch_cloudflared, args=(proc,), daemon=True).start()
    return True


def _run_ngrok(port):
    exe = shutil.which("ngrok")
    if not exe:
        return False
    print("Starting ngrok tunnel…")
    proc = subprocess.Popen([exe, "http", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _procs.append(proc)

    def poll():
        import json, urllib.request
        for _ in range(30):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                    tunnels = json.load(r).get("tunnels", [])
                url = next((t["public_url"] for t in tunnels if t["public_url"].startswith("https")), None)
                if url:
                    _print_links(url)
                    return
            except Exception:
                continue
        print("Couldn't read the ngrok URL — open http://127.0.0.1:4040 to see it.")

    threading.Thread(target=poll, daemon=True).start()
    return True


def main():
    ap = argparse.ArgumentParser(description="Serve Prism behind a public HTTPS tunnel.")
    ap.add_argument("--allow-open-dashboard", action="store_true",
                    help="Expose even without RECRUITER_PASSWORD set (NOT recommended).")
    args = ap.parse_args()

    if not settings.recruiter_password and not args.allow_open_dashboard:
        print("Refusing to go public: RECRUITER_PASSWORD is not set, so the recruiter\n"
              "dashboard and all /api routes would be OPEN to anyone with the link.\n"
              "Set RECRUITER_PASSWORD in your .env first, then re-run.\n"
              "(To override for a quick throwaway test: --allow-open-dashboard)")
        return 1

    port = settings.port
    signal.signal(signal.SIGINT, _shutdown)

    # 1) Start the app (stays on 127.0.0.1 — only the tunnel reaches it).
    print(f"Starting Prism on http://127.0.0.1:{port} …")
    app_proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT)
    _procs.append(app_proc)
    time.sleep(2)

    # 2) Start a tunnel (Cloudflare preferred, ngrok fallback).
    if not (_run_cloudflared(port) or _run_ngrok(port)):
        _shutdown()
        print("No tunnel tool found. Install one:\n"
              "  Cloudflare:  winget install --id Cloudflare.cloudflared   (or https://developers.cloudflare.com/cloudflared/)\n"
              "  ngrok:       https://ngrok.com/download  (then: ngrok config add-authtoken <token>)")
        return 1

    app_proc.wait()          # block until the app exits or Ctrl+C
    _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
