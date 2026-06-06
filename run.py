"""
StromCast master process runner.

Two modes:

* **Local launcher** (``python run.py``): spawns the FastAPI backend and the
  Streamlit dashboard as separate processes, waits for the API to become
  healthy, and streams output until interrupted with Ctrl+C.

* **Streamlit Cloud** (``streamlit run run.py``): Streamlit Cloud executes this
  file *inside its own script thread*, so we cannot spawn a second Streamlit
  process or install POSIX signal handlers (those only work in the main
  thread). In that mode we instead start the FastAPI backend in a background
  daemon thread within the same process and render the dashboard inline.

Usage (local)::

    python run.py                 # start API + dashboard
    python run.py --train         # (re)train the model first, then start
    python run.py --api-only      # backend only
    python run.py --dashboard-only

Ports default to API 8000 / dashboard 8501 (see src/config.py, overridable
via STROMCAST_API_PORT / STROMCAST_DASHBOARD_PORT env vars).
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

PROCS: list[subprocess.Popen] = []


# ──────────────────────────────────────────────────────────────────────────
# Streamlit-Cloud (in-process) mode
# ──────────────────────────────────────────────────────────────────────────
def _running_under_streamlit() -> bool:
    """True when this file is executed via ``streamlit run run.py``."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _api_listening() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", config.API_PORT)) == 0


def _start_api_thread() -> None:
    """Start the FastAPI backend in a daemon thread (idempotent).

    Streamlit reruns this script on every interaction, so we guard against
    starting uvicorn more than once by checking whether the port is already
    bound. Signal handlers are disabled because uvicorn would otherwise try to
    install them off the main thread and crash.
    """
    if _api_listening():
        return

    import uvicorn
    from src.api.main import app

    cfg = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=config.API_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(cfg)
    # Off-thread servers must not touch signals.
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

    threading.Thread(target=server.run, daemon=True, name="stromcast-api").start()

    # Block the first script run until the backend is actually healthy. The
    # API trains the model in its startup event, so "port bound" is not enough;
    # waiting here means the dashboard's first render already has live data
    # instead of flashing the "cannot reach API" error.
    url = f"{config.API_BASE_URL}{config.API_PREFIX}/health"
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)


def _serve_inline() -> None:
    """Streamlit-Cloud entry point: backend in-process, dashboard inline."""
    _start_api_thread()
    import runpy
    runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")


# ──────────────────────────────────────────────────────────────────────────
# Local launcher (separate processes)
# ──────────────────────────────────────────────────────────────────────────
def _spawn(cmd: list[str], name: str) -> subprocess.Popen:
    print(f"[run] starting {name}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    PROCS.append(proc)
    return proc


def _wait_for_api(timeout: float = 180.0) -> bool:
    url = f"{config.API_BASE_URL}{config.API_PREFIX}/health"
    deadline = time.time() + timeout
    print(f"[run] waiting for API at {url} (first run trains the model, be patient)…")
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                print("[run] API is healthy ✔")
                return True
        except requests.RequestException:
            pass
        # Surface a crashed backend early.
        for p in PROCS:
            if p.poll() is not None:
                print(f"[run] a child process exited early (code {p.returncode}).")
                return False
        time.sleep(2)
    print("[run] timed out waiting for the API.")
    return False


def _shutdown(*_args) -> None:
    print("\n[run] shutting down…")
    for p in PROCS:
        if p.poll() is None:
            p.terminate()
    for p in PROCS:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="StromCast launcher")
    parser.add_argument("--train", action="store_true", help="Train before launch")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--dashboard-only", action="store_true")
    args = parser.parse_args()

    # signal handlers only work in the main thread of the main interpreter.
    import signal
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.train:
        print("[run] training model…")
        from src import pipeline
        meta = pipeline.train(force_data=True)
        print(f"[run] trained · MAPE {meta['metrics']['mape']:.2f}% · "
              f"RMSE {meta['metrics']['rmse']:.0f} MW")

    start_api = not args.dashboard_only
    start_dash = not args.api_only

    if start_api:
        _spawn(
            [sys.executable, "-m", "uvicorn", "src.api.main:app",
             "--host", config.API_HOST, "--port", str(config.API_PORT)],
            "FastAPI",
        )
        if start_dash and not _wait_for_api():
            _shutdown()

    if start_dash:
        _spawn(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
             "--server.port", str(config.DASHBOARD_PORT),
             "--server.headless", "true"],
            "Streamlit",
        )
        print(f"\n[run] Dashboard → http://localhost:{config.DASHBOARD_PORT}")
        print(f"[run] API docs  → {config.API_BASE_URL}/docs\n")

    # Block until any child dies or Ctrl+C.
    try:
        while True:
            for p in PROCS:
                if p.poll() is not None:
                    print(f"[run] process exited (code {p.returncode}); stopping all.")
                    _shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


if _running_under_streamlit():
    # Imported/executed by `streamlit run run.py` (e.g. Streamlit Cloud).
    _serve_inline()
elif __name__ == "__main__":
    main()
