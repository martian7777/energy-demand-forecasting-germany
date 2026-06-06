"""
StromCast master process runner.

Launches the FastAPI backend (uvicorn) and the Streamlit dashboard
concurrently, waits for the API to become healthy, and streams both
processes' output until interrupted with Ctrl+C.

Usage::

    python run.py                 # start API + dashboard
    python run.py --train         # (re)train the model first, then start
    python run.py --api-only      # backend only
    python run.py --dashboard-only

Ports default to API 8000 / dashboard 8501 (see src/config.py, overridable
via STROMCAST_API_PORT / STROMCAST_DASHBOARD_PORT env vars).
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

PROCS: list[subprocess.Popen] = []


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


if __name__ == "__main__":
    main()
