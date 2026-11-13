"""One-command demo: trains the models if needed, starts the API, then opens the app.

    python scripts/run_demo.py

Stop everything with Ctrl+C.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def wait_for(url, seconds=30):
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


if __name__ == "__main__":
    if not (ROOT / "saved_models" / "surge_xgb.pkl").exists():
        print("Models not found: running the pipeline first...")
        subprocess.run([PY, "scripts/run_pipeline.py"], cwd=ROOT, check=True)

    api = subprocess.Popen([PY, "-m", "uvicorn", "api.main:app", "--port", "8000"], cwd=ROOT)
    try:
        if not wait_for("http://127.0.0.1:8000/health"):
            sys.exit("The API did not start. Run: uvicorn api.main:app   to see the error.")
        print("\nAPI ready (docs: http://127.0.0.1:8000/docs). Starting the app...\n")
        subprocess.run([PY, "-m", "streamlit", "run", "app/streamlit_app.py"], cwd=ROOT)
    except KeyboardInterrupt:
        pass
    finally:
        api.terminate()
