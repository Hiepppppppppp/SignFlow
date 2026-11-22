"""One-click runner for SignFlow webcam gesture recognition."""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if (ROOT / "src").exists():
    PROJECT_DIR = ROOT
elif (ROOT / "SignFlow-main" / "src").exists():
    PROJECT_DIR = ROOT / "SignFlow-main"
else:
    PROJECT_DIR = ROOT

venv_python = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
python_exec = str(venv_python) if venv_python.exists() else sys.executable

cmd = [python_exec, "-m", "src.webcam"] + sys.argv[1:]
print(f"Starting SignFlow Webcam from: {PROJECT_DIR}")
print(f"Using Python: {python_exec}")
print(f"Command: {' '.join(cmd)}")

try:
    subprocess.run(cmd, cwd=str(PROJECT_DIR))
except KeyboardInterrupt:
    print("\nExited.")
