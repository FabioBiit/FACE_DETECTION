# src/shell_pipe_orchestrator.py
# comando per lanciare le pipe in sequenza: python .\src\shell_pipe_orchestrator.py
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

def main():
    py = sys.executable  # usa python del venv attivo

    run([py, "src/prepare_base_yolo.py"])
    run([py, "src/make_person_crops.py"])
    run([py, "src/train_mask_yolo.py"])

    print("\n[OK] Pipeline completata.")

if __name__ == "__main__":
    main()