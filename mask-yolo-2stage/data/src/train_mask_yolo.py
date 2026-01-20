from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(p: str) -> str:
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str((REPO_ROOT / pp).resolve())


def main():
    # Scegli modello base: yolov8n (veloce) / yolov8s (meglio)
    model = YOLO("yolov8s.pt")

    model.train(
        data=resolve_repo_path("datasets/person_crops_yolo.yaml"),
        imgsz=540,  # era default 640
        epochs=5,   # l'elaborazione richiede molto tempo, quindi ho diminuito le epoche
        batch=5,   # =-1 autobatch
        device=0,   # GPU 0
        workers=4
    )


if __name__ == "__main__":
    main()
