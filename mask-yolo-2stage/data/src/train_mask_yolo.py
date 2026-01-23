# src/train_mask_yolo.py
# comando per addestrare il modello mask/no_mask (one-shot): python src/train_mask_yolo.py
from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(p: str) -> str:
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str((REPO_ROOT / pp).resolve())


def main():
    # Modello base: yolov8n (nano) / yolov8s (small)
    model = YOLO("yolov8s.pt")

    model.train(
        data=resolve_repo_path("datasets/person_crops_yolo.yaml"),
        imgsz=640,  # Default a 640
        epochs=30,  # l'elaborazione richiede molto tempo, quindi ho diminuito le epoche da 50 a 30
        batch=-1,   # -1 autobatch
        device=0,   # GPU 0
        workers=4
    )


if __name__ == "__main__":
    main()
