# src/infer_pipeline.py
# comando per inferenza su immagine singola: 
# python .\src\infer_pipeline.py --img "raw\processed\base_yolo\images\test\NOMEIMG.jpg" 
# --mask_model "runs_trained\detect\train\weights\best.pt"     

from __future__ import annotations
from pathlib import Path
import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--img", type=str, required=True)
    p.add_argument("--mask_model", type=str, required=True)
    p.add_argument("--person_model", type=str, default="yolov8s.pt")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--conf_person", type=float, default=0.25)
    p.add_argument("--conf_mask", type=float, default=0.25)
    args = p.parse_args()

    img_path = resolve_repo_path(args.img)
    mask_model_path = resolve_repo_path(args.mask_model)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    person = YOLO(args.person_model)
    mask = YOLO(str(mask_model_path))

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    H, W = img.shape[:2]

    # Stage 1: person detection
    pres = person.predict(source=img, conf=args.conf_person, verbose=False)[0]
    persons = []
    if pres.boxes is not None:
        for b in pres.boxes:
            cls = int(b.cls.item())
            if pres.names.get(cls) == "person":
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                persons.append((x1, y1, x2, y2))
    if not persons:
        persons = [(0, 0, W - 1, H - 1)]

    # Stage 2: mask/no_mask in each crop
    for (x1, y1, x2, y2) in persons:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        mres = mask.predict(source=crop, conf=args.conf_mask, verbose=False)[0]
        if mres.boxes is None:
            continue

        for b in mres.boxes:
            cls = int(b.cls.item())  # 0 mask, 1 no_mask
            conf = float(b.conf.item())
            fx1, fy1, fx2, fy2 = b.xyxy[0].tolist()

            # remap to original frame
            fx1 += x1
            fx2 += x1
            fy1 += y1
            fy2 += y1
            fx1, fy1, fx2, fy2 = map(int, [fx1, fy1, fx2, fy2])

            label = "MASK" if cls == 0 else "NO_MASK"
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)

            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), color, 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (fx1, max(20, fy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    out_path = out_dir / f"{img_path.stem}__out.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()