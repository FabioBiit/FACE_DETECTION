# src/make_person_crops.py
# comando per creare i crop delle persone (one-shot): python src/make_person_crops.py
from __future__ import annotations
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def yolo_to_xyxy(line: str, w: int, h: int):
    parts = line.strip().split()
    cls_id = int(parts[0])
    xc, yc, bw, bh = map(float, parts[1:5])
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return cls_id, x1, y1, x2, y2


def xyxy_to_yolo(cls_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return f"{cls_id} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def center_inside(box_xyxy, container_xyxy) -> bool:
    _, x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    X1, Y1, X2, Y2 = container_xyxy
    return (X1 <= cx <= X2) and (Y1 <= cy <= Y2)


def pad_box(x1, y1, x2, y2, w, h, pad=0.05):
    bw = x2 - x1
    bh = y2 - y1
    x1 -= bw * pad
    y1 -= bh * pad
    x2 += bw * pad
    y2 += bh * pad
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    return x1, y1, x2, y2


def main():
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--base_yaml", type=str, default="datasets/base_yolo.yaml")
    p.add_argument("--out_root", type=str, default="raw/processed/person_crops_yolo")
    p.add_argument("--person_model", type=str, default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.base_yaml).read_text(encoding="utf-8"))
    base_root = Path(cfg["path"])
    out_root = Path(args.out_root)
    for s in ["train", "val", "test"]:
        (out_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / s).mkdir(parents=True, exist_ok=True)

    person_model = YOLO(args.person_model)

    for split in ["train", "val", "test"]:
        img_dir = base_root / "images" / split
        lab_dir = base_root / "labels" / split
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        for img_path in tqdm(images, desc=f"Cropping persons [{split}]"):
            label_path = lab_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            labels = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            face_boxes = [yolo_to_xyxy(ln, W, H) for ln in labels]  # cls + xyxy

            # Stage 1: detect persons
            res = person_model.predict(source=img, conf=args.conf, verbose=False)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                names = res.names
                for b in res.boxes:
                    cls = int(b.cls.item())
                    if names.get(cls) == "person":
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        boxes.append((x1, y1, x2, y2))

            # fallback: se non trova persone, usa full frame come “crop”
            if not boxes:
                boxes = [(0, 0, W - 1, H - 1)]

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, W, H, pad=0.07)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]

                # trasferisci solo le bbox “mask/no_mask” dentro questa persona
                new_lines = []
                for fb in face_boxes:
                    if center_inside(fb, (x1, y1, x2, y2)):
                        cls_id, fx1, fy1, fx2, fy2 = fb
                        fx1 -= x1; fx2 -= x1
                        fy1 -= y1; fy2 -= y1
                        new_lines.append(xyxy_to_yolo(cls_id, fx1, fy1, fx2, fy2, cw, ch))

                # se non c'è nessuna faccia nel crop, skip
                if not new_lines:
                    continue

                out_name = f"{img_path.stem}__p{i}{img_path.suffix}"
                out_img = out_root / "images" / split / out_name
                out_lab = out_root / "labels" / split / f"{Path(out_name).stem}.txt"

                cv2.imwrite(str(out_img), crop)
                out_lab.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # YAML per training stage2
    out_yaml = Path("datasets") / "person_crops_yolo.yaml"
    out_yaml.write_text(
        f"""path: {out_root.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: mask
  1: no_mask
""",
        encoding="utf-8",
    )
    print(f"[OK] person_crops dataset: {out_root}")
    print(f"[OK] YAML: {out_yaml}")


if __name__ == "__main__":
    main()
