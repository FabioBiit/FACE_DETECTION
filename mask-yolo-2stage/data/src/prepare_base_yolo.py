# src/prepare_base_yolo.py
from __future__ import annotations
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from lxml import etree
from tqdm import tqdm
import yaml


CLASS_NAMES = ["mask", "no_mask"]
MAP_ANDREWMVD = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 1,  # mappo "incorrect" su no_mask
}

MAP_MFVT_DEFAULT = {
    "ok": 0,
    "none": 1,
    "wrong": 1,
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class SplitPaths:
    images: Path
    labels: Path


def ensure_split_dirs(root: Path) -> Dict[str, SplitPaths]:
    splits = {}
    for s in ["train", "val", "test"]:
        img = root / "images" / s
        lab = root / "labels" / s
        img.mkdir(parents=True, exist_ok=True)
        lab.mkdir(parents=True, exist_ok=True)
        splits[s] = SplitPaths(images=img, labels=lab)
    return splits

def yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    # clamp
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"{cls_id} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def split_assign(n: int, seed: int = 42, train=0.8, val=0.1, test=0.1) -> List[str]:
    assert abs(train + val + test - 1.0) < 1e-6
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    out = [None] * n
    n_train = int(n * train)
    n_val = int(n * val)
    for k, i in enumerate(idx):
        if k < n_train:
            out[i] = "train"
        elif k < n_train + n_val:
            out[i] = "val"
        else:
            out[i] = "test"
    return out


def ingest_andrewmvd_voc(src_root: Path) -> List[Tuple[Path, List[str]]]:
    """
    Atteso:
      src_root/
        images/  (o direttamente immagini)
        annotations/ (xml)
    """
    if not src_root.exists():
        print(f"[SKIP] AndrewMVD non trovato: {src_root}")
        return []

    # images e annotations
    images_dir = src_root / "images"
    ann_dir = src_root / "annotations"
    if not images_dir.exists():
        images_dir = src_root
    if not ann_dir.exists():
        ann_dir = src_root

    xml_files = sorted([p for p in ann_dir.rglob("*.xml")])
    if not xml_files:
        print(f"[SKIP] Nessun XML trovato in: {ann_dir}")
        return []

    items = []
    for xml_path in tqdm(xml_files, desc="Ingest AndrewMVD (VOC)"):
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        filename = root.findtext("filename")
        if not filename:
            continue
        img_path = (images_dir / filename)
        if not img_path.exists():
            # prova a cercare immagine con stesso stem
            candidates = list(images_dir.rglob(xml_path.stem + ".*"))
            candidates = [c for c in candidates if c.suffix.lower() in IMG_EXTS]
            if not candidates:
                continue
            img_path = candidates[0]

        size = root.find("size")
        w = int(size.findtext("width"))
        h = int(size.findtext("height"))

        yolo_lines = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in MAP_ANDREWMVD:
                continue
            cls_id = MAP_ANDREWMVD[name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            yolo_lines.append(yolo_line(cls_id, xmin, ymin, xmax, ymax, w, h))
        if yolo_lines:
            items.append((img_path, yolo_lines))
    return items


def ingest_yolo_dataset(src_root: Path) -> List[Tuple[Path, List[str]]]:
    """
    Dataset YOLO-style:
      src_root/
        train/images + train/labels
        val/images + val/labels   (oppure val/...)
        test/images  + test/labels
    """
    if not src_root.exists():
        print(f"[SKIP] YOLO dataset non trovato: {src_root}")
        return []

    split_candidates = [("train", "train"), ("val", "val"), ("val", "val"), ("test", "test")]
    items: List[Tuple[Path, List[str]]] = []

    for split, _ in split_candidates:
        img_dir = src_root / split / "images"
        lab_dir = src_root / split / "labels"
        if not img_dir.exists() or not lab_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lab_path = lab_dir / (img_path.stem + ".txt")
            if not lab_path.exists():
                continue
            lines = [ln.strip() for ln in lab_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                items.append((img_path, lines))

    if items:
        print(f"[YOLO] Ingestiti {len(items)} samples da {src_root}")
    else:
        print(f"[SKIP] Nessuna label YOLO trovata in: {src_root}")
    return items


def ingest_roboflow_yolo(src_root: Path) -> List[Tuple[Path, List[str]]]:
    """
    Atteso Roboflow YOLO classico:
      src_root/
        train/images, train/labels
        val/images, val/labels
        test/images, test/labels
    Nota: qui non convertiamo, ingestiamo già label YOLO e rimappiamo classi se servisse.
    """
    if not src_root.exists():
        print(f"[SKIP] Roboflow non trovato: {src_root}")
        return []

    # prova a trovare split folder
    split_names = [("train", "train"), ("val", "val"), ("test", "test")]
    items = []
    for rf_split, _ in split_names:
        img_dir = src_root / rf_split / "images"
        lab_dir = src_root / rf_split / "labels"
        if not img_dir.exists() or not lab_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lab_path = lab_dir / (img_path.stem + ".txt")
            if not lab_path.exists():
                continue
            lines = [ln.strip() for ln in lab_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            # Roboflow abbia classico ha il mapping fatto come "mask/no_mask"
            # Se le classi sono diverse, va fatto un remapping.
            items.append((img_path, lines))
    if items:
        print(f"[Roboflow] Ingestiti {len(items)} samples (labels YOLO già pronte).")
    return items


def write_dataset(items: List[Tuple[Path, List[str]]], out_root: Path, seed: int = 42):
    out_root.mkdir(parents=True, exist_ok=True)
    splits = ensure_split_dirs(out_root)
    assigns = split_assign(len(items), seed=seed)

    for (img_path, yolo_lines), split in tqdm(list(zip(items, assigns)), desc="Writing base_yolo"):
        dst_img = splits[split].images / img_path.name
        dst_lab = splits[split].labels / (img_path.stem + ".txt")
        # evita collisioni nome file
        if dst_img.exists():
            dst_img = splits[split].images / f"{img_path.stem}__{abs(hash(str(img_path)))%10_000}{img_path.suffix}"
            dst_lab = splits[split].labels / (dst_img.stem + ".txt")

        shutil.copy2(img_path, dst_img)
        dst_lab.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    # YAML dataset
    yaml_path = Path("datasets") / "base_yolo.yaml"
    yaml_path.parent.mkdir(exist_ok=True)
    cfg = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[OK] base_yolo scritto in: {out_root}")
    print(f"[OK] YAML: {yaml_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--andrewmvd", type=str, default="./raw/andrewmvd_voc")
    p.add_argument("--mfvt", type=str, default="./raw/mfvt_coco/")
    p.add_argument("--roboflow", type=str, default="./raw/roboflow_yolo")
    p.add_argument("--out", type=str, default="./raw/processed/base_yolo")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items: List[Tuple[Path, List[str]]] = []
    items += ingest_andrewmvd_voc(Path(args.andrewmvd))
    items += ingest_yolo_dataset(Path(args.mfvt))
    items += ingest_roboflow_yolo(Path(args.roboflow))

    if not items:
        raise SystemExit("Nessun dato ingestito. Controlla i path sotto data/raw/.")

    write_dataset(items, Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()