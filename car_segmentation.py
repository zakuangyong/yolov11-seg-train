from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _default_weights() -> Path:
    candidates = [
        Path("./models/best.pt"),
        Path("./runs/segment/train6/weights/best.pt"),
        Path("./runs/segment/train6/weights/last.pt"),
        Path("./models/yolo11m-seg.pt"),
        Path("./yolo11n.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("未找到可用权重文件，请用 --weights 指定 .pt 路径")


def _iter_images(img_dir: Path) -> Iterable[Path]:
    if img_dir.is_file() and img_dir.suffix.lower() in IMAGE_EXTS:
        yield img_dir
        return
    for p in sorted(img_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _imread_cn(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite_cn(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _stable_color(name: str) -> tuple[int, int, int]:
    v = 0
    for ch in name:
        v = (v * 131 + ord(ch)) & 0xFFFFFFFF
    b = 64 + (v & 0x7F)
    g = 64 + ((v >> 7) & 0x7F)
    r = 64 + ((v >> 14) & 0x7F)
    return int(b), int(g), int(r)


def _draw_label(
    img_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    ty1 = max(0, y1 - th - baseline - 4)
    ty2 = ty1 + th + baseline + 4
    tx1 = x1
    tx2 = min(w, x1 + tw + 6)
    cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(img_bgr, text, (tx1 + 3, ty2 - baseline - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _mask_to_rgba_crop(img_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    if mask01.ndim != 2:
        return None
    if mask01.shape[0] != h or mask01.shape[1] != w:
        mask01 = cv2.resize(mask01.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    m = (mask01 > 0.5).astype(np.uint8)
    if m.max() == 0:
        return None
    ys, xs = np.where(m > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_m = m[y1:y2, x1:x2] * 255

    rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_bgr
    rgba[:, :, 3] = crop_m
    rgba[crop_m == 0, :3] = 0
    return rgba


def _is_light_class(cls_name: str) -> bool:
    return cls_name.endswith("_light") or cls_name in {"front_light", "back_light"}


def _infer_faces_right(cls_names: list[str], centers_x: np.ndarray) -> bool:
    def _mean_x(name: str) -> float | None:
        xs = [float(centers_x[i]) for i, n in enumerate(cls_names) if n == name]
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    pairs = [
        ("front_bumper", "back_bumper"),
        ("hood", "trunk"),
        ("hood", "tailgate"),
        ("front_glass", "back_glass"),
        ("front_light", "back_light"),
    ]
    for a, b in pairs:
        xa = _mean_x(a)
        xb = _mean_x(b)
        if xa is not None and xb is not None and xa != xb:
            return xa > xb

    return True


def segment_car_parts(
    img_dir: str | Path = "./test/img",
    out_root: str | Path = "./result/segment",
    label_dir: str | Path = "./test/complete_label",
    weights: str | Path | None = None,
    conf: float = 0.25,
    save_crops: bool = True,
    save_labels: bool = True,
) -> dict[str, int]:
    img_dir = Path(img_dir)
    base_dir = img_dir.parent if img_dir.is_file() else img_dir
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    label_dir = Path(label_dir)
    if save_labels:
        label_dir.mkdir(parents=True, exist_ok=True)

    if weights is None:
        weights_path = _default_weights()
    else:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights_path))

    counts: dict[str, int] = {}
    for img_path in _iter_images(img_dir):
        img = _imread_cn(img_path)
        if img is None:
            continue

        rel = img_path.relative_to(base_dir)
        stem = rel.stem
        rel_parent = rel.parent
        per_img_dir = out_root / rel_parent / stem
        if save_crops:
            per_img_dir.mkdir(parents=True, exist_ok=True)

        annotated = img.copy()
        annotated_changed = False

        results = model.predict(
            source=str(img_path),
            save=False,
            conf=conf,
            device=device,
            verbose=False,
        )

        for r in results:
            if r.masks is None or r.boxes is None or r.boxes.cls is None:
                continue

            masks = r.masks.data
            clss = r.boxes.cls
            if masks is None or clss is None:
                continue

            masks_np = masks.detach().cpu().numpy()
            clss_np = clss.detach().cpu().numpy().astype(int)
            boxes_np = r.boxes.xyxy.detach().cpu().numpy()
            confs_np = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else None

            n = min(len(masks_np), len(clss_np), len(boxes_np))
            if n <= 0:
                continue

            cls_names = [model.names.get(int(clss_np[i]), str(int(clss_np[i]))) for i in range(n)]
            centers_x = (boxes_np[:n, 0] + boxes_np[:n, 2]) / 2.0
            faces_right = _infer_faces_right(cls_names, centers_x)

            keep = np.ones((n,), dtype=bool)
            for cls_name in set(cls_names):
                if not (
                    cls_name == "wheel"
                    or _is_light_class(cls_name)
                    or cls_name in {"left_mirror", "right_mirror"}
                ):
                    continue
                idxs = [i for i, n2 in enumerate(cls_names) if n2 == cls_name and keep[i]]
                if len(idxs) <= 1:
                    continue
                if cls_name in {"left_mirror", "right_mirror"} and confs_np is not None:
                    scores = confs_np[idxs]
                    best = float(np.max(scores))
                    tie = [idxs[k] for k, s in enumerate(scores) if float(s) == best]
                    if len(tie) == 1:
                        chosen = tie[0]
                    else:
                        xs = centers_x[tie]
                        chosen = tie[int(np.argmax(xs) if faces_right else np.argmin(xs))]
                else:
                    xs = centers_x[idxs]
                    chosen = idxs[int(np.argmax(xs) if faces_right else np.argmin(xs))]
                for j in idxs:
                    if j != chosen:
                        keep[j] = False

            per_class_idx: dict[str, int] = {}
            for i in range(n):
                if not keep[i]:
                    continue
                cls_name = cls_names[i]
                if save_labels:
                    color = _stable_color(cls_name)
                    if confs_np is None:
                        label = cls_name
                    else:
                        label = f"{cls_name} {float(confs_np[i]):.2f}"
                    _draw_label(annotated, boxes_np[i], label, color)
                    annotated_changed = True

                if save_crops:
                    rgba = _mask_to_rgba_crop(img, masks_np[i])
                    if rgba is None:
                        continue

                k = per_class_idx.get(cls_name, 0) + 1
                per_class_idx[cls_name] = k

                if k == 1:
                    out_name = f"{cls_name}_{stem}.png"
                else:
                    out_name = f"{cls_name}_{stem}_{k}.png"

                if save_crops:
                    out_path = per_img_dir / out_name
                    _imwrite_cn(out_path, rgba)

                counts[cls_name] = counts.get(cls_name, 0) + 1

        if save_labels and annotated_changed:
            out_img_path = (label_dir / rel_parent / rel.name).with_suffix(rel.suffix)
            _imwrite_cn(out_img_path, annotated)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="./test/img")
    parser.add_argument("--out_root", default="./result/segment")
    parser.add_argument("--label_dir", default="./test/complete_label")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no_crops", action="store_true")
    parser.add_argument("--no_labels", action="store_true")
    args = parser.parse_args()

    counts = segment_car_parts(
        img_dir=args.img_dir,
        out_root=args.out_root,
        label_dir=args.label_dir,
        weights=args.weights,
        conf=args.conf,
        save_crops=not args.no_crops,
        save_labels=not args.no_labels,
    )
    for k in sorted(counts.keys()):
        print(f"{k}: {counts[k]}")


if __name__ == "__main__":
    main()

