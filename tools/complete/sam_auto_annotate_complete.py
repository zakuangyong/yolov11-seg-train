from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_images(p: Path) -> Iterable[Path]:
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        yield p
        return
    for fp in sorted(p.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
            yield fp


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


def _resolve_sam_weights(p: str | Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    c1 = Path("./models/sam") / p.name
    if c1.exists():
        return c1
    c2 = Path("./models") / p.name
    if c2.exists():
        return c2
    return p


def _init_sam(model_type: str, weights_path: str | Path, device: str):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except Exception as e:
        raise ModuleNotFoundError("未安装 segment-anything，请先安装。") from e

    wp = _resolve_sam_weights(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"SAM 权重不存在: {wp}")

    sam = sam_model_registry[model_type](checkpoint=str(wp))
    sam.to(device=device)
    sam.eval()
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,
    )
    return generator


def _touch(mask_u8: np.ndarray) -> tuple[float, float, float, float, int]:
    h, w = mask_u8.shape[:2]
    m = mask_u8 > 0
    top = float(np.count_nonzero(m[0, :])) / max(1.0, float(w))
    bottom = float(np.count_nonzero(m[-1, :])) / max(1.0, float(w))
    left = float(np.count_nonzero(m[:, 0])) / max(1.0, float(h))
    right = float(np.count_nonzero(m[:, -1])) / max(1.0, float(h))
    cnt = int(top > 0.02) + int(bottom > 0.02) + int(left > 0.02) + int(right > 0.02)
    return top, bottom, left, right, cnt


def _pick_main_mask(masks: list[dict], w: int, h: int, min_ratio: float, max_ratio: float) -> np.ndarray | None:
    if not masks:
        return None
    img_area = float(w * h)
    min_area = img_area * float(min_ratio)
    max_area = img_area * float(max_ratio)
    cx, cy = float(w) / 2.0, float(h) / 2.0

    best_score = -1.0
    best: np.ndarray | None = None
    for d in masks:
        area = float(d.get("area", 0))
        if area < min_area or area > max_area:
            continue
        seg = d.get("segmentation", None)
        bbox = d.get("bbox", None)
        if not isinstance(seg, np.ndarray) or seg.ndim != 2 or bbox is None or len(bbox) != 4:
            continue
        x, y, bw, bh = [float(v) for v in bbox]
        if bw <= 1 or bh <= 1:
            continue
        m = (seg > 0).astype(np.uint8)

        upper_h = max(1, int(h * 0.55))
        upper_ratio = float(np.count_nonzero(m[:upper_h, :])) / max(1.0, area)
        top, bottom, left, right, cnt = _touch(m)

        bx, by = x + bw / 2.0, y + bh / 2.0
        dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        maxd = (cx**2 + cy**2) ** 0.5 + 1e-6
        center_bonus = 1.0 - min(1.0, dist / maxd)
        border_penalty = 1.0 - min(1.0, (top + bottom + left + right) / 2.0)

        iou = float(d.get("predicted_iou", 1.0))
        st = float(d.get("stability_score", 1.0))
        score = area * (0.35 + 0.65 * upper_ratio) * (0.35 + 0.65 * center_bonus) * (0.35 + 0.65 * border_penalty)
        score *= (0.5 + 0.5 * iou) * (0.5 + 0.5 * st)
        if cnt >= 3 and upper_ratio < 0.2:
            score *= 0.25

        if score > best_score:
            best_score = score
            best = m

    if best is None:
        d = max(masks, key=lambda x: int(x.get("area", 0)))
        seg = d.get("segmentation", None)
        if isinstance(seg, np.ndarray) and seg.ndim == 2:
            best = (seg > 0).astype(np.uint8)
    return best


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    cc, lab = cv2.connectedComponents(m, connectivity=8)
    if cc <= 2:
        return m
    best_lbl = 1
    best_area = -1
    for lbl in range(1, cc):
        a = int(np.sum(lab == lbl))
        if a > best_area:
            best_area = a
            best_lbl = lbl
    return (lab == best_lbl).astype(np.uint8)


def _clean_mask(mask_u8: np.ndarray) -> np.ndarray:
    m = _largest_component(mask_u8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    return _largest_component(m)


def _mask_to_polygon_line(mask_u8: np.ndarray, class_id: int = 0, min_points: int = 20) -> str:
    contours, _ = cv2.findContours(mask_u8.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ""
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area <= 10:
        return ""
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, 0.002 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return ""

    if len(approx) < min_points and len(cnt) >= min_points:
        step = max(1, len(cnt) // min_points)
        approx = cnt[::step, 0, :]
        if len(approx) < 3:
            return ""

    h, w = mask_u8.shape[:2]
    xs = np.clip(approx[:, 0].astype(np.float32) / max(1.0, float(w)), 0.0, 1.0)
    ys = np.clip(approx[:, 1].astype(np.float32) / max(1.0, float(h)), 0.0, 1.0)
    coords: list[str] = []
    for x, y in zip(xs, ys):
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return f"{class_id} " + " ".join(coords)


def _prepare_dirs(out_root: Path) -> None:
    for s in ["train", "val", "test"]:
        (out_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / s).mkdir(parents=True, exist_ok=True)
    (out_root / "prelabel_preview").mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_root: Path) -> Path:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "car"},
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def _split(items: list[Path], split: tuple[float, float, float], seed: int) -> dict[Path, str]:
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"split 之和必须为 1.0，当前为 {sum(split)}")
    r = random.Random(seed)
    xs = items[:]
    r.shuffle(xs)
    n = len(xs)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    d: dict[Path, str] = {}
    for i, p in enumerate(xs):
        if i < n_train:
            d[p] = "train"
        elif i < n_train + n_val:
            d[p] = "val"
        else:
            d[p] = "test"
    return d


def build_complete_dataset(
    input_dir: str | Path = "./datasets/raw-img",
    output_dir: str | Path = "./datasets/car-complete-seg",
    sam_weights: str | Path = "./models/sam/sam_vit_h.pth",
    sam_model_type: str = "vit_h",
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    class_id: int = 0,
    device: str | None = None,
) -> dict[str, int | str]:
    in_dir = Path(input_dir)
    out_root = Path(output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {in_dir}")
    imgs = list(_iter_images(in_dir))
    if not imgs:
        raise RuntimeError(f"未找到可用图片: {in_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = _init_sam(sam_model_type, sam_weights, device)
    _prepare_dirs(out_root)
    split_map = _split(imgs, split, seed)

    ok = 0
    skipped = 0
    for img_path in tqdm(imgs, desc="SAM 预标注(整车)"):
        img = _imread_cn(img_path)
        if img is None:
            skipped += 1
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        masks = generator.generate(img_rgb)
        main = _pick_main_mask(masks, w, h, min_area_ratio, max_area_ratio)
        if main is None:
            skipped += 1
            continue
        main = _clean_mask(main)
        line = _mask_to_polygon_line(main, class_id=class_id)
        if not line:
            skipped += 1
            continue

        sp = split_map[img_path]
        stem = img_path.stem
        img_out = out_root / "images" / sp / f"{stem}{img_path.suffix.lower()}"
        lbl_out = out_root / "labels" / sp / f"{stem}.txt"

        img_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(img_path), str(img_out))
        with open(lbl_out, "w", encoding="utf-8") as f:
            f.write(line + "\n")

        # 预览图，便于后续人工修标
        vis = img.copy()
        cts, _ = cv2.findContours(main.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cts, -1, (0, 255, 0), 2)
        overlay = vis.copy()
        overlay[main > 0] = (overlay[main > 0] * 0.55 + np.array([30, 180, 30]) * 0.45).astype(np.uint8)
        pv = out_root / "prelabel_preview" / sp / f"{stem}.jpg"
        _imwrite_cn(pv, overlay)

        ok += 1

    yaml_path = _write_data_yaml(out_root)
    return {
        "total": len(imgs),
        "ok": ok,
        "skipped": skipped,
        "data_yaml": str(yaml_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 预标注整车分割数据集（YOLO-seg 格式）")
    p.add_argument("--input_dir", default="./datasets/raw-img")
    p.add_argument("--output_dir", default="./datasets/car-complete-seg")
    p.add_argument("--sam_weights", default="./models/sam/sam_vit_h.pth")
    p.add_argument("--sam_model_type", default="vit_h")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_area_ratio", type=float, default=0.05)
    p.add_argument("--max_area_ratio", type=float, default=0.95)
    p.add_argument("--class_id", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_complete_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sam_weights=args.sam_weights,
        sam_model_type=args.sam_model_type,
        split=(float(args.split[0]), float(args.split[1]), float(args.split[2])),
        seed=int(args.seed),
        min_area_ratio=float(args.min_area_ratio),
        max_area_ratio=float(args.max_area_ratio),
        class_id=int(args.class_id),
        device=args.device,
    )
    print(
        f"完成: total={stats['total']} ok={stats['ok']} skipped={stats['skipped']} "
        f"data_yaml={stats['data_yaml']}"
    )


if __name__ == "__main__":
    main()
