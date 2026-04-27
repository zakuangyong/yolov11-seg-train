from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


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
    alt = Path("./models/sam") / p.name
    if alt.exists():
        return alt
    alt2 = Path("./models") / p.name
    if alt2.exists():
        return alt2
    return p


def _init_sam(model_type: str, weights_path: str | Path, device: str):
    try:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    except Exception as e:
        raise ModuleNotFoundError("未安装 segment-anything（SAM）。") from e

    weights_path = _resolve_sam_weights(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"SAM 权重文件不存在: {weights_path}")

    sam = sam_model_registry[model_type](checkpoint=str(weights_path))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,
    )
    return predictor, generator


def _border_touches(mask_u8: np.ndarray) -> tuple[float, float, float, float, int]:
    h, w = mask_u8.shape[:2]
    m = mask_u8 > 0
    top = float(np.count_nonzero(m[0, :])) / max(1.0, float(w))
    bottom = float(np.count_nonzero(m[-1, :])) / max(1.0, float(w))
    left = float(np.count_nonzero(m[:, 0])) / max(1.0, float(h))
    right = float(np.count_nonzero(m[:, -1])) / max(1.0, float(h))
    touch_cnt = int(top > 0.02) + int(bottom > 0.02) + int(left > 0.02) + int(right > 0.02)
    return top, bottom, left, right, touch_cnt


def _is_background_like(mask_u8: np.ndarray) -> bool:
    h, w = mask_u8.shape[:2]
    area = float(np.count_nonzero(mask_u8 > 0)) / max(1.0, float(h * w))
    top, bottom, left, right, touch_cnt = _border_touches(mask_u8)
    if touch_cnt >= 3 and area >= 0.45:
        return True
    if top > 0.60 and left > 0.60 and right > 0.60 and area >= 0.40:
        return True
    if bottom > 0.75 and top > 0.40 and area >= 0.55:
        return True
    return False


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


def _maybe_invert(mask_u8: np.ndarray, min_area_ratio: float, max_area_ratio: float) -> np.ndarray:
    h, w = mask_u8.shape[:2]
    area = float(np.count_nonzero(mask_u8 > 0)) / max(1.0, float(h * w))
    inv = (mask_u8 == 0).astype(np.uint8)
    inv = _largest_component(inv)
    inv_area = float(np.count_nonzero(inv > 0)) / max(1.0, float(h * w))
    if _is_background_like(mask_u8) and (min_area_ratio <= inv_area <= max_area_ratio) and not _is_background_like(inv):
        return inv
    if not (min_area_ratio <= area <= max_area_ratio) and (min_area_ratio <= inv_area <= max_area_ratio) and not _is_background_like(inv):
        return inv
    return mask_u8


def _predict_main_mask_points(
    predictor,
    img_rgb: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
) -> np.ndarray | None:
    h, w = img_rgb.shape[:2]
    predictor.set_image(img_rgb)

    pos = np.array(
        [
            [int(w * 0.50), int(h * 0.62)],
            [int(w * 0.40), int(h * 0.62)],
            [int(w * 0.60), int(h * 0.62)],
            [int(w * 0.50), int(h * 0.45)],
        ],
        dtype=np.int32,
    )
    neg = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [int(w * 0.5), 0],
            [int(w * 0.5), h - 1],
            [0, int(h * 0.5)],
            [w - 1, int(h * 0.5)],
        ],
        dtype=np.int32,
    )
    pts = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones((len(pos),), dtype=np.int32), np.zeros((len(neg),), dtype=np.int32)], axis=0)

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        multimask_output=True,
        return_logits=False,
    )
    if masks is None or len(masks) == 0:
        return None

    img_area = float(h * w)
    best_score = -1.0
    best_mask: np.ndarray | None = None
    for i in range(len(masks)):
        m = masks[i].astype(np.uint8)
        area = float(np.count_nonzero(m > 0))
        if area <= 0:
            continue
        ar = area / max(1.0, img_area)
        if ar < float(min_area_ratio) or ar > float(max_area_ratio):
            continue
        upper_y = max(1, int(h * 0.55))
        upper_ratio = float(np.count_nonzero(m[:upper_y, :])) / max(1.0, area)
        top, bottom, left, right, touch_cnt = _border_touches(m)
        border_penalty = 1.0 - min(1.0, (top + bottom + left + right) / 2.0)
        s = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        sc = area * (0.35 + 0.65 * upper_ratio) * (0.35 + 0.65 * border_penalty) * (0.5 + 0.5 * s)
        if touch_cnt >= 3 and upper_ratio < 0.20:
            sc *= 0.25
        if sc > best_score:
            best_score = sc
            best_mask = m

    if best_mask is None:
        best = int(np.argmax(scores)) if scores is not None and len(scores) == len(masks) else 0
        best_mask = masks[best].astype(np.uint8)

    return (best_mask > 0).astype(np.uint8)


def _pick_main_mask(
    masks: list[dict],
    img_w: int,
    img_h: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> np.ndarray | None:
    if not masks:
        return None
    img_area = float(img_w * img_h)
    min_area = img_area * float(min_area_ratio)
    max_area = img_area * float(max_area_ratio)
    cx, cy = float(img_w) / 2.0, float(img_h) / 2.0

    def _score_one(d: dict) -> tuple[float, np.ndarray] | None:
        area = float(d.get("area", 0))
        if area <= 0:
            return None
        if area < min_area or area > max_area:
            return None
        seg = d.get("segmentation", None)
        bbox = d.get("bbox", None)
        if not isinstance(seg, np.ndarray) or seg.ndim != 2:
            return None
        if bbox is None or len(bbox) != 4:
            return None
        x, y, bw, bh = [float(v) for v in bbox]
        if bw <= 1 or bh <= 1:
            return None

        m = seg.astype(bool)
        upper_y = max(1, int(img_h * 0.55))
        upper = float(np.count_nonzero(m[:upper_y, :]))
        upper_ratio = upper / max(1.0, area)

        bx = x + bw / 2.0
        by = y + bh / 2.0
        dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        maxd = (cx**2 + cy**2) ** 0.5 + 1e-6
        center_bonus = 1.0 - min(1.0, dist / maxd)
        by_norm = by / max(1.0, float(img_h))

        top_touch = float(np.count_nonzero(m[0, :])) / max(1.0, float(img_w))
        bottom_touch = float(np.count_nonzero(m[-1, :])) / max(1.0, float(img_w))
        left_touch = float(np.count_nonzero(m[:, 0])) / max(1.0, float(img_h))
        right_touch = float(np.count_nonzero(m[:, -1])) / max(1.0, float(img_h))
        touch_cnt = int(top_touch > 0.02) + int(bottom_touch > 0.02) + int(left_touch > 0.02) + int(right_touch > 0.02)

        stability = float(d.get("stability_score", 1.0))
        iou = float(d.get("predicted_iou", 1.0))

        score = (
            area
            * (0.6 + 0.4 * center_bonus)
            * (0.35 + 0.65 * upper_ratio)
            * (0.5 + 0.5 * stability)
            * (0.5 + 0.5 * iou)
        )

        if by_norm > 0.70:
            score *= 0.35
        if bottom_touch > 0.25 and upper_ratio < 0.12:
            score *= 0.05
        if touch_cnt >= 3 and upper_ratio < 0.20:
            score *= 0.25

        return score, m.astype(np.uint8)

    scored: list[tuple[float, np.ndarray]] = []
    for d in masks:
        r = _score_one(d)
        if r is not None:
            scored.append(r)

    scored.sort(key=lambda t: t[0], reverse=True)
    best_mask: np.ndarray | None = None
    for _, m in scored:
        area = float(np.count_nonzero(m))
        if area <= 0:
            continue
        upper_y = max(1, int(img_h * 0.55))
        upper_ratio = float(np.count_nonzero(m[:upper_y, :])) / max(1.0, area)
        bottom_touch = float(np.count_nonzero(m[-1, :])) / max(1.0, float(img_w))
        if bottom_touch > 0.35 and upper_ratio < 0.10:
            continue
        best_mask = m
        break

    if best_mask is None:
        best = max(masks, key=lambda d: int(d.get("area", 0)))
        seg = best.get("segmentation", None)
        if isinstance(seg, np.ndarray) and seg.ndim == 2:
            best_mask = seg.astype(np.uint8)

    if best_mask is None:
        return None
    return (best_mask > 0).astype(np.uint8)


def _postprocess_mask(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape[:2]
    m0 = (mask_u8 > 0).astype(np.uint8)
    m0 = _largest_component(m0)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(m0, k3, iterations=1)
    core = _largest_component(core)

    near_ksz = int(max(21, (min(h, w) * 0.03)))
    if near_ksz % 2 == 0:
        near_ksz += 1
    k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_ksz, near_ksz))
    near = cv2.dilate(core, k_near, iterations=1)
    m = (m0 > 0) & (near > 0)
    m = m.astype(np.uint8)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    return _largest_component(m)


def _mask_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if len(ys) == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return x1, y1, x2, y2


def cutout_main(
    img_dir: str | Path,
    out_dir: str | Path = "./result/cutout_main",
    sam_weights: str | Path = "./models/sam_vit_h.pth",
    sam_model_type: str = "vit_h",
    device: str | None = None,
    pad: int = 20,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
) -> int:
    img_dir = Path(img_dir)
    base_dir = img_dir.parent if img_dir.is_file() else img_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor, generator = _init_sam(sam_model_type, sam_weights, device)

    ok_cnt = 0
    for img_path in _iter_images(img_dir):
        img = _imread_cn(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        main_mask = _predict_main_mask_points(predictor, img_rgb, min_area_ratio, max_area_ratio)
        if main_mask is None:
            ms = generator.generate(img_rgb)
            main_mask = _pick_main_mask(ms, w, h, min_area_ratio, max_area_ratio)
        if main_mask is None:
            continue
        main_mask = _maybe_invert(main_mask, min_area_ratio, max_area_ratio)
        main_mask = _largest_component(main_mask)
        bb0 = _mask_bbox(main_mask)
        if bb0 is None:
            continue
        x1g, y1g, x2g, y2g = bb0
        bw = max(1, x2g - x1g)
        bh = max(1, y2g - y1g)
        gate = max(40, int(max(bw, bh) * 0.20))
        x1g = max(0, x1g - gate)
        y1g = max(0, y1g - gate)
        x2g = min(w, x2g + gate)
        y2g = min(h, y2g + gate)
        gated = np.zeros_like(main_mask, dtype=np.uint8)
        gated[y1g:y2g, x1g:x2g] = main_mask[y1g:y2g, x1g:x2g]
        main_mask = _postprocess_mask(gated)
        bb = _mask_bbox(main_mask)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        p = max(0, int(pad))
        x1 = max(0, x1 - p)
        y1 = max(0, y1 - p)
        x2 = min(w, x2 + p)
        y2 = min(h, y2 + p)

        crop_bgr = img[y1:y2, x1:x2]
        crop_m = main_mask[y1:y2, x1:x2] * 255
        rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = crop_bgr
        rgba[:, :, 3] = crop_m
        rgba[crop_m == 0, :3] = 0

        rel = img_path.relative_to(base_dir)
        out_path = (out_dir / rel.parent / rel.stem).with_suffix(".png")
        if _imwrite_cn(out_path, rgba):
            ok_cnt += 1

    return ok_cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", default="./result/cutout_main")
    parser.add_argument("--sam_weights", default="./models/sam_vit_h.pth")
    parser.add_argument("--sam_model_type", default="vit_h")
    parser.add_argument("--device", default=None)
    parser.add_argument("--pad", type=int, default=80)
    parser.add_argument("--min_area_ratio", type=float, default=0.05)
    parser.add_argument("--max_area_ratio", type=float, default=0.95)
    args = parser.parse_args()

    n = cutout_main(
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        sam_weights=args.sam_weights,
        sam_model_type=args.sam_model_type,
        device=args.device,
        pad=args.pad,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
    )
    print(f"saved: {n}")


if __name__ == "__main__":
    main()
