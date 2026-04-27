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
        Path("./models/yolo-seg/yolo11m-seg-carpart-epoch70.pt"),
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
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as e:
        raise ModuleNotFoundError(
            "未安装 segment-anything（SAM）。请先安装后再启用 --sam_refine。"
        ) from e

    weights_path = _resolve_sam_weights(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"SAM 权重文件不存在: {weights_path}")

    sam = sam_model_registry[model_type](checkpoint=str(weights_path))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def _sam_refine_instance_mask(
    predictor,
    img_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    yolo_mask01: np.ndarray | None,
) -> np.ndarray | None:
    h, w = img_rgb.shape[:2]
    box = box_xyxy.astype(np.float32).reshape(-1)
    if box.shape[0] != 4:
        return None
    box[0] = np.clip(box[0], 0, w - 1)
    box[2] = np.clip(box[2], 0, w - 1)
    box[1] = np.clip(box[1], 0, h - 1)
    box[3] = np.clip(box[3], 0, h - 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=True,
        return_logits=False,
    )
    if masks is None or len(masks) == 0:
        return None
    best = int(np.argmax(scores)) if scores is not None and len(scores) == len(masks) else 0
    m = masks[best].astype(bool)

    if yolo_mask01 is not None:
        ym = yolo_mask01
        if ym.shape[0] != h or ym.shape[1] != w:
            ym = cv2.resize(ym.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        guard = (ym > 0.5).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        guard = cv2.dilate(guard, k, iterations=1) > 0
        m = m & guard

    if not np.any(m):
        return None
    return m.astype(np.float32)


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
    anchor_xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    occupied_boxes: list[tuple[int, int, int, int]],
) -> None:
    h, w = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_w = tw + 8
    box_h = th + baseline + 8

    ax, ay = anchor_xy
    ax = max(0, min(int(ax), w - 1))
    ay = max(0, min(int(ay), h - 1))

    def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    candidates = [
        (ax + 70, ay - 50),
        (ax + 70, ay + 10),
        (ax - box_w - 70, ay - 50),
        (ax - box_w - 70, ay + 10),
        (ax + 20, ay - box_h - 20),
        (ax + 20, ay + 20),
    ]

    chosen_rect: tuple[int, int, int, int] | None = None
    for cx, cy in candidates:
        tx1 = max(0, min(int(cx), w - box_w))
        ty1 = max(0, min(int(cy), h - box_h))
        rect = (tx1, ty1, tx1 + box_w, ty1 + box_h)
        if any(_overlap(rect, ob) for ob in occupied_boxes):
            continue
        chosen_rect = rect
        break

    if chosen_rect is None:
        tx1 = max(0, min(ax + 20, w - box_w))
        ty1 = max(0, min(ay - box_h - 20, h - box_h))
        chosen_rect = (tx1, ty1, tx1 + box_w, ty1 + box_h)

    tx1, ty1, tx2, ty2 = chosen_rect
    text_org = (tx1 + 4, ty2 - baseline - 3)
    line_end = (tx1, ty1 + box_h // 2) if tx1 > ax else (tx2, ty1 + box_h // 2)

    cv2.line(img_bgr, (ax, ay), line_end, color, 2, cv2.LINE_AA)
    cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(img_bgr, text, text_org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    occupied_boxes.append(chosen_rect)


def _mask_anchor(mask01: np.ndarray, img_w: int, img_h: int) -> tuple[int, int] | None:
    if mask01.ndim != 2:
        return None
    if mask01.shape[0] != img_h or mask01.shape[1] != img_w:
        mask01 = cv2.resize(mask01.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    m = mask01 > 0.5
    if not np.any(m):
        return None
    ys, xs = np.where(m)
    return int(np.mean(xs)), int(np.mean(ys))


def _mask_to_rgba_crop(img_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray | None:
    """将实例掩码裁剪为 RGBA 小图（alpha 来自 mask）。"""
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


def _infer_view_tag(path: Path, cls_names: list[str], confs: np.ndarray | None) -> str:
    s = (path.stem or "").lower()
    parts = [p.lower() for p in path.parts]
    for tok in parts + [s]:
        if "front" in tok:
            return "front"
        if "rear" in tok or "back" in tok:
            return "back"
        if "side" in tok:
            return "side"

    front_names = {
        "front_bumper",
        "hood",
        "front_glass",
        "front_light",
        "front_plate",
    }
    back_names = {
        "back_bumper",
        "trunk",
        "tailgate",
        "back_glass",
        "back_light",
        "back_plate",
    }
    side_names = {
        "left_mirror",
        "right_mirror",
        "left_fender",
        "right_fender",
        "front_left_door",
        "front_right_door",
        "back_left_door",
        "back_right_door",
    }

    scores = confs.astype(float).tolist() if confs is not None else None
    f = b = sd = 0.0
    for i, n in enumerate(cls_names):
        w = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        if n in front_names:
            f += w
        if n in back_names:
            b += w
        if n in side_names:
            sd += w
    if f >= b and f >= sd and f > 0:
        return "front"
    if b >= f and b >= sd and b > 0:
        return "back"
    if sd > 0:
        return "side"
    return "unknown"


def _lr_side_from_name(name: str) -> str | None:
    n = name.lower()
    if "left" in n:
        return "left"
    if "right" in n:
        return "right"
    return None


def _swap_lr_name(name: str) -> str:
    s = name
    s = s.replace("left_", "__tmp__")
    s = s.replace("right_", "left_")
    s = s.replace("__tmp__", "right_")
    s = s.replace("_left", "__tmp__")
    s = s.replace("_right", "_left")
    s = s.replace("__tmp__", "_right")
    return s


def _mask_to_img01(mask01: np.ndarray, img_w: int, img_h: int) -> np.ndarray | None:
    if mask01.ndim != 2:
        return None
    m = mask01.astype(np.float32)
    if m.shape[0] != img_h or m.shape[1] != img_w:
        m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return m


def _mask_bbox_xyxy(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    m = mask01 > 0.5
    if not np.any(m):
        return None
    ys, xs = np.where(m)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return x1, y1, x2, y2


def _mirror_mask(mask01: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask01[:, ::-1])


def _mirror_box_xyxy(box: np.ndarray, img_w: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box.reshape(-1)[:4].tolist()]
    nx1 = float(img_w) - x2
    nx2 = float(img_w) - x1
    return np.array([nx1, y1, nx2, y2], dtype=np.float32)


def segment_car_parts(
    img_dir: str | Path = "./test/img",
    out_root: str | Path = "./result/segment",
    label_dir: str | Path = "./test/complete_label",
    weights: str | Path | None = None,
    conf: float = 0.25,
    save_crops: bool = True,
    save_labels: bool = True,
    sam_refine: bool = False,
    sam_weights: str | Path = "./models/sam_vit_h.pth",
    sam_model_type: str = "vit_h",
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
    sam_device = "cuda" if device != "cpu" else "cpu"
    sam_predictor = _init_sam(sam_model_type, sam_weights, sam_device) if sam_refine else None

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

        img_rgb: np.ndarray | None = None
        if sam_predictor is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(img_rgb)

        annotated = img.copy()
        annotated_changed = False
        occupied_label_boxes: list[tuple[int, int, int, int]] = []

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
            view_tag = _infer_view_tag(rel, cls_names, confs_np[:n] if confs_np is not None else None)
            centers_x = (boxes_np[:n, 0] + boxes_np[:n, 2]) / 2.0
            faces_right = _infer_faces_right(cls_names, centers_x)

            wheel_names = {"wheel", "front_wheel", "rear_wheel"}
            suppress_wheel = view_tag in {"front", "back"}
            wheel_union_native: np.ndarray | None = None
            wheel_union_img: np.ndarray | None = None
            if suppress_wheel:
                wheel_idxs = [i for i, n2 in enumerate(cls_names) if n2 in wheel_names]
                if wheel_idxs:
                    mh, mw = masks_np[0].shape[:2]
                    wheel_union_native = np.zeros((mh, mw), dtype=bool)
                    for wi in wheel_idxs:
                        m0 = masks_np[wi]
                        if m0.shape[:2] != (mh, mw):
                            m0 = cv2.resize(m0.astype(np.float32), (mw, mh), interpolation=cv2.INTER_NEAREST)
                        wheel_union_native |= (m0 > 0.5)
                    ih, iw = img.shape[:2]
                    wheel_union_img = cv2.resize(wheel_union_native.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST) > 0

            keep = np.ones((n,), dtype=bool)
            if suppress_wheel:
                for wi in [i for i, n2 in enumerate(cls_names) if n2 in wheel_names]:
                    keep[wi] = False
            for cls_name in set(cls_names):
                if not (
                    (cls_name in wheel_names and not suppress_wheel)
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
            img_h, img_w = img.shape[:2]

            instances: list[dict[str, object]] = []
            for i in range(n):
                if not keep[i]:
                    continue
                cls_name = cls_names[i]
                mask_i = masks_np[i]
                if sam_predictor is not None and img_rgb is not None:
                    refined = _sam_refine_instance_mask(
                        sam_predictor,
                        img_rgb,
                        boxes_np[i],
                        mask_i,
                    )
                    if refined is not None:
                        mask_i = refined
                mask_img = _mask_to_img01(mask_i, img_w, img_h)
                if mask_img is None:
                    continue
                if suppress_wheel and wheel_union_img is not None and cls_name not in wheel_names:
                    mask_img = np.where(wheel_union_img, 0.0, mask_img)
                if not np.any(mask_img > 0.5):
                    continue
                conf_i = float(confs_np[i]) if confs_np is not None else 1.0
                instances.append(
                    {
                        "name": cls_name,
                        "mask": mask_img,
                        "box": boxes_np[i].astype(np.float32),
                        "conf": conf_i,
                    }
                )

            if view_tag in {"front", "back"} and instances:
                center_x = float(img_w) / 2.0
                processed: list[dict[str, object]] = []

                for inst in instances:
                    name = str(inst["name"])
                    side = _lr_side_from_name(name)
                    if side is not None:
                        box = inst["box"]
                        bx = float((float(box[0]) + float(box[2])) / 2.0)
                        should_right = bx >= center_x
                        if side == "left" and should_right:
                            inst["name"] = _swap_lr_name(name)
                        elif side == "right" and not should_right:
                            inst["name"] = _swap_lr_name(name)
                    processed.append(inst)

                instances = processed

                right_only: list[dict[str, object]] = []
                mirrors = [inst for inst in instances if str(inst["name"]) in {"left_mirror", "right_mirror"}]
                lights = [inst for inst in instances if _is_light_class(str(inst["name"]))]
                others = [inst for inst in instances if inst not in mirrors and inst not in lights]

                right_mirrors = [m for m in mirrors if str(m["name"]) == "right_mirror"]
                left_mirrors = [m for m in mirrors if str(m["name"]) == "left_mirror"]
                if right_mirrors:
                    best = max(right_mirrors, key=lambda d: float(d["conf"]))
                    right_only.append(best)
                elif left_mirrors:
                    best = max(left_mirrors, key=lambda d: float(d["conf"]))
                    nm = _mirror_mask(best["mask"])
                    nb = _mirror_box_xyxy(best["box"], img_w)
                    right_only.append({"name": "right_mirror", "mask": nm, "box": nb, "conf": float(best["conf"])})

                light_groups: dict[str, list[dict[str, object]]] = {}
                for l in lights:
                    nm = str(l["name"])
                    base = nm.replace("left_", "").replace("right_", "").replace("_left", "").replace("_right", "")
                    light_groups.setdefault(base, []).append(l)

                for base, group in light_groups.items():
                    right = [g for g in group if _lr_side_from_name(str(g["name"])) == "right"]
                    left = [g for g in group if _lr_side_from_name(str(g["name"])) == "left"]
                    none = [g for g in group if _lr_side_from_name(str(g["name"])) is None]

                    if right:
                        best = max(right, key=lambda d: float(d["conf"]))
                        right_only.append(best)
                        continue

                    if none:
                        best = max(none, key=lambda d: float(d["conf"]))
                        bx = float((float(best["box"][0]) + float(best["box"][2])) / 2.0)
                        if bx >= center_x:
                            right_only.append(best)
                        else:
                            nm = _mirror_mask(best["mask"])
                            nb = _mirror_box_xyxy(best["box"], img_w)
                            right_only.append({"name": str(best["name"]), "mask": nm, "box": nb, "conf": float(best["conf"])})
                        continue

                    if left:
                        best = max(left, key=lambda d: float(d["conf"]))
                        nm = _mirror_mask(best["mask"])
                        nb = _mirror_box_xyxy(best["box"], img_w)
                        new_name = _swap_lr_name(str(best["name"]))
                        right_only.append({"name": new_name, "mask": nm, "box": nb, "conf": float(best["conf"])})

                central_merge_names = {
                    "front_bumper",
                    "back_bumper",
                    "front_glass",
                    "back_glass",
                    "hood",
                    "trunk",
                    "tailgate",
                }
                merged: list[dict[str, object]] = []
                by_name: dict[str, list[dict[str, object]]] = {}
                for inst in others:
                    by_name.setdefault(str(inst["name"]), []).append(inst)

                for nm, group in by_name.items():
                    if nm in central_merge_names and len(group) > 1:
                        um = np.zeros((img_h, img_w), dtype=np.float32)
                        for g in group:
                            um = np.maximum(um, g["mask"].astype(np.float32))
                        bb = _mask_bbox_xyxy(um)
                        if bb is None:
                            continue
                        x1, y1, x2, y2 = bb
                        if x2 <= int(center_x) or x1 >= int(center_x):
                            um = np.maximum(um, _mirror_mask(um))
                            bb2 = _mask_bbox_xyxy(um)
                            if bb2 is not None:
                                x1, y1, x2, y2 = bb2
                        merged.append(
                            {
                                "name": nm,
                                "mask": um,
                                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                                "conf": max(float(g["conf"]) for g in group),
                            }
                        )
                    else:
                        merged.extend(group)

                instances = right_only + merged

            for inst in instances:
                cls_name = str(inst["name"])
                mask_img = inst["mask"]
                if save_labels:
                    color = _stable_color(cls_name)
                    c = float(inst["conf"])
                    label = cls_name if confs_np is None else f"{cls_name} {c:.2f}"
                    anchor = _mask_anchor(mask_img, img_w, img_h)
                    if anchor is None:
                        box = inst["box"]
                        anchor = (int((float(box[0]) + float(box[2])) / 2), int((float(box[1]) + float(box[3])) / 2))
                    _draw_label(annotated, anchor, label, color, occupied_label_boxes)
                    annotated_changed = True

                if save_crops:
                    rgba = _mask_to_rgba_crop(img, mask_img)
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
    parser.add_argument("--sam_refine", action="store_true")
    parser.add_argument("--sam_weights", default="./models/sam_vit_h.pth")
    parser.add_argument("--sam_model_type", default="vit_h")
    args = parser.parse_args()

    counts = segment_car_parts(
        img_dir=args.img_dir,
        out_root=args.out_root,
        label_dir=args.label_dir,
        weights=args.weights,
        conf=args.conf,
        save_crops=not args.no_crops,
        save_labels=not args.no_labels,
        sam_refine=args.sam_refine,
        sam_weights=args.sam_weights,
        sam_model_type=args.sam_model_type,
    )
    for k in sorted(counts.keys()):
        print(f"{k}: {counts[k]}")


if __name__ == "__main__":
    main()

