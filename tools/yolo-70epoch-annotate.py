"""
Use trained YOLO-seg model (epoch70.pt) to auto-annotate images and export YOLO-seg labels.

Usage:
    python tools/yolo-70epoch-annotate.py
    python tools/yolo-70epoch-annotate.py --input ./raw-img --output ./datasets/carparts-seg-yolo-supplement
"""

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO epoch70 auto-annotation for segmentation")
    parser.add_argument("--input", type=str, default="datasets/raw-img/", help="Input image directory")
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/carparts-seg-yolo-supplement",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/segment/train6/weights/epoch70.pt",
        help="YOLO-seg model weights path",
    )
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="train/val/test split ratios")
    parser.add_argument("--conf", type=float, default=0.25, help="Predict confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/0")
    parser.add_argument("--visualize", action="store_true", help="Save visualized predictions to output/visualize")
    return parser.parse_args()


def resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def find_input_dir(input_arg: str, project_root: Path) -> Path:
    candidates = [
        resolve_path(input_arg, project_root),
        project_root / "raw-img",
        project_root / "datasets" / "raw-img",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"Input directory not found. Tried: {[str(x) for x in candidates]}")


def resolve_device(device_arg: str) -> str:
    """Return a valid device string for ultralytics predict."""
    d = str(device_arg).strip().lower()
    if d == "cpu":
        return "cpu"

    if d == "cuda":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
        print("Warning: CUDA not available, fallback to CPU.")
        return "cpu"

    # Keep explicit device ids such as '0' or '0,1' only when CUDA is available.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return device_arg

    print(f"Warning: Requested device='{device_arg}' but CUDA not available, fallback to CPU.")
    return "cpu"


def list_images(image_dir: Path) -> List[Path]:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    files: List[Path] = []
    for ext in exts:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))
    files = sorted(set(files))
    return files


def split_dataset(file_list: Sequence[Path], split_ratios: Sequence[float]) -> Tuple[List[Path], List[Path], List[Path]]:
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

    rng = np.random.default_rng(42)
    idx = np.arange(len(file_list))
    rng.shuffle(idx)

    n = len(file_list)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])
    train = [file_list[i] for i in idx[:train_end]]
    val = [file_list[i] for i in idx[train_end:val_end]]
    test = [file_list[i] for i in idx[val_end:]]
    return train, val, test


def imread_cn(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def imwrite_cn(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


def copy_image_cn(src: Path, dst: Path) -> None:
    img = imread_cn(src)
    imwrite_cn(dst, img)


def _stable_color(name: str) -> tuple[int, int, int]:
    v = 0
    for ch in name:
        v = (v * 131 + ord(ch)) & 0xFFFFFFFF
    b = 64 + (v & 0x7F)
    g = 64 + ((v >> 7) & 0x7F)
    r = 64 + ((v >> 14) & 0x7F)
    return int(b), int(g), int(r)


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


def _draw_external_label(
    img_bgr: np.ndarray,
    anchor_xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    occupied_boxes: list[tuple[int, int, int, int]],
    forbidden_mask: np.ndarray | None = None,
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

    # Dynamic search around anchor: prefer text boxes outside object (forbidden_mask).
    candidates: list[tuple[int, int]] = []
    radii = [70, 110, 150, 190, 230]
    dirs = [(1, 0), (1, -1), (1, 1), (-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1)]
    for r in radii:
        for dx, dy in dirs:
            candidates.append((ax + dx * r, ay + dy * r))

    def _mask_overlap_ratio(rect: tuple[int, int, int, int]) -> float:
        if forbidden_mask is None:
            return 0.0
        rx1, ry1, rx2, ry2 = rect
        if rx2 <= rx1 or ry2 <= ry1:
            return 1.0
        roi = forbidden_mask[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return 1.0
        return float(np.count_nonzero(roi)) / float(roi.size)

    chosen_rect: tuple[int, int, int, int] | None = None
    best_rect: tuple[int, int, int, int] | None = None
    best_score = 1e9
    for cx, cy in candidates:
        tx1 = max(0, min(int(cx), w - box_w))
        ty1 = max(0, min(int(cy), h - box_h))
        rect = (tx1, ty1, tx1 + box_w, ty1 + box_h)
        if any(_overlap(rect, ob) for ob in occupied_boxes):
            continue
        overlap_ratio = _mask_overlap_ratio(rect)
        # Strictly prefer non-overlapping with object body.
        if overlap_ratio <= 1e-6:
            chosen_rect = rect
            break
        if overlap_ratio < best_score:
            best_score = overlap_ratio
            best_rect = rect

    if chosen_rect is None:
        chosen_rect = best_rect
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


def _draw_visualization(
    img_bgr: np.ndarray,
    classes: np.ndarray,
    masks: np.ndarray,
    names_map,
) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    occupied_boxes: list[tuple[int, int, int, int]] = []
    n = min(len(classes), len(masks))

    # Build a car-body mask to avoid putting text on top of the object.
    forbidden_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(n):
        m0 = masks[i]
        if m0.shape[0] != h or m0.shape[1] != w:
            m0 = cv2.resize(m0.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        forbidden_mask = np.maximum(forbidden_mask, (m0 > 0.5).astype(np.uint8))
    forbidden_mask = cv2.dilate(forbidden_mask, np.ones((15, 15), np.uint8), iterations=1)

    for i in range(n):
        cls_id = int(classes[i])
        cls_name = names_map.get(cls_id, str(cls_id)) if isinstance(names_map, dict) else str(cls_id)
        color = _stable_color(cls_name)
        mask01 = masks[i]
        if mask01.shape[0] != h or mask01.shape[1] != w:
            mask01 = cv2.resize(mask01.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        m = mask01 > 0.5
        if not np.any(m):
            continue
        # keep current color-based part display
        vis[m] = (0.55 * vis[m] + 0.45 * np.array(color)).astype(np.uint8)
        anchor = _mask_anchor(mask01, w, h)
        if anchor is not None:
            _draw_external_label(vis, anchor, cls_name, color, occupied_boxes, forbidden_mask)
    return vis


def _to_names_list(model_names) -> list[str]:
    if isinstance(model_names, dict):
        return [model_names[k] for k in sorted(model_names.keys())]
    return list(model_names)


def _prepare_output_names(model_names) -> tuple[list[str], int | None, int | None]:
    names = _to_names_list(model_names)
    if "front_wheel" in names and "rear_wheel" in names:
        return names, names.index("front_wheel"), names.index("rear_wheel")
    if "wheel" in names:
        wheel_id = names.index("wheel")
        names[wheel_id] = "front_wheel"
        rear_id = len(names)
        names.append("rear_wheel")
        return names, wheel_id, rear_id
    return names, None, None


def _mean_x_for_name(
    classes: np.ndarray,
    centers_x: np.ndarray,
    names_list: list[str],
    target_name: str,
) -> float | None:
    idxs = [i for i, c in enumerate(classes) if names_list[int(c)] == target_name]
    if not idxs:
        return None
    return float(np.mean(centers_x[idxs]))


def _infer_faces_right_from_pred(classes: np.ndarray, centers_x: np.ndarray, names_list: list[str]) -> bool:
    pairs = [
        ("front_bumper", "back_bumper"),
        ("hood", "trunk"),
        ("hood", "tailgate"),
        ("front_glass", "back_glass"),
        ("front_light", "back_light"),
    ]
    for f_name, b_name in pairs:
        xf = _mean_x_for_name(classes, centers_x, names_list, f_name)
        xb = _mean_x_for_name(classes, centers_x, names_list, b_name)
        if xf is not None and xb is not None and xf != xb:
            return xf > xb
    return True


def _infer_view_tag(
    image_path: Path,
    classes: np.ndarray,
    confs: np.ndarray | None,
    names_list: list[str],
) -> str:
    stem = image_path.stem.lower()
    if "front" in stem:
        return "front"
    if "back" in stem or "rear" in stem:
        return "back"
    if "side" in stem:
        return "side"

    front_names = {
        "front_bumper", "hood", "front_glass", "front_light", "front_left_light", "front_right_light"
    }
    back_names = {
        "back_bumper", "trunk", "tailgate", "back_glass", "back_light", "back_left_light", "back_right_light"
    }
    side_names = {
        "front_right_door", "back_right_door", "front_left_door", "back_left_door", "left_mirror", "right_mirror"
    }

    front_score = 0.0
    back_score = 0.0
    side_score = 0.0
    for i, c in enumerate(classes):
        name = names_list[int(c)] if int(c) < len(names_list) else str(int(c))
        s = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        if name in front_names:
            front_score += s
        if name in back_names:
            back_score += s
        if name in side_names:
            side_score += s

    if front_score >= back_score and front_score >= side_score and front_score > 0:
        return "front"
    if back_score >= front_score and back_score >= side_score and back_score > 0:
        return "back"
    return "side"


def _wheel_class_ids(names_list: list[str]) -> set[int]:
    out: set[int] = set()
    for n in ("wheel", "front_wheel", "rear_wheel"):
        if n in names_list:
            out.add(names_list.index(n))
    return out


def _filter_classes_by_view(
    image_path: Path,
    classes: np.ndarray,
    confs: np.ndarray | None,
    polygons: Sequence,
    masks: np.ndarray | None,
    names_list: list[str],
) -> tuple[np.ndarray, list, np.ndarray | None]:
    view_tag = _infer_view_tag(image_path, classes, confs, names_list)
    if view_tag not in {"front", "back"}:
        return classes, list(polygons), masks

    wheel_ids = _wheel_class_ids(names_list)
    if not wheel_ids:
        return classes, list(polygons), masks

    keep_idxs = [i for i, c in enumerate(classes) if int(c) not in wheel_ids]
    if len(keep_idxs) == len(classes):
        return classes, list(polygons), masks

    new_classes = classes[keep_idxs]
    new_polygons = [polygons[i] for i in keep_idxs]
    new_masks = masks[keep_idxs] if masks is not None else None
    return new_classes, new_polygons, new_masks


def _remap_wheel_classes(
    classes: np.ndarray,
    boxes_xyxy: np.ndarray,
    names_list: list[str],
    front_wheel_id: int | None,
    rear_wheel_id: int | None,
) -> np.ndarray:
    if front_wheel_id is None or rear_wheel_id is None:
        return classes

    # Only remap when input prediction still uses legacy "wheel".
    legacy_wheel_id = front_wheel_id if front_wheel_id < len(names_list) and names_list[front_wheel_id] == "front_wheel" else None
    if legacy_wheel_id is None:
        return classes

    # If model already predicts rear_wheel, keep as-is.
    if np.any(classes == rear_wheel_id):
        return classes

    centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    remap = classes.copy()
    wheel_idxs = [i for i, c in enumerate(classes) if int(c) == legacy_wheel_id]
    if not wheel_idxs:
        return remap

    faces_right = _infer_faces_right_from_pred(classes, centers_x, names_list)
    if len(wheel_idxs) == 1:
        i = wheel_idxs[0]
        x_front = np.mean(
            [
                x
                for name in ("front_bumper", "hood", "front_glass")
                if (x := _mean_x_for_name(classes, centers_x, names_list, name)) is not None
            ]
        ) if any(_mean_x_for_name(classes, centers_x, names_list, name) is not None for name in ("front_bumper", "hood", "front_glass")) else None
        x_back = np.mean(
            [
                x
                for name in ("back_bumper", "trunk", "tailgate", "back_glass")
                if (x := _mean_x_for_name(classes, centers_x, names_list, name)) is not None
            ]
        ) if any(_mean_x_for_name(classes, centers_x, names_list, name) is not None for name in ("back_bumper", "trunk", "tailgate", "back_glass")) else None
        if x_front is not None and x_back is not None:
            remap[i] = front_wheel_id if abs(centers_x[i] - x_front) <= abs(centers_x[i] - x_back) else rear_wheel_id
        else:
            remap[i] = front_wheel_id if faces_right else rear_wheel_id
        return remap

    if faces_right:
        front_idx = wheel_idxs[int(np.argmax(centers_x[wheel_idxs]))]
    else:
        front_idx = wheel_idxs[int(np.argmin(centers_x[wheel_idxs]))]
    for i in wheel_idxs:
        remap[i] = front_wheel_id if i == front_idx else rear_wheel_id
    return remap


def polygons_to_yolo_lines(classes: np.ndarray, polygons: Sequence) -> List[str]:
    lines: List[str] = []
    count = min(len(classes), len(polygons))
    for i in range(count):
        cls_id = int(classes[i])
        poly_item = polygons[i]
        segments = poly_item if isinstance(poly_item, list) else [poly_item]

        for seg in segments:
            seg_arr = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            if seg_arr.shape[0] < 3:
                continue
            seg_arr = np.clip(seg_arr, 0.0, 1.0)
            coords = seg_arr.reshape(-1)
            coord_str = " ".join(f"{x:.6f}" for x in coords)
            lines.append(f"{cls_id} {coord_str}")
    return lines


def write_label(label_path: Path, lines: Sequence[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def create_dataset_yaml(output_dir: Path, names: list[str]) -> Path:

    content = [
        "# Auto-generated by yolo-70epoch-annotate.py",
        f"path: {output_dir}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for i, name in enumerate(names):
        content.append(f"  {i}: {name}")

    yaml_path = output_dir / "carparts-seg-yolo-supplement.yaml"
    yaml_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return yaml_path


def process_split(
    model: YOLO,
    split_name: str,
    images: Sequence[Path],
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    visualize: bool,
    names_list: list[str],
    front_wheel_id: int | None,
    rear_wheel_id: int | None,
) -> None:
    img_out = output_dir / "images" / split_name
    lbl_out = output_dir / "labels" / split_name
    vis_out = output_dir / "visualize" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_out.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images, desc=split_name):
        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save=False,
            verbose=False,
            retina_masks=True,
        )
        result = results[0]
        lines: List[str] = []
        if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs_np = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else None
            classes = _remap_wheel_classes(classes, boxes_xyxy, names_list, front_wheel_id, rear_wheel_id)
            polygons = result.masks.xyn
            classes, polygons, _ = _filter_classes_by_view(
                image_path, classes, confs_np, polygons, None, names_list
            )
            lines = polygons_to_yolo_lines(classes, polygons)

        write_label(lbl_out / f"{image_path.stem}.txt", lines)
        copy_image_cn(image_path, img_out / image_path.name)
        if visualize:
            base_img = imread_cn(image_path)
            if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
                cls_np = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
                boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
                confs_np = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else None
                cls_np = _remap_wheel_classes(cls_np, boxes_xyxy, names_list, front_wheel_id, rear_wheel_id)
                mask_np = result.masks.data.detach().cpu().numpy()
                cls_np, _, mask_np = _filter_classes_by_view(
                    image_path, cls_np, confs_np, result.masks.xyn, mask_np, names_list
                )
                vis_names = {i: n for i, n in enumerate(names_list)}
                vis_img = _draw_visualization(base_img, cls_np, mask_np, vis_names)
            else:
                vis_img = base_img
            imwrite_cn(vis_out / image_path.name, vis_img)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    input_dir = find_input_dir(args.input, project_root)
    output_dir = resolve_path(args.output, project_root)
    weights_path = resolve_path(args.weights, project_root)

    if not weights_path.exists():
        print(f"Error: weights not found: {weights_path}")
        sys.exit(1)

    images = list_images(input_dir)
    if not images:
        print(f"Error: no images found in {input_dir}")
        sys.exit(1)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Weights: {weights_path}")
    print(f"Found {len(images)} images")
    run_device = resolve_device(args.device)
    print(f"Device: {run_device}")

    train, val, test = split_dataset(images, args.split)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    model = YOLO(str(weights_path))
    names_list, front_wheel_id, rear_wheel_id = _prepare_output_names(model.names)
    if "wheel" in _to_names_list(model.names) and "rear_wheel" in names_list:
        print("Info: auto-splitting legacy 'wheel' labels to 'front_wheel'/'rear_wheel'.")

    process_split(
        model, "train", train, output_dir, args.conf, args.iou, args.imgsz, run_device, args.visualize,
        names_list, front_wheel_id, rear_wheel_id
    )
    process_split(
        model, "val", val, output_dir, args.conf, args.iou, args.imgsz, run_device, args.visualize,
        names_list, front_wheel_id, rear_wheel_id
    )
    process_split(
        model, "test", test, output_dir, args.conf, args.iou, args.imgsz, run_device, args.visualize,
        names_list, front_wheel_id, rear_wheel_id
    )

    yaml_path = create_dataset_yaml(output_dir, names_list)
    print(f"Done. Dataset saved to: {output_dir}")
    print(f"Config saved to: {yaml_path}")


if __name__ == "__main__":
    main()
