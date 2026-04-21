from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


MODEL_PATH = Path("./runs/segment/train6/weights/epoch70.pt")
IMG_ROOT = Path("./test/img")
OUT_ROOT = Path("./test/result")
CONF = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEW_PARTS = {
    "front": ["front_right_light", "front_bumper", "right_mirror", "front_glass", "hood"],
    "right_side": ["front_right_door", "back_right_door", "right_mirror", "wheel"],
    "back": ["back_light", "back_bumper", "trunk", "back_glass"],
}


def _default_device() -> str | int:
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        return 0
    print("CUDA available: False")
    print("No GPU available, using CPU")
    return "cpu"


def _read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"保存失败: {path}")
    buf.tofile(str(path))


def _iter_images(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _mask_to_rgba_crop(img_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    if mask01.shape[:2] != (h, w):
        mask01 = cv2.resize(mask01.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask01 > 0.5).astype(np.uint8)
    if mask.max() == 0:
        return None

    ys, xs = np.where(mask > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_m = mask[y1:y2, x1:x2] * 255

    rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_bgr
    rgba[:, :, 3] = crop_m
    rgba[crop_m == 0, :3] = 0
    return rgba


def _choose_best_index(part_name: str, indexes: list[int], centers_x: np.ndarray, confs: np.ndarray) -> int:
    if len(indexes) == 1:
        return indexes[0]

    best_local = int(np.argmax(confs[indexes]))
    return indexes[best_local]


def _choose_front_wheel(cls_to_idx: dict[str, list[int]], centers_x: np.ndarray, confs: np.ndarray) -> int | None:
    wheel_idxs = cls_to_idx.get("wheel", [])
    if not wheel_idxs:
        return None
    if len(wheel_idxs) == 1:
        return wheel_idxs[0]

    front_door = cls_to_idx.get("front_right_door", [])
    back_door = cls_to_idx.get("back_right_door", [])
    if front_door and back_door:
        front_x = float(np.mean(centers_x[front_door]))
        back_x = float(np.mean(centers_x[back_door]))
        if front_x >= back_x:
            return wheel_idxs[int(np.argmax(centers_x[wheel_idxs]))]
        return wheel_idxs[int(np.argmin(centers_x[wheel_idxs]))]

    return wheel_idxs[int(np.argmax(confs[wheel_idxs]))]


def _export_parts_for_image(model: YOLO, img_path: Path, view_name: str, device: str | int) -> None:
    img = _read_image(img_path)
    if img is None:
        print(f"skip unreadable image: {img_path}")
        return

    results = model.predict(source=img, save=False, conf=CONF, device=device, verbose=False)
    if not results:
        print(f"no result: {img_path.name}")
        return

    result = results[0]
    if result.masks is None or result.boxes is None or result.boxes.cls is None:
        print(f"no masks: {img_path.name}")
        return

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    n = min(len(masks), len(classes), len(boxes), len(confs))
    if n <= 0:
        print(f"empty prediction: {img_path.name}")
        return

    cls_names = [model.names.get(int(classes[i]), str(int(classes[i]))) for i in range(n)]
    centers_x = (boxes[:n, 0] + boxes[:n, 2]) / 2.0
    cls_to_idx: dict[str, list[int]] = {}
    for i, name in enumerate(cls_names):
        cls_to_idx.setdefault(name, []).append(i)

    out_dir = OUT_ROOT / view_name / img_path.stem
    exported: list[str] = []

    for part_name in VIEW_PARTS[view_name]:
        if part_name == "wheel":
            chosen = _choose_front_wheel(cls_to_idx, centers_x, confs)
        else:
            idxs = cls_to_idx.get(part_name, [])
            chosen = None if not idxs else _choose_best_index(part_name, idxs, centers_x, confs)

        if chosen is None:
            continue

        rgba = _mask_to_rgba_crop(img, masks[chosen])
        if rgba is None:
            continue

        out_path = out_dir / f"{part_name}.png"
        _write_image(out_path, rgba)
        exported.append(part_name)

    print(f"{view_name} | {img_path.name} | exported: {', '.join(exported) if exported else 'none'}")


def main() -> None:
    device = _default_device()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for view_name in ("front", "right_side", "back"):
        view_dir = IMG_ROOT / view_name
        if not view_dir.is_dir():
            print(f"skip missing view dir: {view_dir}")
            continue
        for img_path in _iter_images(view_dir):
            _export_parts_for_image(model, img_path, view_name, device)

    print("All tests completed!")


if __name__ == "__main__":
    main()
