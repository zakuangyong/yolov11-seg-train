from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


DATA_DIR = Path("./datasets/car-full-cls")
BASE_MODEL = Path("./models/yolo11n-cls.pt")
EPOCHS = 50
IMGSZ = 224
BATCH = 64
WORKERS = 8
PROJECT_DIR = Path("./runs/classify")
RUN_NAME = "car_full"
PRETRAINED = True
SAVE_PERIOD = 10
VERBOSE = True


def _default_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def _list_classes(train_dir: Path) -> list[str]:
    if not train_dir.is_dir():
        return []
    names = [p.name for p in train_dir.iterdir() if p.is_dir()]
    return sorted(names)


def train_car_full_cls() -> None:
    data_dir = DATA_DIR
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"验证集目录不存在: {val_dir}")

    classes = _list_classes(train_dir)
    if not classes:
        raise RuntimeError(f"训练集目录下未发现类别子目录: {train_dir}")

    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"基础模型不存在: {BASE_MODEL}")

    device = _default_device()
    model = YOLO(str(BASE_MODEL))
    if VERBOSE:
        print(
            f"data_dir={data_dir} base_model={BASE_MODEL} epochs={EPOCHS} imgsz={IMGSZ} "
            f"batch={BATCH} workers={WORKERS} device={device} project={PROJECT_DIR} name={RUN_NAME} "
            f"pretrained={PRETRAINED} save_period={SAVE_PERIOD}"
        )

    model.train(
        data=str(data_dir),
        epochs=int(EPOCHS),
        imgsz=int(IMGSZ),
        batch=int(BATCH),
        workers=int(WORKERS),
        device=device,
        project=str(PROJECT_DIR),
        name=str(RUN_NAME),
        pretrained=bool(PRETRAINED),
        save_period=int(SAVE_PERIOD),
        verbose=bool(VERBOSE),
    )


def main() -> None:
    train_car_full_cls()


if __name__ == "__main__":
    main()

