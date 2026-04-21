from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


DATA_DIR = Path("./datasets/car-view-cls")
BASE_MODEL = Path("./models/yolo11m-cls.pt")
EPOCHS = 50
IMGSZ = 224
BATCH = 64
WORKERS = 8
PROJECT_DIR = Path("./runs/classify")
RUN_NAME = "car_view"
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


def train_car_view_cls(
    data_dir: str | Path = "./datasets/car-view-cls",
    base_model: str | Path = "./model/yolo11m-cls.pt",
    epochs: int = 50,
    imgsz: int = 224,
    batch: int = 64,
    workers: int = 8,
    device: str | int | None = None,
    project: str | Path = "./runs/classify",
    name: str = "car_view",
    pretrained: bool = True,
    save_period: int = 10,
    verbose: bool = True,
) -> None:
    data_dir = Path(data_dir)
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

    base_model = Path(base_model)
    if not base_model.exists():
        raise FileNotFoundError(
            f"基础模型不存在: {base_model}\n"
            "请确认 ./model/yolo11m-cls.pt 已放置到位，或用 --base_model 指定正确路径。"
        )

    if device is None:
        device = _default_device()

    model = YOLO(str(base_model))
    if verbose:
        print(
            f"data_dir={data_dir} base_model={base_model} epochs={epochs} imgsz={imgsz} "
            f"batch={batch} workers={workers} device={device} project={project} name={name} "
            f"pretrained={pretrained} save_period={save_period}"
        )
    model.train(
        data=str(data_dir),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        device=device,
        project=str(project),
        name=str(name),
        pretrained=bool(pretrained),
        save_period=int(save_period),
        verbose=bool(verbose),
    )


def main() -> None:
    train_car_view_cls(
        data_dir=DATA_DIR,
        base_model=BASE_MODEL,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=_default_device(),
        project=PROJECT_DIR,
        name=RUN_NAME,
        pretrained=PRETRAINED,
        save_period=SAVE_PERIOD,
        verbose=VERBOSE,
    )


if __name__ == "__main__":
    main()
