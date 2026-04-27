from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def _default_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def train_complete_seg(
    data_yaml: str | Path = "./datasets/car-complete-seg/data.yaml",
    base_model: str | Path = "./models/yolo-seg/yolo11m-seg.pt",
    epochs: int = 120,
    imgsz: int = 1024,
    batch: int = 8,
    workers: int = 8,
    device: str | int | None = None,
    project: str | Path = "./runs/segment",
    name: str = "complete_car_seg",
    pretrained: bool = True,
    save_period: int = 10,
    patience: int = 50,
    close_mosaic: int = 10,
    lr0: float = 0.01,
    lrf: float = 0.01,
    verbose: bool = True,
) -> None:
    data_yaml = Path(data_yaml)
    base_model = Path(base_model)

    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集 YAML 不存在: {data_yaml}")
    if not base_model.exists():
        raise FileNotFoundError(f"基础模型不存在: {base_model}")

    if device is None:
        device = _default_device()

    model = YOLO(str(base_model))
    if verbose:
        print(
            f"data={data_yaml} model={base_model} epochs={epochs} imgsz={imgsz} batch={batch} "
            f"workers={workers} device={device} project={project} name={name} "
            f"pretrained={pretrained} save_period={save_period} patience={patience}"
        )

    model.train(
        data=str(data_yaml),
        task="segment",
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        device=device,
        project=str(project),
        name=str(name),
        pretrained=bool(pretrained),
        save_period=int(save_period),
        patience=int(patience),
        close_mosaic=int(close_mosaic),
        lr0=float(lr0),
        lrf=float(lrf),
        verbose=bool(verbose),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="整车 YOLO-seg 训练脚本")
    p.add_argument("--data_yaml", default="./datasets/car-complete-seg/data.yaml")
    p.add_argument("--base_model", default="./models/yolo-seg/yolo11m-seg.pt")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default="./runs/segment")
    p.add_argument("--name", default="complete_car_seg")
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--save_period", type=int, default=10)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--close_mosaic", type=int, default=10)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_complete_seg(
        data_yaml=args.data_yaml,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=bool(args.pretrained),
        save_period=args.save_period,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        lr0=args.lr0,
        lrf=args.lrf,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
