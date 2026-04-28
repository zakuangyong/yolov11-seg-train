from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RenamePair:
    image_src: Path
    label_src: Path
    image_dst: Path
    label_dst: Path
    image_tmp: Path
    label_tmp: Path


def _iter_images(images_root: Path) -> Iterable[Path]:
    for fp in sorted(images_root.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
            yield fp


def _pick_map_path(dataset_root: Path, out_path: str | None) -> Path:
    if out_path:
        p = Path(out_path)
        if not p.is_absolute():
            p = dataset_root / p
        return p

    base = dataset_root / "rename_map.json"
    if not base.exists():
        return base
    for i in range(1, 10000):
        cand = dataset_root / f"rename_map_{i}.json"
        if not cand.exists():
            return cand
    return dataset_root / f"rename_map_{uuid.uuid4().hex}.json"


def _ensure_same_filesystem(a: Path, b: Path) -> None:
    try:
        if os.stat(a).st_dev != os.stat(b).st_dev:
            raise RuntimeError(f"images 与 labels 不在同一分区/盘符，无法安全改名: {a} vs {b}")
    except FileNotFoundError:
        return


def _make_pairs(
    dataset_root: Path,
    prefix: str,
    start_index: int,
    width: int | None,
    label_ext: str,
    strict: bool,
) -> list[RenamePair]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.exists():
        raise FileNotFoundError(f"images 目录不存在: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"labels 目录不存在: {labels_root}")

    _ensure_same_filesystem(images_root, labels_root)

    imgs = list(_iter_images(images_root))
    if not imgs:
        raise RuntimeError(f"未找到任何图片文件: {images_root}")

    if width is None:
        width = max(4, len(str(start_index + len(imgs) - 1)))

    pairs: list[RenamePair] = []
    for i, img_path in enumerate(imgs, start=start_index):
        rel = img_path.relative_to(images_root)
        label_rel = rel.with_suffix(label_ext)
        label_path = labels_root / label_rel
        if strict and not label_path.exists():
            raise FileNotFoundError(f"缺少对应标签: {label_path} (image: {img_path})")
        if not label_path.exists():
            continue

        stem = f"{prefix}_{i:0{width}d}"
        img_dst = img_path.with_name(stem + img_path.suffix)
        label_dst = label_path.with_name(stem + label_path.suffix)

        token = uuid.uuid4().hex
        img_tmp = img_path.with_name(f".__tmp__{token}{img_path.suffix}")
        label_tmp = label_path.with_name(f".__tmp__{token}{label_path.suffix}")

        pairs.append(
            RenamePair(
                image_src=img_path,
                label_src=label_path,
                image_dst=img_dst,
                label_dst=label_dst,
                image_tmp=img_tmp,
                label_tmp=label_tmp,
            )
        )

    if not pairs:
        raise RuntimeError("没有找到任何可配对的 image+label 文件")

    return pairs


def _validate_pairs(pairs: list[RenamePair]) -> None:
    dsts: set[Path] = set()
    for p in pairs:
        if not p.image_src.exists():
            raise FileNotFoundError(f"图片不存在: {p.image_src}")
        if not p.label_src.exists():
            raise FileNotFoundError(f"标签不存在: {p.label_src}")
        for dst in (p.image_dst, p.label_dst):
            if dst in dsts:
                raise RuntimeError(f"目标文件名冲突(同批次重复): {dst}")
            dsts.add(dst)
        for dst in (p.image_dst, p.label_dst):
            if dst.exists() and dst not in {p.image_src, p.label_src}:
                raise FileExistsError(f"目标已存在: {dst}")
        for tmp in (p.image_tmp, p.label_tmp):
            if tmp.exists():
                raise FileExistsError(f"临时文件已存在(可能上次中断遗留): {tmp}")


def _write_map(map_path: Path, pairs: list[RenamePair]) -> None:
    map_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for p in pairs:
        payload.append(
            {
                "image_src": str(p.image_src),
                "image_dst": str(p.image_dst),
                "label_src": str(p.label_src),
                "label_dst": str(p.label_dst),
            }
        )
    map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rename_atomic(pairs: list[RenamePair]) -> None:
    for p in pairs:
        p.image_src.replace(p.image_tmp)
        p.label_src.replace(p.label_tmp)

    for p in pairs:
        p.image_tmp.replace(p.image_dst)
        p.label_tmp.replace(p.label_dst)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="批量重命名 datasets/car-complete-seg 下的 images 与 labels，确保一一对应。默认 dry-run。"
    )
    ap.add_argument("--dataset", type=str, default="./datasets/car-complete-seg", help="数据集根目录")
    ap.add_argument("--prefix", type=str, default="car", help="新文件名前缀")
    ap.add_argument("--start", type=int, default=1, help="起始序号(默认 1)")
    ap.add_argument("--width", type=int, default=0, help="数字宽度(0 表示自动推断，最少 4)")
    ap.add_argument("--label-ext", type=str, default=".txt", help="标签扩展名(默认 .txt)")
    ap.add_argument("--map-out", type=str, default=None, help="输出映射文件路径(相对 dataset 或绝对路径)")
    ap.add_argument("--apply", action="store_true", help="执行改名(不加则只打印预览)")
    ap.add_argument("--strict", action="store_true", help="严格模式：缺失标签直接报错")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个(用于测试，0 表示全部)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    label_ext = args.label_ext
    if not label_ext.startswith("."):
        label_ext = "." + label_ext

    width = None if int(args.width) <= 0 else int(args.width)

    pairs = _make_pairs(
        dataset_root=dataset_root,
        prefix=str(args.prefix),
        start_index=int(args.start),
        width=width,
        label_ext=label_ext,
        strict=bool(args.strict),
    )

    if int(args.limit) > 0:
        pairs = pairs[: int(args.limit)]

    _validate_pairs(pairs)
    map_path = _pick_map_path(dataset_root, args.map_out)
    _write_map(map_path, pairs)

    for p in pairs[: min(20, len(pairs))]:
        print(f"IMG  {p.image_src.name}  ->  {p.image_dst.name}")
        print(f"LAB  {p.label_src.name}  ->  {p.label_dst.name}")

    if not args.apply:
        print(f"\n[dry-run] 共 {len(pairs)} 对，将输出映射: {map_path}")
        print("加 --apply 执行改名。")
        return 0

    _rename_atomic(pairs)
    print(f"\n[done] 已改名 {len(pairs)} 对，映射已写入: {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

