from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO-seg wheel label into front_wheel/rear_wheel")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets/carparts-seg-yolo-supplement",
        help="Dataset root that contains labels/train|val|test and yaml",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="datasets/carparts-seg-yolo-supplement/carparts-seg-yolo-supplement.yaml",
        help="Dataset yaml path with names map",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Label splits to process")
    parser.add_argument("--src_wheel_id", type=int, default=22, help="Old wheel class id to split")
    parser.add_argument("--front_name", type=str, default="front_wheel", help="Front wheel class name in yaml")
    parser.add_argument("--rear_name", type=str, default="rear_wheel", help="Rear wheel class name in yaml")
    parser.add_argument("--backup", action="store_true", help="Create .bak backup before overwrite")
    return parser.parse_args()


def _iter_label_files(labels_root: Path) -> Iterable[Path]:
    if not labels_root.exists():
        return []
    return sorted([p for p in labels_root.rglob("*.txt") if p.is_file()])


def _load_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _name_to_id(names_map: dict[int, str]) -> dict[str, int]:
    return {v: k for k, v in names_map.items()}


def _line_center_x(parts: list[str]) -> float:
    coords = [float(x) for x in parts[1:]]
    xs = coords[0::2]
    return sum(xs) / len(xs) if xs else 0.5


def _mean_of(ids: list[int], entries: list[tuple[int, list[str], float]]) -> float | None:
    vals = [cx for cls_id, _, cx in entries if cls_id in ids]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _infer_faces_right(entries: list[tuple[int, list[str], float]], name2id: dict[str, int]) -> bool:
    def ids(*names: str) -> list[int]:
        return [name2id[n] for n in names if n in name2id]

    pairs = [
        (ids("front_bumper"), ids("back_bumper")),
        (ids("hood"), ids("trunk", "tailgate")),
        (ids("front_glass"), ids("back_glass")),
        (ids("front_light"), ids("back_light")),
    ]
    for fa, ba in pairs:
        if not fa or not ba:
            continue
        x_front = _mean_of(fa, entries)
        x_back = _mean_of(ba, entries)
        if x_front is not None and x_back is not None and x_front != x_back:
            return x_front > x_back
    return True


def _split_one_file(
    file_path: Path,
    src_wheel_id: int,
    front_id: int,
    rear_id: int,
    name2id: dict[str, int],
) -> tuple[bool, int]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return False, 0

    entries: list[tuple[int, list[str], float]] = []
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(float(parts[0]))
        cx = _line_center_x(parts)
        entries.append((cls_id, parts, cx))

    wheel_idxs = [i for i, (cls_id, _, _) in enumerate(entries) if cls_id == src_wheel_id]
    if not wheel_idxs:
        return False, 0

    faces_right = _infer_faces_right(entries, name2id)
    changed = 0

    if len(wheel_idxs) == 1:
        i = wheel_idxs[0]
        front_ref = _mean_of(
            [name2id[n] for n in ("front_bumper", "hood", "front_glass") if n in name2id],
            entries,
        )
        back_ref = _mean_of(
            [name2id[n] for n in ("back_bumper", "trunk", "tailgate", "back_glass") if n in name2id],
            entries,
        )
        wheel_x = entries[i][2]
        if front_ref is not None and back_ref is not None:
            front_like = abs(wheel_x - front_ref) <= abs(wheel_x - back_ref)
            entries[i][1][0] = str(front_id if front_like else rear_id)
        else:
            entries[i][1][0] = str(front_id if faces_right else rear_id)
        changed = 1
    else:
        xs = [(i, entries[i][2]) for i in wheel_idxs]
        if faces_right:
            front_idx = max(xs, key=lambda x: x[1])[0]
        else:
            front_idx = min(xs, key=lambda x: x[1])[0]
        for i in wheel_idxs:
            entries[i][1][0] = str(front_id if i == front_idx else rear_id)
            changed += 1

    out_lines = [" ".join(parts) for _, parts, _ in entries]
    file_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return True, changed


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    yaml_path = Path(args.yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml not found: {yaml_path}")

    names_map = _load_names(yaml_path)
    name2id = _name_to_id(names_map)
    if args.front_name not in name2id or args.rear_name not in name2id:
        raise ValueError(f"yaml names must include '{args.front_name}' and '{args.rear_name}'")

    front_id = name2id[args.front_name]
    rear_id = name2id[args.rear_name]

    total_files = 0
    total_changed_files = 0
    total_changed_labels = 0

    for split in args.splits:
        split_root = dataset_root / "labels" / split
        files = list(_iter_label_files(split_root))
        total_files += len(files)
        for fp in files:
            if args.backup:
                bak = fp.with_suffix(fp.suffix + ".bak")
                if not bak.exists():
                    bak.write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
            changed, num = _split_one_file(fp, args.src_wheel_id, front_id, rear_id, name2id)
            if changed:
                total_changed_files += 1
                total_changed_labels += num

    print(f"Processed files: {total_files}")
    print(f"Changed files: {total_changed_files}")
    print(f"Changed wheel labels: {total_changed_labels}")
    print(f"front_wheel id={front_id}, rear_wheel id={rear_id}, src_wheel_id={args.src_wheel_id}")


if __name__ == "__main__":
    main()
