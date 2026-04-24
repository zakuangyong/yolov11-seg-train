"""
SAM-based Auto-Annotation Script for YOLO-seg
使用 SAM (Segment Anything) 自动生成车辆部件分割标注

Usage:
    python sam_auto_annotate.py --input datasets/raw-img --output datasets/carparts-seg-supplement

Dependencies:
    pip install segment-anything ultralytics opencv-python numpy tqdm shapely
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(config: dict, args: argparse.Namespace) -> dict:
    """Merge YAML config with command line arguments (CLI takes priority over defaults)"""
    if hasattr(args, 'input') and args.input != argparse.SUPPRESS:
        config['input_dir'] = args.input
    if hasattr(args, 'output') and args.output != argparse.SUPPRESS:
        config['output_dir'] = args.output

    if 'sam' not in config:
        config['sam'] = {}
    if hasattr(args, 'model') and args.model != 'vit_h':
        config['sam']['model_type'] = args.model
    if hasattr(args, 'sam_weights') and args.sam_weights != 'sam_vit_h.pth' and 'weights_path' not in config.get('sam', {}):
        config['sam']['weights_path'] = args.sam_weights
    if hasattr(args, 'device') and args.device != 'cuda':
        config['sam']['device'] = args.device

    if hasattr(args, 'split') and args.split != [0.8, 0.1, 0.1]:
        config['split_ratios'] = args.split

    if 'annotation' not in config:
        config['annotation'] = {}
    if hasattr(args, 'min_area') and args.min_area != 500:
        config['annotation']['min_area'] = args.min_area
    if hasattr(args, 'visualize') and args.visualize:
        config['annotation']['visualize'] = True

    return config


def parse_args():
    parser = argparse.ArgumentParser(description='SAM Auto Annotation for YOLO-seg')
    parser.add_argument('--config', '-c', type=str, default='tools/sam_auto_annotate.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--input', '-i', type=str, default=argparse.SUPPRESS,
                        help='Input directory containing raw images (overrides config)')
    parser.add_argument('--output', '-o', type=str, default=argparse.SUPPRESS,
                        help='Output directory for labeled dataset (overrides config)')
    parser.add_argument('--model', '-m', type=str, default='vit_h',
                        help='SAM model type')
    parser.add_argument('--sam-weights', '-w', type=str, default='sam_vit_h.pth',
                        help='Path to SAM model weights')
    parser.add_argument('--split', '-s', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Train/val/test split ratios (e.g., 0.8 0.1 0.1)')
    parser.add_argument('--min-area', '-a', type=int, default=500,
                        help='Minimum mask area in pixels to keep')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Save visualization of annotations')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def init_sam(model_type: str, weights_path: str, device: str):
    """Initialize SAM model"""
    from segment_anything import sam_model_registry, SamPredictor

    print(f"Loading SAM model: {model_type} from {weights_path}")
    sam = sam_model_registry[model_type](checkpoint=weights_path)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image using numpy for Chinese path support"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def generate_grid_points(image_shape: Tuple[int, int], grid_size: int = 8) -> np.ndarray:
    """Generate grid of prompt points covering the image"""
    h, w = image_shape[:2]
    y_points = np.linspace(0, h - 1, grid_size, dtype=int)
    x_points = np.linspace(0, w - 1, grid_size, dtype=int)
    points = []
    for y in y_points:
        for x in x_points:
            points.append([x, y])
    return np.array(points)


def predict_masks(predictor, image_rgb: np.ndarray, image_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Generate masks using SAM with grid points"""
    predictor.set_image(image_rgb)

    h, w = image_shape[:2]
    points = generate_grid_points(image_shape, grid_size=8)
    labels = np.ones(len(points), dtype=int)

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False,
        return_logits=False
    )

    return masks


def mask_to_polygons(mask: np.ndarray) -> List[List[float]]:
    """Convert binary mask to polygon coordinates (normalized)"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    h, w = mask.shape[:2]

    for cnt in contours:
        if len(cnt) < 3:
            continue

        cnt = cnt.reshape(-1, 2)
        normalized = cnt.astype(float)
        normalized[:, 0] /= w
        normalized[:, 1] /= h

        coords = normalized.flatten().tolist()
        polygons.append(coords)

    return polygons


def filter_masks_by_area(masks: np.ndarray, scores: np.ndarray, min_area: int) -> Tuple[List[np.ndarray], List[float]]:
    """Filter masks by minimum area threshold"""
    filtered_masks = []
    filtered_scores = []

    for mask, score in zip(masks, scores):
        area = mask.sum()
        if area >= min_area:
            filtered_masks.append(mask)
            filtered_scores.append(score)

    return filtered_masks, filtered_scores


def create_yolo_seg_label(polygons: List[List[float]], class_id: int = 0) -> str:
    """Create YOLO-seg format label line

    YOLO-seg format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    All coordinates normalized to [0, 1]
    """
    if not polygons:
        return ""

    lines = []
    for poly in polygons:
        coords_str = ' '.join([f'{c:.6f}' for c in poly])
        lines.append(f"{class_id} {coords_str}")

    return '\n'.join(lines)


def process_image(predictor, image_path: str, output_dir: str, visualize: bool = False, class_id: int = 0) -> str:
    """Process single image and save label"""
    img, img_rgb = load_image(image_path)
    h, w = img.shape[:2]

    masks = predict_masks(predictor, img_rgb, img.shape)
    masks = np.array(masks) if not isinstance(masks, np.ndarray) else masks

    if len(masks) == 0:
        print(f"Warning: No masks generated for {image_path}")
        return ""

    all_polygons = []
    for mask in masks:
        polys = mask_to_polygons(mask)
        all_polygons.extend(polys)

    label_content = create_yolo_seg_label(all_polygons, class_id)

    image_name = Path(image_path).stem
    label_path = Path(output_dir) / f"{image_name}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, 'w') as f:
        f.write(label_content)

    if visualize:
        vis_img = img.copy()
        for mask in masks:
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            vis_img[mask > 0] = vis_img[mask > 0] * 0.5 + np.array(color) * 0.5
            vis_img = cv2.drawContours(vis_img, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)

        vis_path = label_path.with_suffix('.jpg')
        cv2.imwrite(str(vis_path), vis_img)

    return label_path


def split_dataset(file_list: List[str], split_ratios: List[float]) -> Tuple[List[str], List[str], List[str]]:
    """Split dataset into train/val/test"""
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

    np.random.seed(42)
    indices = np.arange(len(file_list))
    np.random.shuffle(indices)

    n = len(file_list)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_files = [file_list[i] for i in train_indices]
    val_files = [file_list[i] for i in val_indices]
    test_files = [file_list[i] for i in test_indices]

    return train_files, val_files, test_files


def copy_to_split(image_path: str, split_dir: str, label_path: str, split_label_dir: str) -> Tuple[str, str]:
    """Copy image and label to split directory"""
    import shutil

    image_name = Path(image_path).name
    label_name = Path(label_path).name

    split_image_dir = Path(split_dir) / 'images'
    split_label_dir = Path(split_label_dir) / 'labels'

    split_image_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)

    dest_image = split_image_dir / image_name
    dest_label = split_label_dir / label_name

    shutil.copy2(image_path, dest_image)
    if label_path.exists():
        shutil.copy2(label_path, dest_label)

    return str(dest_image), str(dest_label)


def create_dataset_yaml(output_dir: str, class_names: List[str]):
    """Create dataset YAML configuration file"""
    yaml_content = f"""# Auto-generated YOLO-seg dataset configuration
# Created by sam_auto_annotate.py

path: {output_dir}
train: images/train
val: images/val
test: images/test

# Classes
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"Created dataset config: {yaml_path}")
    return yaml_path


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    config_candidates = [Path(args.config)]
    if not Path(args.config).is_absolute():
        config_candidates.append(script_dir / args.config)
        config_candidates.append(script_dir / Path(args.config).name)

    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is not None:
        config = load_config(str(config_path))
        config = merge_config(config, args)
        print(f"Loaded config from: {config_path}")
    else:
        config = {}

    if 'input_dir' not in config:
        config['input_dir'] = getattr(args, 'input', 'datasets/raw-img')
    if 'output_dir' not in config:
        config['output_dir'] = getattr(args, 'output', 'datasets/carparts-seg-supplement')

    if 'sam' not in config:
        config['sam'] = {
            'model_type': getattr(args, 'model', 'vit_h'),
            'weights_path': getattr(args, 'sam_weights', 'sam_vit_h.pth'),
            'device': getattr(args, 'device', 'cuda')
        }

    if 'split_ratios' not in config:
        config['split_ratios'] = getattr(args, 'split', [0.8, 0.1, 0.1])

    if 'annotation' not in config:
        config['annotation'] = {
            'min_area': getattr(args, 'min_area', 500),
            'visualize': getattr(args, 'visualize', False)
        }

    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for fmt in supported_formats:
        image_files.extend(list(input_dir.glob(f'*{fmt}')))
        image_files.extend(list(input_dir.glob(f'*{fmt.upper()}')))

    if not image_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    sam_weights = Path(config['sam']['weights_path'])
    if not sam_weights.is_absolute():
        weight_candidates = [Path.cwd() / sam_weights]
        if config_path is not None:
            weight_candidates.append(config_path.parent / sam_weights)
            if config_path.parent.name == 'tools':
                weight_candidates.append(config_path.parent.parent / sam_weights)
        sam_weights = next((p for p in weight_candidates if p.exists()), weight_candidates[0])
    if not sam_weights.exists():
        print(f"Error: SAM weights not found: {sam_weights}")
        print("Please download SAM weights from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        sys.exit(1)

    predictor = init_sam(
        config['sam']['model_type'],
        str(sam_weights),
        config['sam']['device']
    )

    visualize = config['annotation'].get('visualize', False)

    print("Processing images...")
    for image_path in tqdm(image_files, desc="Generating annotations"):
        try:
            process_image(
                predictor,
                str(image_path),
                str(output_dir / 'labels'),
                visualize=visualize
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    train_files, val_files, test_files = split_dataset(
        [str(f) for f in image_files],
        config['split_ratios']
    )

    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("Organizing dataset structure...")
    for image_path in tqdm(train_files, desc="Train"):
        label_path = Path(output_dir / 'labels') / f"{Path(image_path).stem}.txt"
        copy_to_split(image_path, str(output_dir / 'train'), str(label_path), str(output_dir))

    for image_path in tqdm(val_files, desc="Val"):
        label_path = Path(output_dir / 'labels') / f"{Path(image_path).stem}.txt"
        copy_to_split(image_path, str(output_dir / 'val'), str(label_path), str(output_dir))

    for image_path in tqdm(test_files, desc="Test"):
        label_path = Path(output_dir / 'labels') / f"{Path(image_path).stem}.txt"
        copy_to_split(image_path, str(output_dir / 'test'), str(label_path), str(output_dir))

    class_names = config.get('class_names', [
        'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
        'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
        'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
        'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
        'tailgate', 'trunk', 'wheel'
    ])

    create_dataset_yaml(str(output_dir), class_names)

    print(f"\nDataset created at: {output_dir}")
    print("To train YOLO-seg model:")
    print(f"  yolo train data={output_dir}/dataset.yaml model=yolo11n-seg.pt")


if __name__ == '__main__':
    main()
