# yolov11-seg-train

基于 Ultralytics YOLOv11 的车辆相关任务工程脚本集合，包含：

- 车辆视角分类（训练/推理/归档）
- 车辆整车分类（训练）
- 车辆部件实例分割导出（截图 PNG）与全图可视化标注（框+标签）

## 环境要求

- Python 3.10+
- 依赖：`ultralytics`、`torch`、`opencv-python`、`numpy`
- Windows 中文路径兼容：图像读写统一采用 `np.fromfile + cv2.imdecode/imencode + tofile`

> 说明：不同机器上 Torch/CUDA 安装方式不同，建议按你当前环境选择对应的 PyTorch 安装命令。

## 目录约定

- `./models/`：放置模型权重（示例：`best.pt`、`yolo11m-seg.pt`、`yolo11m-cls.pt` 等）
- `./datasets/`：训练数据集
  - `datasets/car-view-cls/`：视角分类数据集（`train/<class>/...`、`val/<class>/...`）
  - `datasets/car-full-cls/`：整车分类数据集（`train/<class>/...`、`val/<class>/...`）
  - `datasets/carparts-seg/`：分割数据集（含 `carparts-seg.yaml`）
- `./test/img/`：推理测试图片（可按 `front/back/right_side/...` 分目录）
- `./runs/`：Ultralytics 训练/验证输出

## 脚本一览

- `car_segmentation.py`：车辆部件分割推理
  - 默认同时输出：
    - 部件截图（透明背景 PNG，按图片分别建目录）
    - 全图标注（在原图上画框+标签）
- `car_view_train.py`：车辆视角分类训练（Ultralytics classify）
- `car_full_train.py`：车辆整车分类训练（Ultralytics classify）
- `test.py`：简化版推理脚本（按视角导出指定部件截图）
- `skill_agent/`：Skill/Agent 框架（读取 `.env`，可列出/运行 skill 脚本或直接 chat）

## 部件分割与全图标注（car\_segmentation.py）

### 基本用法

```bash
# 默认：递归处理 ./test/img 下所有图片
# 1) 部件截图输出到 ./result/segment
# 2) 全图标注输出到 ./test/complete_label
python car_segmentation.py --img_dir ./test/img
```

### 只输出全图标注（不截图）

```bash
python car_segmentation.py --img_dir ./test/img --no_crops
```

### 只输出截图（不保存全图标注）

```bash
python car_segmentation.py --img_dir ./test/img --no_labels
```

### 指定输出目录与权重

```bash
python car_segmentation.py ^
  --img_dir ./test/img ^
  --weights ./models/best.pt ^
  --conf 0.25 ^
  --out_root ./result/segment ^
  --label_dir ./test/complete_label
```

### 输出结构

- 全图标注：
  - `./test/complete_label/<相对img_dir的子目录>/原文件名.原后缀`
  - 例：`./test/complete_label/front/xxx.png`
- 部件截图（透明 PNG）：
  - `./result/segment/<相对img_dir的子目录>/<stem>/*.png`
  - 例：`./result/segment/front/xxx/front_bumper_xxx.png`

## 训练脚本

### 视角分类训练（car\_view\_train.py）

确保数据集目录存在：

- `datasets/car-view-cls/train/<class>/*.jpg|png...`
- `datasets/car-view-cls/val/<class>/*.jpg|png...`

运行：

```bash
python car_view_train.py
```

### 整车分类训练（car\_full\_train.py）

确保数据集目录存在：

- `datasets/car-full-cls/train/<class>/*.jpg|png...`
- `datasets/car-full-cls/val/<class>/*.jpg|png...`

运行：

```bash
python car_full_train.py
```

## Skill/Agent（skill\_agent）

项目根目录支持 `.env`（默认 `override=False`），常见字段：

- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_API_KEY`
- 以及各类 `CAR_VIEW_*` 配置项（如有）

常用命令：

```bash
# 列出所有 skills
python -m skill_agent.cli list

# 查看某个 skill 下有哪些脚本
python -m skill_agent.cli scripts car_view

# 运行某个 skill 脚本（--args 原样透传给脚本）
python -m skill_agent.cli run car_view car_view_judge.py --args "--img_dir ./test/img"

# 直接对话（可用参数临时覆盖 .env）
python -m skill_agent.cli chat "把 ./test/img 里的图按视角归档" --llm_base_url http://127.0.0.1:8000/v1
```

## 常见问题

- 运行时报 `ModuleNotFoundError: No module named 'cv2'`
  - 安装 OpenCV：`pip install opencv-python`
- 输出目录不存在
  - 脚本会自动创建（`mkdir(parents=True, exist_ok=True)`），确认进程对目录有写权限

