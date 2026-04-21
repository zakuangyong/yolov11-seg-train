---
name: "car_view"
description: "用本地大模型(Qwen3.5-35B)批量判断车辆视角并按 front/front_side45/side/back_side45/back 归档。用户要用大模型做角度分类整理时调用。"
---

# Car View

## 适用场景

当用户希望用本地多模态大模型对车辆图片进行视角分类，并把图片按类别整理到目录（而不是用 YOLO 分类模型）时调用。

## 入口脚本

- 视角判断与归档脚本：[car_view_judge.py](scripts/car_view_judge.py)

## 分类标签

- 固定 5 类：front、front_side45、side、back_side45、back

## 常用命令

只做目录归档（不生成 CSV）：

```bash
python ./.trae/skills/car_view/scripts/car_view_judge.py --img_dir ./test/img --out_categories_dir ./result/categories --force_json
```

生成 CSV + 目录归档：

```bash
python ./.trae/skills/car_view/scripts/car_view_judge.py --img_dir ./test/img --out_categories_dir ./result/categories --write_csv --out_csv ./result/view_pred.csv --force_json
```

排查 unknown（仅跑少量并打印原始返回）：

```bash
python ./.trae/skills/car_view/scripts/car_view_judge.py --img_dir ./test/img --out_categories_dir ./result/categories --max_images 10 --debug --force_json
```

## 关键参数

- `--llm_base_url`：本地服务地址（支持传到根或 `/v1`）
- `--llm_model`：模型名（例如 Qwen3.5-35B）
- `--labels`：类别列表（默认已设为 5 类）
- `--retries`：解析失败时重试次数（降低 unknown）
- `--force_json`：要求模型输出 JSON（更易解析，推荐开启）
