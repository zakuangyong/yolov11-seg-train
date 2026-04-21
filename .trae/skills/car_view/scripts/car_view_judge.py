from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill_agent.dotenv import apply_dotenv


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_images(img_dir: Path) -> Iterable[Path]:
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _safe_copy_to_dir(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    stem = src.stem
    suf = src.suffix
    for i in range(2, 10000):
        cand = dst_dir / f"{stem}_{i}{suf}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
    raise RuntimeError(f"目标目录重名文件过多，无法写入: {dst_dir}")


def _parse_labels(labels: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(labels, str):
        parts = [p.strip() for p in labels.split(",")]
        return [p for p in parts if p]
    return [str(x).strip() for x in labels if str(x).strip()]


def _read_skill_md(max_chars: int = 2000) -> str:
    skill_root = Path(__file__).resolve().parents[1].parent
    md = skill_root / "SKILL.md"
    if not md.is_file():
        return ""
    try:
        txt = md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if txt.startswith("---"):
        end = txt.find("\n---", 3)
        if end != -1:
            end2 = txt.find("\n", end + 4)
            if end2 != -1:
                txt = txt[end2 + 1 :]
    txt = txt.strip()
    if max_chars > 0 and len(txt) > int(max_chars):
        txt = txt[: int(max_chars)].rstrip()
    return txt


def _llm_prompt(labels: list[str], skill_md: str) -> str:
    label_str = ", ".join(labels)
    skill_md = (skill_md or "").strip()
    if skill_md:
        skill_block = "\n\n以下是技能说明（skill.md），推理时需参考其中对任务/参数/输出的约束，但最终仍必须严格按输出要求只输出一个类别：\n" + skill_md
    else:
        skill_block = ""
    return (
        "你是车辆图像视角分类器。请根据输入图片选择且仅选择一个类别。\n"
        f"可选类别：{label_str}\n"
        "输出要求：只输出类别字符串本身（严格等于可选类别之一），不要输出任何解释、标点、引号、JSON 或额外文本。\n"
        "如果难以判断，也必须从可选类别里选择最接近的一个。\n"
        "提示：front/back 表示正前/正后；side 表示正侧面；front_side45/back_side45 表示前/后 45 度。\n"
        f"{skill_block}\n"
    )


def _cleanup_llm_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_label(text: str, labels: list[str]) -> str | None:
    t0 = _cleanup_llm_text(text)
    if not t0:
        return None

    if t0.startswith("{") and t0.endswith("}"):
        try:
            obj = json.loads(t0)
            if isinstance(obj, dict):
                for k in (
                    "label",
                    "pred_view",
                    "class",
                    "category",
                    "category_selection",
                    "choice",
                    "selected",
                    "result",
                ):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        t0 = v.strip()
                        break
        except Exception:
            pass

    matches = re.findall(r"\b(front_side45|back_side45|front|back|side)\b", t0, flags=re.IGNORECASE)
    if matches:
        cand = matches[-1]
        for lab in labels:
            if cand.lower() == lab.lower():
                return lab

    t = t0.strip().strip('"').strip("'").strip()
    if t in labels:
        return t
    tl = t.lower()
    for lab in labels:
        if tl == lab.lower():
            return lab
    alias = {
        "front side45": "front_side45",
        "front-side45": "front_side45",
        "front45": "front_side45",
        "front 45": "front_side45",
        "back side45": "back_side45",
        "back-side45": "back_side45",
        "back45": "back_side45",
        "back 45": "back_side45",
        "front_view": "front",
        "back_view": "back",
        "side_view": "side",
    }
    if tl in alias:
        mapped = alias[tl]
        for lab in labels:
            if mapped.lower() == lab.lower():
                return lab
    for lab in labels:
        if lab in t:
            return lab
    return None


def _imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _bgr_to_jpeg_bytes(img_bgr: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return buf.tobytes()


def _llm_classify_image_openai_compat(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    img_jpeg: bytes,
    labels: list[str],
    skill_md: str,
    timeout_s: float,
    force_json: bool,
) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"

    data_url = "data:image/jpeg;base64," + base64.b64encode(img_jpeg).decode("ascii")
    payload: dict = {
        "model": model,
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": 64 if force_json else 32,
        "messages": [
            {"role": "system", "content": _llm_prompt(labels, skill_md)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '只输出 JSON：{"label":"<类别>"}（label 必须等于可选类别之一）。'
                            if force_json
                            else "只输出一个类别字符串。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        if force_json and ("response_format" in detail or "json_object" in detail or "unsupported" in detail.lower()):
            return _llm_classify_image_openai_compat(
                base_url=base_url,
                model=model,
                api_key=api_key,
                img_jpeg=img_jpeg,
                labels=labels,
                timeout_s=timeout_s,
                force_json=False,
            )
        raise RuntimeError(f"LLM 请求失败 HTTP {e.code}: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"LLM 请求失败: {e}") from e

    try:
        obj = json.loads(raw.decode("utf-8", errors="ignore"))
        content = obj["choices"][0]["message"]["content"]
        return str(content)
    except Exception as e:
        raise RuntimeError(f"LLM 返回解析失败: {raw[:300]!r}") from e


def judge_views(
    img_dir: str | Path = "./test/img",
    out_categories_dir: str | Path = "./result/categories",
    out_csv: str | Path = "./result/view_pred.csv",
    write_csv: bool = False,
    llm_base_url: str = "http://172.25.150.2:53377/v1",
    llm_model: str = "Qwen3.5-35B",
    llm_api_key: str | None = None,
    labels: str | list[str] = "front,front_side45,side,back_side45,back",
    timeout_s: float = 60.0,
    force_json: bool = True,
    retries: int = 2,
    max_images: int = 0,
    debug: bool = False,
) -> int:
    img_dir = Path(img_dir)
    out_categories_dir = Path(out_categories_dir)
    out_categories_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_csv)
    if write_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    label_list = _parse_labels(labels)
    if not label_list:
        raise ValueError("labels 不能为空")

    skill_md = _read_skill_md(max_chars=2000)

    rows: list[dict[str, str]] = []
    ok = 0
    seen = 0
    for img_path in _iter_images(img_dir):
        if max_images and seen >= int(max_images):
            break
        seen += 1

        img = _imread_unicode(img_path)
        if img is None:
            continue

        img_jpeg = _bgr_to_jpeg_bytes(img)
        if img_jpeg is None:
            continue

        label: str | None = None
        last_text = ""
        for _ in range(max(1, int(retries))):
            text = _llm_classify_image_openai_compat(
                base_url=llm_base_url,
                model=llm_model,
                api_key=llm_api_key,
                img_jpeg=img_jpeg,
                labels=label_list,
                skill_md=skill_md,
                timeout_s=timeout_s,
                force_json=force_json,
            )
            last_text = text
            label = _extract_label(text, label_list)
            if label is None and force_json:
                text_plain = _llm_classify_image_openai_compat(
                    base_url=llm_base_url,
                    model=llm_model,
                    api_key=llm_api_key,
                    img_jpeg=img_jpeg,
                    labels=label_list,
                    skill_md=skill_md,
                    timeout_s=timeout_s,
                    force_json=False,
                )
                last_text = text_plain
                label = _extract_label(text_plain, label_list)
            if label is not None:
                break

        if label is None:
            label = "unknown"
            if debug:
                print(f"[unknown] {img_path.name}: {last_text!r}")
        elif debug:
            print(f"[llm] {img_path.name}: {last_text!r} -> {label}")

        ok += 1
        _safe_copy_to_dir(img_path, out_categories_dir / str(label))
        if write_csv:
            rows.append({"path": str(img_path.as_posix()), "pred_view": str(label), "pred_conf": ""})

    if write_csv:
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "pred_view", "pred_conf"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    return ok


def main() -> None:
    apply_dotenv(PROJECT_ROOT / ".env", override=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None)
    parser.add_argument("--out_categories_dir", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--write_csv", action="store_true")
    parser.add_argument("--llm_base_url", default=None)
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--llm_api_key", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--timeout_s", type=float, default=0.0)
    parser.add_argument("--force_json", action="store_true")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    img_dir = args.img_dir or os.getenv("CAR_VIEW_IMG_DIR", "./test/img")
    out_categories_dir = args.out_categories_dir or os.getenv("CAR_VIEW_OUT_CATEGORIES_DIR", "./result/categories")
    out_csv = args.out_csv or os.getenv("CAR_VIEW_OUT_CSV", "./result/view_pred.csv")
    if bool(args.write_csv):
        write_csv = True
    else:
        write_csv = os.getenv("CAR_VIEW_WRITE_CSV", "").strip() in {"1", "true", "True", "YES", "yes"}
    if int(args.max_images) > 0:
        max_images = int(args.max_images)
    else:
        max_images = int(os.getenv("CAR_VIEW_MAX_IMAGES", "0"))
    if bool(args.debug):
        debug = True
    else:
        debug = os.getenv("CAR_VIEW_DEBUG", "").strip() in {"1", "true", "True", "YES", "yes"}

    llm_base_url = args.llm_base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    llm_model = args.llm_model or os.getenv("LLM_MODEL", "Qwen3.5-35B")
    llm_api_key = args.llm_api_key if args.llm_api_key else (os.getenv("LLM_API_KEY") or None)
    labels = args.labels or os.getenv("CAR_VIEW_LABELS", "front,front_side45,side,back_side45,back")
    timeout_s = float(args.timeout_s) if float(args.timeout_s) > 0 else float(os.getenv("CAR_VIEW_TIMEOUT_S", "60"))
    retries = int(args.retries) if int(args.retries) > 0 else int(os.getenv("CAR_VIEW_RETRIES", "2"))
    if not bool(args.force_json):
        force_json = os.getenv("CAR_VIEW_FORCE_JSON", "").strip() in {"1", "true", "True", "YES", "yes"}
    else:
        force_json = True

    n = judge_views(
        img_dir=str(img_dir),
        out_categories_dir=str(out_categories_dir),
        out_csv=str(out_csv),
        write_csv=bool(write_csv),
        llm_base_url=str(llm_base_url),
        llm_model=str(llm_model),
        llm_api_key=llm_api_key,
        labels=str(labels),
        timeout_s=float(timeout_s),
        force_json=bool(force_json),
        retries=int(retries),
        max_images=int(max_images),
        debug=bool(debug),
    )
    print(f"done: {n}")


if __name__ == "__main__":
    main()
