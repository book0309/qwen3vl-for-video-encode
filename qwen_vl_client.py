from pathlib import Path
from typing import List, Dict, Tuple

import json
import re

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ===== 1. 本地加载 qwen3-vl-2b 模型 =====
# 本地缓存根目录（Hugging Face cache）
_LOCAL_MODEL_CACHE_ROOT = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub\models--Qwen--Qwen3-VL-2B-Instruct"
)


def _resolve_local_model_path(cache_root: Path) -> str:
    if not cache_root.exists():
        raise FileNotFoundError(f"Local model cache not found: {cache_root}")

    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.is_dir():
        snapshot_dirs = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if snapshot_dirs:
            return str(snapshot_dirs[0])

    return str(cache_root)


model_name = _resolve_local_model_path(_LOCAL_MODEL_CACHE_ROOT)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

_processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
)
_model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=_dtype,
    trust_remote_code=True,
)
_model.to(_device)
_model.eval()


def _encode_image_to_pil_or_bytes(image_bgr: np.ndarray) -> Image.Image:
    """
    把 OpenCV BGR 图像转成 PIL.Image (RGB)，用于送入 qwen3-vl-2b。
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return pil_img


def _generate_with_images(
    images: List[Image.Image],
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """
    通用的多图 + 文本推理封装，返回生成的文本。
    """
    # Qwen3-VL-2B 的典型用法：在聊天模板里放入图像占位符和文本
    # 这里用统一的 "user" role，把多张图像按顺序加入。
    messages = [
        {
            "role": "user",
            "content": [
                # 多张图像
                *[
                    {"type": "image", "image": img}
                    for img in images
                ],
                # 文本 prompt
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    # 只保留模型新增内容
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generated_text = _processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return generated_text[0].strip() if generated_text else ""


def get_video_global_description(keyframes: List[np.ndarray]) -> str:
    """
    使用 qwen3-vl-2b 对抽取的关键帧进行多图理解，生成全局视频描述。
    返回一个自然语言字符串。
    """
    if not keyframes:
        return "视频为空或未能抽取关键帧。"

    images = [_encode_image_to_pil_or_bytes(img) for img in keyframes]

    prompt = (
        "这些图像按时间顺序来自同一段视频。"
        "请整体分析视频内容，描述主要场景和关键物体，注意不要有修饰"        
        "用一两句中文进行概括。"
        "注意：只需要描述视频内容,不要输出类似“好的，我看到了视频帧”的话。"
        "一个优秀的输出示例：这是道路场景，关键物体是车辆、行人和交通标志。"
    )

    desc = _generate_with_images(images, prompt, max_new_tokens=256)
    return desc


def get_frame_macroblock_qp(
    image: np.ndarray,
    global_description: str,
    frame_index: int,
    fps: float,
    blocks_meta: List[Dict],
    grid_shape: Tuple[int, int],
    qp_contrast: float = 1.0,
) -> List[List[float]]:
    """
    使用 qwen3-vl-2b 生成当前帧的宏块级 QP 分配。
    返回形状为 [num_blocks_h][num_blocks_w] 的 QP 矩阵（Python 嵌套列表）。
    """
    num_blocks_h, num_blocks_w = grid_shape
    qp_contrast = float(np.clip(qp_contrast, 0.5, 3.0))

    def _extract_json(s: str) -> Dict:
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s.replace("json\n", "", 1).strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and start < end:
            snippet = s[start : end + 1]
            return json.loads(snippet)
        raise ValueError(f"模型输出中未能解析出有效 JSON: {s[:200]}...")

    def _sanitize_qp(qp_src: List[List[float]], h: int, w: int) -> np.ndarray:
        qp_matrix = np.full((h, w), 28.0, dtype=np.float32)
        for r in range(h):
            row_src = qp_src[r] if r < len(qp_src) else []
            for c in range(w):
                try:
                    v = float(row_src[c])
                except Exception:
                    v = 28.0
                qp_matrix[r, c] = max(0.0, min(51.0, v))
        return qp_matrix

    def _recover_qp_from_numbers(s: str, h: int, w: int) -> np.ndarray:
        # 当 JSON 结构损坏或被截断时，尽量从文本里抽取数字恢复粗网格。
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
        if not nums:
            raise ValueError("no numeric tokens found in model output")

        needed = h * w
        if len(nums) < needed:
            nums.extend([28.0] * (needed - len(nums)))
        arr = np.asarray(nums[:needed], dtype=np.float32).reshape(h, w)
        return np.clip(arr, 0.0, 51.0)

    def _texture_fallback_qp(frame_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        texture = cv2.resize(mag, (w, h), interpolation=cv2.INTER_AREA)

        t_min = float(texture.min())
        t_max = float(texture.max())
        if t_max - t_min < 1e-6:
            return np.full((h, w), 28.0, dtype=np.float32)

        texture_n = (texture - t_min) / (t_max - t_min)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        dist = np.sqrt(((xx - cx) / max(cx, 1.0)) ** 2 + ((yy - cy) / max(cy, 1.0)) ** 2)
        dist = np.clip(dist / np.sqrt(2.0), 0.0, 1.0)

        qp = 34.0 - texture_n * 14.0 + dist * 3.0
        return np.clip(qp, 18.0, 42.0).astype(np.float32)

    def _enhance_qp_contrast(qp_map: np.ndarray, texture_hint: np.ndarray) -> np.ndarray:
        qp = np.asarray(qp_map, dtype=np.float32)
        before_std = float(np.std(qp))
        before_range = float(np.max(qp) - np.min(qp))
        center_qp = float(np.median(qp))

        gain = 1.0
        target_std = 2.5 + 1.3 * qp_contrast
        target_range = 10.0 + 6.0 * qp_contrast
        if before_std < target_std:
            gain = max(gain, min(3.8, target_std / max(before_std, 0.2)))
        if before_range < target_range:
            gain = max(gain, min(3.2, target_range / max(before_range, 1.0)))

        if gain > 1.02:
            qp = (qp - center_qp) * gain + center_qp

        # 叠加纹理先验增强空间差异：细节区(低 texture_hint)进一步降 QP，背景区升 QP。
        tex = np.asarray(texture_hint, dtype=np.float32)
        tex_delta = tex - float(np.mean(tex))
        qp = qp + (0.25 + 0.30 * qp_contrast) * tex_delta

        # 若增强后仍偏平，再补一次拉伸。
        mid_std = float(np.std(qp))
        rescue_std = 2.0 + 1.0 * qp_contrast
        if mid_std < rescue_std:
            extra = min(1.8 + 0.2 * qp_contrast, rescue_std / max(mid_std, 0.3))
            qp = (qp - center_qp) * extra + center_qp

        low_clip = 12.0 - 2.0 * min(qp_contrast, 1.0)
        high_clip = 42.0 + 2.0 * min(qp_contrast, 1.0)
        qp = np.clip(qp, low_clip, high_clip)
        after_std = float(np.std(qp))
        after_range = float(np.max(qp) - np.min(qp))
        print(
            f"[INFO] QP contrast enhanced at frame={frame_index}: "
            f"contrast={qp_contrast:.2f}, gain={gain:.2f}, std {before_std:.2f}->{after_std:.2f}, "
            f"range {before_range:.2f}->{after_range:.2f}"
        )
        return qp.astype(np.float32)

    pil_img = _encode_image_to_pil_or_bytes(image)

    # 直接让 2B 模型输出完整宏块矩阵太长，容易退化成常量或解析失败。
    # 改为先预测粗网格，再上采样到完整宏块网格。
    coarse_h = min(num_blocks_h, 8)
    coarse_w = min(num_blocks_w, 16)

    user_prompt = f"""
你是一个视频编码专家。下面这张图像是视频中的一帧。

视频整体语义描述如下：
{global_description}

当前帧信息：
- 帧索引: {frame_index}
- 帧率: {fps}
- 目标宏块网格: {num_blocks_h} 行 x {num_blocks_w} 列
- 请先输出粗网格 QP: {coarse_h} 行 x {coarse_w} 列

任务：
- 为粗网格每个单元分配一个量化参数 QP (0~51)。
- 关键物体区域应明显低于背景区域，不要整图给同一个值。
- 只允许输出 {coarse_h}x{coarse_w} 的 qp 矩阵，不要输出目标大网格。

输出格式（非常重要）：
- 只输出一个 JSON 对象，不要输出任何多余文字。
- JSON 结构必须为：
{{
  "qp": [
    [q00, q01, ..., q0{coarse_w-1}],
    ...,
    [q{coarse_h-1}0, ..., q{coarse_h-1}{coarse_w-1}]
  ]
}}
"""

    coarse_cells = coarse_h * coarse_w
    max_new_tokens = min(4096, max(512, coarse_cells * 6 + 128))

    qp_full: np.ndarray
    try:
        generated = _generate_with_images(
            [pil_img],
            user_prompt,
            max_new_tokens=max_new_tokens,
        )
        try:
            parsed = _extract_json(generated)
            qp_coarse = _sanitize_qp(parsed["qp"], coarse_h, coarse_w)
        except Exception:
            qp_coarse = _recover_qp_from_numbers(generated, coarse_h, coarse_w)
            print(f"[WARN] Recovered coarse QP from numeric tokens at frame={frame_index}.")
        qp_full = cv2.resize(qp_coarse, (num_blocks_w, num_blocks_h), interpolation=cv2.INTER_CUBIC)
    except Exception as ex:
        print(f"[WARN] QP JSON parse failed at frame={frame_index}, fallback to texture map: {ex}")
        qp_full = _texture_fallback_qp(image, num_blocks_h, num_blocks_w)

    texture_hint = _texture_fallback_qp(image, num_blocks_h, num_blocks_w)

    # 如果模型输出几乎常量图（常见为全 28），融合纹理先验打破退化。
    if float(np.std(qp_full)) < 0.35:
        qp_full = 0.4 * qp_full.astype(np.float32) + 0.6 * texture_hint
        qp_full = np.clip(qp_full, 0.0, 51.0)
        print(f"[WARN] Flat QP map detected at frame={frame_index}, applied texture-guided rescue.")

    qp_full = _enhance_qp_contrast(qp_full, texture_hint)

    return qp_full.astype(np.float32).tolist()
