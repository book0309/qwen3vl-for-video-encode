from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import cv2
import numpy as np


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _is_image_sequence_path(source_path: str) -> bool:
    return Path(source_path).is_dir()


def _frame_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), path.name
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits), path.name
    return (10**18, path.name)


def list_frame_paths(sequence_dir: str) -> List[Path]:
    dir_path = Path(sequence_dir)
    if not dir_path.is_dir():
        raise RuntimeError(f"Failed to open image sequence directory: {sequence_dir}")

    frame_paths = [
        path for path in dir_path.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    frame_paths.sort(key=_frame_sort_key)
    return frame_paths


def iter_image_sequence_frames(
    sequence_dir: str,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    逐帧返回 (frame_index, frame[BGR])，适用于按图片文件存储的视频序列。
    """
    for frame_idx, frame_path in enumerate(list_frame_paths(sequence_dir)):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame image: {frame_path}")
        yield frame_idx, frame


def iter_source_frames(
    source_path: str,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    兼容视频文件和图像序列目录的统一帧迭代器。
    """
    if _is_image_sequence_path(source_path):
        yield from iter_image_sequence_frames(source_path)
        return

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")
    try:
        yield from iter_video_frames(cap)
    finally:
        cap.release()


def extract_keyframes(source_path: str, interval: int = 1, fps: float | None = None) -> List[np.ndarray]:
    """
    按时间间隔（秒）从视频文件或图像序列中抽取关键帧，用于全局语义理解。
    返回 BGR 图像列表（np.ndarray）。
    """
    keyframes = []
    if _is_image_sequence_path(source_path):
        source_fps = 10.0 if fps is None else float(fps)
        frame_interval = max(int(source_fps * interval), 1)
        for frame_idx, frame in iter_image_sequence_frames(source_path):
            if frame_idx % frame_interval == 0:
                keyframes.append(frame.copy())
        return keyframes

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) if fps is None else float(fps)
    frame_interval = max(int(source_fps * interval), 1)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                keyframes.append(frame.copy())
            frame_idx += 1
    finally:
        cap.release()

    return keyframes


def iter_video_frames(
    cap: cv2.VideoCapture,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    逐帧返回 (frame_index, frame[BGR])。
    """
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1


def split_frame_into_blocks(
    frame: np.ndarray, block_size: int
) -> Tuple[List[Dict], Tuple[int, int]]:
    """
    将一帧划分为 block_size x block_size 的宏块网格。
    对于 x264，推荐 block_size=16，对应标准的 16x16 宏块。
    返回:
      - blocks: 每个宏块的元信息列表（不返回像素，只返回坐标与大小）
      - grid_shape: (num_blocks_h, num_blocks_w)
    """
    h, w, _ = frame.shape
    num_blocks_h = (h + block_size - 1) // block_size
    num_blocks_w = (w + block_size - 1) // block_size

    blocks = []
    for by in range(num_blocks_h):
        for bx in range(num_blocks_w):
            y0 = by * block_size
            x0 = bx * block_size
            blocks.append(
                {
                    "block_y": by,
                    "block_x": bx,
                    "x": x0,
                    "y": y0,
                    "width": min(block_size, w - x0),
                    "height": min(block_size, h - y0),
                }
            )

    return blocks, (num_blocks_h, num_blocks_w)
