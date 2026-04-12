import json
import struct
import subprocess
from pathlib import Path
from typing import Dict, Iterator, Tuple

import cv2
import numpy as np

from video_utils import iter_source_frames


_MAGIC = b"QPK1"
_VERSION = 1
_DEFAULT_QP = 28


def _load_qp_map(qp_map_path: str) -> Tuple[Dict[int, np.ndarray], int]:
    with Path(qp_map_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    block_size = int(data.get("block_size", 16))
    frame_to_qp: Dict[int, np.ndarray] = {}

    for item in data.get("frames", []):
        frame_idx = int(item["frame_index"])
        qp = np.asarray(item["qp"], dtype=np.float32)
        if qp.ndim != 2:
            raise ValueError(f"Invalid qp matrix at frame {frame_idx}: expected 2D")
        frame_to_qp[frame_idx] = qp

    return frame_to_qp, block_size


def _pad_to_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    padded_height = height + (height % 2)
    padded_width = width + (width % 2)
    if padded_height == height and padded_width == width:
        return frame
    return cv2.copyMakeBorder(
        frame,
        0,
        padded_height - height,
        0,
        padded_width - width,
        borderType=cv2.BORDER_REPLICATE,
    )


def _iter_source_i420_planes(source_path: str) -> Iterator[Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]]:
    for frame_idx, frame in iter_source_frames(source_path):
        padded = _pad_to_even(frame)
        height, width = padded.shape[:2]

        i420 = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
        flat = i420.reshape(-1)

        y_size = width * height
        uv_size = y_size // 4

        y = flat[:y_size]
        u = flat[y_size : y_size + uv_size]
        v = flat[y_size + uv_size : y_size + uv_size + uv_size]

        yield frame_idx, width, height, y, u, v


def _expected_qp_shape(width: int, height: int, block_size: int) -> Tuple[int, int]:
    qp_h = (height + block_size - 1) // block_size
    qp_w = (width + block_size - 1) // block_size
    return qp_h, qp_w


def _make_default_qp(width: int, height: int, block_size: int) -> np.ndarray:
    qp_h, qp_w = _expected_qp_shape(width, height, block_size)
    return np.full((qp_h, qp_w), _DEFAULT_QP, dtype=np.float32)


def _to_qp_u8(qp: np.ndarray) -> np.ndarray:
    qp_clip = np.clip(qp, 0.0, 51.0)
    return np.rint(qp_clip).astype(np.uint8)


def _pack_frame_packet(
    frame_id: int,
    width: int,
    height: int,
    qp_u8: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> bytes:
    if qp_u8.ndim != 2:
        raise ValueError("qp_u8 must be a 2D matrix")

    qp_h, qp_w = qp_u8.shape

    y_bytes = y.tobytes(order="C")
    u_bytes = u.tobytes(order="C")
    v_bytes = v.tobytes(order="C")
    qp_bytes = qp_u8.reshape(-1).tobytes(order="C")

    header = struct.pack(
        "<10I",
        _VERSION,
        frame_id,
        width,
        height,
        qp_w,
        qp_h,
        len(y_bytes),
        len(u_bytes),
        len(v_bytes),
        len(qp_bytes),
    )
    return _MAGIC + header + y_bytes + u_bytes + v_bytes + qp_bytes


def encode_with_x264(
    source_path: str,
    qp_map_path: str,
    output_bitstream: str,
    x264_bin: str,
) -> None:
    """
    A 路径编码：将视频文件或图像序列逐帧转成 I420，并携带每帧 QP 图打包后写入 videocode.exe stdin。
    x264_bin 在该路径下应传入 videocode.exe 的可执行文件路径。
    """
    frame_to_qp, block_size = _load_qp_map(qp_map_path)

    cmd = [x264_bin, "-", output_bitstream]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    sent_frames = 0
    last_frame_idx = -1

    def _read_stderr_text() -> str:
        try:
            if proc.stderr is None:
                return ""
            data = proc.stderr.read()
            return data.decode("utf-8", errors="replace") if data else ""
        except Exception:
            return ""

    try:
        assert proc.stdin is not None

        for frame_idx, width, height, y, u, v in _iter_source_i420_planes(source_path):
            last_frame_idx = frame_idx

            return_code = proc.poll()
            if return_code is not None:
                err_text = _read_stderr_text()
                raise RuntimeError(
                    "videocode encoder exited early before write "
                    f"(returncode={return_code}, frame_idx={frame_idx}, sent_frames={sent_frames}): {err_text}"
                )

            qp = frame_to_qp.get(frame_idx)
            if qp is None:
                qp = _make_default_qp(width, height, block_size)

            qp_u8 = _to_qp_u8(qp)
            packet = _pack_frame_packet(
                frame_id=frame_idx,
                width=width,
                height=height,
                qp_u8=qp_u8,
                y=y,
                u=u,
                v=v,
            )
            try:
                proc.stdin.write(packet)
                sent_frames += 1
            except BrokenPipeError:
                return_code = proc.poll()
                err_text = _read_stderr_text()
                raise RuntimeError(
                    "broken pipe while writing to videocode encoder "
                    f"(returncode={return_code}, frame_idx={frame_idx}, sent_frames={sent_frames}): {err_text}"
                ) from None

        proc.stdin.close()
        _stdout, stderr = proc.communicate()

    except Exception:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        err_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        raise RuntimeError(
            "videocode encoder failed "
            f"(returncode={proc.returncode}, sent_frames={sent_frames}, last_frame_idx={last_frame_idx}): {err_text}"
        )
