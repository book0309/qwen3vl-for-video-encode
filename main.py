import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from video_utils import (
    extract_keyframes,
    iter_source_frames,
    split_frame_into_blocks,
)
from qwen_vl_client import (
    get_video_global_description,
    get_frame_macroblock_qp,
)
from x264_runner import encode_with_x264  # 新增导入


def _log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def process_sequence(
    source_path: str,
    output_path: str,
    block_size: int = 16,
    keyframe_interval: int = 1,
    fps: float = 10.0,
    # 是否在生成 QP JSON 后立刻调用 videocode.exe 编码
    encode_x264: bool = False,
    x264_bin: str = "videocode.exe",
    x264_output: str | None = None,
    max_frames: int | None = None,
    frame_step: int = 1,
    vis_size: int = 224,
    log_interval: int = 10,
    qp_contrast: float = 1.6,
):
    _log(
        "Start sequence | "
        f"source={source_path} fps={fps} block={block_size} "
        f"frame_step={frame_step} max_frames={max_frames} encode={encode_x264} qp_contrast={qp_contrast}"
    )

    # 1. 抽取关键帧并做全局视频理解
    keyframes = extract_keyframes(source_path, interval=keyframe_interval, fps=fps)
    _log(f"Keyframe extraction done: {len(keyframes)} keyframes")

    global_desc = get_video_global_description(keyframes)
    _log("Global description generated")

    frame_results = []
    processed_frames = 0

    # 2. 逐帧处理
    for frame_idx, frame in iter_source_frames(source_path):
        if frame_step > 1 and (frame_idx % frame_step != 0):
            continue

        if max_frames is not None and processed_frames >= max_frames:
            break

        # 2.1 宏块划分（这里只负责坐标和裁剪，不做 QP 决策）
        blocks, grid_shape = split_frame_into_blocks(frame, block_size)

        # 2.2 可以选一个下采样版本作为视觉输入，减少开销
        vis_frame = cv2.resize(frame, (vis_size, vis_size))

        # 2.3 调用 qwen2-vl-2b 让其为每个宏块生成 QP
        qp_matrix = get_frame_macroblock_qp(
            image=vis_frame,
            global_description=global_desc,
            frame_index=frame_idx,
            fps=fps,
            blocks_meta=blocks,
            grid_shape=grid_shape,
            qp_contrast=qp_contrast,
        )
        qp_matrix = np.asarray(qp_matrix, dtype=np.float32)

        frame_results.append(
            {
                "frame_index": frame_idx,
                "qp": qp_matrix.tolist(),
            }
        )

        processed_frames += 1
        if processed_frames == 1 or (processed_frames % log_interval == 0):
            _log(
                "Frame progress | "
                f"processed={processed_frames} last_frame_idx={frame_idx} grid={grid_shape[0]}x{grid_shape[1]}"
            )

    # 3. 保存 QP JSON
    result = {
        "source": str(source_path),
        "encoder": "x264",            # 指明目标编码器标准
        "block_size": block_size,     # 推荐为 16, 对应 x264 宏块 16x16
        "macroblock_size": [block_size, block_size],
        "fps": fps,
        "qp_contrast": qp_contrast,
        "global_description": global_desc,
        "frames": frame_results,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(f"QP JSON saved: {out_path}")

    # 4. 可选: 直接调用 videocode.exe 使用该 QP 图进行编码
    if encode_x264:
        if x264_output is None:
            # 默认输出到同目录, 后缀 .h264
            x264_output = str(Path(output_path).with_suffix(".h264"))
        _log(f"Encoding started: {x264_output}")
        encode_with_x264(
            source_path=str(source_path),
            qp_map_path=str(out_path),
            output_bitstream=x264_output,
            x264_bin=x264_bin,
        )
        _log(f"Encoding finished: {x264_output}")

    _log("Sequence completed")


def process_dataset(
    input_root: str,
    output_dir: str,
    block_size: int = 16,
    keyframe_interval: int = 1,
    fps: float = 10.0,
    encode_x264: bool = False,
    x264_bin: str = "videocode.exe",
    max_frames: int | None = None,
    frame_step: int = 1,
    vis_size: int = 224,
    log_interval: int = 10,
    qp_contrast: float = 1.0,
) -> None:
    root_path = Path(input_root)
    if not root_path.is_dir():
        raise RuntimeError(f"Failed to open dataset directory: {input_root}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_dirs = [path for path in sorted(root_path.iterdir()) if path.is_dir()]
    if not sequence_dirs:
        raise RuntimeError(f"No sequence folders found under: {input_root}")

    for idx, sequence_dir in enumerate(sequence_dirs, start=1):
        _log(f"Dataset progress | sequence={idx}/{len(sequence_dirs)} name={sequence_dir.name}")
        sequence_output = output_root / f"{sequence_dir.name}.json"
        sequence_bitstream = output_root / f"{sequence_dir.name}.h264"
        process_sequence(
            source_path=str(sequence_dir),
            output_path=str(sequence_output),
            block_size=block_size,
            keyframe_interval=keyframe_interval,
            fps=fps,
            encode_x264=encode_x264,
            x264_bin=x264_bin,
            x264_output=str(sequence_bitstream),
            max_frames=max_frames,
            frame_step=frame_step,
            vis_size=vis_size,
            log_interval=log_interval,
            qp_contrast=qp_contrast,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use qwen3-vl-2b to assign macroblock-level QP for each frame "
            "from either a video file or a KITTI-style image sequence dataset."
        )
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Input video path (legacy mode).",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Input dataset root containing sequence folders like 0000/0001/....",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for single video mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/qp",
        help="Output directory for dataset mode (one JSON/H.264 per sequence).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Macroblock size in pixels (default: 16 for x264 16x16 MB).",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=1,
        help="Interval (in seconds) to sample keyframes for global understanding.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate used to interpret image-sequence time spacing (default: 10 fps).",
    )
    # videocode 编码相关命令行参数
    parser.add_argument(
        "--encode-x264",
        action="store_true",
        help=(
            "After generating QP JSON, call videocode.exe to encode "
            "the video with per-frame macroblock QP map."
        ),
    )
    parser.add_argument(
        "--x264-bin",
        type=str,
        default="videocode.exe",
        help="Path to videocode encoder binary (default: videocode.exe in PATH).",
    )
    parser.add_argument(
        "--x264-output",
        type=str,
        default=None,
        help="Output H.264 path (default: <output JSON>.h264).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only process at most N sampled frames per sequence.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Sample one frame every N frames (default: 1).",
    )
    parser.add_argument(
        "--vis-size",
        type=int,
        default=224,
        help="Resize each frame to vis-size x vis-size before VLM inference.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print frame progress every N processed frames.",
    )
    parser.add_argument(
        "--qp-contrast",
        type=float,
        default=1.0,
        help="QP contrast strength (0.5~3.0). Larger means larger macroblock QP differences.",
    )
    parser.add_argument(
        "--quick-validate",
        action="store_true",
        help="Fast verification mode: fewer frames and coarser sampling/logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick_validate:
        # 快速验证模式：优先缩短推理总时间，便于快速检查端到端链路。
        args.max_frames = 1
        args.frame_step = 1
        args.vis_size = 160
        args.keyframe_interval = max(args.keyframe_interval, 2)
        args.log_interval = 1
        _log("Quick validation mode enabled (max_frames=1, frame_step=1, vis_size=160)")

    if args.input_root:
        process_dataset(
            input_root=args.input_root,
            output_dir=args.output_dir,
            block_size=args.block_size,
            keyframe_interval=args.keyframe_interval,
            fps=args.fps,
            encode_x264=args.encode_x264,
            x264_bin=args.x264_bin,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
            vis_size=args.vis_size,
            log_interval=args.log_interval,
            qp_contrast=args.qp_contrast,
        )
    elif args.video and args.output:
        process_sequence(
            source_path=args.video,
            output_path=args.output,
            block_size=args.block_size,
            keyframe_interval=args.keyframe_interval,
            fps=args.fps,
            encode_x264=args.encode_x264,
            x264_bin=args.x264_bin,
            x264_output=args.x264_output,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
            vis_size=args.vis_size,
            log_interval=args.log_interval,
            qp_contrast=args.qp_contrast,
        )
    else:
        raise SystemExit("Specify either --input-root for dataset mode or --video with --output for single video mode.")
