import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from x264_runner import encode_with_x264


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_frame_count(source_path: str) -> int:
    src = Path(source_path)
    if src.is_dir():
        return sum(1 for p in src.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
    raise RuntimeError(f"Only image-sequence source is supported now: {source_path}")


def _bitrate_kbps(size_bytes: int, frame_count: int, fps: float) -> float:
    if frame_count <= 0 or fps <= 0:
        return 0.0
    return (size_bytes * 8.0 * fps) / (frame_count * 1000.0)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _global_qp_mean(data: Dict) -> float:
    vals: List[float] = []
    for frame in data.get("frames", []):
        qp = frame.get("qp", [])
        for row in qp:
            vals.extend(float(v) for v in row)
    if not vals:
        return 28.0
    return float(np.mean(np.asarray(vals, dtype=np.float32)))


def _apply_qp_delta(data: Dict, delta: float) -> Dict:
    out = json.loads(json.dumps(data))
    for frame in out.get("frames", []):
        qp = frame.get("qp", [])
        new_qp = []
        for row in qp:
            arr = np.asarray(row, dtype=np.float32)
            arr = np.clip(arr + delta, 0.0, 51.0)
            new_qp.append(arr.tolist())
        frame["qp"] = new_qp
    return out


def _run_encode_and_measure(
    source_path: str,
    qp_json_path: Path,
    output_h264_path: Path,
    x264_bin: str,
    frame_count: int,
    fps: float,
) -> Tuple[int, float]:
    output_h264_path.parent.mkdir(parents=True, exist_ok=True)
    encode_with_x264(
        source_path=source_path,
        qp_map_path=str(qp_json_path),
        output_bitstream=str(output_h264_path),
        x264_bin=x264_bin,
    )
    size_bytes = output_h264_path.stat().st_size
    return size_bytes, _bitrate_kbps(size_bytes, frame_count, fps)


def _search_delta_for_target_bitrate(
    base_data: Dict,
    source_path: str,
    out_json_path: Path,
    out_h264_path: Path,
    x264_bin: str,
    frame_count: int,
    fps: float,
    target_kbps: float,
    init_delta: float,
    tol_percent: float,
    max_iters: int,
) -> Dict:
    lo = -20.0
    hi = 20.0
    best = None
    history = []

    # First try init delta from mean-QP alignment.
    candidate_deltas: List[float] = [init_delta]

    # Add binary-search probes.
    for _ in range(max_iters):
        if not candidate_deltas:
            candidate_deltas.append((lo + hi) * 0.5)
        delta = candidate_deltas.pop(0)

        new_data = _apply_qp_delta(base_data, delta)
        _dump_json(out_json_path, new_data)
        size_bytes, kbps = _run_encode_and_measure(
            source_path=source_path,
            qp_json_path=out_json_path,
            output_h264_path=out_h264_path,
            x264_bin=x264_bin,
            frame_count=frame_count,
            fps=fps,
        )

        err_percent = 0.0
        if target_kbps > 1e-9:
            err_percent = (kbps - target_kbps) / target_kbps * 100.0

        item = {
            "delta": delta,
            "size_bytes": size_bytes,
            "bitrate_kbps": kbps,
            "error_percent": err_percent,
        }
        history.append(item)

        if best is None or abs(err_percent) < abs(best["error_percent"]):
            best = item

        if abs(err_percent) <= tol_percent:
            break

        # Higher bitrate => need larger delta (raise QP); lower bitrate => reduce delta.
        if kbps > target_kbps:
            lo = max(lo, delta)
        else:
            hi = min(hi, delta)

        next_delta = (lo + hi) * 0.5
        candidate_deltas.append(next_delta)

    assert best is not None
    return {
        "best": best,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate adaptive outputs matching bitrate of base_qp10/22/34/46 and save logs into each new JSON."
    )
    parser.add_argument("--adaptive-json", required=True, help="Source adaptive json, e.g. outputs/qp/0000.json")
    parser.add_argument("--outputs-root", default="outputs", help="Outputs root containing base_qpXX and qp_qpXX folders")
    parser.add_argument("--targets", nargs="+", default=["qp10", "qp22", "qp34", "qp46"], help="Target baseline labels")
    parser.add_argument("--x264-bin", required=True, help="Path to videocode.exe")
    parser.add_argument("--tol-percent", type=float, default=1.5, help="Target bitrate tolerance percent")
    parser.add_argument("--max-iters", type=int, default=8, help="Max iterations for each target")
    args = parser.parse_args()

    adaptive_json_path = Path(args.adaptive_json)
    outputs_root = Path(args.outputs_root)

    base_data = _load_json(adaptive_json_path)
    source_path = str(base_data["source"])
    fps = float(base_data.get("fps", 10.0))
    frame_count = _list_frame_count(source_path)

    seq_name = adaptive_json_path.stem
    adaptive_mean_qp = _global_qp_mean(base_data)

    print(f"sequence={seq_name} frame_count={frame_count} fps={fps} adaptive_mean_qp={adaptive_mean_qp:.3f}")

    for label in args.targets:
        # label like qp10 -> base_qp10, qp_qp10
        baseline_h264 = outputs_root / f"base_{label}" / f"{seq_name}.h264"
        if not baseline_h264.exists():
            print(f"[WARN] baseline not found, skip: {baseline_h264}")
            continue

        target_size = baseline_h264.stat().st_size
        target_kbps = _bitrate_kbps(target_size, frame_count, fps)

        try:
            target_qp_value = float(label.replace("qp", ""))
        except Exception:
            target_qp_value = adaptive_mean_qp

        init_delta = target_qp_value - adaptive_mean_qp

        out_dir = outputs_root / f"qp_{label}"
        out_json = out_dir / f"{seq_name}.json"
        out_h264 = out_dir / f"{seq_name}.h264"

        print(
            f"[INFO] target={label} target_kbps={target_kbps:.3f} "
            f"init_delta={init_delta:.3f} out_dir={out_dir}"
        )

        result = _search_delta_for_target_bitrate(
            base_data=base_data,
            source_path=source_path,
            out_json_path=out_json,
            out_h264_path=out_h264,
            x264_bin=args.x264_bin,
            frame_count=frame_count,
            fps=fps,
            target_kbps=target_kbps,
            init_delta=init_delta,
            tol_percent=args.tol_percent,
            max_iters=args.max_iters,
        )

        # Write record table info into new json.
        out_data = _load_json(out_json)
        out_data["rate_match"] = {
            "target_label": label,
            "baseline_h264": str(baseline_h264),
            "target_size_bytes": target_size,
            "target_bitrate_kbps": target_kbps,
            "adaptive_source_json": str(adaptive_json_path),
            "adaptive_global_qp_mean": adaptive_mean_qp,
            "init_delta": init_delta,
            "final_delta": result["best"]["delta"],
            "final_size_bytes": result["best"]["size_bytes"],
            "final_bitrate_kbps": result["best"]["bitrate_kbps"],
            "final_error_percent": result["best"]["error_percent"],
            "tol_percent": args.tol_percent,
            "history": result["history"],
        }
        _dump_json(out_json, out_data)

        print(
            f"[DONE] {label}: final_delta={result['best']['delta']:.3f}, "
            f"bitrate={result['best']['bitrate_kbps']:.3f} kbps, "
            f"err={result['best']['error_percent']:+.3f}%"
        )


if __name__ == "__main__":
    main()
