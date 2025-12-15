import argparse
import os
import sys
from pathlib import Path

import cv2


def parse_time(value):
    """Parse seconds or HH:MM:SS(.mmm) strings into float seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Cannot parse time string '{value}'")

    parts = [p.strip() for p in parts]
    secs = 0.0
    for idx, part in enumerate(parts):
        last = idx == len(parts) - 1
        if last:
            secs = secs * 60 + float(part)
        else:
            secs = secs * 60 + int(part)
    return secs


def clamp(value, lower, upper):
    if value is None:
        return None
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file at a fixed interval."
    )
    parser.add_argument("--video", required=True, help="Path to the source video file.")
    parser.add_argument(
        "--output",
        default=os.path.join("output", "frames"),
        help="Directory where extracted frames will be written.",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=1.0,
        help="Seconds between extracted frames (must be > 0).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start time (seconds or HH:MM:SS).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end time (seconds or HH:MM:SS).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Stop after saving this many frames (0 = unlimited).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix for saved frames.",
    )
    args = parser.parse_args()

    if args.every <= 0:
        print("[ERROR] --every must be greater than 0 seconds.", file=sys.stderr)
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else -1
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else None

    try:
        start_seconds = parse_time(args.start)
        end_seconds = parse_time(args.end)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    start_seconds = 0.0 if start_seconds is None else max(0.0, start_seconds)
    if duration is not None:
        start_seconds = clamp(start_seconds, 0.0, duration)
        end_seconds = clamp(end_seconds, 0.0, duration)

    if end_seconds is not None and end_seconds <= start_seconds:
        print("[ERROR] --end must be greater than --start.", file=sys.stderr)
        sys.exit(1)

    if fps > 0:
        start_frame = int(start_seconds * fps)
    else:
        # Fall back to beginning if FPS is unknown.
        start_frame = 0
        start_seconds = 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_saved = 0
    frame_index = start_frame
    next_capture = start_seconds
    epsilon = 1e-6

    print(f"[INFO] Video FPS: {fps:.3f}" if fps > 0 else "[INFO] Video FPS unavailable.")
    if duration:
        print(f"[INFO] Video duration: {duration:.2f}s")
    print(f"[INFO] Extracting frames to '{output_dir}' every {args.every:.3f}s")

    while True:
        if args.max_frames and frames_saved >= args.max_frames:
            break

        if end_seconds is not None and fps > 0:
            current_time = frame_index / fps
            if current_time - epsilon > end_seconds:
                break

        ret, frame = cap.read()
        if not ret:
            break

        if fps > 0:
            timestamp = frame_index / fps
        else:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if timestamp is None:
            timestamp = 0.0

        if timestamp + epsilon >= next_capture:
            filename = f"{args.prefix}_{frames_saved:05d}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            print(f"[INFO] Saved {filename} at {timestamp:.3f}s")
            frames_saved += 1
            next_capture += args.every

        frame_index += 1

    cap.release()
    print(f"[INFO] Extraction complete. {frames_saved} frame(s) saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
