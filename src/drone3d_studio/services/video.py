"""Bounded-memory video sampling with real blur and duplicate measurements."""
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from drone3d_studio.domain.models import AnalysisConfig, Frame, Video
from drone3d_studio.persistence.store import atomic_write, reference


class Cancelled(Exception):
    pass


def metadata(path: Path) -> Video:
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read()
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or fps <= 0 or count <= 0:
            raise ValueError("Cannot decode this video. Try an MP4/H.264 or AVI file supported by OpenCV.")
        return Video(path=str(path.resolve()), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=fps,
                     frame_count=count, duration=count / fps, size=path.stat().st_size)
    finally:
        cap.release()


def gray_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 960:
        gray = cv2.resize(gray, (960, round(gray.shape[0] * 960 / gray.shape[1])))
    return gray


def blur_score(frame) -> float:
    return float(cv2.Laplacian(gray_frame(frame), cv2.CV_64F).var())


def signature(frame):
    return cv2.resize(gray_frame(frame), (64, 64)).astype(np.float32)


def duplicate_score(a, b) -> float:
    return float(np.mean(np.abs(a - b)))


def write_image(path: Path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise OSError(f"Could not encode {path.name}")
    encoded.tofile(str(path))  # Unicode-safe on Windows.


def analyze(path: Path, root: Path, config: AnalysisConfig, cancel, progress):
    started = time.monotonic()
    info = metadata(path)
    step = max(1, round(config.interval * info.fps))
    indices = range(0, min(info.frame_count, step * config.max_frames), step)
    run = uuid4().hex[:12]
    cap = cv2.VideoCapture(str(path))
    records, last_accepted = [], None
    try:
        for number, index in enumerate(indices):
            if cancel.is_set():
                raise Cancelled("Analysis cancelled. Previously completed analysis is retained.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Decoding failed at frame {index}; try converting the video to H.264.")
            blur, sig = blur_score(frame), signature(frame)
            reason = "Blurry" if blur < config.blur_threshold else ""
            if not reason and last_accepted is not None and duplicate_score(sig, last_accepted) < config.duplicate_threshold:
                reason = "Near duplicate"
            accepted = not reason
            if accepted:
                last_accepted = sig
            category = "accepted" if accepted else "rejected"
            target = root / "frames" / category / run / f"frame_{index:08d}.jpg"
            # Keep full accepted frames for photogrammetry; rejected frames are previews only.
            thumb = cv2.resize(frame, (240, max(1, round(frame.shape[0] * 240 / frame.shape[1]))))
            write_image(target, frame if accepted else thumb)
            thumbnail = root / "thumbnails" / run / target.name
            write_image(thumbnail, thumb)
            records.append(Frame(index=index, time=index / info.fps, path=reference(root, target),
                                 thumbnail=reference(root, thumbnail), blur=blur, accepted=accepted, reason=reason))
            progress(round((number + 1) / len(indices) * 100), f"Sampled {number + 1}/{len(indices)} · {category} · sharpness {blur:.1f}")
        elapsed = time.monotonic() - started
        import json
        atomic_write(root / "frames" / f"analysis-{run}.json", json.dumps({"seconds": elapsed, "frames": [r.model_dump() for r in records]}, indent=2))
        return records, elapsed
    finally:
        cap.release()


def synthetic_video(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (320, 240))
    if not writer.isOpened():
        raise OSError("MJPG encoding is unavailable")
    rng = np.random.default_rng(42)
    texture = rng.integers(30, 200, (240, 320, 3), dtype=np.uint8)
    try:
        for i in range(48):
            frame = np.roll(texture, i * 3, axis=1).copy()
            cv2.rectangle(frame, (30 + i * 2, 70), (100 + i * 2, 140), (70, 230, 180), -1)
            cv2.putText(frame, f"SYNTHETIC DEMO {i:02d}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            writer.write(frame)
    finally:
        writer.release()
