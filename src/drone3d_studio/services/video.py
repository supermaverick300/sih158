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


def quality(frame):
    gray = gray_frame(frame)
    count = len(cv2.ORB_create(nfeatures=1000).detect(gray, None))
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    return count, clipped


def motion(previous, current):
    if previous is None:
        return 0.
    points = cv2.goodFeaturesToTrack(previous, 200, .01, 8)
    if points is None:
        return 0.
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if tracked is None or status.sum() < 8:
        return 0.
    valid = status.ravel().astype(bool)
    return float(np.median(np.linalg.norm(tracked[valid] - points[valid], axis=2)))


def next_step(step, displacement, base):
    return max(1, round(np.clip(step * np.clip(12 / max(displacement, 1), .5, 2), max(1, base / 4), base * 4)))


def analyze(path: Path, root: Path, config: AnalysisConfig, cancel, progress, telemetry=None):
    started = time.monotonic()
    info = metadata(path)
    from drone3d_studio.services.telemetry import at_time, distance
    if config.gps_spacing_m > 0 and not telemetry:
        raise ValueError("GPS keyframe spacing is enabled but telemetry is missing. Import a synchronized CSV or set spacing to zero.")
    step = base_step = max(1, round(config.interval * info.fps))
    if config.cover_entire_video and info.frame_count > base_step * config.max_frames:
        progress(0, "Sample budget spreads frames across the whole pass. Increase maximum samples if the gaps remove needed overlap.")
    run = uuid4().hex[:12]
    cap = cv2.VideoCapture(str(path))
    records, last_accepted, previous = [], None, None
    accepted_gps = []
    index = 0
    try:
        while index < info.frame_count and len(records) < config.max_frames:
            if cancel.is_set():
                raise Cancelled("Analysis cancelled. Previously completed analysis is retained.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Decoding failed at frame {index}; try converting the video to H.264.")
            blur, sig = blur_score(frame), signature(frame)
            features, clipped = quality(frame)
            gray = cv2.resize(gray_frame(frame), (640, 360))
            displacement = motion(previous, gray)
            previous = gray
            fix = at_time(telemetry, index / info.fps + config.telemetry_offset_s, config.telemetry_max_gap_s)
            reason = "Blurry" if blur < config.blur_threshold else ""
            if not reason and clipped > config.max_clipped_fraction:
                reason = "Exposure clipping"
            if not reason and features < config.min_features:
                reason = "Too few visual features"
            if not reason and last_accepted is not None and duplicate_score(sig, last_accepted) < config.duplicate_threshold:
                reason = "Near duplicate"
            if not reason and config.gps_spacing_m > 0:
                if fix is None:
                    reason = "Missing synchronized GPS"
                elif accepted_gps and min(distance(fix, other) for other in accepted_gps) < config.gps_spacing_m:
                    reason = "GPS spacing too small"
            accepted = not reason
            if accepted:
                last_accepted = sig
                if fix:
                    accepted_gps.append(fix)
            category = "accepted" if accepted else "rejected"
            target = root / "frames" / category / run / f"frame_{index:08d}.jpg"
            # Keep full accepted frames for photogrammetry; rejected frames are previews only.
            thumb = cv2.resize(frame, (240, max(1, round(frame.shape[0] * 240 / frame.shape[1]))))
            write_image(target, frame if accepted else thumb)
            thumbnail = root / "thumbnails" / run / target.name
            write_image(thumbnail, thumb)
            records.append(Frame(index=index, time=index / info.fps, path=reference(root, target),
                                 thumbnail=reference(root, thumbnail), blur=blur, accepted=accepted, reason=reason, features=features, clipped_fraction=clipped,
                                 quality_score=100 * (.4 * min(1, blur / max(3 * config.blur_threshold, 100)) + .3 * min(1, features / 500) + .3 * (1 - clipped)), motion_px=displacement, gps=fix))
            percent = max((index + 1) / info.frame_count, len(records) / config.max_frames)
            progress(min(99, round(percent * 100)), f"Sampled {len(records)} · {category} · sharpness {blur:.1f} · features {features} · clipped {clipped:.1%}")
            if config.sampling_mode == "Adaptive" and len(records) > 1:
                step = next_step(step, displacement, base_step)
            if config.cover_entire_video:
                if index == info.frame_count - 1:
                    break
                remaining = config.max_frames - len(records)
                if remaining > 0:
                    step = max(step, int(np.ceil((info.frame_count - 1 - index) / remaining)))
                index = min(index + step, info.frame_count - 1)
            else:
                index += step
        elapsed = time.monotonic() - started
        import json
        atomic_write(root / "frames" / f"analysis-{run}.json", json.dumps({"seconds": elapsed, "configuration":config.model_dump(), "sampled_span_seconds":records[-1].time-records[0].time if records else 0, "frames": [r.model_dump() for r in records]}, indent=2))
        progress(100, f"Analysis complete: {len(records)} candidates, {sum(r.accepted for r in records)} keyframes")
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
