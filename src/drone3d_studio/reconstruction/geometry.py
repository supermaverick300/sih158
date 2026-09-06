"""COLMAP text camera geometry shared by alignment and calibrated AI depth."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


@dataclass
class CameraView:
    name: str
    camera_id: int
    R: np.ndarray
    t: np.ndarray
    observations: list

    @property
    def center(self):
        return -self.R.T @ self.t


def read_views(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    views, index = [], 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 10:
            raise ValueError("Invalid COLMAP image pose record")
        observations = lines[index].split() if index < len(lines) else []
        index += 1
        triples = [(float(observations[j]), float(observations[j+1]), int(observations[j+2])) for j in range(0, len(observations), 3)]
        views.append(CameraView(" ".join(values[9:]), int(values[8]), rotation([float(x) for x in values[1:5]]), np.array([float(x) for x in values[5:8]]), triples))
    return views


def read_points(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            values = line.split()
            result[int(values[0])] = np.array(list(map(float, values[1:4])))
    return result


def read_intrinsics(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        camera_id, kind, width, height = int(values[0]), values[1], int(values[2]), int(values[3])
        params = list(map(float, values[4:]))
        if kind == "PINHOLE":
            fx, fy, cx, cy = params
        elif kind == "SIMPLE_PINHOLE":
            fx, cx, cy = params
            fy = fx
        else:
            raise ValueError(f"AI depth fusion requires undistorted pinhole cameras, got {kind}")
        result[camera_id] = (width, height, fx, fy, cx, cy)
    return result
