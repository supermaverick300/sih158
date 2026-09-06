"""Depth Anything V2 Small, calibrated against SfM, with multi-view fusion.

Weights are local-only. Relative predictions are never mislabeled as metric.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from drone3d_studio.persistence.store import atomic_write
from drone3d_studio.services.video import Cancelled, write_image
from drone3d_studio.reconstruction.geometry import read_views, read_points, read_intrinsics


def check_installation(weights):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise ValueError("AI dependencies are missing. Run scripts\\setup_ai_windows.cmd, then restart the app.") from exc
    weights = Path(weights)
    missing = [name for name in ("config.json", "preprocessor_config.json", "model.safetensors") if not (weights / name).is_file()]
    if missing:
        raise ValueError(f"Depth Anything V2 weights missing in {weights}: {', '.join(missing)}. Run scripts\\download_depth_model.py or select the downloaded folder.")


class DepthPredictor:
    def __init__(self, weights, device="Auto"):
        check_installation(weights)
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        torch.set_num_threads(4)
        self.device = "cuda" if device == "CUDA" or device == "Auto" and torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("This PyTorch installation has no CUDA support. Select CPU or install the CUDA PyTorch build.")
        self.processor = AutoImageProcessor.from_pretrained(str(weights), local_files_only=True, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(str(weights), local_files_only=True, use_safetensors=True).to(self.device).eval()

    def predict(self, rgb):
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            result = self.torch.nn.functional.interpolate(output[:, None], size=rgb.shape[:2], mode="bicubic", align_corners=False)
        return result[0, 0].cpu().numpy().astype(np.float32)


def calibrate_inverse_depth(relative, pixels, depths):
    """Robust affine fit of predicted inverse depth to SfM inverse Z."""
    pixels, depths = np.asarray(pixels), np.asarray(depths)
    if len(depths) < 20:
        raise ValueError("Fewer than 20 triangulated depth anchors")
    x = relative[pixels[:, 1].astype(int), pixels[:, 0].astype(int)].astype(float)
    y = 1 / depths
    valid = np.isfinite(x) & np.isfinite(y) & (depths > 0)
    if valid.sum() < 20 or np.std(x[valid]) < 1e-6:
        raise ValueError("Insufficient valid depth variation")
    mask = valid.copy()
    for _ in range(5):
        a, b = np.linalg.lstsq(np.column_stack((x[mask], np.ones(mask.sum()))), y[mask], rcond=None)[0]
        residual = np.abs(a * x + b - y)
        cut = max(3 * np.median(residual[mask]), 1e-8)
        updated = valid & (residual <= cut)
        if updated.sum() < 20:
            break
        mask = updated
    a, b = np.linalg.lstsq(np.column_stack((x[mask], np.ones(mask.sum()))), y[mask], rcond=None)[0]
    inverse = a * relative + b
    predicted = a * x[mask] + b
    error = float(np.median(np.abs(predicted - y[mask]) / y[mask]))
    if a <= 0:
        raise ValueError(f"AI and SfM disagree on near/far ordering (inverse-depth slope {a:.6g}). Check camera calibration and capture geometry.")
    if error > .35:
        raise ValueError(f"AI/SfM calibration is unreliable (relative inverse-depth error {error:.2f})")
    low, high = np.percentile(depths[mask], [1, 99])
    valid_map = np.isfinite(inverse) & (inverse > 1 / (high * 2)) & (inverse < 1 / max(low * .5, 1e-9))
    depth = np.zeros_like(relative, dtype=np.float32)
    depth[valid_map] = 1 / inverse[valid_map]
    return depth, {"a": float(a), "b": float(b), "anchors": int(mask.sum()), "median_relative_inverse_error": error}


def consistent_points(world, views, current, maps, intrinsics):
    counts = np.zeros(len(world), dtype=int)
    neighbors = sorted([i for i in range(len(views)) if i != current], key=lambda i: np.linalg.norm(views[i].center - views[current].center))[:4]
    for i in neighbors:
        other = views[i]
        camera = world @ other.R.T + other.t
        width, height, fx, fy, cx, cy = intrinsics[other.camera_id]
        z = camera[:, 2]
        u = np.rint(fx * camera[:, 0] / np.maximum(z, 1e-9) + cx).astype(int)
        v = np.rint(fy * camera[:, 1] / np.maximum(z, 1e-9) + cy).astype(int)
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(valid)
        measured = maps[i][v[indices], u[indices]]
        counts[indices] += (measured > 0) & (np.abs(measured - z[indices]) / np.maximum(z[indices], 1e-9) < .08)
    return counts >= min(2, len(neighbors))


def fuse(views, depth_paths, images, intrinsics, output, config, cancel, progress):
    maps = [np.load(path, mmap_mode="r") for path in depth_paths]
    voxels = {}
    try:
        for i, view in enumerate(views):
            if cancel.is_set():
                raise Cancelled("AI depth fusion cancelled")
            width, height, fx, fy, cx, cy = intrinsics[view.camera_id]
            yy, xx = np.mgrid[0:height:config.depth_stride, 0:width:config.depth_stride]
            z = maps[i][yy, xx].ravel()
            xx, yy = xx.ravel(), yy.ravel()
            valid = np.isfinite(z) & (z > 0)
            xx, yy, z = xx[valid], yy[valid], z[valid]
            camera = np.column_stack(((xx - cx) / fx * z, (yy - cy) / fy * z, z))
            world = (camera - view.t) @ view.R
            keep = consistent_points(world, views, i, maps, intrinsics)
            rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(str(images / view.name), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            colors = rgb[yy, xx]
            for point, color in zip(world[keep], colors[keep]):
                key = tuple(np.floor(point / config.fusion_voxel_size).astype(np.int64))
                if key in voxels:
                    entry = voxels[key]
                    entry[:3] += point
                    entry[3:6] += color
                    entry[6] += 1
                else:
                    if len(voxels) >= 500000:
                        raise ValueError("AI fusion exceeded 500,000 voxels. Increase voxel size or pixel stride and retry.")
                    voxels[key] = np.r_[point, color.astype(float), 1.]
            progress(-1, f"AI fusion {i + 1}/{len(views)}: {len(voxels)} occupied voxels")
    finally:
        for depth in maps:
            depth._mmap.close()
    if len(voxels) < 100:
        raise ValueError("Too few multi-view-consistent AI depth points. Use sharper overlapping views or COLMAP stereo.")
    array = np.array(list(voxels.values()))
    cloud = trimesh.points.PointCloud(array[:, :3] / array[:, 6:7], colors=np.clip(array[:, 3:6] / array[:, 6:7], 0, 255).astype(np.uint8))
    cloud.export(str(output))
    return len(cloud.vertices)


def run_depth_pipeline(dense, text_model, weights, config, cancel, progress):
    views = read_views(text_model / "images.txt")
    intrinsics = read_intrinsics(text_model / "cameras.txt")
    points = read_points(text_model / "points3D.txt")
    folder = dense / "ai-depth"
    folder.mkdir(exist_ok=True)
    predictor = DepthPredictor(weights, config.device)
    accepted, paths, reports = [], [], []
    for i, view in enumerate(views):
        if cancel.is_set():
            raise Cancelled("AI depth inference cancelled")
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(str(dense / "images" / view.name), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        relative = predictor.predict(rgb)
        np.save(folder / (view.name + ".relative.npy"), relative)
        normalized = cv2.normalize(relative, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        write_image(folder / (view.name + ".preview.jpg"), cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO))
        pixels, depths = [], []
        for u, v, point_id in view.observations:
            if point_id in points and 0 <= u < relative.shape[1] and 0 <= v < relative.shape[0]:
                z = (view.R @ points[point_id] + view.t)[2]
                if z > 0:
                    pixels.append((u, v))
                    depths.append(z)
        try:
            depth, report = calibrate_inverse_depth(relative, pixels, depths)
            path = folder / (view.name + ".depth.npy")
            np.save(path, depth)
            accepted.append(view)
            paths.append(path)
            reports.append({"image": view.name, "status": "calibrated", **report})
        except ValueError as exc:
            reports.append({"image": view.name, "status": "rejected", "reason": str(exc)})
            progress(-1, f"AI depth rejected {view.name}: {exc}")
        progress(-1, f"Depth Anything V2 {i + 1}/{len(views)}")
    manifest = weights / "download-manifest.json"
    provenance = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {"weights_path": str(weights), "revision": "unverified"}
    atomic_write(folder / "calibration.json", json.dumps({"model": "Depth-Anything-V2-Small-hf", "provenance": provenance, "units": "ENU metres" if config.align_gps else "SfM units (not georeferenced)", "frames": reports}, indent=2))
    del predictor
    if len(accepted) < 3:
        raise ValueError("Fewer than three AI depth maps passed SfM calibration. See ai-depth/calibration.json; no cloud was fabricated.")
    if config.capture_mode == "Single pass":
        from drone3d_studio.reconstruction.visible_surface import build_surface
        maps = [np.load(path,mmap_mode="r") for path in paths]
        try:
            return build_surface(accepted,maps,dense/"images",intrinsics,dense/"ai-visible-surface.ply",config.depth_stride,cancel,progress)
        finally:
            for depth in maps:
                depth._mmap.close()
    output = dense / "ai-fused.ply"
    count = fuse(accepted, paths, dense / "images", intrinsics, output, config, cancel, progress)
    progress(-1, f"Fused {count} colored AI depth points")
    return output
