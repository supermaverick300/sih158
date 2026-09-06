"""Align registered camera centers to local East/North/Up metres."""
import json
import numpy as np
from drone3d_studio.persistence.store import atomic_write
from drone3d_studio.services.telemetry import ecef
from drone3d_studio.reconstruction.geometry import read_views


def to_enu(fixes, origin):
    lat, lon = np.radians([origin.latitude, origin.longitude])
    basis = np.array([[-np.sin(lon), np.cos(lon), 0], [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)], [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return (np.array([ecef(f) for f in fixes]) - ecef(origin)) @ basis.T


def align(model, work, frames, executable, config, cancel, log, runner):
    source_text = work / "alignment-source"
    source_text.mkdir()
    runner([executable, "model_converter", "--input_path", str(model), "--output_path", str(source_text), "--output_type", "TXT"], cancel, log, work)
    views = read_views(source_text / "images.txt")
    registered = {v.name for v in views}
    tagged = [(f"{i:06d}.jpg", frame.gps) for i, frame in enumerate(frames) if frame.gps is not None and f"{i:06d}.jpg" in registered]
    if len(tagged) < 3:
        raise ValueError("GPS alignment requires at least three registered camera positions with synchronized GPS.")
    # All disconnected components must use the same ENU origin in one scene.
    origin = next(frame.gps for frame in frames if frame.gps is not None)
    xyz = to_enu([fix for _, fix in tagged], origin)
    singular = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    if singular[0] < 1 or singular[1] < max(.1, singular[0] * .001):
        raise ValueError("GPS trajectory is too short or nearly collinear to determine a reliable 3D alignment.")
    refs = work / "camera-enu.txt"
    refs.write_text("".join(f"{name} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n" for (name, _), p in zip(tagged, xyz)), encoding="utf-8")
    aligned = work / "aligned"
    aligned.mkdir()
    runner([executable, "model_aligner", "--input_path", str(model), "--output_path", str(aligned), "--ref_images_path", str(refs), "--ref_is_gps", "0", "--alignment_type", "custom", "--alignment_max_error", str(config.alignment_max_error_m), "--transform_path", str(work / "sfm-to-enu.txt")], cancel, log, work)
    aligned_text = work / "alignment-result"
    aligned_text.mkdir()
    runner([executable, "model_converter", "--input_path", str(aligned), "--output_path", str(aligned_text), "--output_type", "TXT"], cancel, log, work)
    centers = {view.name: view.center for view in read_views(aligned_text / "images.txt")}
    residuals = np.array([np.linalg.norm(centers[name] - p) for (name, _), p in zip(tagged, xyz)])
    inliers = residuals <= config.alignment_max_error_m
    if inliers.sum() < max(3, int(np.ceil(len(tagged) * .5))):
        raise ValueError("GPS alignment failed residual verification; at least three and half of registered GPS references must agree.")
    inlier_xyz = xyz[inliers]
    spread = np.linalg.svd(inlier_xyz - inlier_xyz.mean(axis=0), compute_uv=False)
    if spread[0] < 1 or spread[1] < max(.1, spread[0] * .001):
        raise ValueError("GPS alignment inliers are too close or collinear to verify the transform.")
    report = {"status": "Aligned", "coordinate_system": "Local ENU", "units": "metres", "origin_wgs84": origin.model_dump(), "inliers": int(inliers.sum()), "references": len(tagged), "inlier_rmse_m": float(np.sqrt(np.mean(residuals[inliers]**2))), "camera_residuals_m": {name: float(error) for (name, _), error in zip(tagged, residuals)}, "note": "Accuracy is limited by telemetry, timing and altitude datum; not survey certified."}
    atomic_write(work / "georeference.json", json.dumps(report, indent=2))
    log(f"GPS aligned: {report['inliers']}/{len(tagged)} camera references, RMSE {report['inlier_rmse_m']:.3f} m")
    return aligned
