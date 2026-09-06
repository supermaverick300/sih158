"""Deterministic geometry/telemetry tests; fixtures are not real flight results."""
import threading
import cv2
import numpy as np
import pytest
from drone3d_studio.domain.models import AnalysisConfig, GPSFix, PipelineConfig, Project
from drone3d_studio.persistence import store
from drone3d_studio.services import telemetry, video
from drone3d_studio.reconstruction.ai_depth import calibrate_inverse_depth, consistent_points, fuse
from drone3d_studio.reconstruction.alignment import to_enu
from drone3d_studio.reconstruction.geometry import CameraView, read_views, rotation


def fix(time=0, lat=0, lon=0, altitude=100):
    return GPSFix(time=time, latitude=lat, longitude=lon, altitude_m=altitude)


def test_telemetry_validation_interpolation_and_dateline(tmp_path):
    path = tmp_path / "flight.csv"
    path.write_text("time_s,latitude,longitude,altitude_m\n0,0,179.9,100\n1,0,-179.9,102\n2,0,-179.8,104\n")
    fixes = telemetry.load_csv(path)
    halfway = telemetry.at_time(fixes, .5)
    assert abs(halfway.longitude) == pytest.approx(180)
    assert halfway.altitude_m == 101
    assert telemetry.at_time(fixes, -1) is None
    assert telemetry.at_time(fixes, 3) is None
    assert telemetry.at_time(fixes, .5, .1) is None
    path.write_text("time_s,latitude,longitude,altitude_m\n0,0,0,1\n0,0,0,2\n1,0,0,3\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        telemetry.load_csv(path)
    with pytest.raises(ValueError, match="timestamps"):
        Project(name="bad", telemetry=[fix(1), fix(0)])


def test_enu_axes_and_distance():
    origin = fix(altitude=0)
    positions = to_enu([origin, fix(lon=.0001, altitude=0), fix(lat=.0001, altitude=0), fix(altitude=10)], origin)
    np.testing.assert_allclose(positions[0], 0, atol=1e-8)
    assert positions[1, 0] == pytest.approx(11.131949, rel=1e-5)
    assert positions[2, 1] == pytest.approx(11.057428, rel=1e-5)
    assert positions[3, 2] == pytest.approx(10)
    assert telemetry.distance(origin, fix(altitude=10)) == pytest.approx(10)


def test_adaptive_sampling_and_quality_rejection(tmp_path):
    path = tmp_path / "video.avi"
    video.synthetic_video(path)
    config = AnalysisConfig(interval=.5, blur_threshold=0, duplicate_threshold=0, sampling_mode="Adaptive")
    frames, _ = video.analyze(path, tmp_path, config, threading.Event(), lambda *a: None)
    assert len(set(np.diff([f.index for f in frames]))) > 1
    assert all(f.features > 0 and 0 <= f.quality_score <= 100 for f in frames)
    assert video.next_step(12, 40, 12) < 12
    assert video.next_step(12, 1, 12) > 12
    config.min_features = 2000
    rejected, _ = video.analyze(path, tmp_path, config, threading.Event(), lambda *a: None)
    assert all(f.reason == "Too few visual features" for f in rejected)
    features, clipped = video.quality(np.full((100, 100, 3), 255, np.uint8))
    assert features == 0 and clipped == 1


def test_gps_spacing_and_missing_telemetry(tmp_path):
    path = tmp_path / "video.avi"
    video.synthetic_video(path)
    config = AnalysisConfig(interval=.5, blur_threshold=0, duplicate_threshold=0, gps_spacing_m=1.5)
    with pytest.raises(ValueError, match="telemetry is missing"):
        video.analyze(path, tmp_path, config, threading.Event(), lambda *a: None)
    fixes = [fix(t, lon=t / 111319.49) for t in range(5)]
    frames, _ = video.analyze(path, tmp_path, config, threading.Event(), lambda *a: None, fixes)
    accepted = [f for f in frames if f.accepted]
    assert 2 <= len(accepted) < len(frames)
    assert any(f.reason == "GPS spacing too small" for f in frames)
    assert all(telemetry.distance(a.gps, b.gps) >= 1.5 for a, b in zip(accepted, accepted[1:]))


def test_depth_calibration_recovers_scale_shift_and_rejects_bad_fit():
    relative = np.linspace(1, 10, 120, dtype=np.float32).reshape(10, 12)
    yy, xx = np.indices(relative.shape)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    truth = 1 / (.02 * relative + .03)
    depths = truth.ravel().copy()
    depths[::15] *= 10  # Triangulation outliers must not control the fit.
    calibrated, report = calibrate_inverse_depth(relative, pixels, depths)
    np.testing.assert_allclose(calibrated, truth, rtol=1e-5)
    assert report["a"] == pytest.approx(.02, rel=1e-5)
    with pytest.raises(ValueError, match="near/far"):
        calibrate_inverse_depth(relative, pixels, (relative + 5).ravel())
    with pytest.raises(ValueError, match="20"):
        calibrate_inverse_depth(relative, pixels[:3], depths[:3])


def test_camera_pose_parser_and_rotation(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("# COLMAP\n1 1 0 0 0 -2 -3 -4 1 000000.jpg\n2 3 7 4 5 -1\n2 1 0 0 0 0 0 0 1 000001.jpg\n\n")
    views = read_views(path)
    np.testing.assert_allclose(views[0].center, [2, 3, 4])
    assert views[0].observations[0] == (2, 3, 7)
    assert views[1].observations == []
    np.testing.assert_allclose(rotation([np.sqrt(.5), 0, 0, np.sqrt(.5)]) @ [1, 0, 0], [0, 1, 0], atol=1e-7)


def test_multiview_fusion_rejects_inconsistent_depth_and_preserves_color(tmp_path):
    views = [CameraView(f"{i}.jpg", 1, np.eye(3), np.array([-i*.1, 0, 0]), []) for i in range(3)]
    intrinsics = {1: (40, 40, 40., 40., 20., 20.)}
    maps = [np.full((40, 40), 5, np.float32) for _ in views]
    world = np.array([[0, 0, 5], [0, 0, 8]])
    np.testing.assert_array_equal(consistent_points(world, views, 0, maps, intrinsics), [True, False])
    paths = []
    for i, view in enumerate(views):
        path = tmp_path / f"{i}.npy"
        np.save(path, maps[i])
        paths.append(path)
        video.write_image(tmp_path / view.name, np.full((40, 40, 3), [30, 60, 180], np.uint8))
    output = tmp_path / "fused.ply"
    count = fuse(views, paths, tmp_path, intrinsics, output, PipelineConfig(depth_stride=2, fusion_voxel_size=.05), threading.Event(), lambda *a: None)
    import trimesh
    cloud = trimesh.load(output)
    assert count >= 100
    np.testing.assert_allclose(cloud.vertices[:, 2], 5)
    np.testing.assert_allclose(cloud.colors[:, :3].mean(axis=0), [180, 60, 30], atol=3)


def test_pipeline_and_telemetry_survive_json_export(tmp_path):
    root = tmp_path / "project"
    project = store.create(root, "GPS test")
    project.pipeline.depth_method = "Depth Anything V2"
    project.pipeline.align_gps = True
    project.telemetry = [fix(0), fix(1), fix(2)]
    project.telemetry_source = "source/flight.csv"
    path = tmp_path / "export.json"
    store.export_json(path, root, project)
    imported = store.import_json(path, root)
    assert imported.telemetry_source == project.telemetry_source
    assert imported.pipeline == project.pipeline
    assert imported.telemetry == project.telemetry


def test_ai_backend_orchestration_and_failure_report(tmp_path, monkeypatch):
    """Contract test with mocked inference/SfM; real inference is separately validated."""
    import json
    import trimesh
    from drone3d_studio.domain.models import Frame
    from drone3d_studio.reconstruction import colmap, ai_depth
    root = tmp_path / "contract"
    store.create(root, "Contract")
    path = root / "frames" / "accepted" / "source.jpg"
    video.write_image(path, np.zeros((16, 16, 3), np.uint8))
    frames = [Frame(index=i, time=i, path=str(path), thumbnail="", blur=50, accepted=True) for i in range(3)]
    monkeypatch.setattr(colmap, "detect", lambda _: "colmap.exe")
    monkeypatch.setattr(ai_depth, "check_installation", lambda _: None)
    commands = []
    def runner(args, cancel, log, cwd):
        commands.append(args[1])
        if args[1] == "mapper":
            sparse = cwd / "sparse" / "0"
            sparse.mkdir()
            (sparse / "cameras.bin").write_bytes(b"contract")
        elif args[1] == "model_converter" and args[-1] == "PLY":
            trimesh.points.PointCloud([[0, 0, 0], [1, 1, 1]]).export(args[args.index("--output_path") + 1])
    monkeypatch.setattr(colmap, "run_command", runner)
    def inference(dense, text, weights, config, cancel, progress):
        output = dense / "ai-fused.ply"
        trimesh.points.PointCloud([[0, 0, 0], [1, 1, 1]]).export(str(output))
        return output
    monkeypatch.setattr(ai_depth, "run_depth_pipeline", inference)
    backend = colmap.ColmapBackend(pipeline=PipelineConfig(depth_method="Depth Anything V2"))
    result = backend.run(root, frames, threading.Event(), lambda *a: None)
    assert result[-1][0].origin == "Depth Anything V2"
    assert not result[0][0].visible
    assert "image_undistorter" in commands and "patch_match_stereo" not in commands
    report = json.loads((backend.work / "pipeline-report.json").read_text())
    assert report["status"] == "Succeeded" and report["units"] == "Arbitrary SfM units"
    def fail(*args):
        raise ValueError("Incompatible depth ordering")
    monkeypatch.setattr(ai_depth, "run_depth_pipeline", fail)
    with pytest.raises(ValueError, match="ordering"):
        backend.run(root, frames, threading.Event(), lambda *a: None)
    report = json.loads((backend.work / "pipeline-report.json").read_text())
    assert report["status"] == "Failed" and "ordering" in report["error"]
