import threading
import sys
import cv2
import numpy as np
import pytest
from drone3d_studio.domain.models import AnalysisConfig, Transform
from drone3d_studio.persistence import store
from drone3d_studio.services import video, meshes
from drone3d_studio.reconstruction.colmap import detect, run_command


def test_metadata_sampling_and_report(tmp_path):
    root = tmp_path / "flight"
    store.create(root, "Flight")
    path = root / "source" / "tiny.avi"
    video.synthetic_video(path)
    info = video.metadata(path)
    assert (info.width, info.height, info.frame_count) == (320, 240, 48)
    assert info.duration == pytest.approx(4)
    updates = []
    frames, elapsed = video.analyze(path, root, AnalysisConfig(interval=.5, max_frames=5, blur_threshold=0, duplicate_threshold=0), threading.Event(), lambda *args: updates.append(args))
    assert [f.index for f in frames] == [0, 6, 12, 18, 24]
    assert all(f.accepted and store.resolve(root, f.path).exists() for f in frames)
    assert updates[-1][0] == 100
    assert elapsed > 0
    assert list((root / "frames").glob("analysis-*.json"))


def test_blur_and_duplicate():
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(sharp, (15, 15), 4)
    assert video.blur_score(sharp) > video.blur_score(blurry) * 10
    assert video.duplicate_score(video.signature(sharp), video.signature(sharp)) == 0
    assert video.duplicate_score(video.signature(sharp), video.signature(blurry)) > 0


def test_rejections(tmp_path):
    path = tmp_path / "same.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
    for _ in range(5):
        writer.write(np.full((100, 100, 3), 100, dtype=np.uint8))
    writer.release()
    frames, _ = video.analyze(path, tmp_path, AnalysisConfig(interval=.1, blur_threshold=0), threading.Event(), lambda *a: None)
    assert frames[0].accepted
    assert all(f.reason == "Near duplicate" for f in frames[1:])
    frames, _ = video.analyze(path, tmp_path, AnalysisConfig(interval=.1, blur_threshold=1), threading.Event(), lambda *a: None)
    assert all(f.reason == "Blurry" for f in frames)


def test_cancellation(tmp_path):
    path = tmp_path / "tiny.avi"
    video.synthetic_video(path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(video.Cancelled):
        video.analyze(path, tmp_path, AnalysisConfig(), cancel, lambda *a: None)
    cancel.clear()
    with pytest.raises(video.Cancelled):
        video.analyze(path, tmp_path, AnalysisConfig(interval=.1), cancel, lambda *a: cancel.set())


def test_model_validation_demo_transforms(tmp_path):
    scene = meshes.demo_scene(tmp_path, threading.Event(), lambda *a: None)
    assert len(scene) == 3
    for model, mesh in scene:
        assert model.origin == "Demo" and model.faces > 0
        assert len(meshes.load_mesh(store.resolve(tmp_path, model.path)).vertices) > 0
    transform = Transform(position=(2, 3, 4), rotation=(0, 0, 90), scale=(2, 2, 2))
    actual = meshes.matrix(transform) @ np.array([1, 0, 0, 1])
    np.testing.assert_allclose(actual, [2, 5, 4, 1], atol=1e-6)
    for suffix in (".obj", ".stl", ".ply", ".glb"):
        path = tmp_path / ("model" + suffix)
        meshes.primitive("Box").export(str(path))
        assert len(meshes.load_mesh(path).vertices) > 0


def test_invalid_files(tmp_path):
    with pytest.raises(ValueError):
        video.metadata(tmp_path / "missing.mp4")
    path = tmp_path / "broken.avi"
    path.write_text("not a video")
    with pytest.raises(ValueError):
        video.metadata(path)
    with pytest.raises(ValueError):
        meshes.load_mesh(tmp_path / "bad.txt")
    path = tmp_path / "broken.ply"
    path.write_text("not a mesh")
    with pytest.raises(ValueError):
        meshes.load_mesh(path)


def test_external_process_logs_errors_cancel(tmp_path):
    lines = []
    run_command([sys.executable, "-c", "print('hello')"], threading.Event(), lines.append, tmp_path)
    assert "hello" in lines and "Exit code: 0" in lines
    with pytest.raises(RuntimeError, match="code 7"):
        run_command([sys.executable, "-c", "raise SystemExit(7)"], threading.Event(), lines.append, tmp_path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(video.Cancelled):
        run_command([sys.executable, "-c", "import time; time.sleep(10)"], cancel, lines.append, tmp_path)
    with pytest.raises(ValueError, match="COLMAP"):
        detect(str(tmp_path / "missing.exe"))


def test_gltf_and_pointcloud_import(tmp_path):
    import trimesh
    exported = trimesh.exchange.gltf.export_gltf(trimesh.Scene(meshes.primitive("Box")))
    for name, contents in exported.items():
        (tmp_path / name).write_bytes(contents)
    assert len(meshes.load_mesh(tmp_path / "model.gltf").vertices) == 8
    cloud = trimesh.points.PointCloud(np.array([[0, 0, 0], [1, 2, 3], [3, 2, 1]]))
    path = tmp_path / "cloud.ply"
    cloud.export(str(path))
    record, loaded = meshes.model_record(tmp_path, path)
    assert record.vertices == 3 and record.faces == 0


@pytest.mark.parametrize("dense", [False, True])
def test_colmap_pipeline_contract_with_fake_runner(tmp_path, monkeypatch, dense):
    """Validate orchestration; this is explicitly not a real COLMAP reconstruction."""
    import trimesh
    from drone3d_studio.domain.models import Frame
    from drone3d_studio.reconstruction import colmap
    root = tmp_path / "p"
    store.create(root, "P")
    frames = []
    for i in range(3):
        path = root / "frames" / "accepted" / f"{i}.jpg"
        video.write_image(path, np.zeros((16, 16, 3), dtype=np.uint8))
        frames.append(Frame(index=i, time=i, path=store.reference(root, path), thumbnail="", blur=50, accepted=True))
    monkeypatch.setattr(colmap, "detect", lambda configured: "colmap.exe")
    commands = []
    def runner(args, cancel, log, cwd):
        commands.append(args)
        if "-h" in args:
            log("FeatureExtraction.use_gpu FeatureMatching.use_gpu")
        elif args[1] == "mapper":
            output = cwd / "sparse" / "0"
            output.mkdir()
            (output / "cameras.bin").write_bytes(b"fake")
        elif args[1] == "model_converter":
            trimesh.points.PointCloud([[0, 0, 0], [1, 1, 1]]).export(args[args.index("--output_path") + 1])
        elif args[1] == "poisson_mesher":
            meshes.primitive("Box").export(args[args.index("--output_path") + 1])
        log("Exit code: 0")
    monkeypatch.setattr(colmap, "run_command", runner)
    result = colmap.ColmapBackend(output="Dense mesh (CUDA)" if dense else "Sparse cloud (CPU)").run(root, frames, threading.Event(), lambda *a: None)
    assert result[0][0].origin == "COLMAP"
    assert result[0][0].faces == 0
    assert any("--FeatureExtraction.use_gpu" in command and command[-1] == "0" for command in commands)
    assert any("--FeatureMatching.use_gpu" in command and command[-1] == "0" for command in commands)
    if dense:
        assert result[1][0].faces > 0 and not result[0][0].visible
        assert [c[1] for c in commands[-4:]] == ["image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher"]


def test_colmap_environment_is_child_only(tmp_path, monkeypatch):
    from drone3d_studio.reconstruction import colmap
    import os
    binary = tmp_path / "bin"
    binary.mkdir()
    (tmp_path / "plugins").mkdir()
    monkeypatch.setenv("QT_PLUGIN_PATH", "pyside-plugins")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    real_popen = colmap.subprocess.Popen
    seen = []
    def popen(args, **kwargs):
        seen.append(kwargs["env"])
        return real_popen([sys.executable, "-c", "print('ok')"], **kwargs)
    monkeypatch.setattr(colmap.subprocess, "Popen", popen)
    colmap.run_command([str(binary / "colmap.exe"), "-h"], threading.Event(), lambda s: None, tmp_path)
    assert seen[0]["QT_PLUGIN_PATH"] == str(tmp_path / "plugins")
    assert "QT_QPA_PLATFORM" not in seen[0]
    assert os.environ["QT_PLUGIN_PATH"] == "pyside-plugins"


def test_inspection_preserves_measured_colors():
    original = meshes.primitive("Box")
    original.visual.vertex_colors = np.array([[20 + i, 100, 200, 255] for i in range(len(original.vertices))], dtype=np.uint8)
    inspection = original.copy()
    before = inspection.vertices.copy()
    meshes.transfer_vertex_colors(original, inspection, threading.Event())
    np.testing.assert_array_equal(inspection.visual.vertex_colors, original.visual.vertex_colors)
    np.testing.assert_array_equal(inspection.vertices, before)
