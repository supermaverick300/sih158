import json
import threading
import numpy as np
import pytest
from drone3d_studio.domain.models import PipelineConfig, Frame
from drone3d_studio.reconstruction import midas, colmap
from drone3d_studio.services.video import write_image, Cancelled


def test_relief_is_complete_colored_grid_with_near_points_forward():
    rgb = np.zeros((30, 50, 3), dtype=np.uint8)
    rgb[:, :, 0] = 230
    depth = np.tile(np.linspace(0, 1, 50), (30, 1))
    mesh = midas.relief_mesh(rgb, depth)
    assert len(mesh.faces) == 2 * 29 * 49
    assert np.all(mesh.visual.vertex_colors[:, 0] == 230)
    assert mesh.vertices[49, 1] < mesh.vertices[0, 1]
    assert not mesh.is_watertight
    with pytest.raises(ValueError, match='variation'):
        midas.relief_mesh(rgb, np.ones((30, 50)))


def test_relief_bypasses_colmap_and_reports_nonmetric(tmp_path, monkeypatch):
    image = tmp_path / 'image.jpg'
    write_image(image, np.full((40, 60, 3), 100, dtype=np.uint8))
    frames = [Frame(index=0, time=0, path=str(image), thumbnail='', blur=100, accepted=True)]
    class Predictor:
        def __init__(self, path): pass
        def predict(self, rgb): return np.tile(np.linspace(0, 1, 60), (40, 1))
    monkeypatch.setattr(midas, 'MidasPredictor', Predictor)
    monkeypatch.setattr(colmap, 'detect', lambda _: pytest.fail('Relief must not launch COLMAP'))
    backend = colmap.ColmapBackend(pipeline=PipelineConfig(depth_method='MiDaS ONNX relief'))
    weights = tmp_path / 'fixture.onnx'
    weights.write_bytes(b'mocked weights')
    backend.pipeline.midas_weights_path = str(weights)
    result = backend.run(tmp_path, frames, threading.Event(), lambda *a: None)
    assert result[0][0].origin == 'MiDaS estimated relief'
    assert result[0][0].faces > 0
    report = json.loads((backend.work / 'pipeline-report.json').read_text())
    assert report['units'] == 'Normalized presentation units; not metric'
    assert (backend.work / 'source.jpg').is_file()
    cancel = threading.Event(); cancel.set()
    with pytest.raises(Cancelled):
        backend.run(tmp_path, frames, cancel, lambda *a: None)
    backend.pipeline.align_gps = True
    with pytest.raises(ValueError, match='georeferenced'):
        backend.run(tmp_path, frames, threading.Event(), lambda *a: None)


def test_prediction_cache_invalidates_content_and_recovers_corruption(tmp_path, monkeypatch):
    calls = []
    class Predictor:
        def __init__(self, path): pass
        def predict(self, rgb):
            calls.append(1)
            return np.tile(np.linspace(0, 1, 20), (10, 1)).astype(np.float32)
    monkeypatch.setattr(midas, 'MidasPredictor', Predictor)
    weights = tmp_path / 'model.onnx'; weights.write_bytes(b'one')
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    cache = tmp_path / 'cache'
    first, hit, key = midas.cached_prediction(rgb, weights, cache, threading.Event())
    assert not hit
    second, hit, _ = midas.cached_prediction(rgb, weights, cache, threading.Event())
    np.testing.assert_array_equal(first, second)
    assert hit and len(calls) == 1
    weights.write_bytes(b'two')
    assert not midas.cached_prediction(rgb, weights, cache, threading.Event())[1]
    rgb[0, 0, 0] = 20
    _, hit, key = midas.cached_prediction(rgb, weights, cache, threading.Event())
    assert not hit
    (cache / (key + '.npy')).write_bytes(b'corrupt')
    assert not midas.cached_prediction(rgb, weights, cache, threading.Event())[1]
    assert len(calls) == 4
    (cache / (key + '.npy')).write_bytes(b'')
    assert not midas.cached_prediction(rgb, weights, cache, threading.Event())[1]


def test_textured_glb_preserves_image_and_uv(tmp_path):
    from drone3d_studio.services.meshes import load_mesh
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[:16, :32, 0] = 255
    rgb[16:, 32:, 2] = 255
    depth = np.tile(np.linspace(0, 1, 64), (32, 1))
    mesh = midas.texture_relief(midas.relief_mesh(rgb, depth, 16), rgb)
    target = tmp_path / 'mesh.glb'
    mesh.export(target)
    restored = load_mesh(target)
    assert len(restored.faces) == len(mesh.faces)
    np.testing.assert_allclose(restored.visual.uv, mesh.visual.uv, atol=1e-6)
    pixels = np.asarray(restored.visual.material.baseColorTexture)
    np.testing.assert_array_equal(pixels[:, :, :3], rgb)


def test_textured_viewport_preserves_orientation(qtbot):
    from drone3d_studio.viewer.canvas import SceneCanvas
    from drone3d_studio.domain.models import Model
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[:16, :32, 0] = 255
    rgb[16:, 32:, 2] = 255
    depth = np.tile(np.linspace(0, 1, 64), (32, 1))
    mesh = midas.texture_relief(midas.relief_mesh(rgb, depth, 16), rgb)
    record = Model(name='Texture', path='test.glb', format='GLB', vertices=len(mesh.vertices), faces=len(mesh.faces), origin='MiDaS estimated relief')
    canvas = SceneCanvas(); qtbot.addWidget(canvas); canvas.resize(800, 600)
    canvas.set_scene([record], {record.id: mesh}); canvas.frame_all(); canvas.view('Front'); canvas.show()
    qtbot.waitExposed(canvas)
    image = canvas.grab().toImage()
    for point, channel in (([-1, -.175, .5], 0), ([1, -.525, -.5], 2)):
        position, _ = canvas.project(np.asarray([point]))
        color = image.pixelColor(*map(int, position[0]))
        values = [color.red(), color.green(), color.blue()]
        assert values[channel] > 220 and sum(values) - values[channel] < 30
