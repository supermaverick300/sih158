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
