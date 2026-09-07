"""Local MiDaS ONNX inference and explicitly non-metric image relief.

This is a single-frame depth visualization, not fused video photogrammetry.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from drone3d_studio.services.video import Cancelled, write_image
from drone3d_studio.services.meshes import model_record
from drone3d_studio.persistence.store import resolve, atomic_write


class MidasPredictor:
    def __init__(self, path):
        if not Path(path).is_file():
            raise ValueError('MiDaS weights missing. Run scripts\\download_midas.py or set the MiDaS ONNX path in Settings.')
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError('Install requirements-midas.txt, then restart the app.') from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        if self.input.shape != [1, 3, 256, 256]:
            raise ValueError('Expected the MiDaS v2.1 Small 256 ONNX export.')

    def predict(self, rgb):
        # Upstream tf/run_onnx.py: RGB / 255; normalization is in the ONNX graph.
        tensor = cv2.resize(rgb.astype(np.float32) / 255, (256, 256), interpolation=cv2.INTER_CUBIC)
        prediction = self.session.run(None, {self.input.name: np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])})[0]
        prediction = np.asarray(prediction).squeeze()
        if prediction.shape != (256, 256) or not np.isfinite(prediction).all():
            raise ValueError('MiDaS returned invalid depth.')
        return cv2.resize(prediction, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)


def relief_mesh(rgb, relative, max_side=256):
    """Orthographic colored relief; normalized inverse depth extrudes toward viewer.

    Full grid intentionally connects depth boundaries like an image relief.
    Coordinates are presentation units, not camera-calibrated distances.
    """
    if relative.shape != rgb.shape[:2] or not np.isfinite(relative).all():
        raise ValueError('Invalid relative-depth map')
    height, width = rgb.shape[:2]
    scale = min(1, max_side / max(height, width))
    w, h = max(2, round(width * scale)), max(2, round(height * scale))
    depth = cv2.resize(relative.astype(np.float32), (w, h))
    low, high = np.percentile(relative, [2, 98])
    if high - low < 1e-6:
        raise ValueError('Depth prediction has no usable variation')
    near = np.clip((depth - low) / (high - low), 0, 1)
    x, z = np.meshgrid(np.linspace(-width / height, width / height, w), np.linspace(1, -1, h))
    vertices = np.column_stack((x.ravel(), (-.7 * near).ravel(), z.ravel()))
    indices = np.arange(w * h).reshape(h, w)
    a, b, c, d = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel(), indices[1:, :-1].ravel(), indices[1:, 1:].ravel()
    faces = np.concatenate((np.column_stack((a, c, b)), np.column_stack((b, c, d))))
    colors = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)


def run_relief(root, frames, config, work, cancel, progress):
    accepted = [f for f in frames if f.accepted]
    if not accepted:
        raise ValueError('Analyze footage and select an accepted frame first.')
    frame = max(accepted, key=lambda f: (f.quality_score, f.blur))
    if cancel.is_set():
        raise Cancelled('MiDaS cancelled')
    path = resolve(root, frame.path)
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Cannot read frame: {path}')
    # Bound processing memory while retaining the original source externally.
    ratio = min(1, 1280 / max(bgr.shape[:2]))
    bgr = cv2.resize(bgr, (round(bgr.shape[1] * ratio), round(bgr.shape[0] * ratio)))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    weights = Path(config.midas_weights_path)
    if not weights.is_absolute():
        weights = Path(__file__).resolve().parents[3] / weights
    progress(10, f'MiDaS: estimating frame at {frame.time:.2f}s; single-frame non-metric relief')
    relative = MidasPredictor(weights).predict(rgb)
    if cancel.is_set():
        raise Cancelled('MiDaS cancelled')
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / 'relative-depth.npy', relative)
    write_image(work / 'source.jpg', bgr)
    normalized = cv2.normalize(relative, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    write_image(work / 'depth-preview.jpg', cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO))
    full = relief_mesh(rgb, relative)
    full.export(work / 'midas-relief.ply')
    # A complete coarser grid avoids dropping random triangles in the CPU viewer.
    aspect = max(rgb.shape[:2]) / min(rgb.shape[:2])
    preview = relief_mesh(rgb, relative, max_side=min(256, int(np.sqrt(6000 * aspect))))
    preview.export(work / 'midas-relief-preview.ply')
    atomic_write(work / 'relief-report.json', json.dumps({
        'source_frame': str(path), 'time_seconds': frame.time, 'weights': str(weights),
        'type': 'Estimated single-frame 2.5D relief', 'units': 'Normalized presentation units',
        'projection': 'Orthographic image grid with normalized inverse-depth extrusion',
        'limitations': 'Not a fused video model; no metric scale, unseen surfaces or segmentation. Depth boundaries are stretched connections.',
        'vertices': len(full.vertices), 'faces': len(full.faces)}, indent=2))
    record, mesh = model_record(root, work / 'midas-relief-preview.ply', 'MiDaS estimated relief')
    record.name = f'MiDaS 2.5D relief · frame {frame.time:.2f}s · estimated'
    progress(100, f'Filled MiDaS relief: {len(full.faces):,} full faces. Files: {work}')
    return [(record, mesh)]
