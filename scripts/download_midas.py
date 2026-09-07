"""Download a pinned snapshot of the MiDaS ONNX mirror, recording provenance."""
from pathlib import Path
import hashlib
import json
from huggingface_hub import HfApi, snapshot_download

if __name__ == '__main__':
    repository = 'Heliosoph/midas-small-onnx'
    destination = Path(__file__).resolve().parents[1] / 'models' / 'midas-small-onnx'
    revision = HfApi().model_info(repository).sha
    snapshot_download(repository, revision=revision, local_dir=destination,
                      allow_patterns=['*.onnx', 'README.md', 'LICENSE*'])
    weights = destination / 'midas_v21_small_256.onnx'
    manifest = {'repository': repository, 'revision': revision,
                'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}
    (destination / 'download-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(manifest)
