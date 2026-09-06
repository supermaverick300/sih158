"""Explicit one-time download; normal reconstruction never downloads files."""
from pathlib import Path
import hashlib
import json
from huggingface_hub import HfApi, snapshot_download

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "models" / "depth-anything-v2-small"
    repository = "depth-anything/Depth-Anything-V2-Small-hf"
    revision = HfApi().model_info(repository).sha
    snapshot_download(repo_id=repository, revision=revision, local_dir=destination,
                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md"])
    weights = destination / "model.safetensors"
    manifest = {"repository": repository, "revision": revision, "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    (destination / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded to {destination}\nRevision: {revision}")
