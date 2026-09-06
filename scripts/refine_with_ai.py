"""Reuse a COLMAP undistorted workspace; no repeat of feature matching/SfM."""
import argparse
from pathlib import Path
import shutil
import signal
import sys
import threading
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone3d_studio.domain.models import PipelineConfig
from drone3d_studio.persistence import store
from drone3d_studio.reconstruction.colmap import detect, run_command
from drone3d_studio.reconstruction.ai_depth import run_depth_pipeline
from drone3d_studio.services.meshes import model_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("dense_workspace", type=Path)
    parser.add_argument("--colmap", default="")
    parser.add_argument("--device", choices=["CPU", "CUDA", "Auto"], default="CPU")
    parser.add_argument("--voxel-size", type=float, default=.05)
    args = parser.parse_args()
    root = args.project.resolve()
    project = store.load(root)
    source = args.dense_workspace.resolve()
    if not (source / "sparse" / "images.bin").is_file():
        parser.error("Expected an undistorted COLMAP workspace with sparse/images.bin and images/")
    work = root / "reconstruction" / ("ai-refine-" + uuid4().hex[:12])
    work.mkdir(parents=True)
    shutil.copytree(source / "images", work / "images")
    text_model = work / "text-model"
    text_model.mkdir()
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    with (work / "processing.log").open("w", encoding="utf-8") as logfile:
        def log(message):
            print(message, flush=True)
            logfile.write(message + "\n")
            logfile.flush()
        run_command([detect(args.colmap), "model_converter", "--input_path", str(source / "sparse"), "--output_path", str(text_model), "--output_type", "TXT"], cancel, log, work)
        # This utility does not have verified telemetry provenance for its input.
        config = PipelineConfig(depth_method="Depth Anything V2", device=args.device, fusion_voxel_size=args.voxel_size)
        weights = Path(__file__).resolve().parents[1] / config.weights_path
        cloud = run_depth_pipeline(work, text_model, weights, config, cancel, lambda _, message: log(message))
        record, geometry = model_record(root, cloud, "Depth Anything V2")
        record.name = "AI fused cloud · SfM units (unverified georeference)"
        for old in project.models:
            old.visible = False
        project.models.append(record)
        project.pipeline = config
        project.status = "Scene ready"
        project.reconstruction_status = f"Succeeded — Depth Anything V2: {record.vertices:,} colored points · georeference unverified"
        store.save(root, project)
        log(project.reconstruction_status)
        log(str(cloud))


if __name__ == "__main__":
    main()
