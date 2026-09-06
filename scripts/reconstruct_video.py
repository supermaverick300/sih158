"""Run the same real backend without a GUI; useful for reproducible diagnostics."""
import argparse
from pathlib import Path
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone3d_studio.domain.models import AnalysisConfig
from drone3d_studio.persistence import store
from drone3d_studio.services.video import metadata, analyze
from drone3d_studio.reconstruction.colmap import ColmapBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("project", type=Path)
    parser.add_argument("--colmap", default="")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--blur", type=float, default=35)
    args = parser.parse_args()
    root = args.project.resolve()
    project = store.create(root, args.video.stem)
    project.video = metadata(args.video)
    project.analysis = AnalysisConfig(interval=args.interval, max_frames=args.max_frames, blur_threshold=args.blur)
    project.settings.executable = args.colmap
    project.settings.reconstruction_output = "Sparse cloud (CPU)" if args.sparse else "Dense mesh (CUDA)"
    cancel = threading.Event()
    import signal
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    def log(value, message):
        print(message, flush=True)
    try:
        project.status = "Analysis running"
        store.save(root, project)
        project.frames, project.analysis_seconds = analyze(args.video, root, project.analysis, cancel, log)
        project.status = "Reconstruction running"
        store.save(root, project)
        result = ColmapBackend(args.colmap, project.settings.reconstruction_output).run(root, project.frames, cancel, log)
        project.models = [m for m, geometry in result]
        project.status = "Scene ready"
        project.reconstruction_status = "Succeeded — real COLMAP " + project.settings.reconstruction_output
        print([(m.name, m.vertices, m.faces) for m in project.models])
    except Exception as exc:
        project.status = "Cancelled" if cancel.is_set() else "Failed"
        project.reconstruction_status = str(exc)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.save(root, project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
