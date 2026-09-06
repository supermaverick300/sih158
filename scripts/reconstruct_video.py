"""Run the same real backend without a GUI; useful for reproducible diagnostics."""
import argparse
from pathlib import Path
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone3d_studio.domain.models import AnalysisConfig, PipelineConfig
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
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--min-features", type=int, default=40)
    parser.add_argument("--max-clipped", type=float, default=.35)
    parser.add_argument("--ai", action="store_true", help="Depth Anything V2 colored cloud; local weights required")
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--gps-spacing", type=float, default=0)
    parser.add_argument("--telemetry-offset", type=float, default=0)
    parser.add_argument("--align-gps", action="store_true")
    parser.add_argument("--general", action="store_true", help="Use the previous general/Poisson workflow")
    parser.add_argument("--horizontal-fov", type=float, default=0, help="Known horizontal FOV of rectified video; 0 estimates intrinsics")
    args = parser.parse_args()
    info = metadata(args.video)
    root = args.project.resolve()
    project = store.create(root, args.video.stem)
    project.video = info
    project.analysis = AnalysisConfig(interval=args.interval, max_frames=args.max_frames, blur_threshold=args.blur, sampling_mode="Adaptive" if args.adaptive else "Fixed", min_features=args.min_features, max_clipped_fraction=args.max_clipped, gps_spacing_m=args.gps_spacing, telemetry_offset_s=args.telemetry_offset)
    project.pipeline = PipelineConfig(capture_mode="General" if args.general else "Single pass", horizontal_fov_deg=args.horizontal_fov, depth_method="Depth Anything V2" if args.ai else "COLMAP stereo", align_gps=args.align_gps)
    project.analysis.cover_entire_video = not args.general
    if args.telemetry:
        from drone3d_studio.services.telemetry import load_csv
        project.telemetry = load_csv(args.telemetry)
        project.telemetry_source = str(args.telemetry.resolve())
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
        project.frames, project.analysis_seconds = analyze(args.video, root, project.analysis, cancel, log, project.telemetry)
        project.status = "Reconstruction running"
        store.save(root, project)
        result = ColmapBackend(args.colmap, project.settings.reconstruction_output, project.pipeline).run(root, project.frames, cancel, log)
        project.models = [m for m, geometry in result]
        project.status = "Scene ready"
        project.reconstruction_status = "Succeeded — " + ("Depth Anything V2 colored cloud" if args.ai else "COLMAP " + project.settings.reconstruction_output) + (" · GPS-aligned ENU meters" if args.align_gps else " · arbitrary SfM units")
        if not args.general:
            project.reconstruction_status = "Succeeded — single-pass " + ("sparse cloud" if args.sparse and not args.ai else "visible surface") + "; unseen surfaces not reconstructed" + (" · ENU meters" if args.align_gps else " · arbitrary SfM units")
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
