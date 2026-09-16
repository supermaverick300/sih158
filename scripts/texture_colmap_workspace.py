"""Continue a saved undistorted COLMAP workspace into a new textured project."""
import argparse
import sys
import signal
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from drone3d_studio.domain.models import PipelineConfig
from drone3d_studio.persistence import store
from drone3d_studio.reconstruction.colmap import run_command
from drone3d_studio.reconstruction.openmvs import installation, run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    parser.add_argument('project', type=Path)
    parser.add_argument('--openmvs-dir', default='')
    parser.add_argument('--blender', default='')
    args = parser.parse_args()
    config = PipelineConfig(depth_method='OpenMVS textured mesh', openmvs_directory=args.openmvs_dir, blender_executable=args.blender)
    installation(config)
    dense = args.workspace.resolve()
    if not (dense / 'images').is_dir() or not (dense / 'sparse').is_dir():
        raise ValueError('Expected a COLMAP undistorted workspace with images/ and sparse/.')
    root = args.project.resolve()
    project = store.create(root, 'OpenMVS textured reconstruction')
    project.pipeline = config
    project.description = f'Reuses existing COLMAP cameras/images from {dense}. Geographic provenance unverified.'
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    try:
        project.status = 'Reconstruction running'; store.save(root, project)
        with (root / 'logs/openmvs.log').open('w', encoding='utf-8') as stream:
            def log(message):
                print(message, flush=True); stream.write(message + '\n'); stream.flush()
            result = run_pipeline(root, dense, root / 'reconstruction/openmvs', config, cancel, log, run_command)
            from drone3d_studio.reconstruction.confidence import attach_run
            try:
                attach_run(root,root/'reconstruction/openmvs',[],[result],cancel,log)
            except Exception as exc:
                if cancel.is_set():
                    raise
                log(f'Confidence unavailable: {exc}. Completed reconstruction retained.')
        project.models = [result[0]]
        project.status = 'Scene ready'
        project.reconstruction_status = 'Succeeded — textured OpenMVS mesh and three GLBs; original scale, georeference unverified'
    except Exception as exc:
        project.status = 'Cancelled' if cancel.is_set() else 'Failed'
        project.reconstruction_status = str(exc)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.save(root, project)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
