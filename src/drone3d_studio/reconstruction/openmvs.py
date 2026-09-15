"""COLMAP workspace -> OpenMVS -> textured OBJ/atlas -> Blender GLB LODs."""
import json
import shutil
import time
from pathlib import Path
from drone3d_studio.persistence.store import atomic_write
from drone3d_studio.services.meshes import model_record
from drone3d_studio.services.video import Cancelled

APPS = ('InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh')


def installation(config):
    root = Path(__file__).resolve().parents[3]
    directory = Path(config.openmvs_directory) if config.openmvs_directory else root / 'output/tools/OpenMVS/vc17/x64/Release'
    result = {}
    missing = []
    for app in APPS:
        path = directory / (app + '.exe')
        found = str(path.resolve()) if path.is_file() else shutil.which(app) if not config.openmvs_directory else None
        if not found:
            missing.append(app + '.exe')
        else:
            result[app] = found
    blender = config.blender_executable or shutil.which('blender')
    if not blender:
        candidates = list((root / 'output/tools/Blender').glob('*/blender.exe'))
        candidates += list(Path('C:/Program Files/Blender Foundation').glob('Blender */blender.exe'))
        blender = str(sorted(candidates)[-1]) if candidates else ''
    if not blender or not Path(blender).is_file():
        missing.append('blender.exe')
    if missing:
        raise ValueError('Textured pipeline tools missing: ' + ', '.join(missing) + '. Set OpenMVS folder and Blender executable in Settings. See docs/OPENMVS_PIPELINE.md.')
    result['Blender'] = str(Path(blender).resolve())
    return result


def run_pipeline(root, dense, work, config, cancel, log, runner):
    tools = installation(config)
    work.mkdir(parents=True, exist_ok=False)
    report = {'status': 'Running', 'stages': [], 'tools': tools, 'colmap_workspace': str(dense),
              'units': 'ENU metres' if config.align_gps else 'Arbitrary SfM units'}
    def stage(name, args, expected):
        if cancel.is_set():
            raise Cancelled('Textured pipeline cancelled')
        entry = {'name': name, 'command': args, 'status': 'Running'}
        report['stages'].append(entry)
        atomic_write(work / 'textured-report.json', json.dumps(report, indent=2))
        start = time.perf_counter()
        log('Running ' + name)
        try:
            runner(args, cancel, log, work)
            for path in expected:
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f'{name} did not produce {path.name}')
            entry['status'] = 'Succeeded'
        except Exception:
            entry['status'] = 'Cancelled' if cancel.is_set() else 'Failed'
            raise
        finally:
            entry['seconds'] = time.perf_counter() - start
    try:
        scene = work / 'scene.mvs'
        dense_scene = work / 'scene_dense.mvs'
        mesh_scene = work / 'scene_mesh.mvs'
        refined = work / 'scene_refined.mvs'
        textured = work / 'scene_textured.mvs'
        def command(app, *args):
            return [tools[app], *map(str, args), '--max-threads', '4']
        stage('InterfaceCOLMAP', command('InterfaceCOLMAP', '-i', dense, '-o', scene, '--image-folder', dense / 'images'), [scene])
        stage('DensifyPointCloud', command('DensifyPointCloud', '-i', scene, '-o', dense_scene, '--resolution-level', '1'), [dense_scene, work / 'scene_dense.ply'])
        # Interface-format archives keep cameras in scene_dense.mvs and meshes
        # in separate PLYs. Do not assume the mesher writes another .mvs archive.
        stage('ReconstructMesh', command('ReconstructMesh', '-i', dense_scene, '-p', work / 'scene_dense.ply', '-o', mesh_scene, '--close-holes', '0'), [work / 'scene_mesh.ply'])
        stage('RefineMesh', command('RefineMesh', '-i', dense_scene, '-m', work / 'scene_mesh.ply', '-o', refined, '--resolution-level', '1'), [work / 'scene_refined.ply'])
        # This Windows build produced black atlas patches with seam leveling.
        # Retain measured image colors, accepting visible exposure seams.
        stage('TextureMesh', command('TextureMesh', '-i', dense_scene, '-m', work / 'scene_refined.ply', '-o', textured, '--export-type', 'obj', '--max-texture-size', '4096', '--close-holes', '0', '--global-seam-leveling', '0', '--local-seam-leveling', '0'), [work / 'scene_textured.obj', work / 'scene_textured.mtl'])
        atlases = list(work.glob('scene_textured*.png')) + list(work.glob('scene_textured*.jpg'))
        if not atlases:
            raise RuntimeError('TextureMesh produced no texture atlas images')
        script = Path(__file__).resolve().parents[3] / 'scripts/blender_export_lods.py'
        paths = [work / f'scene_{level}.glb' for level in ('high', 'medium', 'low')]
        stage('Blender GLB LOD export', [tools['Blender'], '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(script), '--', str(work / 'scene_textured.obj'), str(work)], paths + [work / 'lod-report.json'])
        record, mesh = model_record(root, paths[-1], 'OpenMVS')
        if not record.faces:
            raise RuntimeError('Exported low LOD contains no surface')
        record.name = 'OpenMVS textured surface · low LOD'
        report['outputs'] = {level: str(path) for level, path in zip(('high', 'medium', 'low'), paths)}
        report['atlases'] = [str(path) for path in atlases]
        report['status'] = 'Succeeded'
        return record, mesh
    except Exception as exc:
        report['status'] = 'Cancelled' if isinstance(exc, Cancelled) else 'Failed'
        report['error'] = str(exc)
        raise
    finally:
        atomic_write(work / 'textured-report.json', json.dumps(report, indent=2))
