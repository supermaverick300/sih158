import json
import threading
from pathlib import Path
import numpy as np
import pytest
import trimesh
from PIL import Image
from drone3d_studio.domain.models import PipelineConfig
from drone3d_studio.reconstruction import openmvs


def test_missing_tools_fail_before_processing(tmp_path):
    config = PipelineConfig(openmvs_directory=str(tmp_path), blender_executable=str(tmp_path / 'missing.exe'))
    with pytest.raises(ValueError, match='InterfaceCOLMAP.exe.*blender.exe'):
        openmvs.installation(config)


def fake_tools(monkeypatch):
    monkeypatch.setattr(openmvs, 'installation', lambda _: {app: app + '.exe' for app in (*openmvs.APPS, 'Blender')})


def fake_runner(calls):
    def run(args, cancel, log, cwd):
        name = Path(args[0]).stem
        calls.append(name)
        if name == 'Blender':
            mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]], process=False)
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=[[0,0],[1,0],[0,1]], image=Image.fromarray(np.full((4,4,3),100,np.uint8)))
            for level in ('high','medium','low'):
                mesh.export(cwd / f'scene_{level}.glb')
            (cwd / 'lod-report.json').write_text('{}')
        else:
            output = Path(args[args.index('-o') + 1])
            if name in ('InterfaceCOLMAP', 'DensifyPointCloud'):
                output.write_bytes(b'mocked scene')
            if name in ('DensifyPointCloud','ReconstructMesh','RefineMesh'):
                output.with_suffix('.ply').write_bytes(b'mocked points or mesh')
            if name == 'TextureMesh':
                for suffix in ('.obj','.mtl','.png'):
                    output.with_suffix(suffix).write_bytes(b'mocked atlas fixture')
    return run


def test_stage_order_lods_and_no_procedural_fallback(tmp_path, monkeypatch):
    fake_tools(monkeypatch)
    calls=[]
    work=tmp_path/'pipeline'
    record, mesh = openmvs.run_pipeline(tmp_path,tmp_path/'dense',work,PipelineConfig(),threading.Event(),lambda _:None,fake_runner(calls))
    assert calls == [*openmvs.APPS, 'Blender']
    assert record.origin == 'OpenMVS' and record.path.endswith('scene_low.glb') and record.faces == 1
    assert mesh.visual.uv is not None
    report=json.loads((work/'textured-report.json').read_text())
    assert report['status']=='Succeeded' and set(report['outputs'])=={'high','medium','low'}
    assert all(s['status']=='Succeeded' for s in report['stages'])
    assert report['units']=='Arbitrary SfM units'


def test_stage_failure_retains_outputs_and_stops(tmp_path, monkeypatch):
    fake_tools(monkeypatch)
    calls=[]
    normal=fake_runner(calls)
    def runner(args,*rest):
        if Path(args[0]).stem == 'RefineMesh':
            raise RuntimeError('refinement failed')
        normal(args,*rest)
    work=tmp_path/'failed'
    with pytest.raises(RuntimeError, match='refinement failed'):
        openmvs.run_pipeline(tmp_path,tmp_path/'dense',work,PipelineConfig(),threading.Event(),lambda _:None,runner)
    assert (work/'scene_mesh.ply').is_file()
    assert 'TextureMesh' not in calls
    report=json.loads((work/'textured-report.json').read_text())
    assert report['status']=='Failed' and report['stages'][-1]['status']=='Failed'


def test_exit_zero_without_output_is_failure(tmp_path, monkeypatch):
    fake_tools(monkeypatch)
    with pytest.raises(RuntimeError, match='did not produce scene.mvs'):
        openmvs.run_pipeline(tmp_path,tmp_path/'dense',tmp_path/'empty',PipelineConfig(),threading.Event(),lambda _:None,lambda *args:None)


@pytest.mark.parametrize('stereo_first', [False, True])
def test_colmap_routes_to_textured_pipeline(tmp_path, monkeypatch, stereo_first):
    from drone3d_studio.reconstruction import colmap
    from drone3d_studio.persistence import store
    from drone3d_studio.domain.models import Frame, Model
    from drone3d_studio.services.video import write_image
    fake_tools(monkeypatch)
    root=tmp_path/'project'; store.create(root,'Textured')
    image=root/'source.jpg'; write_image(image,np.full((32,32,3),100,np.uint8))
    frames=[Frame(index=i,time=i,path=str(image),thumbnail='',blur=100,accepted=True) for i in range(3)]
    monkeypatch.setattr(colmap,'detect',lambda _:'colmap.exe')
    commands=[]
    def runner(args,cancel,log,cwd):
        commands.append(args[1])
        if args[1]=='mapper':
            p=cwd/'sparse/0';p.mkdir();(p/'cameras.bin').write_bytes(b'fixture')
        if args[1]=='model_converter':
            trimesh.points.PointCloud([[0,0,0],[1,0,0],[0,1,0]]).export(args[args.index('--output_path')+1])
    monkeypatch.setattr(colmap,'run_command',runner)
    seen=[]
    def pipeline(root,dense,work,config,*rest):
        seen.append(config)
        assert dense.is_dir() and 'image_undistorter' in commands
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
        return Model(name='Textured',path='scene_low.glb',format='GLB',vertices=3,faces=1,origin='OpenMVS'),mesh
    monkeypatch.setattr(openmvs,'run_pipeline',pipeline)
    config=PipelineConfig(depth_method='OpenMVS textured mesh',colmap_stereo_first=stereo_first)
    result=colmap.ColmapBackend(output='Sparse cloud (CPU)',pipeline=config).run(root,frames,threading.Event(),lambda *a:None)
    assert seen==[config] and result[-1][0].origin=='OpenMVS' and not result[0][0].visible
    assert ('patch_match_stereo' in commands)==stereo_first
    assert 'poisson_mesher' not in commands
