import threading
import numpy as np
import pytest
import trimesh
from drone3d_studio.domain.models import PipelineConfig, Project
from drone3d_studio.reconstruction.visible_surface import read_colmap_depth, grid_faces, build_surface
from drone3d_studio.reconstruction.geometry import CameraView
from drone3d_studio.services.video import write_image, Cancelled


def test_depth_binary_layout_and_corrupt_input(tmp_path):
    path=tmp_path/'depth.bin'
    expected=np.array([[1,2,3],[4,5,6]],dtype='<f4')
    path.write_bytes(b'3&2&1&'+expected.tobytes())
    np.testing.assert_array_equal(read_colmap_depth(path),expected)
    path.write_bytes(b'3&2&1&'+b'bad')
    with pytest.raises(ValueError,match='Truncated'):
        read_colmap_depth(path)
    path.write_bytes(b'broken')
    with pytest.raises(ValueError,match='header'):
        read_colmap_depth(path)


def test_depth_surface_keeps_holes_and_discontinuities():
    z=np.ones((5,5),dtype=float)*5
    valid=np.ones_like(z,dtype=bool)
    valid[2,2]=False
    faces=grid_faces(z,valid)
    assert 12 not in faces
    z[:,3:]=20
    faces=grid_faces(z,valid)
    assert np.all(np.ptp(z.ravel()[faces],axis=1)==0)


def test_single_straight_pass_builds_open_surface(tmp_path):
    # Fixed viewing direction, collinear translating centers: no orbit/multiple passes.
    views=[CameraView(f'{i}.jpg',1,np.eye(3),np.array([-i*.1,0.,0.]),[]) for i in range(3)]
    maps=[np.full((40,40),5,dtype=np.float32) for _ in views]
    for view in views:
        write_image(tmp_path/view.name,np.full((40,40,3),(50,100,150),dtype=np.uint8))
    out=tmp_path/'visible.ply'
    build_surface(views,maps,tmp_path,{1:(40,40,40,40,20,20)},out,2,threading.Event(),lambda *a:None)
    mesh=trimesh.load(out,process=False)
    assert len(mesh.faces)>100 and not mesh.is_watertight
    np.testing.assert_allclose(mesh.vertices[:,2],5)
    assert mesh.visual.vertex_colors[:,0].mean()==pytest.approx(150,abs=3)
    event=threading.Event();event.set()
    with pytest.raises(Cancelled):
        build_surface(views,maps,tmp_path,{1:(40,40,40,40,20,20)},out,2,event,lambda *a:None)


def test_defaults_preserve_existing_explicit_pipeline():
    assert Project(name='New').pipeline.capture_mode=='Single pass'
    assert Project.model_validate({'name':'Old','pipeline':{}}).pipeline.capture_mode=='General'
    assert PipelineConfig.model_validate_json(PipelineConfig(capture_mode='Single pass',horizontal_fov_deg=75).model_dump_json()).horizontal_fov_deg==75


def test_sample_budget_spans_whole_pass(tmp_path):
    from drone3d_studio.domain.models import AnalysisConfig
    from drone3d_studio.services.video import synthetic_video, analyze
    path=tmp_path/'pass.avi'
    synthetic_video(path)
    frames,_=analyze(path,tmp_path,AnalysisConfig(interval=.05,max_frames=5,blur_threshold=0,duplicate_threshold=0,cover_entire_video=True),threading.Event(),lambda *a:None)
    assert len(frames)==5
    assert frames[0].index==0 and frames[-1].index==47
    assert all(a.index<b.index for a,b in zip(frames,frames[1:]))


def test_single_pass_command_contract(tmp_path,monkeypatch):
    from drone3d_studio.reconstruction import colmap,visible_surface
    from drone3d_studio.domain.models import Frame
    from drone3d_studio.persistence import store
    root=tmp_path/'project'
    store.create(root,'Test')
    image=root/'frames'/'frame.jpg'
    write_image(image,np.full((16,16,3),100,dtype=np.uint8))
    frames=[Frame(index=i,time=i,path=str(image),thumbnail='',blur=100,accepted=True) for i in range(3)]
    monkeypatch.setattr(colmap,'detect',lambda _:'colmap.exe')
    calls=[]
    def runner(args,cancel,log,cwd):
        calls.append(args)
        if '-h' in args:
            log('FeatureExtraction.use_gpu FeatureMatching.use_gpu')
        elif args[1]=='mapper':
            p=cwd/'sparse'/'0';p.mkdir()
            (p/'cameras.bin').write_bytes(b'fixture')
        elif args[1]=='model_converter' and args[-1]=='PLY':
            trimesh.points.PointCloud([[0,0,1],[1,0,1]]).export(args[args.index('--output_path')+1])
    monkeypatch.setattr(colmap,'run_command',runner)
    def surface(dense,*args):
        path=dense/'visible-surface.ply'
        trimesh.Trimesh(vertices=[[0,0,1],[1,0,1],[0,1,1]],faces=[[0,1,2]],process=False).export(path)
        return path
    monkeypatch.setattr(visible_surface,'from_stereo',surface)
    result=colmap.ColmapBackend(output='Dense mesh (CUDA)',pipeline=PipelineConfig(capture_mode='Single pass',horizontal_fov_deg=90)).run(root,frames,threading.Event(),lambda *a:None)
    assert result[-1][0].faces==1 and not result[0][0].visible
    assert 'poisson_mesher' not in [c[1] for c in calls]
    matcher=next(c for c in calls if c[1]=='sequential_matcher' and '-h' not in c)
    assert matcher[matcher.index('--SequentialMatching.loop_detection')+1]=='0'
    mapper=next(c for c in calls if c[1]=='mapper')
    assert mapper[mapper.index('--Mapper.ba_refine_focal_length')+1]=='0'
    extraction=next(c for c in calls if c[1]=='feature_extractor' and '-h' not in c)
    params=extraction[extraction.index('--ImageReader.camera_params')+1]
    np.testing.assert_allclose(list(map(float,params.split(','))),[8,8,8,8])
