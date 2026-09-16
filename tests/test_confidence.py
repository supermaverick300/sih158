import json
import struct
import threading
import numpy as np
import pytest
from drone3d_studio.domain.models import Frame, GPSFix
from drone3d_studio.reconstruction import confidence as c


def frames(n=10):
    return [Frame(index=i,time=i,path=f'{i}.jpg',thumbnail='',accepted=True,blur=100,features=300,
                  quality_score=80,clipped_fraction=.02,motion_px=4) for i in range(n)]


def evidence(n=8):
    cameras={i:dict(name=f'{i:06d}.jpg',center=np.array([i*.1,0,0]),R=np.eye(3),keypoints=10,observations=8) for i in range(n)}
    return dict(cameras=cameras,xyz=np.array([[0.,0,2],[1,0,2],[0,1,2]]),errors=np.array([.5,1.,2.]),
                counts=np.array([2,4,6]),ids=np.array([11,20,90]),tracks=[np.array(list(cameras))]*3)


def test_registration_ratio_and_raw_statistics():
    report=c.build_report(evidence(),frames())
    assert report['components']['registration']['ratio']==.8
    assert report['components']['reprojection']['median_px']==1
    assert report['components']['multi_view']['mean_observations']==4
    assert report['coverage']['triangulated_keypoint_fraction']==.8


def test_missing_depth_renormalizes_instead_of_zero():
    components={k:dict(score=.8) for k in c.DEFAULT.weights}
    components['depth_consistency']['score']=None
    overall=c.combine(components)
    assert overall['score']==pytest.approx(80)
    assert sum(overall['effective_weights'].values())==pytest.approx(1)
    assert 'depth_consistency' not in overall['effective_weights']
    assert overall['available_weight']==pytest.approx(.8)


def test_all_missing_and_frame_quality_alone_are_not_reconstruction():
    assert c.combine({})['score'] is None
    assert c.combine({'frame_quality':dict(score=1)})['score'] is None


@pytest.mark.parametrize('value,label',[(0,'LOW'),(39.9,'LOW'),(40,'LIMITED'),(65,'GOOD'),(85,'HIGH'),(100,'HIGH')])
def test_labels(value,label):
    assert c.combine({'registration':dict(score=value/100)})['label']==label


def test_config_ranges_and_weights():
    assert c.error_score(.5)==1
    assert c.error_score(4)==0
    config=c.ConfidenceConfig(reprojection_good_px=1,reprojection_weak_px=3)
    assert c.error_score(2,config)==.5
    assert c.support_score(1)==0 and c.support_score(5)==1
    with pytest.raises(ValueError):
        c.ConfidenceConfig(weights={'registration':.5})


def test_gps_never_changes_geometry_score():
    first=c.build_report(evidence(),frames())
    tagged=frames()
    tagged[0].gps=GPSFix(time=0,latitude=10,longitude=20,altitude_m=30)
    second=c.build_report(evidence(),tagged,alignment={'status':'Aligned','camera_residuals_m':{'a':1,'b':3}})
    assert first['overall']==second['overall']
    assert first['metric_status']['scale']=='Arbitrary/unverified units'
    assert second['metric_status']['median_residual_m']==2
    assert not any('GPS' in w for w in first['warnings'])


def test_ai_provenance_survives_high_score():
    report=c.build_report(evidence(),frames(),provenance='AI_ASSISTED')
    assert report['provenance']['model_class']=='AI_ASSISTED'
    assert report['provenance']['ai_assisted_fraction'] is None
    assert any('AI-assisted' in w for w in report['warnings'])


def test_warnings_and_deterministic_explanation_json():
    data=evidence(2); data['errors'][:]=5
    report=c.build_report(data,frames(),depth={'score':.1})
    assert any('20%' in w for w in report['warnings'])
    assert any('High reprojection' in w for w in report['warnings'])
    assert any('Depth disagreement' in w for w in report['warnings'])
    assert c.explain(report)==c.explain(report)
    loaded=json.loads(json.dumps(report,allow_nan=False))
    assert c.explain(loaded)==report['explanation']


def test_unknown_denominator_and_legacy_frame_score():
    report=c.build_report(evidence(),[])
    assert report['components']['registration']['score'] is None
    legacy=Frame(index=0,time=0,path='a',thumbnail='',blur=50,accepted=True)
    assert c.capture_suitability([legacy])['score'] is None


def write_sparse(folder):
    folder.mkdir()
    with (folder/'images.bin').open('wb') as file:
        file.write(struct.pack('<Q',2))
        for i in (7,42):
            file.write(struct.pack('<i7di',i,1,0,0,0,-i*.01,0,0,1))
            file.write(f'{i:06d}.jpg'.encode()+b'\0')
            file.write(struct.pack('<Qddq',1,10,20,100))
    with (folder/'points3D.bin').open('wb') as file:
        file.write(struct.pack('<Q',1))
        file.write(struct.pack('<Q3d3Bd',100,0,0,2,200,100,50,.75))
        file.write(struct.pack('<Q4i',2,7,0,42,0))


def test_binary_tracks_grid_and_far_regions_are_unknown(tmp_path):
    folder=tmp_path/'sparse';write_sparse(folder)
    data=c.read_sparse(folder,threading.Event())
    assert len(data['cameras'])==2
    assert list(data['counts'])==[2] and list(data['ids'])==[100]
    dest=tmp_path/'metadata'
    path=c.inspect_component(folder,None,frames(2),dest,threading.Event())
    report,grid=c.load_sidecar(path)
    scores,support=c.sample_grid(grid,np.array([[0,0,2],[100,100,100]]))
    assert np.isfinite(scores[0]) and support[0]==2
    assert np.isnan(scores[1])
    assert report['components']['depth_consistency']['score'] is None
    assert len(grid['scores'])==32**3


def test_truncated_and_cancelled_evidence(tmp_path):
    folder=tmp_path/'sparse';write_sparse(folder)
    cancelled=threading.Event();cancelled.set()
    with pytest.raises(c.Cancelled):
        c.read_sparse(folder,cancelled)
    (folder/'points3D.bin').write_bytes(b'bad')
    with pytest.raises(ValueError,match='Truncated'):
        c.read_sparse(folder,threading.Event())


def test_depth_agreement_measures_pairs_not_missing_as_zero(tmp_path):
    sparse=tmp_path/'sparse';sparse.mkdir()
    (sparse/'cameras.txt').write_text('1 PINHOLE 8 8 4 4 4 4\n')
    (sparse/'images.txt').write_text('1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 0 0 0 1 b.jpg\n\n')
    folder=tmp_path/'ai-depth';folder.mkdir()
    for name in ('a','b'):
        np.save(folder/f'{name}.jpg.depth.npy',np.full((8,8),2.))
    measured=c.depth_evidence(tmp_path,threading.Event())
    assert measured['score']==1 and measured['evaluable_pairs']==128
    np.save(folder/'b.jpg.depth.npy',np.full((8,8),3.))
    assert c.depth_evidence(tmp_path,threading.Event())['score']==0
    for name in ('a','b'):
        np.save(folder/f'{name}.jpg.depth.npy',np.zeros((8,8)))
    assert c.depth_evidence(tmp_path,threading.Event())['score'] is None


def test_confidence_reference_json_rebase(tmp_path):
    from drone3d_studio.persistence import store
    from drone3d_studio.domain.models import Model
    root=tmp_path/'project';project=store.create(root,'Evidence')
    project.models.append(Model(name='Model',path='model.ply',format='PLY',vertices=1,faces=0,confidence_path='reconstruction/run/metadata/confidence.json'))
    destination=tmp_path/'external';destination.mkdir()
    store.export_json(destination/'export.json',root,project)
    copy=store.import_json(destination/'export.json',root)
    assert store.resolve(root,copy.models[0].confidence_path)==root/'reconstruction/run/metadata/confidence.json'


def test_attach_run_and_lod_share_original_evidence(tmp_path):
    from drone3d_studio.domain.models import Model
    root=tmp_path; work=root/'reconstruction/run';work.mkdir(parents=True)
    (work/'sparse').mkdir();write_sparse(work/'sparse/0')
    models=[(Model(name='Sparse',path='reconstruction/run/sparse-0.ply',format='PLY',vertices=1,faces=0,origin='COLMAP'),None),
            (Model(name='Textured',path='reconstruction/run/openmvs-0/scene_low.glb',format='GLB',vertices=3,faces=1,origin='OpenMVS'),None)]
    c.attach_run(root,work,frames(2),models,threading.Event(),lambda _:None)
    assert models[0][0].confidence_path==models[1][0].confidence_path
    report,grid=c.load_sidecar(work/'metadata/confidence.json')
    assert report['overall']['score'] is not None and grid is not None
    assert report['components']['registration']['ratio']==1


def test_ai_without_tracks_is_unknown_not_photogrammetric(tmp_path):
    from drone3d_studio.domain.models import Model
    work=tmp_path/'reconstruction/midas';work.mkdir(parents=True)
    model=Model(name='Relief',path='reconstruction/midas/relief.glb',format='GLB',vertices=3,faces=1,origin='MiDaS estimated relief')
    c.attach_run(tmp_path,work,frames(),[(model,None)],threading.Event(),lambda _:None)
    report,_=c.load_sidecar(tmp_path/model.confidence_path)
    assert report['overall']['score'] is None
    assert report['provenance']['model_class']=='AI_ASSISTED'
