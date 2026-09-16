"""Measured reconstruction evidence. Scores are engineering heuristics, not probabilities.

COLMAP binary layout: https://colmap.github.io/format.html (images/points3D).
Sidecars never modify geometry or its photographic textures.
"""
from dataclasses import asdict, dataclass, field
from collections import Counter
import json
import struct
from pathlib import Path
import numpy as np
from drone3d_studio.persistence.store import atomic_write, reference, resolve
from drone3d_studio.reconstruction.geometry import rotation
from drone3d_studio.services.video import Cancelled


@dataclass
class ConfidenceConfig:
    weights: dict = field(default_factory=lambda: dict(registration=.25, reprojection=.25, multi_view=.20, depth_consistency=.20, frame_quality=.10))
    labels: tuple = ((85, 'HIGH'), (65, 'GOOD'), (40, 'LIMITED'), (0, 'LOW'))
    reprojection_good_px: float = .5
    reprojection_weak_px: float = 4.
    support_strong: int = 5
    low_registration: float = .5
    weak_depth: float = .6
    weak_parallax_deg: float = 2.
    excessive_rotation_deg: float = 20.
    low_translation_ratio: float = .01
    grid_resolution: int = 32
    grid_trim_fraction: float = .01
    depth_max_views: int = 12
    depth_samples_per_view: int = 2048

    def __post_init__(self):
        if any(v < 0 for v in self.weights.values()) or not np.isclose(sum(self.weights.values()), 1):
            raise ValueError('Confidence weights must be nonnegative and sum to one')
        if self.reprojection_weak_px <= self.reprojection_good_px or self.support_strong <= 1:
            raise ValueError('Invalid confidence normalization range')
        if not 2 <= self.grid_resolution <= 64:
            raise ValueError('Confidence grid resolution must be 2–64')
        if not 0 <= self.grid_trim_fraction < .25:
            raise ValueError('Invalid local grid trimming fraction')


DEFAULT = ConfidenceConfig()


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)), count=len(a)) if len(a) else dict(mean=None, median=None, p95=None, count=0)


def error_score(error, config=DEFAULT):
    return np.clip((config.reprojection_weak_px - np.asarray(error)) / (config.reprojection_weak_px - config.reprojection_good_px), 0, 1)


def support_score(count, config=DEFAULT):
    return np.clip((np.asarray(count) - 1) / (config.support_strong - 1), 0, 1)


def combine(components, config=DEFAULT):
    available = {k: v for k, v in config.weights.items() if components.get(k, {}).get('score') is not None and v > 0}
    total = sum(available.values())
    weights = {k: v / total for k, v in available.items()} if total else {}
    score = 100 * sum(components[k]['score'] * v for k, v in weights.items()) if weights else None
    # Frame quality alone is suitability, not reconstruction evidence.
    if not any(k != 'frame_quality' for k in weights):
        score = None
    label = next((label for threshold, label in config.labels if score is not None and score >= threshold), 'N/A')
    return dict(score=score, label=label, method='weighted_evidence_v1', effective_weights=weights,
                available_weight=total, interpretation='Engineering heuristic; not real-world accuracy or probability')


def capture_suitability(frames):
    accepted = [f for f in frames if f.accepted]
    # Legacy frames without a stored score must not silently become zero-quality.
    measured = [f for f in accepted if 'quality_score' in f.model_fields_set]
    component = dict(score=float(np.mean([f.quality_score for f in measured])) / 100 if measured else None,
                     source='Existing stored frame quality_score / 100, accepted keyframes only',
                     measured_frames=len(measured), accepted_frames=len(accepted))
    raw = [dict(time=f.time, accepted=f.accepted, reason=f.reason, sharpness=f.blur,
                features=f.features if 'features' in f.model_fields_set else None,
                clipped_fraction=f.clipped_fraction if 'clipped_fraction' in f.model_fields_set else None,
                motion_px=f.motion_px if 'motion_px' in f.model_fields_set else None,
                quality_score=f.quality_score if 'quality_score' in f.model_fields_set else None) for f in frames]
    reasons = dict(Counter(f.reason for f in frames if not f.accepted))
    return dict(score=None if component['score'] is None else component['score'] * 100,
                method='Existing accepted-frame quality heuristic; not a calibrated prediction',
                frame_quality=component, sampled_frames=len(frames), accepted_frames=len(accepted),
                rejection_reasons=reasons, duplicate_ratio=reasons.get('Near duplicate', 0) / len(frames) if frames else None,
                sharpness=stats([f.blur for f in frames]), motion=stats([f.motion_px for f in frames if 'motion_px' in f.model_fields_set]),
                estimated_overlap=None, motion_diversity=None, raw_frames=raw)


def _read(file, fmt):
    size = struct.calcsize('<' + fmt)
    data = file.read(size)
    if len(data) != size:
        raise ValueError('Truncated COLMAP evidence file')
    return struct.unpack('<' + fmt, data)


def read_sparse(folder, cancel):
    """Read actual IDs/tracks; IDs need not be contiguous. Retain raw point evidence in NPZ."""
    folder = Path(folder)
    cameras = {}
    points, errors, counts, ids, tracks = [], [], [], [], []
    with (folder / 'images.bin').open('rb') as file:
        for _ in range(_read(file, 'Q')[0]):
            if cancel.is_set():
                raise Cancelled('Confidence cancelled')
            values = _read(file, 'i7di')
            name = bytearray()
            while True:
                char = file.read(1)
                if not char:
                    raise ValueError('Truncated COLMAP image name')
                if char == b'\0':
                    break
                name.extend(char)
            n = _read(file, 'Q')[0]
            obs = np.fromfile(file, dtype=np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')]), count=n)
            if len(obs) != n:
                raise ValueError('Truncated COLMAP observations')
            R = rotation(values[1:5]); t = np.array(values[5:8])
            if not np.isfinite(R).all() or not np.isfinite(t).all():
                raise ValueError('Invalid COLMAP camera pose')
            cameras[values[0]] = dict(name=name.decode('utf-8'), center=-R.T @ t, R=R,
                                      t=t,camera_id=values[8],
                                      observations=int(np.count_nonzero(obs['id'] >= 0)), keypoints=n)
    with (folder / 'points3D.bin').open('rb') as file:
        for i in range(_read(file, 'Q')[0]):
            if i % 4096 == 0 and cancel.is_set():
                raise Cancelled('Confidence cancelled')
            v = _read(file, 'Q3d3Bd')
            n = _read(file, 'Q')[0]
            track = np.fromfile(file, dtype='<i4', count=n * 2)
            if len(track) != n * 2:
                raise ValueError('Truncated COLMAP track')
            image_ids = np.unique(track.reshape(-1, 2)[:, 0])
            if any(int(image_id) not in cameras for image_id in image_ids):
                raise ValueError('COLMAP point track references an unavailable camera')
            ids.append(v[0]); points.append(v[1:4]); errors.append(v[7]); counts.append(len(image_ids)); tracks.append(image_ids)
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all() or not np.isfinite(errors).all() or any(e < 0 for e in errors):
        raise ValueError('Invalid COLMAP point evidence')
    return dict(cameras=cameras, xyz=xyz, errors=np.array(errors), counts=np.array(counts), ids=np.array(ids, dtype=np.uint64), tracks=tracks)


def metric_status(frames, alignment=None):
    aligned = alignment is not None and alignment.get('status') == 'Aligned'
    residuals = list((alignment or {}).get('camera_residuals_m', {}).values())
    return dict(telemetry='Available' if any(f.gps is not None for f in frames) or aligned else 'Unavailable or unrecorded',
                alignment='Verified residual check passed' if aligned else 'Not verified',
                scale='Local ENU metres' if aligned else 'Arbitrary/unverified units',
                median_residual_m=stats(residuals)['median'], alignment_evidence=alignment,
                warning='Telemetry residuals are not ground-truth accuracy.' if aligned else 'No verified GPS alignment; metric scale and vertical are unverified.')


def explain(report):
    c = report['components']; pieces = []
    r = c['registration']
    if r.get('ratio') is not None:
        pieces.append(f"{r['registered_frames']} of {r['accepted_frames']} accepted keyframes registered ({r['ratio']:.1%}).")
    else:
        pieces.append('Registration ratio is N/A: the accepted-keyframe denominator is unavailable.')
    if c['reprojection'].get('median_px') is not None:
        pieces.append(f"Median stored COLMAP point reprojection error is {c['reprojection']['median_px']:.3f} px.")
    if c['multi_view'].get('mean_observations') is not None:
        pieces.append(f"Sparse points have {c['multi_view']['mean_observations']:.2f} distinct camera observations on average.")
    missing = [k for k in DEFAULT.weights if c[k]['score'] is None]
    if missing:
        pieces.append('Unavailable components excluded: ' + ', '.join(missing) + '.')
    pieces.extend(report['warnings'])
    pieces.append('This is an engineering evidence score, not measured real-world accuracy. Sparse regional evidence does not validate every mesh face.')
    return ' '.join(pieces)


def build_report(evidence, frames, config=DEFAULT, depth=None, alignment=None, provenance='PHOTOGRAMMETRIC'):
    capture = capture_suitability(frames)
    cameras = evidence['cameras']; errors = evidence['errors']; counts = evidence['counts']
    accepted = sum(f.accepted for f in frames)
    registered = len(cameras)
    ratio = registered / accepted if accepted and registered <= accepted else None
    err = stats(errors)
    components = dict(frame_quality=capture['frame_quality'],
        registration=dict(score=ratio, ratio=ratio, accepted_frames=accepted or None, registered_frames=registered),
        reprojection=dict(score=float(error_score(err['median'], config)) if err['median'] is not None else None,
                          median_px=err['median'], mean_px=err['mean'], p95_px=err['p95'], point_count=err['count'],
                          semantics='Statistics of COLMAP per-point stored ERROR, not individual residual quantiles'),
        multi_view=dict(score=float(np.mean(support_score(counts, config))) if len(counts) else None,
                        mean_observations=float(np.mean(counts)) if len(counts) else None,
                        observation_histogram=dict(Counter(map(int, counts))), semantics='Distinct sparse track image IDs'),
        depth_consistency=depth or dict(score=None, reason='No supported depth measurements evaluated'))
    warnings = []
    if ratio is not None and ratio < config.low_registration:
        warnings.append(f'Only {ratio:.0%} of selected frames were registered by COLMAP.')
    if accepted and registered > accepted:
        warnings.append('Registered-image count exceeds saved accepted frames; registration ratio is N/A.')
    if err['median'] is not None and err['median'] >= config.reprojection_weak_px:
        warnings.append('High reprojection error relative to the configured heuristic range.')
    if len(counts) and np.mean(counts <= 2) > .5:
        warnings.append('Most sparse points have at most two camera observations.')
    if components['depth_consistency']['score'] is not None and components['depth_consistency']['score'] < config.weak_depth:
        warnings.append('Depth disagreement detected between evaluated neighboring views.')
    if provenance == 'AI_ASSISTED':
        warnings.append('AI depth contributed to this model; its provenance remains AI-assisted.')
    if capture['rejection_reasons'].get('Blurry', 0) > len(frames) / 2:
        warnings.append('More than half of analyzed samples were rejected as blurry.')
    if (capture['duplicate_ratio'] or 0) > .5:
        warnings.append('More than half of analyzed samples were rejected as near duplicates.')
    ordered = sorted(cameras.values(), key=lambda c: c['name'])
    angles = []
    for xyz, track in zip(evidence['xyz'][::max(1, len(counts)//5000)], evidence['tracks'][::max(1, len(counts)//5000)]):
        available = [cameras[int(i)]['center'] for i in track if int(i) in cameras]
        if len(available) >= 2:
            # First/last ID rays: explicit bounded proxy, not maximum triangulation angle.
            rays = np.array([available[0], available[-1]]) - xyz
            norms = np.linalg.norm(rays, axis=1)
            if np.all(norms > 1e-12):
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(rays[0], rays[1]) / np.prod(norms), -1, 1)))))
    rotations = [float(np.degrees(np.arccos(np.clip((np.trace(a['R'] @ b['R'].T) - 1) / 2, -1, 1)))) for a, b in zip(ordered, ordered[1:])]
    if angles and np.median(angles) < config.weak_parallax_deg:
        warnings.append('Limited measured ray-angle diversity: first/last track-camera parallax is low.')
    if rotations and np.median(rotations) > config.excessive_rotation_deg:
        warnings.append('Large camera rotation between filename-ordered registered frames.')
    translation_ratio=None
    if ordered and len(evidence['xyz']):
        centers=np.array([camera['center'] for camera in ordered])
        scene_center=np.median(evidence['xyz'],axis=0)
        distance=float(np.median(np.linalg.norm(centers-scene_center,axis=1)))
        if distance>1e-12:
            translation_ratio=float(np.linalg.norm(np.ptp(centers,axis=0))/distance)
            if translation_ratio<config.low_translation_ratio:
                warnings.append('Low camera translation relative to distance from the sparse scene.')
    feature_total = sum(c['keypoints'] for c in ordered)
    report = dict(version=1, overall=combine(components, config), components=components, configuration=asdict(config),
                  capture_suitability=capture, metric_status=metric_status(frames, alignment), warnings=warnings,
                  provenance=dict(model_class=provenance, photogrammetric_fraction=None, ai_assisted_fraction=None,
                                  reason='No surface-area provenance fractions measured'),
                  coverage=dict(triangulated_keypoint_fraction=sum(c['observations'] for c in ordered)/feature_total if feature_total else None,
                                vertical_surface_coverage=None, reason='No verified vertical surface classification'),
                  single_pass=dict(parallax_deg=stats(angles), parallax_method='Sampled sparse tracks; first/last image ID rays',
                                   camera_span_to_scene_distance=translation_ratio,
                                   adjacent_rotation_deg=stats(rotations), camera_order='Filename order; staged names follow video time'),
                  local=dict(method='Sparse evidence in bounded spatial cells; not per-face validation'))
    report['explanation'] = explain(report)
    return report


def write_grid(path, evidence, config=DEFAULT):
    xyz = evidence['xyz']; n = config.grid_resolution
    origin = np.quantile(xyz,config.grid_trim_fraction,axis=0) if len(xyz) else np.zeros(3)
    upper = np.quantile(xyz,1-config.grid_trim_fraction,axis=0) if len(xyz) else np.ones(3)
    cell = max(float((upper-origin).max()) / (n - 1), 1e-9)
    cells = np.floor((xyz-origin)/cell).astype(int)
    valid=((cells>=0)&(cells<n)).all(axis=1)
    cells=cells[valid]
    keys = cells[:, 0]*n*n + cells[:, 1]*n + cells[:, 2]
    count = np.bincount(keys, minlength=n**3)
    score = ((error_score(evidence['errors'],config)+support_score(evidence['counts'],config))/2)[valid]
    sums = np.bincount(keys,weights=score,minlength=n**3)
    supports = np.bincount(keys,weights=evidence['counts'][valid],minlength=n**3)
    ordered = sorted(evidence['cameras'].values(),key=lambda c:c['name'])
    np.savez_compressed(path, origin=origin, cell=cell, resolution=n,
                        scores=np.divide(sums,count,out=np.full(n**3,np.nan),where=count>0).astype('f4'),
                        supporting_views=np.divide(supports,count,out=np.zeros(n**3),where=count>0).astype('f4'),
                        point_count=count, point_ids=evidence['ids'], point_xyz=xyz.astype('f4'),
                        point_errors=evidence['errors'].astype('f4'), point_observations=evidence['counts'].astype('i4'),
                        cameras=np.array([c['center'] for c in ordered]).reshape(-1,3),
                        camera_directions=np.array([c['R'][2] for c in ordered]).reshape(-1,3))


def sample_grid(grid, vertices):
    n = int(grid['resolution']); cells = np.floor((vertices-grid['origin']) / float(grid['cell'])).astype(np.int64)
    valid = ((cells >= 0)&(cells < n)).all(axis=1)
    scores = np.full(len(vertices), np.nan); support = np.zeros(len(vertices))
    k = cells[valid,0]*n*n+cells[valid,1]*n+cells[valid,2]
    scores[valid] = grid['scores'][k]; support[valid] = grid['supporting_views'][k]
    return scores, support


def load_sidecar(path):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    grid_path = Path(path).parent / report.get('local',{}).get('file','local-confidence.npz')
    grid = None
    if grid_path.is_file():
        with np.load(grid_path, allow_pickle=False) as data:
            grid = {k:data[k].copy() for k in ('origin','cell','resolution','scores','supporting_views','cameras','camera_directions')}
    return report, grid


def depth_evidence(dense, cancel, config=DEFAULT):
    """Bounded re-evaluation using the existing calibrated-depth consistency check."""
    from drone3d_studio.reconstruction.geometry import read_views, read_intrinsics
    from drone3d_studio.reconstruction.visible_surface import read_colmap_depth
    from drone3d_studio.reconstruction.ai_depth import consistent_support
    missing = dict(score=None, reason='No readable undistorted text cameras and depth maps')
    if dense is None:
        return missing
    dense = Path(dense)
    text = next((p for p in (dense/'text-model',dense/'sparse') if (p/'images.txt').is_file() and (p/'cameras.txt').is_file()), None)
    if text is None:
        from drone3d_studio.reconstruction.geometry import CameraView
        sparse=dense/'sparse'
        if not (sparse/'images.bin').is_file() or not (sparse/'cameras.bin').is_file():
            return missing
        evidence=read_sparse(sparse,cancel)
        views=[CameraView(c['name'],c['camera_id'],c['R'],c['t'],[]) for c in evidence['cameras'].values()]
        intrinsics={}
        with (sparse/'cameras.bin').open('rb') as file:
            for _ in range(_read(file,'Q')[0]):
                camera_id,kind,w,h=_read(file,'iiQQ')
                if kind not in (0,1):
                    return dict(score=None,reason='Depth evidence requires undistorted pinhole cameras')
                params=_read(file,'3d' if kind==0 else '4d')
                fx,fy,cx,cy=(params[0],params[0],params[1],params[2]) if kind==0 else params
                intrinsics[camera_id]=(w,h,fx,fy,cx,cy)
    else:
        views = read_views(text/'images.txt'); intrinsics = read_intrinsics(text/'cameras.txt')
    pairs = []
    for view in sorted(views, key=lambda v:v.name):
        stereo = dense/'stereo/depth_maps'/(view.name+'.geometric.bin')
        ai = dense/'ai-depth'/(view.name+'.depth.npy')
        if stereo.is_file() or ai.is_file():
            pairs.append((view, stereo if stereo.is_file() else ai))
    if len(pairs) < 2:
        return missing
    indices = np.unique(np.linspace(0,len(pairs)-1,min(config.depth_max_views,len(pairs)),dtype=int))
    selected = [pairs[i] for i in indices]; maps=[]
    try:
        for view,path in selected:
            if cancel.is_set():
                raise Cancelled('Depth evidence cancelled')
            depth = np.load(path,mmap_mode='r') if path.suffix=='.npy' else read_colmap_depth(path)
            w,h,*_ = intrinsics[view.camera_id]
            if depth.shape != (h,w):
                return dict(score=None,reason='Depth/camera dimensions differ; not compared')
            maps.append(depth)
            if sum(m.nbytes for m in maps)>256*1024*1024:
                return dict(score=None,reason='Depth evidence exceeds 256 MB budget')
        agreements=evaluations=samples=survived=0
        histogram=Counter()
        for i,(view,_) in enumerate(selected):
            if cancel.is_set():
                raise Cancelled('Depth evidence cancelled')
            w,h,fx,fy,cx,cy=intrinsics[view.camera_id]
            stride=max(1,int(np.ceil(np.sqrt(w*h/config.depth_samples_per_view))))
            yy,xx=np.mgrid[0:h:stride,0:w:stride]; z=maps[i][yy,xx]
            valid=np.isfinite(z)&(z>0); z=z[valid]; xx=xx[valid]; yy=yy[valid]
            world=(np.column_stack(((xx-cx)/fx*z,(yy-cy)/fy*z,z))-view.t)@view.R
            support, evaluated, neighbors=consistent_support(world,[v for v,_ in selected],i,maps,intrinsics)
            agreements+=int(support.sum()); evaluations+=int(evaluated.sum()); samples+=len(z)
            survived+=int(np.count_nonzero(support>=min(2,neighbors)))
            histogram.update(map(int,support))
        return dict(score=agreements/evaluations if evaluations else None, agreeing_pairs=agreements,evaluable_pairs=evaluations,
                    sampled_valid_points=samples, sampled_surviving_points=survived,
                    sampled_survival_ratio=survived/samples if samples else None,
                    supporting_view_count_histogram=dict(histogram), selected_views=[v.name for v,_ in selected],
                    available_depth_views=len(pairs), relative_depth_tolerance=.08,
                    semantics='Sampled valid-depth agreement with up to four nearest selected cameras; not whole-surface coverage')
    finally:
        for depth in maps:
            if hasattr(depth,'_mmap'):
                depth._mmap.close()


def inspect_component(sparse, dense, frames, destination, cancel, provenance='PHOTOGRAMMETRIC', alignment=None, config=DEFAULT):
    evidence=read_sparse(sparse,cancel)
    try:
        depth=depth_evidence(dense,cancel,config)
    except (ValueError,OSError,KeyError) as exc:
        depth=dict(score=None,reason=str(exc))
    report=build_report(evidence,frames,config,depth,alignment,provenance)
    report['sources']=dict(sparse=str(sparse),dense=str(dense) if dense else None)
    report['ai_agreement']=dict(anchors=None,median_relative_inverse_error=None,reason='No calibration report')
    if dense:
        fusion=Path(dense)/'fusion-evidence.json'
        if fusion.is_file():
            report['fusion_filter']=json.loads(fusion.read_text(encoding='utf-8'))
        calibration=Path(dense)/'ai-depth/calibration.json'
        if calibration.is_file():
            raw=json.loads(calibration.read_text(encoding='utf-8'))
            calibrated=[r for r in raw.get('frames',[]) if r.get('status')=='calibrated']
            report['ai_agreement']=dict(calibrated_frames=len(calibrated),attempted_frames=len(raw.get('frames',[])),
                anchors=sum(r['anchors'] for r in calibrated),
                median_relative_inverse_error=stats([r['median_relative_inverse_error'] for r in calibrated])['median'],
                raw_calibration=raw,semantics='Calibration observations can reuse the same SfM point across images; no metric truth claim')
    destination.mkdir(parents=True,exist_ok=True)
    write_grid(destination/'local-confidence.npz',evidence,config)
    report['local'].update(file='local-confidence.npz',resolution=config.grid_resolution,
                          formula='Mean of (reprojection score + sparse support score)/2 in each occupied cell',
                          limitation='Spatial approximation of sparse evidence, not dense vertex observation counts. Grid bounds use configured coordinate quantiles to limit outlier influence; points outside the grid and empty cells are insufficient evidence.')
    atomic_write(destination/'confidence.json',json.dumps(report,indent=2,allow_nan=False))
    return destination/'confidence.json'


def attach_run(root,work,frames,result,cancel,log,config=DEFAULT):
    """Post-process existing evidence; component coordinates are never mixed."""
    groups={}
    for record,mesh in result:
        if record.origin=='Demo':
            continue
        path=resolve(root,record.path)
        component=None; dense=None; sparse=None
        for parent in (path.parent,*path.parents):
            if parent==work.parent:
                break
            if parent.name.startswith(('openmvs-','ai-dense-','dense-')):
                component=parent.name.rsplit('-',1)[-1]
                break
        if path.stem.startswith('sparse-'):
            component=path.stem.split('-')[-1]
        if component is not None:
            source=work/f'component-{component}'
            sparse=source/'aligned' if (source/'aligned/images.bin').is_file() else work/'sparse'/component
            sources=work/'metadata/sources.json'
            if sources.is_file():
                saved=json.loads(sources.read_text(encoding='utf-8')).get(component)
                if saved:
                    sparse=resolve(work,saved)
            dense=next((p for p in (work/f'ai-dense-{component}',work/f'dense-{component}',work/f'mvs-input-{component}') if p.is_dir()),None)
        elif (work/'textured-report.json').is_file():
            raw=json.loads((work/'textured-report.json').read_text(encoding='utf-8'))
            dense=Path(raw['colmap_workspace']); sparse=dense/'sparse'; component='0'; source=work
        if sparse is None or not (sparse/'images.bin').is_file():
            # MiDaS has no measured multiview evidence: explicit N/A, AI provenance.
            report=build_report(dict(cameras={},xyz=np.empty((0,3)),errors=np.array([]),counts=np.array([]),tracks=[],ids=np.array([])),[],config=config,provenance='AI_ASSISTED' if record.origin in ('MiDaS estimated relief','Depth Anything V2') else 'INSUFFICIENT_EVIDENCE')
            report['warnings'].append('No readable COLMAP camera/track evidence is associated with this model.')
            report['explanation']=explain(report)
            dest=work/'metadata'/record.id/'confidence.json'
            atomic_write(dest,json.dumps(report,indent=2,allow_nan=False)); record.confidence_path=reference(root,dest)
            continue
        provenance='AI_ASSISTED' if record.origin=='Depth Anything V2' else 'PHOTOGRAMMETRIC'
        key=(str(sparse),provenance)
        if key not in groups:
            alignment_path=source/'georeference.json'
            alignment=json.loads(alignment_path.read_text(encoding='utf-8')) if alignment_path.is_file() else None
            dest=work/'metadata'/f'component-{component}-{provenance.lower()}'
            groups[key]=inspect_component(sparse,dense,frames,dest,cancel,provenance,alignment,config)
            # Native OpenMVS stage completion is context, never a confidence bonus.
            native=work/f'openmvs-{component}'/'textured-report.json'
            if not native.is_file():
                native=work/'textured-report.json'
            if native.is_file():
                report=json.loads(groups[key].read_text(encoding='utf-8'))
                native_report=json.loads(native.read_text(encoding='utf-8'))
                report['openmvs']=dict(stages=[dict(name=s['name'],status=s['status']) for s in native_report.get('stages',[])],
                    texture_coverage=None,mesh_support=None,note='Successful execution is not geometric evidence; unsupported native depth formats are not scored')
                for filename,label in (('scene_dense.ply','dense_points'),('scene_refined.ply','refined_mesh')):
                    ply=native.parent/filename
                    if ply.is_file():
                        with ply.open('rb') as file:
                            header=[]
                            for _ in range(100):
                                line=file.readline(1024).decode('ascii',errors='replace').strip()
                                header.append(line)
                                if line=='end_header':
                                    break
                        report['openmvs'][label]={line.split()[1]:int(line.split()[2]) for line in header if line.startswith(('element vertex ','element face '))}
                atomic_write(groups[key],json.dumps(report,indent=2,allow_nan=False))
        record.confidence_path=reference(root,groups[key])
        log('Evidence confidence: '+str(groups[key]))
    # Root report is an index when components differ; never average disconnected frames twice.
    reports=list(dict.fromkeys(m.confidence_path for m,_ in result if m.confidence_path))
    if len(reports)==1:
        root_report=json.loads(resolve(root,reports[0]).read_text(encoding='utf-8'))
        root_report['component_report']=reports[0]
        root_report['local']['file']=str(resolve(root,reports[0]).parent.relative_to(work/'metadata')/'local-confidence.npz')
    else:
        root_report=dict(version=1,overall=dict(score=None,label='N/A'),component_reports=reports,
                         reason='Multiple provenance classes/components; inspect individual component evidence instead of mixing coordinate systems')
    atomic_write(work/'metadata/confidence.json',json.dumps(root_report,indent=2,allow_nan=False))
