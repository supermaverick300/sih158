"""Open surfaces from supported depth samples; never close unobserved backsides."""
from pathlib import Path
import json
import cv2
import numpy as np
import trimesh
from drone3d_studio.persistence.store import atomic_write
from drone3d_studio.services.video import Cancelled
from drone3d_studio.reconstruction.geometry import read_views, read_intrinsics
from drone3d_studio.reconstruction.ai_depth import consistent_points


def read_colmap_depth(path):
    """COLMAP width&height&channels& header and little-endian row-major floats."""
    with Path(path).open("rb") as file:
        header = bytearray()
        while header.count(b"&") < 3:
            char = file.read(1)
            if not char or len(header) > 80:
                raise ValueError(f"Invalid COLMAP depth header: {path}")
            header.extend(char)
        width, height, channels = map(int, header[:-1].split(b"&"))
        if channels != 1 or min(width, height) < 1 or width * height > 100_000_000:
            raise ValueError("Unsupported depth dimensions")
        values = np.fromfile(file, dtype="<f4")
    if values.size != width * height:
        raise ValueError("Truncated or oversized COLMAP depth map")
    return values.reshape(height, width)


def grid_faces(depth, valid, jump=.08):
    """Only adjacent, valid pixels at compatible depths may be connected."""
    h, w = depth.shape
    ids = np.arange(h*w).reshape(h, w)
    a, b, c, d = ids[:-1,:-1].ravel(), ids[:-1,1:].ravel(), ids[1:,:-1].ravel(), ids[1:,1:].ravel()
    faces = np.concatenate((np.column_stack((a,c,b)), np.column_stack((b,c,d))))
    z = depth.ravel()[faces]
    keep = valid.ravel()[faces].all(axis=1) & ((z.max(axis=1)-z.min(axis=1)) <= jump * np.maximum(z.min(axis=1),1e-9))
    return faces[keep]


def build_surface(views, maps, images, intrinsics, output, stride, cancel, progress):
    if len(views) < 3:
        raise ValueError("A single pass needs at least three overlapping registered frames with depth.")
    vertices, colors, faces, count = [], [], [], 0
    for i, view in enumerate(views):
        if cancel.is_set():
            raise Cancelled("Visible-surface reconstruction cancelled")
        w,h,fx,fy,cx,cy = intrinsics[view.camera_id]
        if maps[i].shape != (h,w):
            raise ValueError("Depth dimensions do not match the undistorted camera")
        yy,xx = np.mgrid[0:h:stride,0:w:stride]
        z = maps[i][yy,xx]
        valid = np.isfinite(z) & (z>0)
        safe = np.where(valid,z,0)
        camera = np.column_stack((((xx-cx)/fx*safe).ravel(),((yy-cy)/fy*safe).ravel(),safe.ravel()))
        world = (camera-view.t)@view.R
        support = np.zeros(valid.size,dtype=bool)
        ids = np.flatnonzero(valid.ravel())
        support[ids] = consistent_points(world[ids],views,i,maps,intrinsics)
        triangles = grid_faces(safe,support.reshape(z.shape))
        if not len(triangles):
            continue
        used, remap = np.unique(triangles,return_inverse=True)
        if count+len(used)>2_000_000:
            raise ValueError("Surface exceeds two million vertices. Increase depth pixel stride and retry.")
        image = cv2.imdecode(np.fromfile(str(images/view.name),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Missing undistorted image: {view.name}")
        vertices.append(world[used])
        colors.append(image[yy.ravel()[used],xx.ravel()[used],::-1])
        faces.append(remap.reshape(-1,3)+count)
        count += len(used)
        progress(-1,f"Single-pass surface {i+1}/{len(views)}: {count:,} supported vertices")
    if not faces or sum(len(f) for f in faces)<100:
        raise ValueError("Too few consistent depth triangles. A translating camera and overlapping sharp frames are needed; an orbit is not required.")
    mesh = trimesh.Trimesh(vertices=np.concatenate(vertices),faces=np.concatenate(faces),vertex_colors=np.concatenate(colors),process=False)
    # Independent view patches preserve openings; no watertightness or completion claim.
    mesh.export(str(output))
    atomic_write(output.with_suffix(".json"),json.dumps({"capture":"Single pass","surface":"Observed depth patches","vertices":len(mesh.vertices),"faces":len(mesh.faces),"registered_depth_views":len(views),"pixel_stride":stride,"neighbor_depth_tolerance":.08,"unseen_surfaces":"Not generated","limitations":"Overlapping patches can remain; boundaries and depth discontinuities remain open. Colors are sampled from the video, not a texture atlas."},indent=2))
    return output


def from_stereo(dense,text_model,config,cancel,progress,output=None):
    views = read_views(text_model/"images.txt")
    intrinsics = read_intrinsics(text_model/"cameras.txt")
    selected, maps = [], []
    for view in views:
        path = dense/"stereo"/"depth_maps"/(view.name+".geometric.bin")
        if path.is_file():
            if cancel.is_set():
                raise Cancelled("Depth loading cancelled")
            maps.append(read_colmap_depth(path))
            selected.append(view)
            if sum(depth.nbytes for depth in maps)>512*1024*1024:
                raise ValueError("Depth workspace exceeds the 512 MB surface budget. Use fewer keyframes or lower-resolution depth maps.")
    try:
        return build_surface(selected,maps,dense/"images",intrinsics,output or dense/"visible-surface.ply",config.depth_stride,cancel,progress)
    finally:
        maps.clear()
