"""Model utilities independent of the UI. Original geometry is preserved on disk."""
from pathlib import Path
from uuid import uuid4
import numpy as np
import trimesh

from drone3d_studio.domain.models import Model, Transform
from drone3d_studio.persistence.store import reference

SUPPORTED = {".obj", ".stl", ".ply", ".glb", ".gltf"}


def load_mesh(path: Path):
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError("Supported models: OBJ, STL, PLY, GLB and GLTF")
    if not path.is_file():
        raise ValueError(f"Model file missing: {path}")
    try:
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            raise ValueError("No geometry found")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("Geometry has non-finite coordinates")
        return mesh
    except Exception as exc:
        raise ValueError(f"Could not load {path.name}: {exc}") from exc


def model_record(root: Path, path: Path, origin="Imported"):
    mesh = load_mesh(path)
    return Model(name=path.stem, path=reference(root, path), format=path.suffix[1:].upper(),
                 vertices=len(mesh.vertices), faces=len(getattr(mesh, "faces", [])), origin=origin), mesh


def matrix(transform: Transform):
    result = trimesh.transformations.euler_matrix(*np.radians(transform.rotation), axes="sxyz")
    result[:3, :3] = result[:3, :3] @ np.diag(transform.scale)
    result[:3, 3] = transform.position
    return result


def primitive(kind: str):
    if kind == "Sphere":
        return trimesh.creation.icosphere(subdivisions=2, radius=.7)
    if kind == "Box":
        return trimesh.creation.box(extents=(1.5, 1.5, 1.5))
    if kind == "Drone":
        parts = [trimesh.creation.box(extents=(1.2, .7, .3))]
        for x in (-.85, .85):
            for y in (-.85, .85):
                arm = trimesh.creation.box(extents=(1.9, .1, .1))
                arm.apply_transform(trimesh.transformations.rotation_matrix(np.sign(x * y) * np.pi / 4, (0, 0, 1)))
                parts.append(arm)
                rotor = trimesh.creation.cylinder(radius=.4, height=.04, sections=24)
                rotor.apply_translation((x, y, .17))
                parts.append(rotor)
        return trimesh.util.concatenate(parts)
    raise ValueError(f"Unknown primitive: {kind}")


def add_primitive(root: Path, kind: str):
    mesh = primitive(kind)
    path = root / "models" / f"demo_{kind.lower()}_{uuid4().hex[:8]}.ply"
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    record, _ = model_record(root, path, "Demo")
    record.name = f"{kind} · DEMO"
    return record, mesh


def demo_scene(root: Path, cancel, progress):
    result = []
    for i, kind in enumerate(("Drone", "Box", "Sphere")):
        from drone3d_studio.services.video import Cancelled
        if cancel.is_set():
            raise Cancelled("Demo generation cancelled")
        record, mesh = add_primitive(root, kind)
        record.transform.position = ((i - 1) * 3, 0, 1)
        result.append((record, mesh))
        progress((i + 1) * 100 // 3, f"Generated {kind} · demonstration geometry only")
    return result
