"""Atomic project storage and portable references."""
import json
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from drone3d_studio.domain.models import Project, now

FILE_NAME = "project.drone3d.json"
FOLDERS = ("backups", "source", "frames/accepted", "frames/rejected", "models", "reconstruction", "thumbnails", "logs", "exports")


def resolve(root: Path, reference: str) -> Path:
    p = Path(reference)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def reference(root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".save-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save(root: Path, project: Project):
    validated = Project.model_validate_json(project.model_dump_json())
    validated.modified = now()
    for folder in FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    target = root / FILE_NAME
    if target.exists():
        previous = target.read_text(encoding="utf-8")
        try:
            Project.model_validate_json(previous)
        except ValueError:
            pass  # Never overwrite a valid backup with corrupt data.
        else:
            atomic_write(root / "backups" / FILE_NAME, previous)
    atomic_write(target, validated.model_dump_json(indent=2))
    project.modified = validated.modified


def load(root: Path) -> Project:
    project = Project.model_validate_json((root / FILE_NAME).read_text(encoding="utf-8"))
    if project.status in ("Analysis running", "Reconstruction running"):
        project.status = "Cancelled"
        project.logs.append("Previous processing was interrupted; run it again.")
    return project


def create(root: Path, name: str, description: str = "") -> Project:
    project = Project(name=name.strip(), description=description)
    root.mkdir(parents=True, exist_ok=False)
    save(root, project)
    return project


def duplicate(root: Path, destination: Path, project: Project) -> Project:
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Choose a destination outside the original project.")
    shutil.copytree(root, destination)
    copy = project.model_copy(deep=True)
    copy.id, copy.created = uuid4().hex, now()
    copy.name += " (copy)"
    save(destination, copy)
    return copy


def import_json(path: Path, root: Path) -> Project:
    project = Project.model_validate_json(path.read_text(encoding="utf-8"))
    # Rebase references to the JSON's directory before saving in a new location.
    if project.video:
        project.video.path = reference(root, resolve(path.parent, project.video.path))
    for model in project.models:
        model.path = reference(root, resolve(path.parent, model.path))
    for frame in project.frames:
        frame.path = reference(root, resolve(path.parent, frame.path))
        frame.thumbnail = reference(root, resolve(path.parent, frame.thumbnail))
    return project


def export_json(path: Path, root: Path, project: Project):
    copy = project.model_copy(deep=True)
    if copy.video:
        copy.video.path = reference(path.parent, resolve(root, copy.video.path))
    for model in copy.models:
        model.path = reference(path.parent, resolve(root, model.path))
    for frame in copy.frames:
        frame.path = reference(path.parent, resolve(root, frame.path))
        frame.thumbnail = reference(path.parent, resolve(root, frame.thumbnail))
    atomic_write(path, copy.model_dump_json(indent=2))
