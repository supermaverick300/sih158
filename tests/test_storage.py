import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from drone3d_studio.domain.models import Project, Transform, Settings
from drone3d_studio.persistence import store


def test_creation_roundtrip_backup(tmp_path):
    root = tmp_path / "flight"
    project = store.create(root, "Flight", "Coast survey")
    assert all((root / d).is_dir() for d in store.FOLDERS)
    project.description = "Edited"
    store.save(root, project)
    assert store.load(root).description == "Edited"
    backup = Project.model_validate_json((root / "backups" / store.FILE_NAME).read_text())
    assert backup.description == "Coast survey"
    assert not list(root.glob(".save-*.tmp"))


def test_atomic_failure_preserves_file(tmp_path, monkeypatch):
    root = tmp_path / "flight"
    project = store.create(root, "Flight")
    before = (root / store.FILE_NAME).read_bytes()
    real_replace = store.os.replace
    def fail(source, destination):
        if Path(destination) == root / store.FILE_NAME:
            raise OSError("simulated disk failure")
        real_replace(source, destination)
    monkeypatch.setattr(store.os, "replace", fail)
    project.name = "New name"
    with pytest.raises(OSError):
        store.save(root, project)
    assert (root / store.FILE_NAME).read_bytes() == before
    assert not list(root.glob(".save-*.tmp"))


@pytest.mark.parametrize("data", [{"name": ""}, {"name": "x", "schema_version": 2}, {"name": "x", "unknown": 1}, {"name": "x", "analysis": {"interval": 0}}, {"name": "x", "analysis": {"blur_threshold": float("nan")}}])
def test_schema_rejects_invalid(data):
    with pytest.raises(ValidationError):
        Project.model_validate(data)


def test_invalid_transform_and_settings():
    with pytest.raises(ValidationError):
        Transform(scale=(0, 1, 1))
    with pytest.raises(ValidationError):
        Settings(background="red")


def test_relative_paths_and_moved_project(tmp_path):
    root = tmp_path / "a"
    project = store.create(root, "A")
    file = root / "models" / "test.ply"
    file.write_text("test")
    assert store.reference(root, file) == "models/test.ply"
    assert store.resolve(root, "models/test.ply") == file
    assert Path(store.reference(root, tmp_path / "external.avi")).is_absolute()
    moved = tmp_path / "b"
    root.rename(moved)
    assert store.resolve(moved, "models/test.ply").exists()


def test_json_export_import(tmp_path):
    from drone3d_studio.services.meshes import add_primitive
    root = tmp_path / "a"
    project = store.create(root, "A")
    model, _ = add_primitive(root, "Box")
    project.models.append(model)
    exported = tmp_path / "exports" / "export.json"
    store.export_json(exported, root, project)
    imported = store.import_json(exported, root)
    assert imported.models[0].path == model.path
    assert imported.name == "A"
    exported.write_text('{"name": 5, "schema_version": 99}')
    with pytest.raises(ValidationError):
        store.import_json(exported, root)


def test_duplicate(tmp_path):
    root = tmp_path / "a"
    original = store.create(root, "A")
    copied = store.duplicate(root, tmp_path / "b", original)
    assert copied.id != original.id
    assert store.load(tmp_path / "b").name == "A (copy)"
    with pytest.raises(ValueError):
        store.duplicate(root, root / "child", original)


def test_interrupted_load_and_invalid_file(tmp_path):
    root = tmp_path / "a"
    project = store.create(root, "A")
    project.status = "Analysis running"
    store.save(root, project)
    assert store.load(root).status == "Cancelled"
    (root / store.FILE_NAME).write_text("broken")
    with pytest.raises(ValidationError):
        store.load(root)
    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "missing")


def test_corrupt_file_does_not_replace_valid_backup(tmp_path):
    root = tmp_path / "a"
    project = store.create(root, "A")
    store.save(root, project)
    before = (root / "backups" / store.FILE_NAME).read_bytes()
    (root / store.FILE_NAME).write_text("bad json")
    store.save(root, project)
    assert (root / "backups" / store.FILE_NAME).read_bytes() == before
