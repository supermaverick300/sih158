from pathlib import Path
import threading
from drone3d_studio.persistence import store
from drone3d_studio.services.meshes import demo_scene


def make_sample(root: Path):
    project = store.create(root, "Harbor flight · Demo", "A procedural scene for testing inspection and transforms. This is not a photogrammetry reconstruction.")
    for model, _ in demo_scene(root, threading.Event(), lambda value, message: None):
        project.models.append(model)
    project.status = "Scene ready"
    project.reconstruction_status = "Demo scene — NOT photogrammetry"
    store.save(root, project)
    return project
