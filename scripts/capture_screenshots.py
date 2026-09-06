"""Render real app pages offscreen using a disposable generated project."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys
import tempfile
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from drone3d_studio.application import Studio
from drone3d_studio.persistence import store
from drone3d_studio.services import meshes, video


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    output = Path("docs/screenshots")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, temporary)
        root = Path(temporary) / "Harbor"
        project = store.create(root, "Harbor flight", "Scene inspection workspace · synthetic input and procedural demo")
        window = Studio()
        window.activate(root, project)
        clip = root / "source" / "synthetic.avi"
        video.synthetic_video(clip)
        info = video.metadata(clip)
        info.path = store.reference(root, clip)
        window.video_imported(info)
        window.analysis_done(video.analyze(clip, root, project.analysis, threading.Event(), lambda *a: None))
        window.scene_done(meshes.demo_scene(root, threading.Event(), lambda *a: None))
        window.select_id(project.models[0].id)
        window.show()
        app.processEvents()
        for page, name in ((1, "dashboard"), (2, "video"), (4, "scene")):
            window.nav.setCurrentRow(page)
            app.processEvents()
            if page == 2:
                window.seek(0)
            window.grab().save(str(output / (name + ".png")))
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
