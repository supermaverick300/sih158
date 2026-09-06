import threading
from PySide6.QtCore import Qt
from drone3d_studio.application import Studio
from drone3d_studio.domain.models import Project
from drone3d_studio.persistence import store
from drone3d_studio.persistence.autosave import Autosave
from drone3d_studio.services import meshes, video


def test_autosave_debounce_and_failure(qtbot):
    calls, states = [], []
    saver = Autosave(lambda: calls.append(1))
    saver.state.connect(states.append)
    saver.trigger(20)
    saver.trigger(20)
    qtbot.waitUntil(lambda: len(calls) == 1)
    assert states[-2:] == ["Saving…", "Saved"]
    def failure():
        raise OSError("disk full")
    saver.callback = failure
    saver.trigger()
    assert not saver.flush() and saver.dirty
    assert states[-1].startswith("Save failed")
    saver.timer.stop()


def test_complete_gui_workflow(qtbot, tmp_path):
    window = Studio()
    qtbot.addWidget(window)
    root = tmp_path / "project"
    window.activate(root, store.create(root, "Test flight"))
    window.synthetic()
    qtbot.waitUntil(lambda: window.job is None, timeout=15000)
    assert window.project.video.frame_count == 48
    window.blur.setValue(0)
    window.start_analysis()
    qtbot.waitUntil(lambda: window.job is None, timeout=15000)
    assert window.project.status == "Ready for reconstruction"
    assert window.frames.count() == 4
    window.reconstruct()
    qtbot.waitUntil(lambda: window.job is None, timeout=15000)
    assert len(window.project.models) == 3
    key = window.project.models[0].id
    window.select_id(key)
    window.transform_boxes["position"][0].setValue(7)
    window.uniform.setChecked(True)
    window.transform_boxes["scale"][1].setValue(2)
    assert window.project.models[0].transform.scale == (2, 2, 2)
    window.model_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not window.project.models[0].visible
    window.manual_save()
    assert store.load(root).models[0].transform.position[0] == 7
    window.refresh_json()
    assert Project.model_validate_json(window.json_view.toPlainText()).models[0].id == key
    for i in range(7):
        window.nav.setCurrentRow(i)
        window.stack.currentWidget().grab()
    window.close()
    reopened = Studio()
    qtbot.addWidget(reopened)
    reopened.open_path(root)
    qtbot.waitUntil(lambda: reopened.job is None, timeout=15000)
    assert reopened.project.models[0].transform.position[0] == 7
    assert len(reopened.meshes) == 3
    reopened.close()


def test_selection_nudge_remove(qtbot, tmp_path):
    window = Studio()
    qtbot.addWidget(window)
    root = tmp_path / "p"
    window.activate(root, store.create(root, "P"))
    model, mesh = meshes.add_primitive(root, "Box")
    window.models_added([(model, mesh)])
    window.select_id(model.id)
    window.nudge(2, .1)
    assert model.transform.position[2] == .1
    window.canvas.resize(700, 500)
    window.canvas.grab()
    assert window.canvas.hit
    window.remove_model()
    assert not window.project.models
    assert store.resolve(root, model.path).exists()
    window.close()


def test_close_cancels_worker_and_persists(qtbot, tmp_path):
    from drone3d_studio.services.video import Cancelled
    window = Studio()
    qtbot.addWidget(window)
    root = tmp_path / "p"
    window.activate(root, store.create(root, "P"))
    def work(cancel, progress):
        cancel.wait(5)
        if cancel.is_set():
            raise Cancelled("Stopped for close")
    window.project.status = "Analysis running"
    window.run_job(work, lambda result: None)
    window.close()
    qtbot.waitUntil(lambda: window.job is None, timeout=3000)
    assert store.load(root).status == "Cancelled"


def test_missing_model_and_config_autosave(qtbot, tmp_path):
    window = Studio()
    qtbot.addWidget(window)
    root = tmp_path / "p"
    project = store.create(root, "P")
    model, _ = meshes.add_primitive(root, "Box")
    project.models.append(model)
    store.resolve(root, model.path).unlink()
    window.activate(root, project)
    qtbot.waitUntil(lambda: window.job is None)
    assert "missing" in project.models[0].status.lower()
    window.interval.setValue(2.5)
    window.project.settings.autosave_ms = 200
    window.maximum.setValue(42)
    qtbot.waitUntil(lambda: not window.autosave.dirty)
    assert store.load(root).analysis.max_frames == 42
    assert store.load(root).analysis.interval == 2.5
    window.close()
