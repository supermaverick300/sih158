"""Desktop shell and main-thread workflow coordination."""
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import sys

import cv2
from PySide6.QtCore import Qt, QTimer, QSettings, QSize, QUrl
from PySide6.QtGui import QAction, QImage, QPixmap, QIcon, QKeySequence, QFontDatabase, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QStackedWidget, QLineEdit,
    QTextEdit, QPlainTextEdit, QFileDialog, QMessageBox, QInputDialog, QSlider,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QProgressBar, QSplitter,
    QGroupBox, QScrollArea,
)

from drone3d_studio.domain.models import Project, Settings, AnalysisConfig, PipelineConfig, Transform, now
from drone3d_studio.persistence import store
from drone3d_studio.persistence.autosave import Autosave
from drone3d_studio.services import video, meshes, telemetry
from drone3d_studio.reconstruction.colmap import ColmapBackend, detect, probe
from drone3d_studio.viewer.canvas import SceneCanvas
from drone3d_studio.workers.jobs import Job

DARK = """
QWidget {background:#111923; color:#e4edf7; font-family:'Segoe UI'; font-size:13px;}
QMainWindow, QStackedWidget {background:#111923;}
QLabel#title {font-size:27px; font-weight:700; padding:4px 0 10px;}
QLabel#muted {color:#91a4ba;}
QListWidget {background:#162230; border:1px solid #2a3b4e; border-radius:8px; padding:6px;}
QListWidget::item {padding:12px; border-radius:6px;}
QListWidget::item:selected {background:#214b50; color:#78edd1;}
QListWidget::item:hover {background:#25364a;}
QPushButton {background:#233449; border:1px solid #34495e; border-radius:6px; padding:9px 14px;}
QPushButton:hover {background:#314b64; border-color:#63d7bc;}
QPushButton:disabled {color:#61758b; background:#192330;}
QPushButton#primary {background:#38b79c; color:#081c19; font-weight:700; border:0;}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {background:#182534; border:1px solid #34465b; border-radius:5px; padding:7px; selection-background-color:#376f72;}
QGroupBox {border:1px solid #2c4053; border-radius:9px; margin-top:14px; padding:16px; font-weight:600;}
QGroupBox::title {subcontrol-origin:margin; left:12px; padding:0 5px;}
QProgressBar {border:1px solid #34465b; border-radius:4px; text-align:center; min-height:16px;}
QProgressBar::chunk {background:#38b79c;}
QStatusBar {background:#162230; color:#9bb1c8;}
QToolTip {background:#24364a; color:white; border:1px solid #54718d;}
"""


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.clicked.connect(lambda checked=False: callback())
    widget.setToolTip(text)
    if primary:
        widget.setObjectName("primary")
    return widget


def spin(value, minimum, maximum, decimals=2):
    widget = QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def row(*widgets):
    layout = QHBoxLayout()
    for widget in widgets:
        layout.addWidget(widget)
    return layout


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        # The Windows offscreen Qt platform does not enumerate system fonts.
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and os.name == "nt":
            fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for name in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf"):
                QFontDatabase.addApplicationFont(str(fonts / name))
        self.setWindowTitle("Drone 3D Studio")
        self.resize(1360, 880)
        self.project, self.root = None, None
        self.meshes = {}
        self.job = None
        self.cap = None
        self.loading = False
        self.close_pending = False
        self.prefs = QSettings("Drone3DStudio", "Desktop")
        self.defaults = Settings()
        try:
            self.defaults = Settings.model_validate_json(self.prefs.value("settings", self.defaults.model_dump_json()))
        except ValueError:
            logging.warning("Invalid saved settings; using defaults")
        self.defaults.reconstruction_mode = "COLMAP"
        try:
            self.defaults.executable = detect(self.defaults.executable)
        except ValueError:
            pass
        self.autosave = Autosave(self.save_current, self)
        self.autosave.state.connect(self.statusBar().showMessage)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_tick)
        shell = QWidget()
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(12, 12, 12, 12)
        sidebar = QVBoxLayout()
        logo = QLabel("DRONE 3D\nSTUDIO")
        logo.setStyleSheet("font-size:20px;font-weight:700;color:#68e1c4;padding:15px 5px")
        sidebar.addWidget(logo)
        self.nav = QListWidget()
        self.nav.setFixedWidth(190)
        self.nav.addItems(["Projects", "Dashboard", "Video", "Models", "Scene", "JSON Data", "Settings"])
        sidebar.addWidget(self.nav)
        sidebar.addWidget(QLabel("LOCAL WORKSPACE\nNo cloud · No accounts"))
        layout.addLayout(sidebar)
        content = QVBoxLayout()
        self.header = QLabel("Create a project to begin")
        self.header.setObjectName("muted")
        content.addWidget(self.header)
        self.stack = QStackedWidget()
        content.addWidget(self.stack)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.cancel_button = button("Cancel processing", self.cancel_job)
        self.cancel_button.setVisible(False)
        content.addLayout(row(self.progress, self.cancel_button))
        layout.addLayout(content, 1)
        self.setCentralWidget(shell)
        self.build_projects()
        self.build_dashboard()
        self.build_video()
        self.build_models()
        self.build_scene()
        self.build_json()
        self.build_settings()
        for control in (self.interval, self.blur, self.duplicate_threshold, self.maximum):
            control.valueChanged.connect(self.analysis_settings_changed)
        for control in self.analysis_widgets.values():
            signal = control.currentTextChanged if isinstance(control, QComboBox) else control.valueChanged
            signal.connect(self.analysis_settings_changed)
        self.nav.currentRowChanged.connect(self.navigate)
        self.nav.setCurrentRow(0)
        self.apply_theme(self.defaults.theme)
        action = QAction("Save", self)
        action.setShortcut(QKeySequence.StandardKey.Save)
        action.triggered.connect(self.manual_save)
        self.addAction(action)
        self.refresh_recent()

    def page(self, title, subtitle):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(22, 12, 22, 12)
        heading = QLabel(title)
        heading.setObjectName("title")
        layout.addWidget(heading)
        help_text = QLabel(subtitle)
        help_text.setWordWrap(True)
        help_text.setObjectName("muted")
        layout.addWidget(help_text)
        self.stack.addWidget(widget)
        return layout

    def build_projects(self):
        layout = self.page("Your projects", "A local workspace for each flight. Create a project or pick up where you left off.")
        layout.addLayout(row(button("+ New project", self.new_project, True), button("Open project…", self.open_project), button("Open sample scene", self.open_sample)))
        self.recent = QListWidget()
        self.recent.itemDoubleClicked.connect(lambda item: self.open_path(Path(item.data(Qt.ItemDataRole.UserRole))))
        layout.addWidget(self.recent, 1)
        details = QGroupBox("Current project")
        form = QFormLayout(details)
        self.project_name = QLineEdit()
        self.description = QTextEdit()
        self.description.setMaximumHeight(85)
        form.addRow("Name", self.project_name)
        form.addRow("Description", self.description)
        self.project_details = QLabel("No project selected")
        self.project_details.setWordWrap(True)
        form.addRow(self.project_details)
        form.addRow(button("Save project details", self.rename_project))
        layout.addWidget(details)
        layout.addLayout(row(button("Duplicate…", self.duplicate_project), button("Delete project…", self.delete_project), button("Save now  Ctrl+S", self.manual_save)))

    def build_dashboard(self):
        layout = self.page("Flight to scene", "Import → analyze → reconstruct → inspect. All processing stays on this computer.")
        self.stage = QLabel("No project")
        self.stage.setStyleSheet("font-size:22px;color:#72dfc5;padding:20px;background:#182c32;border-radius:9px")
        layout.addWidget(self.stage)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet("font-size:15px;padding:20px;background:#1a2635;border-radius:9px")
        layout.addWidget(self.stats)
        layout.addLayout(row(button("Import video", self.import_video, True), button("Analyze frames", self.start_analysis), button("Generate scene", self.reconstruct), button("Inspect scene", lambda: self.nav.setCurrentRow(4))))
        guidance = QLabel("Single-pass workflow: fly steadily over or past a static scene with overlapping sharp frames. No orbit is required. Camera translation supplies depth information; pure rotation does not. Only visible surfaces can be reconstructed.")
        guidance.setWordWrap(True)
        layout.addWidget(guidance)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setPlaceholderText("Processing logs will appear here.")
        layout.addWidget(self.logs, 1)

    def build_video(self):
        layout = self.page("Video & frame analysis", "Sharp, overlapping frames make better input. Rejected images are stored as small previews.")
        layout.addLayout(row(button("Import / replace video…", self.import_video, True), button("Generate synthetic test video", self.synthetic)))
        self.preview = QLabel("No video loaded")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(190)
        self.preview.setMaximumHeight(280)
        self.preview.setStyleSheet("background:#080e16;border-radius:8px")
        layout.addWidget(self.preview)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.sliderMoved.connect(self.seek)
        self.time_label = QLabel("0:00 / 0:00")
        layout.addLayout(row(button("Play / pause", self.toggle_play), self.timeline, self.time_label))
        self.video_info = QLabel("MP4, MOV and AVI, subject to your OpenCV codec support. Audio is not played.")
        self.video_info.setWordWrap(True)
        layout.addWidget(self.video_info)
        controls = QGroupBox("Extraction settings")
        form = QHBoxLayout(controls)
        self.interval = spin(1, .05, 3600)
        self.blur = spin(35, 0, 100000)
        self.duplicate_threshold = spin(2.5, 0, 255)
        self.maximum = QSpinBox()
        self.maximum.setRange(1, 2000)
        self.maximum.setValue(150)
        for label, widget in (("Interval (s)", self.interval), ("Min sharpness", self.blur), ("Duplicate Δ", self.duplicate_threshold), ("Max samples", self.maximum)):
            form.addWidget(QLabel(label))
            form.addWidget(widget)
        form.addWidget(button("Analyze", self.start_analysis, True))
        layout.addWidget(controls)
        self.frame_summary = QLabel("No frames extracted yet")
        layout.addWidget(self.frame_summary)
        self.frames = QListWidget()
        self.frames.setViewMode(QListWidget.ViewMode.IconMode)
        self.frames.setIconSize(QSize(160, 90))
        self.frames.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.frames.setMinimumHeight(145)
        layout.addWidget(self.frames, 1)

    def build_models(self):
        layout = self.page("Model library", "Import geometry or create a labelled demo. Removing an object leaves its file intact.")
        self.model_list = QListWidget()
        self.model_list.currentRowChanged.connect(self.model_selected)
        self.model_list.itemChanged.connect(self.visibility_changed)
        layout.addLayout(row(button("Import model…", self.import_model, True), button("Reconstruct footage", self.reconstruct), button("Generate demo (test only)", lambda: self.reconstruct(demo=True))))
        layout.addLayout(row(*[button("+ " + kind, lambda k=kind: self.add_primitive(k)) for kind in ("Drone", "Box", "Sphere")]))
        layout.addWidget(self.model_list, 1)
        self.model_info = QLabel("Select a model to view its details.")
        self.model_info.setWordWrap(True)
        layout.addWidget(self.model_info)
        layout.addLayout(row(button("Rename model", self.rename_model), button("Remove from scene", self.remove_model), button("Open in scene", lambda: self.nav.setCurrentRow(4)), button("Reconstruction files", self.open_reconstruction_files)))

    def build_scene(self):
        layout = self.page("Scene workspace", "Select an object to edit its transform. Image colors remain visible when selected. Demo objects are not reconstructed footage.")
        self.views = QComboBox()
        self.views.addItems(["Perspective", "Front", "Back", "Left", "Right", "Top", "Bottom"])
        self.canvas = SceneCanvas()
        self.views.currentTextChanged.connect(self.canvas.view)
        layout.addLayout(row(self.views, button("Frame all", self.canvas.frame_all), button("Face surface", self.canvas.face_surface), button("Reset camera", self.reset_camera)))
        split = QSplitter()
        split.addWidget(self.canvas)
        panel = QWidget()
        properties = QVBoxLayout(panel)
        self.hierarchy = QListWidget()
        self.hierarchy.currentRowChanged.connect(self.hierarchy_selected)
        self.hierarchy.itemChanged.connect(self.hierarchy_visibility)
        properties.addWidget(QLabel("SCENE OBJECTS"))
        properties.addWidget(self.hierarchy)
        self.transform_boxes = {}
        for name in ("position", "rotation", "scale"):
            group = QGroupBox(name.capitalize())
            form = QFormLayout(group)
            boxes = []
            for axis in "XYZ":
                widget = spin(1 if name == "scale" else 0, .001 if name == "scale" else -1e9, 10000 if name == "scale" else 1e9, 3)
                widget.valueChanged.connect(lambda value, n=name, a=len(boxes): self.transform_changed(n, a, value))
                form.addRow(axis, widget)
                boxes.append(widget)
            self.transform_boxes[name] = boxes
            properties.addWidget(group)
        self.uniform = QCheckBox("Uniform scale")
        properties.addWidget(self.uniform)
        properties.addWidget(button("Reset transform", self.reset_transform))
        properties.addWidget(button("Remove selected object", self.remove_model))
        panel.setMaximumWidth(275)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMaximumWidth(295)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        layout.addWidget(split, 1)
        self.canvas.selected.connect(self.select_id)
        self.canvas.nudge.connect(self.nudge)

    def build_json(self):
        layout = self.page("Project JSON", "Validated schema version 1. JSON exports reference assets; they do not bundle video or geometry files.")
        layout.addLayout(row(button("Refresh", self.refresh_json), button("Copy", self.copy_json), button("Export JSON…", self.export_json), button("Import validated JSON…", self.import_json)))
        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setStyleSheet("font-family:Consolas;font-size:12px")
        layout.addWidget(self.json_view)

    def build_settings(self):
        layout = self.page("Settings", "Settings are saved with the project and used as defaults for new projects.")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        scroll.setWidget(panel)
        layout.addWidget(scroll)
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.setting_widgets = {}
        options = {"theme": ["Dark", "Light"], "reconstruction_mode": ["COLMAP"], "reconstruction_output": ["Dense mesh (CUDA)", "Sparse cloud (CPU)"], "logging_level": ["DEBUG", "INFO", "WARNING", "ERROR"]}
        labels = {"theme": "Theme", "autosave_ms": "Autosave debounce (ms)", "default_project_directory": "Default project directory", "reconstruction_mode": "Reconstruction backend", "reconstruction_output": "Reconstruction output", "executable": "COLMAP executable (.exe)", "background": "Viewer background (#RRGGBB)", "logging_level": "Logging level"}
        for key, value in self.defaults.model_dump().items():
            if key in options:
                widget = QComboBox()
                widget.addItems(options[key])
                widget.setCurrentText(value)
            elif key == "autosave_ms":
                widget = QSpinBox()
                widget.setRange(200, 60000)
                widget.setValue(value)
            else:
                widget = QLineEdit(str(value))
            self.setting_widgets[key] = widget
            form.addRow(labels[key], widget)
        layout.addLayout(form)
        extraction_help = QLabel("Frame interval, sharpness, duplicate threshold and maximum samples are configured on the Video page.")
        extraction_help.setWordWrap(True)
        layout.addWidget(extraction_help)
        layout.addLayout(row(button("Choose COLMAP executable…", self.choose_executable), button("Check COLMAP installation", self.check_colmap)))
        info = QLabel("Reconstruct footage uses real COLMAP photogrammetry. Dense mesh requires a CUDA GPU and produces a PLY surface. Sparse cloud runs on CPU. Dense processing is capped at 1000-pixel images with 1 GB caches. Procedural demos are separate test actions in Models. Download: https://github.com/colmap/colmap/releases")
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info)
        group = QGroupBox("1–3 · Adaptive sampling, quality and GPS keyframes")
        form = QFormLayout(group)
        self.analysis_widgets = {}
        defaults = Project(name="Defaults").analysis
        for key, label, minimum, maximum, decimals in (
            ("sampling_mode", "Sampling method", 0, 0, 0),
            ("min_features", "Minimum ORB features", 0, 2000, 0),
            ("max_clipped_fraction", "Maximum clipped pixel fraction", 0, 1, 2),
            ("gps_spacing_m", "Minimum GPS spacing (m; 0 disables)", 0, 1000, 2),
            ("telemetry_offset_s", "Telemetry time = video time + offset (s)", -86400, 86400, 2),
            ("telemetry_max_gap_s", "Maximum telemetry interpolation gap (s)", .01, 60, 2),
        ):
            if key == "sampling_mode":
                widget = QComboBox()
                widget.addItems(["Adaptive", "Fixed"])
                widget.setCurrentText(defaults.sampling_mode)
            else:
                widget = spin(getattr(defaults, key), minimum, maximum, decimals)
            self.analysis_widgets[key] = widget
            form.addRow(label, widget)
        self.telemetry_info = QLabel("No telemetry loaded. GPS stages require synchronized flight positions.")
        self.telemetry_info.setWordWrap(True)
        form.addRow(self.telemetry_info)
        form.addRow(button("Import telemetry CSV…", self.import_telemetry))
        form.addRow(button("Clear telemetry", self.clear_telemetry))
        layout.addWidget(group)
        group = QGroupBox("4–7 · Reconstruction pipeline (current project)")
        form = QFormLayout(group)
        self.pipeline_widgets = {}
        config = PipelineConfig()
        for key, label in (("capture_mode", "Capture workflow"), ("horizontal_fov_deg", "Known horizontal FOV (degrees; 0 estimates)"), ("depth_method", "Depth method"), ("weights_path", "Local Depth Anything weights folder"), ("midas_weights_path", "MiDaS Small ONNX file"), ("device", "AI inference device"), ("align_gps", "Align SfM to GPS before depth fusion"), ("alignment_max_error_m", "GPS inlier threshold (m)"), ("fusion_voxel_size", "AI cloud voxel size (General workflow)"), ("depth_stride", "Depth surface pixel stride")):
            value = getattr(config, key)
            if key in ("depth_method", "device", "capture_mode"):
                widget = QComboBox()
                widget.addItems(["Single pass", "General"] if key == "capture_mode" else ["COLMAP stereo", "Depth Anything V2", "MiDaS ONNX relief"] if key == "depth_method" else ["Auto", "CPU", "CUDA"])
            elif key == "horizontal_fov_deg":
                widget = spin(0,0,150,2)
            elif key == "align_gps":
                widget = QCheckBox()
            elif key in ("weights_path", "midas_weights_path"):
                widget = QLineEdit(value)
            else:
                widget = spin(value, 2 if key == "depth_stride" else .001, 32 if key == "depth_stride" else 100, 0 if key == "depth_stride" else 3)
            self.pipeline_widgets[key] = widget
            form.addRow(label, widget)
        form.addRow(button("Choose AI weights folder…", self.choose_weights))
        form.addRow(button("Check AI installation", self.check_ai))
        help_text = QLabel("Single pass reconstructs visible depth surfaces from one continuous translating flight, leaving unknown areas open. COLMAP stereo requires CUDA; Depth Anything V2 uses calibrated estimated depth and overrides the output choice above. General retains the previous meshing/cloud workflow. Supply horizontal FOV only if known for this exact rectified video and crop; 0 lets COLMAP estimate it. Without telemetry leave GPS spacing at 0 and alignment unchecked. AI setup: scripts\\setup_ai_windows.cmd.")
        help_text.setWordWrap(True)
        help_text.setText(help_text.text() + " MiDaS ONNX relief builds a filled 2.5D surface from the selected accepted frame (or the highest-quality frame). It runs on CPU without COLMAP, ignores FOV/stride, and requires GPS alignment off. This is estimated image depth, not fused video geometry. Setup: requirements-midas.txt and scripts/download_midas.py.")
        form.addRow(help_text)
        layout.addWidget(group)
        layout.addWidget(button("Save settings", self.save_settings, True))
        layout.addStretch()

    def error(self, text):
        logging.error(text)
        QMessageBox.warning(self, "Drone 3D Studio", str(text))

    def confirm(self, text):
        return QMessageBox.question(self, "Please confirm", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def require_project(self):
        if self.project is None:
            self.error("Create or open a project first.")
            self.nav.setCurrentRow(0)
            return False
        return True

    def idle(self):
        if self.job is not None:
            self.error("Wait for processing to finish or cancel it first.")
            return False
        return True

    def can_switch(self):
        if not self.idle():
            return False
        if not self.autosave.flush():
            self.error("Save failed. Resolve the save error before switching projects.")
            return False
        return True

    def new_project(self):
        if not self.can_switch():
            return
        name, ok = QInputDialog.getText(self, "New project", "Project name")
        if not ok or not name.strip():
            return
        description, ok = QInputDialog.getMultiLineText(self, "Project description", "Describe this flight (optional)")
        if not ok:
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent directory", self.defaults.default_project_directory)
        if not parent:
            return
        import re
        folder = re.sub(r'[^\w -]', '_', name).strip(' .') or "DroneProject"
        try:
            root = Path(parent) / folder
            project = store.create(root, name, description)
            project.settings = self.defaults.model_copy(deep=True)
            try:
                project.analysis = AnalysisConfig.model_validate_json(self.prefs.value("analysis_defaults", project.analysis.model_dump_json()))
            except ValueError:
                pass
            self.activate(root, project)
            self.changed()
        except Exception as exc:
            self.error(f"Could not create project: {exc}")

    def open_project(self):
        if not self.can_switch():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Drone project (project.drone3d.json)")
        if path:
            self.open_path(Path(path).parent)

    def open_path(self, root):
        if not self.can_switch():
            return
        try:
            project = store.load(root)
        except Exception as exc:
            backup = root / "backups" / store.FILE_NAME
            if backup.exists() and self.confirm(f"Project could not be opened: {exc}\nRecover the previous backup?"):
                try:
                    project = Project.model_validate_json(backup.read_text(encoding="utf-8"))
                except Exception as backup_exc:
                    self.error(f"Backup is also unreadable: {backup_exc}")
                    return
            else:
                self.error(f"Could not open project: {exc}")
                return
        self.activate(root, project)

    def activate(self, root, project):
        self.stop_video()
        self.root, self.project, self.meshes = root.resolve(), project, {}
        self.canvas.geometry.clear()
        self.project_name.setText(project.name)
        self.description.setPlainText(project.description)
        for control, value in ((self.interval, project.analysis.interval), (self.blur, project.analysis.blur_threshold), (self.maximum, project.analysis.max_frames), (self.duplicate_threshold, project.analysis.duplicate_threshold)):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self.load_settings_widgets(project.settings)
        for key, widget in self.analysis_widgets.items():
            widget.blockSignals(True)
            value = getattr(project.analysis, key)
            widget.setCurrentText(value) if isinstance(widget, QComboBox) else widget.setValue(value)
            widget.blockSignals(False)
        for key, widget in self.pipeline_widgets.items():
            value = getattr(project.pipeline, key)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(value)
            else:
                widget.setText(value)
        self.refresh_telemetry()
        self.apply_theme(project.settings.theme)
        self.canvas.background = project.settings.background
        logging.getLogger().setLevel(project.settings.logging_level)
        self.refresh()
        self.refresh_frames()
        self.remember(root)
        self.nav.setCurrentRow(1)
        if project.video:
            self.open_capture()
        if project.models:
            records = list(project.models)
            def loading(cancel, progress):
                result = []
                for i, model in enumerate(records):
                    if cancel.is_set():
                        raise video.Cancelled("Model loading cancelled; reopen the project to retry")
                    try:
                        mesh = meshes.load_mesh(store.resolve(self.root, model.path))
                        result.append((model.id, mesh, "Ready"))
                    except Exception as exc:
                        result.append((model.id, None, str(exc)))
                    progress((i + 1) * 100 // len(records), f"Loading {model.name}")
                return result
            self.run_job(loading, self.loaded_models)

    def loaded_models(self, result):
        for key, mesh, status in result:
            model = next(m for m in self.project.models if m.id == key)
            model.status = status
            if mesh is not None:
                self.meshes[key] = mesh
            else:
                self.log(status)
        self.refresh()
        self.canvas.frame_all()
        if any(m.origin == "COLMAP" and m.faces for m in self.project.models):
            self.canvas.face_surface()

    def remember(self, root):
        recent = self.prefs.value("recent", [], type=list)
        recent = [str(root)] + [p for p in recent if p != str(root)]
        self.prefs.setValue("recent", recent[:15])
        self.refresh_recent()

    def refresh_recent(self):
        self.recent.clear()
        for path in self.prefs.value("recent", [], type=list):
            item = QListWidgetItem(f"{Path(path).name}\n{path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.recent.addItem(item)
        if self.recent.count() == 0:
            item = QListWidgetItem("No recent projects. Create your first workspace above.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(item)

    def rename_project(self):
        if not self.require_project():
            return
        try:
            self.project.name = self.project_name.text().strip()
            self.project.description = self.description.toPlainText()
            self.changed()
        except ValueError as exc:
            self.error(exc)

    def duplicate_project(self):
        if not self.require_project() or not self.can_switch():
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent for duplicate")
        if not parent:
            return
        destination = Path(parent) / (self.root.name + "-copy")
        try:
            copy = store.duplicate(self.root, destination, self.project)
            self.activate(destination, copy)
        except Exception as exc:
            self.error(f"Could not duplicate project: {exc}")

    def delete_project(self):
        if not self.require_project() or not self.can_switch():
            return
        root = self.root.resolve()
        if not (root / store.FILE_NAME).is_file() or root == Path(root.anchor):
            self.error("Refusing to delete an unrecognized project directory.")
            return
        if not self.confirm(f"Permanently delete this entire project directory?\n{root}\nThis includes copied source videos, generated models and frames inside it. External source files are retained."):
            return
        self.stop_video()
        try:
            shutil.rmtree(root)
            self.project, self.root, self.meshes = None, None, {}
            self.prefs.setValue("recent", [p for p in self.prefs.value("recent", [], type=list) if Path(p).resolve() != root])
            self.refresh_recent()
            self.refresh()
            self.nav.setCurrentRow(0)
        except Exception as exc:
            self.error(f"Deletion failed: {exc}")

    def open_sample(self):
        if not self.can_switch():
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent for a new sample project")
        if not parent:
            return
        try:
            from drone3d_studio.services.samples import make_sample
            root = Path(parent) / "Drone3D-Sample"
            make_sample(root)
            self.open_path(root)
        except Exception as exc:
            self.error(exc)

    def changed(self):
        if self.project:
            self.autosave.trigger(self.project.settings.autosave_ms)
            self.refresh()

    def save_current(self):
        if self.project and self.root:
            store.save(self.root, self.project)
            store.atomic_write(self.root / "source" / "video_reference.json", self.project.video.model_dump_json(indent=2) if self.project.video else "null")
            self.refresh()

    def manual_save(self):
        if self.require_project():
            self.autosave.dirty = True
            if not self.autosave.flush():
                self.error("Project could not be saved. Check permissions and available disk space.")

    def log(self, text):
        logging.info(text)
        if self.project:
            self.project.logs = (self.project.logs + [f"{now()[11:19]}  {text}"])[-200:]
            self.logs.setPlainText("\n".join(self.project.logs))
            self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def navigate(self, index):
        self.stack.setCurrentIndex(index)
        if index != 2:
            self.play_timer.stop()
        if index == 5:
            self.refresh_json()

    def refresh(self):
        project = self.project
        self.header.setText(f"{project.name}  /  {project.status}" if project else "Create a project to begin")
        self.stage.setText(project.status if project else "No project")
        self.setWindowTitle(f"{project.name} — Drone 3D Studio" if project else "Drone 3D Studio")
        self.loading = True
        selected = self.canvas.selection
        self.model_list.clear()
        self.hierarchy.clear()
        if project:
            accepted = sum(f.accepted for f in project.frames)
            self.stats.setText(f"{project.name}\n\nVideo: {Path(project.video.path).name if project.video else 'Not imported'}\nFrames: {len(project.frames)} sampled · {accepted} accepted · {len(project.frames) - accepted} rejected\nMode: {project.settings.reconstruction_mode} · Models: {len(project.models)}\nReconstruction: {project.reconstruction_status}\nLast saved: {project.modified}")
            self.project_details.setText(f"Created: {project.created}\nModified: {project.modified}\nVideo: {project.video.path if project.video else 'None'}\nStatus: {project.status}\nFolder: {self.root}")
            self.logs.setPlainText("\n".join(project.logs))
            for model in project.models:
                for listing in (self.model_list, self.hierarchy):
                    item = QListWidgetItem(f"{model.name}\n{model.origin} · {model.vertices:,} vertices")
                    item.setData(Qt.ItemDataRole.UserRole, model.id)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if model.visible else Qt.CheckState.Unchecked)
                    item.setToolTip(f"{model.path}\n{model.status}")
                    listing.addItem(item)
                    if model.id == selected:
                        listing.setCurrentItem(item)
        else:
            self.stats.setText("Create a project to import your first video.")
            self.project_details.setText("No project selected")
            self.project_name.clear()
            self.description.clear()
            self.logs.clear()
            self.frames.clear()
            self.json_view.clear()
        self.canvas.set_scene(project.models if project else [], self.meshes)
        self.loading = False
        self.update_properties()

    def import_video(self):
        if not self.require_project() or not self.idle():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import drone video", "", "Video (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if not path:
            return
        if self.project.video and not self.confirm("Replace the video and clear its analysis records? Existing scene objects and source files will be retained."):
            return
        copy = self.confirm("Copy this video into the project? Choose No to keep a reference and avoid copying a large file.")
        def importing(cancel, progress):
            info = video.metadata(Path(path))
            target = Path(path)
            if copy:
                from uuid import uuid4
                target = self.root / "source" / (uuid4().hex[:8] + "_" + Path(path).name)
                try:
                    with open(path, "rb") as source, target.open("wb") as destination:
                        copied = 0
                        while chunk := source.read(1024 * 1024):
                            if cancel.is_set():
                                raise video.Cancelled("Video import cancelled")
                            destination.write(chunk)
                            copied += len(chunk)
                            progress(int(copied / max(info.size, 1) * 100), "Copying source video")
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
            info.path = store.reference(self.root, target)
            return info
        self.run_job(importing, self.video_imported)

    def video_imported(self, info):
        self.stop_video()
        self.project.video = info
        self.project.frames = []
        self.project.telemetry = []
        self.project.telemetry_source = ""
        self.refresh_telemetry()
        self.project.analysis_seconds = 0
        self.project.reconstruction_status = "Not started for current video"
        self.project.status = "Ready for analysis"
        store.atomic_write(self.root / "source" / "video_reference.json", info.model_dump_json(indent=2))
        self.open_capture()
        self.refresh_frames()
        self.changed()
        self.nav.setCurrentRow(2)

    def synthetic(self):
        if not self.require_project() or not self.idle():
            return
        if self.project.video and not self.confirm("Replace the current video reference with a generated synthetic test clip and reset analysis?"):
            return
        def generate(cancel, progress):
            from uuid import uuid4
            target = self.root / "source" / f"synthetic-{uuid4().hex[:8]}.avi"
            video.synthetic_video(target)
            info = video.metadata(target)
            info.path = store.reference(self.root, target)
            return info
        self.run_job(generate, self.video_imported)

    def stop_video(self):
        self.play_timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.preview.clear()
        self.preview.setText("No video loaded")

    def open_capture(self):
        self.stop_video()
        info = self.project.video
        path = store.resolve(self.root, info.path)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.preview.setText("Video missing or unreadable. Use Import / replace to relink it.")
            self.cap.release()
            self.cap = None
            self.log(f"Video unavailable: {path}")
            return
        self.timeline.setRange(0, max(0, info.frame_count - 1))
        self.video_info.setText(f"{path.name}  ·  {info.width} × {info.height}  ·  {info.fps:.2f} fps  ·  {info.frame_count:,} frames  ·  {info.size / 1048576:.1f} MB  ·  {info.duration:.1f}s")
        self.seek(0)

    def seek(self, index):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.play_tick()

    def play_tick(self):
        if self.cap is None:
            return
        index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = self.cap.read()
        if not ok:
            self.play_timer.stop()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.timeline.setValue(index)
        self.time_label.setText(f"{index / self.project.video.fps:.1f}s / {self.project.video.duration:.1f}s")

    def toggle_play(self):
        if self.cap is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
        else:
            if self.timeline.value() >= self.timeline.maximum():
                self.seek(0)
            self.play_timer.start(max(1, round(1000 / self.project.video.fps)))

    def start_analysis(self):
        if not self.require_project() or not self.idle():
            return
        if not self.project.video:
            self.error("Import a video before analyzing frames.")
            return
        config = self.analysis_config()
        self.project.analysis = config
        self.project.status = "Analysis running"
        path = store.resolve(self.root, self.project.video.path)
        fixes = list(self.project.telemetry)
        self.run_job(lambda cancel, progress: video.analyze(path, self.root, config, cancel, progress, fixes), self.analysis_done)
        self.changed()

    def analysis_settings_changed(self):
        if self.project:
            self.project.analysis = self.analysis_config()
            self.retag_frames()
            self.autosave.trigger(self.project.settings.autosave_ms)

    def analysis_config(self):
        values = {key: widget.currentText() if isinstance(widget, QComboBox) else widget.value() for key, widget in self.analysis_widgets.items()}
        values["cover_entire_video"] = self.project.pipeline.capture_mode == "Single pass" if self.project else True
        return AnalysisConfig(interval=self.interval.value(), blur_threshold=self.blur.value(), duplicate_threshold=self.duplicate_threshold.value(), max_frames=self.maximum.value(), **values)

    def analysis_done(self, result):
        self.project.frames, self.project.analysis_seconds = result
        accepted = sum(f.accepted for f in self.project.frames)
        self.project.status = "Ready for reconstruction" if accepted else "Ready for analysis"
        self.log(f"Analysis finished in {self.project.analysis_seconds:.2f}s: {accepted} accepted, {len(self.project.frames) - accepted} rejected.")
        if not accepted:
            self.log("No suitable frames. Reduce the sharpness threshold or choose clearer footage.")
        self.refresh_frames()
        self.changed()

    def refresh_frames(self):
        self.frames.clear()
        if not self.project:
            return
        for frame in self.project.frames:
            item = QListWidgetItem(f"{'Accepted' if frame.accepted else 'Rejected'} · {frame.time:.1f}s\nSharpness {frame.blur:.1f}\n{frame.reason}")
            item.setIcon(QIcon(str(store.resolve(self.root, frame.thumbnail))))
            item.setToolTip(f"{frame.path}\n{frame.reason or 'Passed quality checks'}\nQuality: {frame.quality_score:.0f}/100 · Features: {frame.features}\nClipped pixels: {frame.clipped_fraction:.1%} · Motion: {frame.motion_px:.1f}px\nGPS: {frame.gps.model_dump() if frame.gps else 'Unavailable'}")
            self.frames.addItem(item)
        accepted = sum(f.accepted for f in self.project.frames)
        self.frame_summary.setText(f"{len(self.project.frames)} sampled · {accepted} accepted · {len(self.project.frames) - accepted} rejected · {self.project.analysis_seconds:.2f}s")

    def reconstruct(self, demo=False):
        if not self.require_project() or not self.idle():
            return
        if not self.project.frames:
            self.error("Analyze your video first. You can also add demo primitives from Models without a video.")
            return
        mode = "Demo" if demo else "COLMAP"
        self.project.settings.reconstruction_mode = mode
        self.project.status = "Reconstruction running"
        self.project.reconstruction_status = f"{mode} running"
        if mode == "Demo":
            function = lambda cancel, progress: meshes.demo_scene(self.root, cancel, progress)
        else:
            backend = ColmapBackend(self.project.settings.executable, self.project.settings.reconstruction_output, self.project.pipeline.model_copy(deep=True))
            frames = [frame.model_copy(deep=True) for frame in self.project.frames]
            if self.project.pipeline.depth_method == "MiDaS ONNX relief" and self.frames.currentRow() >= 0:
                selected = frames[self.frames.currentRow()]
                if selected.accepted:
                    frames = [selected]
            function = lambda cancel, progress: backend.run(self.root, frames, cancel, progress)
        self.run_job(function, self.scene_done)
        self.changed()

    def scene_done(self, result):
        if result and result[0][0].origin != "Demo":
            for existing in self.project.models:
                if existing.origin in ("COLMAP", "Depth Anything V2", "MiDaS estimated relief"):
                    existing.visible = False
        for model, mesh in result:
            self.project.models.append(model)
            self.meshes[model.id] = mesh
        self.project.status = "Scene ready"
        self.project.reconstruction_status = "Demo scene — procedural geometry, NOT photogrammetry" if result and result[0][0].origin == "Demo" else "Succeeded — real COLMAP surface mesh" if any(m.faces for m, mesh in result) else "Succeeded — real COLMAP sparse point cloud"
        if any(m.origin == "Depth Anything V2" for m, mesh in result):
            self.project.reconstruction_status = "Succeeded — SfM-calibrated Depth Anything V2 colored cloud"
        if result and result[0][0].origin != "Demo" and self.project.pipeline.capture_mode == "Single pass":
            self.project.reconstruction_status = "Succeeded — single-pass visible surface; unseen areas not reconstructed" if any(m.faces for m, mesh in result) else "Succeeded — single-pass sparse cloud; unseen areas not reconstructed"
        if result and result[0][0].origin != "Demo":
            self.project.reconstruction_status += " · GPS-aligned ENU meters" if self.project.pipeline.align_gps else " · arbitrary SfM units (no GPS)"
        if any(m.origin == "MiDaS estimated relief" for m, mesh in result):
            self.project.reconstruction_status = "Succeeded — MiDaS estimated 2.5D image relief; not fused video geometry or metric scale"
        self.log(self.project.reconstruction_status)
        self.changed()
        self.canvas.frame_all()
        if any(m.origin == "COLMAP" and m.faces for m, mesh in result):
            self.canvas.face_surface()
        if any(m.origin == "MiDaS estimated relief" for m, mesh in result):
            self.canvas.view("Front")
        self.nav.setCurrentRow(4)

    def import_model(self):
        if not self.require_project() or not self.idle():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import model", "", "3D geometry (*.obj *.stl *.ply *.glb *.gltf)")
        if path:
            self.run_job(lambda cancel, progress: [meshes.model_record(self.root, Path(path))], self.models_added)

    def open_reconstruction_files(self):
        if self.require_project():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root / "reconstruction")))

    def add_primitive(self, kind):
        if self.require_project() and self.idle():
            self.run_job(lambda cancel, progress: [meshes.add_primitive(self.root, kind)], self.models_added)

    def models_added(self, result):
        for model, mesh in result:
            self.project.models.append(model)
            self.meshes[model.id] = mesh
        self.project.status = "Scene ready"
        self.changed()
        self.canvas.frame_all()

    def selected_model(self):
        return next((m for m in self.project.models if m.id == self.canvas.selection), None) if self.project else None

    def select_id(self, key):
        self.canvas.selection = key
        self.refresh()

    def model_selected(self, index):
        if not self.loading and self.project and 0 <= index < len(self.project.models):
            self.select_id(self.project.models[index].id)

    def hierarchy_selected(self, index):
        self.model_selected(index)

    def visibility_changed(self, item):
        if not self.loading and self.project:
            model = next(m for m in self.project.models if m.id == item.data(Qt.ItemDataRole.UserRole))
            model.visible = item.checkState() == Qt.CheckState.Checked
            self.changed()

    def hierarchy_visibility(self, item):
        self.visibility_changed(item)

    def update_properties(self):
        model = self.selected_model()
        self.loading = True
        for name, boxes in self.transform_boxes.items():
            values = getattr(model.transform, name) if model else (1, 1, 1) if name == "scale" else (0, 0, 0)
            for box, value in zip(boxes, values):
                box.setEnabled(model is not None)
                box.setValue(value)
        self.model_info.setText(f"{model.name}\n{model.format} · {model.vertices:,} vertices · {model.faces:,} faces\n{model.path}\nStatus: {model.status}" if model else "Select a model to view its details.")
        self.loading = False

    def transform_changed(self, name, axis, value):
        model = self.selected_model()
        if self.loading or model is None:
            return
        values = list(getattr(model.transform, name))
        values[axis] = value
        if name == "scale" and self.uniform.isChecked():
            values = [value] * 3
        setattr(model.transform, name, tuple(values))
        self.canvas.update()
        self.update_properties()
        self.autosave.trigger(self.project.settings.autosave_ms)

    def nudge(self, axis, delta):
        model = self.selected_model()
        if model:
            self.transform_changed("position", axis, model.transform.position[axis] + delta)

    def reset_transform(self):
        model = self.selected_model()
        if model:
            model.transform = Transform()
            self.changed()

    def reset_camera(self):
        self.views.setCurrentText("Perspective")
        self.canvas.view("Perspective")
        self.canvas.frame_all()

    def rename_model(self):
        model = self.selected_model()
        if model:
            name, ok = QInputDialog.getText(self, "Rename model", "Name", text=model.name)
            if ok and name.strip():
                try:
                    model.name = name.strip()
                    self.changed()
                except ValueError as exc:
                    self.error(exc)

    def remove_model(self):
        if not self.idle():
            return
        model = self.selected_model()
        if model:
            self.project.models.remove(model)
            self.meshes.pop(model.id, None)
            self.canvas.selection = ""
            self.changed()

    def refresh_json(self):
        self.json_view.setPlainText(self.project.model_dump_json(indent=2) if self.project else "{}")

    def copy_json(self):
        self.refresh_json()
        QApplication.clipboard().setText(self.json_view.toPlainText())

    def export_json(self):
        if not self.require_project():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", str(self.root / "exports" / "project.json"), "JSON (*.json)")
        if path:
            try:
                store.export_json(Path(path), self.root, self.project)
            except Exception as exc:
                self.error(exc)

    def import_json(self):
        if not self.require_project() or not self.can_switch():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import validated JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            project = store.import_json(Path(path), self.root)
        except Exception as exc:
            self.error(f"JSON validation failed:\n{exc}")
            return
        if self.confirm(f"Replace current project data with '{project.name}'? Asset references are resolved relative to the imported JSON. Current files remain on disk."):
            self.activate(self.root, project)
            self.changed()

    def load_settings_widgets(self, settings):
        for key, value in settings.model_dump().items():
            widget = self.setting_widgets[key]
            if isinstance(widget, QComboBox):
                widget.setCurrentText(value)
            elif isinstance(widget, QSpinBox):
                widget.setValue(value)
            else:
                widget.setText(value)

    def save_settings(self):
        if not self.idle():
            return
        values = {}
        for key, widget in self.setting_widgets.items():
            values[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.value() if isinstance(widget, QSpinBox) else widget.text()
        try:
            settings = Settings.model_validate(values)
        except ValueError as exc:
            self.error(f"Settings validation failed:\n{exc}")
            return
        self.defaults = settings
        self.prefs.setValue("settings", settings.model_dump_json())
        analysis_defaults = self.analysis_config()
        self.prefs.setValue("analysis_defaults", analysis_defaults.model_dump_json())
        self.apply_theme(settings.theme)
        self.canvas.background = settings.background
        self.canvas.update()
        logging.getLogger().setLevel(settings.logging_level)
        if self.project:
            self.project.settings = settings.model_copy(deep=True)
            values = {}
            for key, widget in self.pipeline_widgets.items():
                values[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.isChecked() if isinstance(widget, QCheckBox) else widget.value() if isinstance(widget, QDoubleSpinBox) else widget.text()
            self.project.pipeline = PipelineConfig.model_validate(values)
            self.changed()
        self.statusBar().showMessage("Settings saved")

    def apply_theme(self, theme):
        QApplication.instance().setStyleSheet(DARK if theme == "Dark" else "QWidget {font-family:'Segoe UI';font-size:13px;} QPushButton {padding:9px;} QLabel#title {font-size:27px;font-weight:700;}")

    def choose_executable(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose COLMAP executable", "", "Executable (*.exe)")
        if path:
            self.setting_widgets["executable"].setText(path)

    def choose_weights(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Depth Anything V2 Small weights folder")
        if path:
            self.pipeline_widgets["weights_path"].setText(path)

    def check_ai(self):
        if not self.idle():
            return
        if self.pipeline_widgets["depth_method"].currentText() == "MiDaS ONNX relief":
            from drone3d_studio.reconstruction.midas import MidasPredictor
            weights = Path(self.pipeline_widgets["midas_weights_path"].text())
            if not weights.is_absolute():
                weights = Path(__file__).resolve().parents[2] / weights
            def checking_midas(cancel, progress):
                MidasPredictor(weights)
                return f"MiDaS ONNX loaded on CPU: {weights}\nEstimated image relief; no metric scale."
            self.run_job(checking_midas, lambda message: QMessageBox.information(self, "AI installation", message))
            return
        from drone3d_studio.reconstruction.ai_depth import check_installation
        weights = Path(self.pipeline_widgets["weights_path"].text())
        if not weights.is_absolute():
            weights = Path(__file__).resolve().parents[2] / weights
        def checking(cancel, progress):
            check_installation(weights)
            import torch
            return f"Local weights found: {weights}\nPyTorch {torch.__version__}\nCUDA available: {torch.cuda.is_available()}\nCPU inference works with the optional AI setup."
        self.run_job(checking, lambda message: QMessageBox.information(self, "AI installation", message))

    def refresh_telemetry(self):
        fixes = self.project.telemetry if self.project else []
        self.telemetry_info.setText(f"{len(fixes)} GPS fixes · {fixes[0].time:.2f}–{fixes[-1].time:.2f}s\n{self.project.telemetry_source}" if fixes else "No telemetry loaded. GPS stages require synchronized flight positions.")

    def import_telemetry(self):
        if not self.require_project() or not self.idle():
            return
        if not self.project.video:
            self.error("Import the matching video before its telemetry.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Telemetry: time_s, latitude, longitude, altitude_m", "", "Flight telemetry (*.csv)")
        if not path:
            return
        try:
            from uuid import uuid4
            fixes = telemetry.load_csv(Path(path))
            target = self.root / "source" / f"telemetry-{uuid4().hex[:8]}.csv"
            shutil.copy2(path, target)
            self.project.telemetry = fixes
            self.project.telemetry_source = store.reference(self.root, target)
            self.retag_frames()
            self.refresh_telemetry()
            self.log("Telemetry imported. Re-analyze frames to apply GPS spacing; existing frames have updated GPS timestamps.")
            self.changed()
        except Exception as exc:
            self.error(str(exc))

    def retag_frames(self):
        for frame in self.project.frames:
            frame.gps = telemetry.at_time(self.project.telemetry, frame.time + self.project.analysis.telemetry_offset_s, self.project.analysis.telemetry_max_gap_s)
        self.refresh_frames()

    def clear_telemetry(self):
        if self.require_project() and self.idle():
            self.project.telemetry = []
            self.project.telemetry_source = ""
            self.retag_frames()
            self.refresh_telemetry()
            self.changed()

    def check_colmap(self):
        if self.idle():
            executable = self.setting_widgets["executable"].text()
            self.run_job(lambda cancel, progress: probe(executable, cancel, progress, self.root or Path.cwd()), lambda message: QMessageBox.information(self, "COLMAP check", message))

    def run_job(self, function, success):
        if self.job is not None:
            return
        self.play_timer.stop()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.cancel_button.setVisible(True)
        self.job = Job(function, self)
        self.job.progress.connect(self.job_progress)
        self.job.succeeded.connect(lambda result: self.deliver_result(success, result))
        self.job.failed.connect(self.job_failed)
        self.job.cancelled.connect(self.job_cancelled)
        self.job.finished.connect(self.job_finished)
        self.job.start()

    def deliver_result(self, callback, result):
        try:
            callback(result)
        except Exception as exc:
            logging.exception("Could not apply worker result")
            self.job_failed(str(exc))

    def job_progress(self, value, message):
        self.progress.setRange(0, 0 if value < 0 else 100)
        if value >= 0:
            self.progress.setValue(value)
        self.log(message)

    def job_failed(self, message):
        if self.project:
            if self.project.status == "Reconstruction running":
                self.project.reconstruction_status = "Failed: " + message
            self.project.status = "Failed"
            self.log(message)
            self.changed()
        self.error(message)

    def job_cancelled(self, message):
        if self.project:
            if self.project.status == "Reconstruction running":
                self.project.reconstruction_status = "Cancelled"
            self.project.status = "Cancelled"
            self.log(message)
            self.changed()

    def job_finished(self):
        job, self.job = self.job, None
        if job:
            job.deleteLater()
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        if self.close_pending:
            self.close_pending = False
            self.close()

    def cancel_job(self):
        if self.job:
            self.job.cancel()
            self.statusBar().showMessage("Cancelling… waiting for worker to stop safely")

    def closeEvent(self, event):
        if self.job:
            self.close_pending = True
            self.cancel_job()
            event.ignore()
            return
        if not self.autosave.flush():
            self.error("Save failed. The window will remain open so you can resolve the error.")
            event.ignore()
            return
        self.stop_video()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Create, render and close the UI offscreen")
    parser.add_argument("--screenshot", type=Path, help="Save a screenshot during the smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    from PySide6.QtCore import QStandardPaths
    logdir = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)) / "Drone3DStudio" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logdir / "application.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(message)s")
    window = Studio()
    def unexpected(kind, value, trace):
        logging.error("Unexpected exception", exc_info=(kind, value, trace))
        window.error(f"An unexpected error occurred: {value}\nSee the application log in {logdir}")
    sys.excepthook = unexpected
    window.show()
    if args.smoke_test:
        def finish():
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(args.screenshot))
            window.close()
            app.quit()
        QTimer.singleShot(300, finish)
    return app.exec()
