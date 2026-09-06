"""Qt CPU 3D viewport: perspective projection, shaded triangles and point clouds.

The display is capped per object for responsiveness; stored geometry remains intact.
"""
import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from drone3d_studio.services.meshes import matrix


class SceneCanvas(QWidget):
    selected = Signal(str)
    nudge = Signal(int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 330)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Left drag: orbit · right/middle drag: pan · wheel: zoom · click: select · arrows: move XY · PageUp/Down: move Z")
        self.models, self.geometry, self.hit = [], {}, []
        self.vertex_colors = {}
        self.selection = ""
        self.background = "#101923"
        self.target = np.zeros(3)
        self.distance, self.yaw, self.pitch = 12., 40., 30.
        self.last = None
        self.dragged = False

    def set_scene(self, models, meshes):
        self.models = models
        for key, mesh in meshes.items():
            if key not in self.geometry:
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(getattr(mesh, "faces", []), dtype=int)
                visual = getattr(mesh, "visual", None)
                raw_colors = getattr(mesh, "colors", None) if not len(faces) else getattr(visual, "vertex_colors", None) if getattr(visual, "defined", False) else None
                raw_colors = np.asarray(raw_colors) if raw_colors is not None else None
                if raw_colors is not None and len(raw_colors) != len(vertices):
                    raw_colors = None
                if len(faces):
                    faces = faces[::max(1, math.ceil(len(faces) / 1800))]
                    used, inverse = np.unique(faces, return_inverse=True)
                    self.geometry[key] = (vertices[used], inverse.reshape((-1, 3)))
                    self.vertex_colors[key] = raw_colors[used] if raw_colors is not None else None
                else:
                    step = max(1, math.ceil(len(vertices) / 6000))
                    self.geometry[key] = (vertices[::step], np.empty((0, 3), dtype=int))
                    self.vertex_colors[key] = raw_colors[::step] if raw_colors is not None else None
        self.geometry = {k: v for k, v in self.geometry.items() if any(m.id == k for m in models)}
        self.vertex_colors = {k: v for k, v in self.vertex_colors.items() if k in self.geometry}
        self.update()

    def transformed(self, model):
        vertices, faces = self.geometry[model.id]
        transform = matrix(model.transform)
        return vertices @ transform[:3, :3].T + transform[:3, 3], faces

    def frame_all(self):
        arrays = [self.transformed(m)[0] for m in self.models if m.visible and m.id in self.geometry]
        if arrays:
            low = np.min([v.min(axis=0) for v in arrays], axis=0)
            high = np.max([v.max(axis=0) for v in arrays], axis=0)
            self.target = (low + high) / 2
            self.distance = max(float(np.linalg.norm(high - low)) * 1.7, 2)
        else:
            self.target, self.distance = np.zeros(3), 12.
        self.update()

    def view(self, name):
        self.yaw, self.pitch = {"Perspective": (40, 30), "Front": (-90, 0), "Back": (90, 0), "Left": (180, 0), "Right": (0, 0), "Top": (0, 89.9), "Bottom": (0, -89.9)}[name]
        self.update()

    def face_surface(self):
        """Look towards the principal surface without altering model coordinates."""
        arrays = [self.transformed(m)[0] for m in self.models if m.visible and m.faces and m.id in self.geometry]
        if not arrays:
            return
        vertices = np.concatenate(arrays)
        _, _, axes = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
        direction = axes[-1]
        if direction[2] < 0:
            direction = -direction
        self.yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
        self.pitch = float(np.degrees(np.arcsin(np.clip(direction[2], -1, 1))))
        self.frame_all()

    def basis(self):
        yaw, pitch = np.radians([self.yaw, self.pitch])
        forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        up = np.cross(forward, right)
        return forward, right, up

    def project(self, vertices):
        forward, right, up = self.basis()
        relative = vertices - self.target
        depth = self.distance - relative @ forward
        scale = min(self.width(), self.height()) * .9 / np.maximum(depth, .01)
        screen = np.column_stack((self.width() / 2 + relative @ right * scale, self.height() / 2 - relative @ up * scale))
        return screen, depth

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.background))
        extent = max(5, self.distance / 2)
        def line(a, b, color):
            pts, depth = self.project(np.array([a, b]))
            if np.all(depth > .05):
                painter.setPen(QPen(QColor(color), 1))
                painter.drawLine(QPointF(*pts[0]), QPointF(*pts[1]))
        step = max(1, round(extent / 10))
        for i in range(-10, 11):
            line((i * step, -10 * step, 0), (i * step, 10 * step, 0), "#273443")
            line((-10 * step, i * step, 0), (10 * step, i * step, 0), "#273443")
        for axis, color in zip(np.eye(3) * 2, ("#ff7474", "#72e4a6", "#72aaff")):
            line((0, 0, 0), axis, color)
        triangles, points = [], []
        self.hit = []
        colors = ("#56cbb6", "#689ced", "#dcad72", "#b299ef")
        for index, model in enumerate(self.models):
            if not model.visible or model.id not in self.geometry:
                continue
            vertices, faces = self.transformed(model)
            screen, depth = self.project(vertices)
            color = QColor("#ffcb70" if model.id == self.selection else colors[index % len(colors)])
            rgb = self.vertex_colors.get(model.id) if model.id != self.selection else None
            if len(faces):
                for face in faces:
                    if np.any(depth[face] <= .05):
                        continue
                    polygon = QPolygonF([QPointF(*p) for p in screen[face]])
                    if rgb is not None:
                        color = QColor(*map(int, rgb[face, :3].mean(axis=0)))
                    normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
                    shade = .55 + .45 * abs(float(normal @ np.array([.3, .4, .85]))) / max(np.linalg.norm(normal), 1e-9)
                    tint = QColor.fromRgbF(min(1, color.redF() * shade), min(1, color.greenF() * shade), min(1, color.blueF() * shade))
                    triangles.append((float(depth[face].mean()), polygon, tint, model.id))
            else:
                points.extend((float(d), QPointF(*p), QColor(*map(int, rgb[i, :3])) if rgb is not None else color, model.id) for i, (p, d) in enumerate(zip(screen, depth)) if d > .05)
        for depth, shape, color, key in sorted(triangles + points, key=lambda x: -x[0]):
            if isinstance(shape, QPolygonF):
                painter.setPen(QPen(color.darker(115), .4))
                painter.setBrush(color)
                painter.drawPolygon(shape)
            else:
                painter.setPen(QPen(color, 3))
                painter.drawPoint(shape)
            self.hit.append((depth, shape, key))
        painter.setPen(QColor("#a2b2c7"))
        painter.drawText(18, 27, "SCENE / CPU VIEWPORT")
        painter.drawText(18, self.height() - 18, "Orbit: drag   •   Pan: right drag   •   Zoom: wheel   •   Select: click")
        if not self.models:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Your scene starts here\nAnalyze your footage, then reconstruct with COLMAP")
        painter.end()

    def mousePressEvent(self, event):
        self.setFocus()
        self.last, self.dragged = event.position(), False

    def mouseMoveEvent(self, event):
        if self.last is None:
            return
        delta = event.position() - self.last
        if abs(delta.x()) + abs(delta.y()) > 2:
            self.dragged = True
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.yaw -= delta.x() * .5
            self.pitch = max(-89.9, min(89.9, self.pitch + delta.y() * .5))
        elif event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            _, right, up = self.basis()
            self.target += (-right * delta.x() + up * delta.y()) * self.distance / max(self.height(), 1)
        self.last = event.position()
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.dragged and event.button() == Qt.MouseButton.LeftButton:
            for _, shape, key in sorted(self.hit, key=lambda x: x[0]):
                match = shape.containsPoint(event.position(), Qt.FillRule.OddEvenFill) if isinstance(shape, QPolygonF) else (shape - event.position()).manhattanLength() < 7
                if match:
                    self.selected.emit(key)
                    break
        self.last = None

    def wheelEvent(self, event):
        self.distance = max(.1, min(1e8, self.distance * math.exp(-event.angleDelta().y() / 1200)))
        self.update()

    def keyPressEvent(self, event):
        mapping = {Qt.Key.Key_Left: (0, -.1), Qt.Key.Key_Right: (0, .1), Qt.Key.Key_Up: (1, .1), Qt.Key.Key_Down: (1, -.1), Qt.Key.Key_PageUp: (2, .1), Qt.Key.Key_PageDown: (2, -.1)}
        if event.key() in mapping:
            self.nudge.emit(*mapping[event.key()])
        else:
            super().keyPressEvent(event)
