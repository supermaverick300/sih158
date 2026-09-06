from PySide6.QtCore import QObject, QTimer, Signal


class Autosave(QObject):
    state = Signal(str)

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.callback = callback
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.flush)
        self.dirty = False

    def trigger(self, interval=800):
        self.dirty = True
        self.state.emit("Unsaved changes")
        self.timer.start(interval)

    def flush(self):
        self.timer.stop()
        if not self.dirty:
            return True
        self.state.emit("Saving…")
        try:
            self.callback()
        except Exception as exc:
            self.state.emit(f"Save failed: {exc}")
            return False
        self.dirty = False
        self.state.emit("Saved")
        return True
