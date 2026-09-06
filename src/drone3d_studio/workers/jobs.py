"""Cooperative worker ownership; GUI state is changed only by main-thread slots."""
import logging
import threading
from PySide6.QtCore import QThread, Signal
from drone3d_studio.services.video import Cancelled


class Job(QThread):
    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = self.function(self.cancel_event, self.progress.emit)
            if self.cancel_event.is_set():
                raise Cancelled("Processing cancelled")
            self.succeeded.emit(result)
        except Cancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            logging.exception("Background operation failed")
            self.failed.emit(str(exc))

    def cancel(self):
        self.cancel_event.set()
