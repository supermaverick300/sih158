import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pytest
from PySide6.QtCore import QSettings


@pytest.fixture(autouse=True)
def isolated_preferences(tmp_path):
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "prefs"))
