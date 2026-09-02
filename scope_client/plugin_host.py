from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

from PyQt5 import QtCore, QtGui, QtWidgets


class PluginHost:
    """Small, explicit plugin loader. Plugins execute only after the user loads them."""

    def __init__(self, main_window: QtWidgets.QMainWindow) -> None:
        self.main_window = main_window
        self.loaded: Dict[Path, ModuleType] = {}

    def plugin_directory(self) -> Path:
        executable_dir = Path(sys.argv[0]).resolve().parent
        if getattr(sys, "frozen", False) or (executable_dir / "OpenScope.exe").exists():
            return executable_dir / "plugins"
        return Path(__file__).resolve().parent.parent / "plugins"

    def ensure_directory(self) -> Path:
        path = self.plugin_directory()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def discover(self) -> List[Path]:
        path = self.ensure_directory()
        return sorted(
            item for item in path.glob("*.py")
            if item.is_file() and not item.name.startswith("_")
        )

    def load(self, path: Path) -> str:
        path = path.resolve()
        if path in self.loaded:
            return f"{path.stem} is already loaded."
        module_name = f"openscope_user_plugin_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create an import spec for {path.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if not callable(register):
                raise RuntimeError("Plugin must define register(main_window).")
            register(self.main_window)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        self.loaded[path] = module
        return f"Loaded {path.stem}."


class PluginManagerDialog(QtWidgets.QDialog):
    def __init__(self, host: PluginHost, parent=None) -> None:
        super().__init__(parent)
        self.host = host
        self.setWindowTitle("OpenScope Plugins")
        self.resize(560, 390)

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Plugins are local Python files and run with the same permissions as OpenScope. "
            "Only load plugins you trust."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        self.path_label = QtWidgets.QLineEdit(str(self.host.ensure_directory()))
        self.path_label.setReadOnly(True)
        layout.addWidget(self.path_label)

        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.open_btn = QtWidgets.QPushButton("Open folder")
        self.load_btn = QtWidgets.QPushButton("Load selected")
        buttons.addWidget(self.refresh_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.load_btn)
        layout.addLayout(buttons)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)

        self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn.clicked.connect(self.open_folder)
        self.load_btn.clicked.connect(self.load_selected)
        self.list.itemDoubleClicked.connect(lambda _item: self.load_selected())
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        for path in self.host.discover():
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            if path.resolve() in self.host.loaded:
                item.setText(path.name + "  [loaded]")
            self.list.addItem(item)
        if self.list.count() == 0:
            self.status.setText("No .py plugins found. See plugins/README.md for the minimal API.")
        else:
            self.status.setText(f"{self.list.count()} plugin file(s) found.")

    def open_folder(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.host.ensure_directory())))

    def load_selected(self) -> None:
        item = self.list.currentItem()
        if item is None:
            self.status.setText("Select a plugin first.")
            return
        path = Path(str(item.data(QtCore.Qt.UserRole)))
        try:
            self.status.setText(self.host.load(path))
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Plugin error", f"{path.name}\n\n{exc}")
