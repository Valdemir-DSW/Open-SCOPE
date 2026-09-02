# OpenScope plugins

Place trusted Python plugin files (`.py`) in this folder and load them from **Plugins > Plugin manager**.
Plugins are not auto-loaded.

Minimal API:

```python
from PyQt5 import QtWidgets


def register(main_window):
    action = QtWidgets.QAction("My tool", main_window)
    action.triggered.connect(lambda: QtWidgets.QMessageBox.information(main_window, "Plugin", "Hello"))
    main_window.tools_menu.addAction(action)
```

A plugin receives the live `MainWindow`, so it can add actions/dialogs and read `main_window.current_capture` and `main_window._configs()`.
Plugins execute with the same user permissions as OpenScope. Load only code you trust.
