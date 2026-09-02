APP_QSS = r"""
QWidget {
    background: #10161d;
    color: #e6edf6;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow, QDialog { background: #0b1016; }
QToolBar {
    background: #121923;
    border: 0;
    border-bottom: 1px solid #253142;
    spacing: 8px;
    padding: 6px;
}
QToolButton, QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #243246, stop:1 #1a2433);
    border: 1px solid #3a4b64;
    border-radius: 7px;
    padding: 7px 12px;
    min-height: 22px;
}
QToolButton:hover, QPushButton:hover { background: #2b3c52; }
QToolButton:pressed, QPushButton:pressed { background: #16202c; }
QPushButton#runButton { background: #163b2b; border-color: #286447; }
QPushButton#pauseButton { background: #3f3519; border-color: #75622d; }
QPushButton#stopButton { background: #402128; border-color: #6d3642; }
QPushButton#triggerButton { background: #2d2520; border-color: #725d42; }
QPushButton#triggerButton:checked { background: #163b2b; border-color: #2f7b55; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QListWidget {
    background: #0d131a;
    border: 1px solid #334257;
    border-radius: 6px;
    padding: 5px 7px;
    selection-background-color: #315b86;
}
QComboBox::drop-down { border: 0; width: 18px; }
QComboBox QAbstractItemView, QAbstractItemView {
    background: #0d131a;
    color: #e6edf6;
    border: 1px solid #334257;
    outline: 0;
    selection-background-color: #315b86;
    selection-color: #ffffff;
}
QComboBox QAbstractItemView::item {
    min-height: 22px;
    padding: 3px 6px;
    background: #0d131a;
    color: #e6edf6;
}
QComboBox QAbstractItemView::item:selected { background: #315b86; color: #ffffff; }
QComboBox QAbstractItemView::item:disabled { background: #10161d; color: #6f8094; }
QListWidget::item { padding: 5px 6px; border-radius: 4px; }
QListWidget::item:alternate { background: #101820; }
QListWidget::item:selected { background: #27405c; color: #f7fbff; }
QListWidget::item:disabled { background: #0d131a; color: #66788c; }
QScrollBar:vertical { background: #0c1219; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #334257; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar {
    background: #0d131a;
    border: 1px solid #334257;
    border-radius: 6px;
    text-align: center;
    min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e7f5a, stop:1 #39a97d);
    border-radius: 5px;
}
QSlider::groove:horizontal {
    border: 1px solid #334257;
    background: #121b24;
    height: 8px;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #67b6ff;
    border: 1px solid #245280;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QDial {
    background: #141b24;
    min-width: 64px;
    min-height: 64px;
}
QDial::groove {
    background: #202936;
    border: 1px solid #344154;
    border-radius: 32px;
}
QGroupBox {
    border: 1px solid #273241;
    border-radius: 9px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #dce8f5; }
QLabel#mutedLabel { color: #97a7bb; font-size: 9pt; }
QLabel#heroTitle { color: #f3f8ff; font-size: 14pt; font-weight: 700; }
QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }
QDockWidget::title {
    background: #111923;
    border-bottom: 1px solid #263344;
    padding: 7px;
    font-weight: 600;
}
QDockWidget#acquisitionDock QComboBox,
QDockWidget#acquisitionDock QSpinBox,
QDockWidget#acquisitionDock QDoubleSpinBox {
    padding: 3px 5px;
    min-height: 18px;
}
QDockWidget#acquisitionDock QPushButton {
    padding: 4px 6px;
    min-height: 19px;
}
QTableWidget {
    background: #0d131a;
    alternate-background-color: #121a24;
    gridline-color: #243140;
    border: 1px solid #273241;
}
QHeaderView::section {
    background: #16202b;
    border: 0;
    border-right: 1px solid #273241;
    border-bottom: 1px solid #273241;
    padding: 6px;
    font-weight: 600;
}
QTabWidget::pane { border: 1px solid #273241; }
QTabBar::tab {
    background: #131b25;
    padding: 5px 6px;
    border: 1px solid #263344;
    border-bottom: 0;
}
QTabBar::tab:selected { background: #22354c; color: #ffffff; }
QStatusBar { background: #111923; border-top: 1px solid #263344; }
QMenuBar { background: #101720; }
QMenuBar::item:selected, QMenu::item:selected { background: #29405a; }
QMenu { background: #131b24; border: 1px solid #334257; }
QCheckBox::indicator { width: 15px; height: 15px; }
QSplitter::handle { background: #28313d; }
"""


COMPACT_QSS = r"""
QWidget {
    font-size: 9pt;
}
QToolBar {
    spacing: 4px;
    padding: 4px;
}
QToolButton, QPushButton {
    padding: 5px 9px;
    min-height: 18px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QListWidget {
    padding: 4px 6px;
    min-height: 18px;
}
QGroupBox {
    margin-top: 7px;
    padding-top: 7px;
}
QDockWidget::title {
    padding: 4px 6px;
}
QTabBar::tab {
    padding: 4px 5px;
}
QHeaderView::section {
    padding: 3px 4px;
}
QStatusBar {
    min-height: 18px;
}
"""
