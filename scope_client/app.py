from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
import serial
from pyqtgraph.exporters import ImageExporter
from PyQt5 import QtCore, QtGui, QtWidgets
from serial.tools import list_ports

from .demo import demo_capture
from .measurements import analyze, fmt
from .protocol import (
    Capture, CaptureHeader, MAGIC_U32, PACKET_CAPTURE, PACKET_STREAM,
    FLAG_DISCONTINUITY, make_device_command,
)
from .serial_worker import SerialWorker
from .stream_buffer import RollingHistory
from .pc_trigger import find_crossings
from .theme import APP_QSS, COMPACT_QSS
from .ardustim_protocol import ArduStimConfig, ArduStimError, ArduStimFirmwareMismatch, ArduStimProtocol
from .ui_preferences import apply_theme, set_windows_dark_titlebar, translate_ui, tr
from .electrical_tools import ElectricalNetworkDialog, PhaseSequenceDialog
from .plugin_host import PluginHost, PluginManagerDialog

pg.setConfigOptions(antialias=False, useOpenGL=False)

CHANNEL_COLORS = ["#ffd34e", "#51d7ff", "#ff65c3", "#8ee35a"]
MAX_SCOPE_CHANNELS = 4
TIME_DIVS = [
    1e-6, 2e-6, 2.5e-6, 5e-6, 10e-6, 20e-6, 25e-6, 50e-6,
    100e-6, 200e-6, 250e-6, 500e-6, 1e-3, 2e-3, 2.5e-3, 5e-3,
    10e-3, 20e-3, 25e-3, 50e-3, 100e-3, 200e-3, 250e-3, 500e-3,
    1.0, 2.0, 2.5, 5.0,
]
V_DIVS = [
    0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
]
RECORD_LENGTHS = [1.0, 2.0, 5.0, 10.0, 30.0]
VERTICAL_SPANS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]
PROFILE_AUTO = "Auto sample rate"
PROFILE_STANDARD = "Standard"
PROFILE_HIGH = "High speed"
PROFILE_LONG = "Long record"
PROFILE_MANUAL = "Manual"
AUTO_TARGET_POINTS = 20_000
APP_DISPLAY_NAME = "OpenScope"
APP_VERSION = "4.0.1"
APP_ORGANIZATION_NAME = "Valdemir "
OFFICIAL_PROJECT_URL = "https://github.com/Valdemir-DSW/Open-SCOPE"


def resource_path(filename: str) -> Path:
    """Resolve optional UI assets from a Nuitka dist or the source tree."""
    executable_dir = Path(sys.argv[0]).resolve().parent
    dist_candidate = executable_dir / "resources" / filename
    if dist_candidate.exists():
        return dist_candidate
    return Path(__file__).resolve().parent.parent / "resources" / filename


def apply_optional_window_icon(widget: QtWidgets.QWidget) -> None:
    icon_path = resource_path("OpenScope.ico")
    if icon_path.exists():
        widget.setWindowIcon(QtGui.QIcon(str(icon_path)))


def _port_detail(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"n/a", "none", "unknown"}:
        return ""
    return text


def format_serial_port_label(port_info: object) -> str:
    device = _port_detail(getattr(port_info, "device", ""))
    description = _port_detail(getattr(port_info, "description", ""))
    product = _port_detail(getattr(port_info, "product", ""))
    manufacturer = _port_detail(getattr(port_info, "manufacturer", ""))

    primary = description or product or device or "Serial port"
    extras = []
    if device and primary != device:
        extras.append(device)
    if manufacturer and manufacturer not in {primary, description, product}:
        extras.append(manufacturer)
    return primary if not extras else f"{primary} - {' - '.join(extras)}"


def stabilize_tab_widget(tab_widget: QtWidgets.QTabWidget, initialized_attr: str) -> None:
    current = max(0, tab_widget.currentIndex())
    if not getattr(tab_widget, initialized_attr, False) and tab_widget.count() > 1:
        blocker = QtCore.QSignalBlocker(tab_widget)
        tab_widget.setCurrentIndex((current + 1) % tab_widget.count())
        tab_widget.setCurrentIndex(current)
        del blocker
        setattr(tab_widget, initialized_attr, True)
    stack = tab_widget.findChild(QtWidgets.QStackedWidget)
    if stack is not None:
        stack.setUpdatesEnabled(False)
        stack.setCurrentIndex(current)
        for index in range(stack.count()):
            page = stack.widget(index)
            visible = index == current
            page.setVisible(visible)
            if visible:
                page.raise_()
        layout = stack.layout()
        if layout is not None:
            layout.activate()
        stack.setUpdatesEnabled(True)
        stack.updateGeometry()
        stack.update()
    tab_widget.updateGeometry()
    tab_widget.repaint()


ARDUSTIM_WHEEL_PATTERNS = [
    "4 cylinder dizzy",
    "6 cylinder dizzy",
    "8 cylinder dizzy",
    "60-2 crank only",
    "60-2 crank and cam",
    "60-2 crank and half moon cam",
    "36-1 crank only",
    "24-1 crank only",
    "4-1 crank with cam",
    "8-1 crank only (R6)",
    "6-1 crank with cam",
    "12-1 crank with cam",
    "40-1 crank only",
    "Odd fire 90 deg",
    "GM OptiSpark LT1 360 and 8",
    "36-2-2-2 H4 crank only",
    "36-2-2-2 H6 crank only",
    "36-2-2-2 crank and cam",
    "GM 4200 crank wheel",
    "Mazda FE3 36-1 with cam",
    "Mitsubishi 6G72 with cam",
    "Buell Oddfire cam wheel",
    "GM LS1 crank and cam",
    "GM 58x crank and 4x cam",
    "Honda RC51 with cam",
    "36-1 with 2nd trigger on 33-34",
]


def si_time(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:g} µs/div"
    if value < 1:
        return f"{value * 1e3:g} ms/div"
    return f"{value:g} s/div"


def si_volt(value: float) -> str:
    if value < 1:
        return f"{value * 1000:g} mV/div"
    return f"{value:g} V/div"


@dataclass
class ChannelConfig:
    enabled: bool = True
    name: str = "CH"
    full_scale: float = 3.3
    calibration_offset: float = 0.0
    probe_factor: float = 1.0
    v_div: float = 1.0
    x_offset_div: float = 0.0
    position: float = 0.0
    bias_voltage: float = 0.0
    fine_gain: float = 1.0
    invert: bool = False
    software_ac: bool = False
    input_mode: str = "Direct"
    divider_r1: float = 0.0
    divider_r2: float = 1.0

    def raw_to_volts(self, raw: np.ndarray, adc_bits: int = 12, *, apply_ac: bool = True) -> np.ndarray:
        max_count = float((1 << adc_bits) - 1)
        adc_v = raw.astype(np.float64) / max_count * self.full_scale
        values = (
            (adc_v - self.bias_voltage)
            * self.probe_factor
            * self.fine_gain
            + self.calibration_offset
        )
        if self.invert:
            values = -values
        if apply_ac and self.software_ac and values.size:
            values = values - float(np.mean(values))
        return values

    def raw_scalar_to_volts(self, raw: float, adc_bits: int = 12) -> float:
        max_count = float((1 << adc_bits) - 1)
        adc_v = raw / max_count * self.full_scale
        value = (
            (adc_v - self.bias_voltage)
            * self.probe_factor
            * self.fine_gain
            + self.calibration_offset
        )
        return -value if self.invert else value

    def volts_scalar_to_raw(self, volts: float, adc_bits: int = 12) -> int:
        max_count = float((1 << adc_bits) - 1)
        value = -float(volts) if self.invert else float(volts)
        gain = max(abs(self.probe_factor * self.fine_gain), 1e-12)
        adc_v = (value - self.calibration_offset) / gain + self.bias_voltage
        raw = adc_v / max(self.full_scale, 1e-12) * max_count
        return int(round(max(0.0, min(max_count, raw))))


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, cfg: ChannelConfig, adc_bits: int = 12, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{cfg.name} — Input calibration")
        self.setModal(True)
        self.resize(420, 430)
        self._adc_bits = adc_bits

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Direct", "Divider", "Biased divider", "Custom"])
        idx = self.mode.findText(cfg.input_mode)
        self.mode.setCurrentIndex(max(0, idx))

        self.vref = QtWidgets.QDoubleSpinBox()
        self.vref.setRange(0.1, 20.0)
        self.vref.setDecimals(5)
        self.vref.setSuffix(" V")
        self.vref.setValue(cfg.full_scale)

        self.r1 = QtWidgets.QDoubleSpinBox()
        self.r1.setRange(0.0, 100000000.0)
        self.r1.setDecimals(1)
        self.r1.setSuffix(" Ω")
        self.r1.setValue(cfg.divider_r1)

        self.r2 = QtWidgets.QDoubleSpinBox()
        self.r2.setRange(0.001, 100000000.0)
        self.r2.setDecimals(1)
        self.r2.setSuffix(" Ω")
        self.r2.setValue(max(cfg.divider_r2, 0.001))

        self.ratio = QtWidgets.QDoubleSpinBox()
        self.ratio.setRange(0.0001, 10000.0)
        self.ratio.setDecimals(6)
        self.ratio.setValue(cfg.probe_factor)
        self.ratio.setSuffix(" x")

        self.bias = QtWidgets.QDoubleSpinBox()
        self.bias.setRange(-20.0, 20.0)
        self.bias.setDecimals(5)
        self.bias.setSuffix(" V ADC")
        self.bias.setValue(cfg.bias_voltage)

        self.gain = QtWidgets.QDoubleSpinBox()
        self.gain.setRange(0.0001, 1000.0)
        self.gain.setDecimals(6)
        self.gain.setValue(cfg.fine_gain)
        self.gain.setSuffix(" x")

        self.offset = QtWidgets.QDoubleSpinBox()
        self.offset.setRange(-1000.0, 1000.0)
        self.offset.setDecimals(6)
        self.offset.setSuffix(" V")
        self.offset.setValue(cfg.calibration_offset)

        self.invert = QtWidgets.QCheckBox("Invert polarity")
        self.invert.setChecked(cfg.invert)
        self.ac = QtWidgets.QCheckBox("Software AC (remove DC mean)")
        self.ac.setChecked(cfg.software_ac)

        self.range_label = QtWidgets.QLabel()
        self.range_label.setWordWrap(True)

        form.addRow("Input mode", self.mode)
        form.addRow("ADC reference", self.vref)
        form.addRow("Divider R1 (top)", self.r1)
        form.addRow("Divider R2 (bottom)", self.r2)
        form.addRow("Input / ADC ratio", self.ratio)
        form.addRow("Electrical bias", self.bias)
        form.addRow("Fine gain", self.gain)
        form.addRow("Final offset", self.offset)
        form.addRow(self.invert)
        form.addRow(self.ac)
        form.addRow("Calculated range", self.range_label)
        layout.addLayout(form)

        note = QtWidgets.QLabel(
            "Direct input: ratio 1x, bias 0 V. For a resistor divider, enter R1/R2 "
            "and the ratio is calculated automatically. Electrical bias is subtracted "
            "before the divider ratio; it does not create a hardware bias."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.reset_btn = buttons.addButton("Reset direct", QtWidgets.QDialogButtonBox.ResetRole)
        layout.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.reset_btn.clicked.connect(self._reset_direct)

        self.mode.currentTextChanged.connect(self._mode_changed)
        self.r1.valueChanged.connect(self._divider_changed)
        self.r2.valueChanged.connect(self._divider_changed)
        for w in (self.vref, self.ratio, self.bias, self.gain, self.offset):
            w.valueChanged.connect(self._update_range)
        self.invert.toggled.connect(self._update_range)
        self._mode_changed(self.mode.currentText())
        self._update_range()

    def _mode_changed(self, mode: str) -> None:
        divider = mode in ("Divider", "Biased divider")
        self.r1.setEnabled(divider)
        self.r2.setEnabled(divider)
        self.bias.setEnabled(mode in ("Biased divider", "Custom"))
        if mode == "Direct":
            self.ratio.setValue(1.0)
            self.bias.setValue(0.0)
        elif divider:
            self._divider_changed()
        self.ratio.setEnabled(mode in ("Direct", "Custom"))
        self._update_range()

    def _divider_changed(self) -> None:
        if self.mode.currentText() not in ("Divider", "Biased divider"):
            return
        r2 = max(self.r2.value(), 1e-9)
        self.ratio.setValue((self.r1.value() + r2) / r2)
        if self.mode.currentText() == "Divider":
            self.bias.setValue(0.0)
        self._update_range()

    def _reset_direct(self) -> None:
        self.mode.setCurrentText("Direct")
        self.vref.setValue(5.0 if self._adc_bits <= 10 else 3.3)
        self.r1.setValue(0.0)
        self.r2.setValue(1.0)
        self.ratio.setValue(1.0)
        self.bias.setValue(0.0)
        self.gain.setValue(1.0)
        self.offset.setValue(0.0)
        self.invert.setChecked(False)
        self.ac.setChecked(False)
        self._update_range()

    def _update_range(self) -> None:
        lo = (0.0 - self.bias.value()) * self.ratio.value() * self.gain.value() + self.offset.value()
        hi = (self.vref.value() - self.bias.value()) * self.ratio.value() * self.gain.value() + self.offset.value()
        if self.invert.isChecked():
            lo, hi = -lo, -hi
        lo, hi = sorted((lo, hi))
        self.range_label.setText(f"{lo:.4g} V … {hi:.4g} V")

    def apply_to(self, cfg: ChannelConfig) -> ChannelConfig:
        cfg.input_mode = self.mode.currentText()
        cfg.full_scale = self.vref.value()
        cfg.divider_r1 = self.r1.value()
        cfg.divider_r2 = self.r2.value()
        cfg.probe_factor = self.ratio.value()
        cfg.bias_voltage = self.bias.value()
        cfg.fine_gain = self.gain.value()
        cfg.calibration_offset = self.offset.value()
        cfg.invert = self.invert.isChecked()
        cfg.software_ac = self.ac.isChecked()
        return cfg


class ChannelPanel(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(f"CH{index + 1}", parent)
        self.index = index
        self.adc_bits = 12
        self.bias_voltage = 0.0
        self.fine_gain = 1.0
        self.invert = False
        self.software_ac = False
        self.input_mode = "Direct"
        self.divider_r1 = 0.0
        self.divider_r2 = 1.0

        self.setCheckable(True)
        self.setChecked(True)
        self.setStyleSheet(
            f"QGroupBox{{border-top:2px solid {CHANNEL_COLORS[index]};}}"
        )

        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(8, 12, 8, 8)

        self.name = QtWidgets.QLineEdit(f"CH{index + 1}")
        self.vdiv = QtWidgets.QComboBox()
        for v in V_DIVS:
            self.vdiv.addItem(si_volt(v), v)
        self.vdiv.setCurrentIndex(V_DIVS.index(1.0))
        self.vdiv_zoom_in = QtWidgets.QToolButton()
        self.vdiv_zoom_in.setText("−")
        self.vdiv_zoom_in.setFixedWidth(26)
        self.vdiv_zoom_in.setToolTip("More vertical detail: decrease V/div")
        self.vdiv_zoom_out = QtWidgets.QToolButton()
        self.vdiv_zoom_out.setText("+")
        self.vdiv_zoom_out.setFixedWidth(26)
        self.vdiv_zoom_out.setToolTip("More vertical range: increase V/div")
        vdiv_widget = QtWidgets.QWidget()
        vdiv_layout = QtWidgets.QHBoxLayout(vdiv_widget)
        vdiv_layout.setContentsMargins(0, 0, 0, 0)
        vdiv_layout.setSpacing(3)
        vdiv_layout.addWidget(self.vdiv, 1)
        vdiv_layout.addWidget(self.vdiv_zoom_in)
        vdiv_layout.addWidget(self.vdiv_zoom_out)
        self.vdiv_widget = vdiv_widget

        self.position = QtWidgets.QDoubleSpinBox()
        self.position.setRange(-12.0, 12.0)
        self.position.setDecimals(2)
        self.position.setSingleStep(0.25)
        self.position.setSuffix(" div")
        self.position.setValue((1 - index) * 2.0)

        self.full_scale = QtWidgets.QDoubleSpinBox()
        self.full_scale.setRange(0.01, 1000.0)
        self.full_scale.setDecimals(5)
        self.full_scale.setValue(3.3)
        self.full_scale.setSuffix(" V")

        self.probe = QtWidgets.QDoubleSpinBox()
        self.probe.setRange(0.0001, 10000.0)
        self.probe.setDecimals(4)
        self.probe.setValue(1.0)
        self.probe.setSuffix(" x")

        self.x_shift = QtWidgets.QDoubleSpinBox()
        self.x_shift.setRange(-5.0, 5.0)
        self.x_shift.setDecimals(2)
        self.x_shift.setSingleStep(0.05)
        self.x_shift.setSuffix(" div")

        self.cal_offset = QtWidgets.QDoubleSpinBox()
        self.cal_offset.setRange(-1000.0, 1000.0)
        self.cal_offset.setDecimals(6)
        self.cal_offset.setValue(0.0)
        self.cal_offset.setSuffix(" V")

        self.calibration_btn = QtWidgets.QPushButton("Advanced calibration…")
        self.calibration_summary = QtWidgets.QLabel("Direct · bias 0 V · gain 1x")
        self.calibration_summary.setWordWrap(True)

        form.addRow("Name", self.name)
        form.addRow("V/div", vdiv_widget)
        form.addRow("X shift", self.x_shift)
        form.addRow("Position", self.position)
        form.addRow("ADC reference", self.full_scale)
        form.addRow("Input ratio", self.probe)
        form.addRow("Final offset", self.cal_offset)
        form.addRow(self.calibration_btn)
        form.addRow(self.calibration_summary)

        for widget in (self.full_scale, self.cal_offset):
            label = form.labelForField(widget)
            if label is not None:
                label.hide()
            widget.hide()

        self.toggled.connect(self.changed)
        self.name.textChanged.connect(self.changed)
        self.vdiv.currentIndexChanged.connect(self.changed)
        self.vdiv_zoom_in.clicked.connect(lambda: self._step_vdiv(-1))
        self.vdiv_zoom_out.clicked.connect(lambda: self._step_vdiv(+1))
        self.x_shift.valueChanged.connect(self.changed)
        self.position.valueChanged.connect(self.changed)
        self.full_scale.valueChanged.connect(self._basic_calibration_changed)
        self.probe.valueChanged.connect(self._basic_calibration_changed)
        self.cal_offset.valueChanged.connect(self._basic_calibration_changed)
        self.calibration_btn.clicked.connect(self._advanced_calibration)
        self._update_calibration_summary()

    def _step_vdiv(self, step: int) -> None:
        index = max(0, min(self.vdiv.count() - 1, self.vdiv.currentIndex() + int(step)))
        self.vdiv.setCurrentIndex(index)

    def _basic_calibration_changed(self) -> None:
        self._update_calibration_summary()
        self.changed.emit()

    def _update_calibration_summary(self) -> None:
        self.calibration_summary.setText(f"{self.input_mode} calibration configured")

    def _advanced_calibration(self) -> None:
        cfg = self.config()
        dlg = CalibrationDialog(cfg, self.adc_bits, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        cfg = dlg.apply_to(cfg)
        blockers = [
            QtCore.QSignalBlocker(self.full_scale),
            QtCore.QSignalBlocker(self.probe),
            QtCore.QSignalBlocker(self.cal_offset),
        ]
        self.full_scale.setValue(cfg.full_scale)
        self.probe.setValue(cfg.probe_factor)
        self.cal_offset.setValue(cfg.calibration_offset)
        self.bias_voltage = cfg.bias_voltage
        self.fine_gain = cfg.fine_gain
        self.invert = cfg.invert
        self.software_ac = cfg.software_ac
        self.input_mode = cfg.input_mode
        self.divider_r1 = cfg.divider_r1
        self.divider_r2 = cfg.divider_r2
        del blockers
        self._update_calibration_summary()
        self.changed.emit()

    def set_adc_bits(self, bits: int) -> None:
        self.adc_bits = int(bits)

    def reset_for_device(self, adc_bits: int) -> None:
        blockers = [
            QtCore.QSignalBlocker(self.full_scale),
            QtCore.QSignalBlocker(self.probe),
            QtCore.QSignalBlocker(self.cal_offset),
        ]
        self.adc_bits = int(adc_bits)
        self.full_scale.setValue(5.0 if adc_bits <= 10 else 3.3)
        self.probe.setValue(1.0)
        self.cal_offset.setValue(0.0)
        self.bias_voltage = 0.0
        self.fine_gain = 1.0
        self.invert = False
        self.software_ac = False
        self.input_mode = "Direct"
        self.divider_r1 = 0.0
        self.divider_r2 = 1.0
        del blockers
        self._update_calibration_summary()
        self.changed.emit()

    def config(self) -> ChannelConfig:
        return ChannelConfig(
            enabled=self.isChecked(),
            name=self.name.text().strip() or f"CH{self.index + 1}",
            full_scale=self.full_scale.value(),
            calibration_offset=self.cal_offset.value(),
            probe_factor=self.probe.value(),
            v_div=float(self.vdiv.currentData()),
            x_offset_div=self.x_shift.value(),
            position=self.position.value(),
            bias_voltage=self.bias_voltage,
            fine_gain=self.fine_gain,
            invert=self.invert,
            software_ac=self.software_ac,
            input_mode=self.input_mode,
            divider_r1=self.divider_r1,
            divider_r2=self.divider_r2,
        )

    def set_config(self, cfg: dict) -> None:
        self.setChecked(bool(cfg.get("enabled", True)))
        self.name.setText(str(cfg.get("name", f"CH{self.index + 1}")))
        target = float(cfg.get("v_div", 1.0))
        idx = min(range(len(V_DIVS)), key=lambda i: abs(V_DIVS[i] - target))
        self.vdiv.setCurrentIndex(idx)
        self.x_shift.setValue(float(cfg.get("x_offset_div", 0.0)))
        self.position.setValue(float(cfg.get("position", 0.0)))
        self.full_scale.setValue(float(cfg.get("full_scale", 3.3)))
        self.probe.setValue(float(cfg.get("probe_factor", 1.0)))
        self.cal_offset.setValue(float(cfg.get("calibration_offset", 0.0)))
        self.bias_voltage = float(cfg.get("bias_voltage", 0.0))
        self.fine_gain = float(cfg.get("fine_gain", 1.0))
        self.invert = bool(cfg.get("invert", False))
        self.software_ac = bool(cfg.get("software_ac", False))
        if "input_mode" in cfg:
            self.input_mode = str(cfg.get("input_mode", "Direct"))
        else:
            legacy_ratio = float(cfg.get("probe_factor", 1.0))
            legacy_bias = float(cfg.get("bias_voltage", 0.0))
            self.input_mode = "Custom" if (abs(legacy_ratio - 1.0) > 1e-9 or abs(legacy_bias) > 1e-12) else "Direct"
        self.divider_r1 = float(cfg.get("divider_r1", 0.0))
        self.divider_r2 = float(cfg.get("divider_r2", 1.0))
        self._update_calibration_summary()


class VoltageAxis(pg.AxisItem):
    """Left axis that keeps the internal graticule in divisions but labels it in volts."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.v_div = 1.0
        self.position_div = 0.0
        self.channel_name = "CH1"
        self.setLabel(text="CH1 Voltage (V)")

    def set_reference(self, cfg: ChannelConfig) -> None:
        self.v_div = max(float(cfg.v_div), 1e-12)
        self.position_div = float(cfg.position)
        self.channel_name = cfg.name or "Channel"
        self.setLabel(text=f"{self.channel_name} Voltage (V)")
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        labels = []
        for div_value in values:
            volts = (float(div_value) - self.position_div) * self.v_div
            av = abs(volts)
            if av < 1e-3 and av > 0:
                labels.append(f"{volts * 1e6:.3g} µV")
            elif av < 1.0 and av > 0:
                labels.append(f"{volts * 1e3:.3g} mV")
            else:
                labels.append(f"{volts:.4g} V")
        return labels


class LockedViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, enableMenu=False, **kwargs)
        self.soft_touch_enabled = False

    def set_soft_touch_enabled(self, enabled: bool) -> None:
        self.soft_touch_enabled = bool(enabled)
        self.setMouseEnabled(x=self.soft_touch_enabled, y=self.soft_touch_enabled)

    def wheelEvent(self, ev, axis=None):
        if not self.soft_touch_enabled:
            ev.ignore()
            return
        super().wheelEvent(ev, axis=axis)

    def mouseDragEvent(self, ev, axis=None):
        if not self.soft_touch_enabled:
            ev.ignore()
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if not self.soft_touch_enabled and ev.button() in (
            QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton, QtCore.Qt.RightButton
        ):
            ev.ignore()
            return
        super().mouseClickEvent(ev)


class SlimDial(QtWidgets.QDial):
    """Thin custom rotary control with smooth pointer movement."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setNotchesVisible(False)
        self.setWrapping(False)
        self.setTracking(True)
        self.setFixedSize(54, 54)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(7.0, 7.0, -7.0, -7.0)

        track = self.palette().color(QtGui.QPalette.Mid)
        active = self.palette().color(QtGui.QPalette.Highlight)
        handle = self.palette().color(QtGui.QPalette.ButtonText)
        if not self.isEnabled():
            active = self.palette().color(QtGui.QPalette.Disabled, QtGui.QPalette.Text)
            handle = active

        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(track, 2.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawArc(rect, 225 * 16, -270 * 16)

        span = max(1, self.maximum() - self.minimum())
        ratio = (self.value() - self.minimum()) / float(span)
        painter.setPen(QtGui.QPen(active, 2.6, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawArc(rect, 225 * 16, int(round(-270.0 * ratio * 16.0)))

        angle = math.radians(225.0 - 270.0 * ratio)
        center = rect.center()
        radius = rect.width() * 0.5
        point = QtCore.QPointF(
            center.x() + math.cos(angle) * radius,
            center.y() - math.sin(angle) * radius,
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(handle)
        painter.drawEllipse(point, 2.3, 2.3)
        painter.end()


class RotarySpin(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int = 2,
        suffix: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._scale = max(1, int(round(1.0 / step)))
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QtWidgets.QLabel(label)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setObjectName("mutedLabel")
        layout.addWidget(title)

        self.dial = SlimDial()
        self.dial.setRange(int(round(minimum * self._scale)), int(round(maximum * self._scale)))
        self.dial.setSingleStep(max(1, int(round(step * self._scale))))
        self.dial.setPageStep(max(1, int(round(step * self._scale * 4))))
        layout.addWidget(self.dial, 0, QtCore.Qt.AlignHCenter)

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin.setAlignment(QtCore.Qt.AlignCenter)
        self.spin.setFixedWidth(78)
        layout.addWidget(self.spin, 0, QtCore.Qt.AlignHCenter)

        self.dial.valueChanged.connect(self._dial_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def _dial_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.spin.setValue(value / self._scale)
        self._updating = False
        self.valueChanged.emit(self.spin.value())

    def _spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.dial.setValue(int(round(value * self._scale)))
        self._updating = False
        self.valueChanged.emit(float(value))

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)


class RotaryChoice(QtWidgets.QWidget):
    currentIndexChanged = QtCore.pyqtSignal(int)

    def __init__(self, label: str, items: List[str], parent=None) -> None:
        super().__init__(parent)
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QtWidgets.QLabel(label)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setObjectName("mutedLabel")
        layout.addWidget(title)

        self._dial_steps_per_item = 8
        self.dial = SlimDial()
        self.dial.setRange(0, max(0, (len(items) - 1) * self._dial_steps_per_item))
        self.dial.setSingleStep(1)
        self.dial.setPageStep(self._dial_steps_per_item)
        layout.addWidget(self.dial, 0, QtCore.Qt.AlignHCenter)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(items)
        self.combo.setFixedWidth(104)
        layout.addWidget(self.combo, 0, QtCore.Qt.AlignHCenter)

        self.dial.valueChanged.connect(self._dial_changed)
        self.combo.currentIndexChanged.connect(self._combo_changed)

    def _dial_changed(self, value: int) -> None:
        if self._updating:
            return
        index = int(round(value / float(self._dial_steps_per_item)))
        index = max(0, min(self.combo.count() - 1, index))
        if index == self.combo.currentIndex():
            return
        self._updating = True
        self.combo.setCurrentIndex(index)
        self._updating = False
        self.currentIndexChanged.emit(index)

    def _combo_changed(self, index: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.dial.setValue(index * self._dial_steps_per_item)
        self._updating = False
        self.currentIndexChanged.emit(index)

    def setCurrentIndex(self, index: int) -> None:
        if self.combo.count() <= 0:
            return
        index = max(0, min(self.combo.count() - 1, int(index)))
        self._updating = True
        self.combo.setCurrentIndex(index)
        self.dial.setValue(index * self._dial_steps_per_item)
        self._updating = False


class ScopePlot(QtWidgets.QWidget):
    cursor_changed = QtCore.pyqtSignal(float, float, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.voltage_axis = VoltageAxis(orientation="left")
        self.view_box = LockedViewBox()
        self.plot = pg.PlotWidget(axisItems={"left": self.voltage_axis}, viewBox=self.view_box)
        self.plot.setBackground("#080b10")
        self.plot.showGrid(x=True, y=True, alpha=0.28)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Time", units="s")
        self.vertical_span = 8.0
        self.view_y_offset = 0.0
        self.view_x_offset_div = 0.0
        self.plot.setYRange(-4.0, 4.0, padding=0)
        self.plot.getViewBox().setLimits(yMin=-64.0, yMax=64.0)
        layout.addWidget(self.plot)

        # Reserved top-right logo slot. The actual asset will be added later.
        self._logo_pixmap = QtGui.QPixmap()
        self.logo_overlay = QtWidgets.QLabel(self.plot)
        self.logo_overlay.setObjectName("plotLogoOverlay")
        self.logo_overlay.setFixedSize(132, 42)
        self.logo_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.logo_overlay.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.logo_overlay.setStyleSheet("background: transparent; border: 0;")
        self._load_optional_logo()

        self.curves = []
        self.history_curves: List[List[pg.PlotDataItem]] = []
        for color in CHANNEL_COLORS:
            curve = self.plot.plot(pen=pg.mkPen(color, width=1))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self.curves.append(curve)
            history = []
            for _ in range(8):
                hc = pg.mkColor(color)
                hc.setAlpha(55)
                h = self.plot.plot(pen=pg.mkPen(hc, width=1))
                h.setClipToView(True)
                h.setDownsampling(auto=True, method="peak")
                h.hide()
                history.append(h)
            self.history_curves.append(history)

        self.trigger_x = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#e56b6f", width=1))
        self.trigger_x.setZValue(20)
        self.plot.addItem(self.trigger_x)

        self.trigger_y = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#e56b6f", width=1, style=QtCore.Qt.DashLine))
        self.trigger_y.setZValue(20)
        self.plot.addItem(self.trigger_y)

        cursor_pen = pg.mkPen("#e5e7eb", width=1, style=QtCore.Qt.DashLine)
        self.t1 = pg.InfiniteLine(pos=-0.001, angle=90, movable=True, pen=cursor_pen, label="T1")
        self.t2 = pg.InfiniteLine(pos=0.001, angle=90, movable=True, pen=cursor_pen, label="T2")
        self.v1 = pg.InfiniteLine(pos=-1.0, angle=0, movable=True, pen=cursor_pen, label="V1")
        self.v2 = pg.InfiniteLine(pos=1.0, angle=0, movable=True, pen=cursor_pen, label="V2")
        for line in (self.t1, self.t2, self.v1, self.v2):
            line.setZValue(30)
            self.plot.addItem(line)
            line.hide()
            line.sigPositionChanged.connect(self._cursor_emit)

        self.delta_text = pg.TextItem(anchor=(0, 0), color="#e5e7eb")
        self.delta_text.setZValue(40)
        self.plot.addItem(self.delta_text)
        self.delta_text.hide()

        self.last_x: Optional[np.ndarray] = None
        self.last_display: List[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * MAX_SCOPE_CHANNELS
        self.persistence = 0
        self.view_box.set_soft_touch_enabled(False)
        self._update_grid_divisions(0.001)

    def _update_grid_divisions(self, time_div: float) -> None:
        x_major = max(float(time_div), 1e-9)
        x_minor = max(x_major / 5.0, 1e-9)
        self.plot.getAxis("bottom").setTickSpacing(major=x_major, minor=x_minor)
        self.voltage_axis.setTickSpacing(major=1.0, minor=0.2)

    def _cursor_emit(self) -> None:
        self.cursor_changed.emit(self.t1.value(), self.t2.value(), self.v1.value(), self.v2.value())

    def set_time_cursors(self, enabled: bool) -> None:
        self.t1.setVisible(enabled)
        self.t2.setVisible(enabled)
        self.delta_text.setVisible(enabled or self.v1.isVisible())

    def set_voltage_cursors(self, enabled: bool) -> None:
        self.v1.setVisible(enabled)
        self.v2.setVisible(enabled)
        self.delta_text.setVisible(enabled or self.t1.isVisible())

    def set_delta_text(self, text: str) -> None:
        self.delta_text.setText(text)
        xr, yr = self.plot.viewRange()
        self.delta_text.setPos(xr[0] + 0.02 * (xr[1] - xr[0]), yr[1] - 0.25)

    def set_persistence(self, count: int) -> None:
        self.persistence = max(0, min(8, int(count)))
        if self.persistence == 0:
            self.clear_persistence_history()

    def clear_persistence_history(self) -> None:
        self.last_display = [None] * MAX_SCOPE_CHANNELS
        for channel in self.history_curves:
            for curve in channel:
                curve.hide()

    def render_capture(
        self,
        capture: Capture,
        configs: List[ChannelConfig],
        time_div: float,
        *,
        rolling: bool = False,
        warmup: bool = False,
        update_persistence: bool = True,
    ) -> None:
        x = capture.time_axis()
        if rolling and warmup and capture.sample_rate > 0:
            # During the first screen fill, draw from left to right instead of
            # showing a tiny trace glued to the right edge. Once one complete
            # visible window exists, normal rolling coordinates take over.
            x = np.arange(capture.frame_count, dtype=np.float64) / float(capture.sample_rate)
        self.last_x = x
        for ch in range(MAX_SCOPE_CHANNELS):
            if ch < capture.channel_count and configs[ch].enabled:
                cfg = configs[ch]
                volts = cfg.raw_to_volts(capture.raw[:, ch], capture.header.adc_bits)
                x_channel = x + cfg.x_offset_div * float(time_div)
                y = volts / max(cfg.v_div, 1e-12) + cfg.position

                if update_persistence and self.persistence and self.last_display[ch] is not None:
                    # Keep X and Y together. The old code paired the *new* X
                    # array with the previous Y array, which crashed whenever
                    # a rolling window changed point count (e.g. 87 vs 39).
                    hist = self.history_curves[ch]
                    for i in range(min(self.persistence - 1, 7), 0, -1):
                        if hist[i - 1].isVisible():
                            xd, yd = hist[i - 1].getData()
                            if xd is not None and yd is not None and xd.shape == yd.shape:
                                hist[i].setData(xd, yd)
                                hist[i].show()
                    prev_x, prev_y = self.last_display[ch]
                    if prev_x.shape == prev_y.shape:
                        hist[0].setData(prev_x, prev_y)
                        hist[0].show()
                    for i in range(self.persistence, 8):
                        hist[i].hide()

                self.last_display[ch] = (x_channel.copy(), y.copy())
                self.curves[ch].setData(x_channel, y)
                self.curves[ch].show()
            else:
                self.curves[ch].hide()
                self.last_display[ch] = None
                for h in self.history_curves[ch]:
                    h.hide()

        # Time/div is authoritative: ten horizontal divisions are always shown.
        # Record/history only controls how much signal is retained on the PC; it
        # must never silently change the visible timebase.
        visible = max(float(time_div) * 10.0, 1e-9)
        x_shift = self.view_x_offset_div * float(time_div)
        if rolling:
            if warmup:
                x_min, x_max = 0.0, visible
            else:
                # Live stream: newest sample is at the right edge (t = 0).
                x_min, x_max = -visible, 0.0
        else:
            # Triggered/snapshot view: place t=0 according to the actual
            # pre-trigger fraction instead of assuming a hard-coded 50 %.
            denom = max(capture.frame_count - 1, 1)
            pre_fraction = max(0.0, min(1.0, capture.pretrigger_frames / denom))
            x_min = -visible * pre_fraction
            x_max = visible * (1.0 - pre_fraction)
        self.plot.setXRange(x_min + x_shift, x_max + x_shift, padding=0)
        self._update_grid_divisions(time_div)

    def set_voltage_reference(self, cfg: ChannelConfig) -> None:
        self.voltage_axis.set_reference(cfg)

    def set_trigger_level(self, y_div: float) -> None:
        self.trigger_y.setValue(y_div)

    def set_trigger_visible(self, enabled: bool, show_time_marker: bool = True) -> None:
        self.trigger_y.setVisible(bool(enabled))
        self.trigger_x.setVisible(bool(enabled and show_time_marker))

    def set_vertical_span(self, span: float) -> None:
        self.vertical_span = max(2.0, float(span))
        half = self.vertical_span * 0.5
        self.plot.setYRange(self.view_y_offset - half, self.view_y_offset + half, padding=0)

    def reset_vertical_view(self) -> None:
        self.view_y_offset = 0.0
        self.set_vertical_span(self.vertical_span)

    def set_view_y_offset(self, offset_div: float) -> None:
        self.view_y_offset = float(offset_div)
        self.set_vertical_span(self.vertical_span)

    def set_view_x_offset_div(self, offset_div: float) -> None:
        self.view_x_offset_div = float(offset_div)

    def set_soft_touch_enabled(self, enabled: bool) -> None:
        self.view_box.set_soft_touch_enabled(enabled)

    def _load_optional_logo(self) -> None:
        logo_path = resource_path("OpenScope_logo.png")
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path))
            if not pixmap.isNull():
                self._logo_pixmap = pixmap
                self.logo_overlay.setPixmap(
                    pixmap.scaled(128, 38, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                )
        self._position_logo_overlay()

    def _position_logo_overlay(self) -> None:
        margin = 12
        x = max(margin, self.plot.width() - self.logo_overlay.width() - margin)
        self.logo_overlay.move(x, margin)
        self.logo_overlay.raise_()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_logo_overlay()

    def export_png(self, path: str) -> None:
        exporter = ImageExporter(self.plot.plotItem)
        exporter.parameters()["width"] = 1920
        exporter.export(path)
        if not self._logo_pixmap.isNull():
            image = QtGui.QImage(path)
            if not image.isNull():
                painter = QtGui.QPainter(image)
                target_w = max(96, min(240, image.width() // 8))
                scaled = self._logo_pixmap.scaledToWidth(target_w, QtCore.Qt.SmoothTransformation)
                painter.drawPixmap(image.width() - scaled.width() - 24, 20, scaled)
                painter.end()
                image.save(path, "PNG")


class LobbyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_DISPLAY_NAME} Lobby")
        self.setModal(True)
        apply_optional_window_icon(self)
        self.setFixedSize(444, 444)

        self.selected_port: Optional[str] = None
        self.selected_baud = 2_000_000
        self.use_demo = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Choose a device")
        title.setObjectName("heroTitle")
        subtitle = QtWidgets.QLabel("Open through serial or enter Demo mode to test the main screen.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("mutedLabel")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        device_box = QtWidgets.QGroupBox("Devices")
        device_layout = QtWidgets.QVBoxLayout(device_box)
        device_layout.setContentsMargins(8, 10, 8, 8)
        device_layout.setSpacing(8)
        self.device_list = QtWidgets.QListWidget()
        self.device_list.setAlternatingRowColors(True)
        self.device_list.itemDoubleClicked.connect(lambda _item: self._accept_serial())
        device_layout.addWidget(self.device_list, 1)
        layout.addWidget(device_box, 1)

        baud_row = QtWidgets.QHBoxLayout()
        baud_row.setSpacing(6)
        baud_row.addWidget(QtWidgets.QLabel("Baud"))
        self.baud = QtWidgets.QComboBox()
        for b in (115200, 460800, 921600, 1000000, 2000000):
            self.baud.addItem(f"{b:,}", b)
        self.baud.setCurrentIndex(self.baud.count() - 1)
        baud_row.addWidget(self.baud, 1)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        baud_row.addWidget(self.refresh_btn)
        layout.addLayout(baud_row)

        self.status = QtWidgets.QLabel("Procurando portas seriais…")
        self.status.setWordWrap(True)
        self.status.setObjectName("mutedLabel")
        layout.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        self.cancel_btn = QtWidgets.QPushButton("Exit")
        self.demo_btn = QtWidgets.QPushButton("Demo")
        self.open_btn = QtWidgets.QPushButton("Open")
        self.open_btn.setDefault(True)
        self.cancel_btn.clicked.connect(self.reject)
        self.demo_btn.clicked.connect(self._accept_demo)
        self.open_btn.clicked.connect(self._accept_serial)
        buttons.addWidget(self.cancel_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.demo_btn)
        buttons.addWidget(self.open_btn)
        layout.addLayout(buttons)

        self.refresh_ports()

    def refresh_ports(self) -> None:
        current = self.device_list.currentItem().data(QtCore.Qt.UserRole) if self.device_list.currentItem() else None
        self.device_list.clear()
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        for p in ports:
            label = format_serial_port_label(p)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, p.device)
            item.setToolTip(label)
            self.device_list.addItem(item)
            if current == p.device:
                self.device_list.setCurrentItem(item)
        if self.device_list.count() and self.device_list.currentRow() < 0:
            self.device_list.setCurrentRow(0)
        language = str(QtCore.QSettings("SCP", "STM32Scope").value("language", "pt"))
        if self.device_list.count():
            if language == "pt":
                self.status.setText(f"{self.device_list.count()} dispositivo(s) encontrados.")
            else:
                self.status.setText(f"{self.device_list.count()} device(s) found.")
        else:
            self.status.setText("Nenhuma porta serial encontrada. Você ainda pode usar Demo." if language == "pt" else "No serial port found. You can still use Demo.")

    def _accept_serial(self) -> None:
        item = self.device_list.currentItem()
        if item is None:
            self.status.setText("Selecione um dispositivo para abrir." if str(QtCore.QSettings("SCP", "STM32Scope").value("language", "pt")) == "pt" else "Select a device to open.")
            return
        self.selected_port = str(item.data(QtCore.Qt.UserRole))
        self.selected_baud = int(self.baud.currentData())
        self.use_demo = False
        self.accept()

    def _accept_demo(self) -> None:
        self.selected_port = None
        self.selected_baud = int(self.baud.currentData())
        self.use_demo = True
        self.accept()


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        language = str(QtCore.QSettings("SCP", "STM32Scope").value("language", "pt"))
        is_pt = language == "pt"
        self.setWindowTitle("Sobre o OpenScope" if is_pt else "About OpenScope")
        apply_optional_window_icon(self)
        self.resize(620, 520)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(f"OpenScope {APP_VERSION}")
        title.setObjectName("heroTitle")
        layout.addWidget(title)

        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(self._about_html(is_pt))
        layout.addWidget(browser, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close_button = buttons.button(QtWidgets.QDialogButtonBox.Close)
        if close_button is not None:
            close_button.setText("Fechar" if is_pt else "Close")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _about_html(self, is_pt: bool) -> str:
        if is_pt:
            return f'''<h3>OpenScope</h3>
<p>Osciloscópio e analisador de sinais de motor construído com Qt 5 / PyQt5.</p>
<p><b>Versão:</b> {APP_VERSION}<br>
<b>Desenvolvimento:</b> Valdemir<br>
<b>Qt:</b> {QtCore.QT_VERSION_STR}<br>
<b>PyQt5:</b> {QtCore.PYQT_VERSION_STR}<br>
<b>PyQtGraph:</b> {getattr(pg, '__version__', 'desconhecida')}<br>
<b>NumPy:</b> {np.__version__}<br>
<b>pySerial:</b> {getattr(serial, '__version__', 'desconhecida')}</p>
<p><b>Projeto:</b> <a href="{OFFICIAL_PROJECT_URL}">{OFFICIAL_PROJECT_URL}</a></p>
<h4>Projetos, bibliotecas e criadores originais</h4>
<ul>
  <li><a href="https://www.qt.io/">Qt — The Qt Company</a></li>
  <li><a href="https://www.riverbankcomputing.com/software/pyqt/">PyQt5 — Riverbank Computing</a></li>
  <li><a href="https://www.pyqtgraph.org/">PyQtGraph</a></li>
  <li><a href="https://numpy.org/">NumPy</a></li>
  <li><a href="https://pyserial.readthedocs.io/">pySerial</a></li>
  <li><a href="https://www.arduino.cc/">Arduino</a></li>
  <li><a href="https://github.com/speeduino/Ardu-Stim">Ardu-Stim — fork Speeduino</a></li>
  <li><a href="https://gitlab.com/libreems-suite/ardu-stim">Ardu-Stim original — David Andruczyk / LibreEMS</a></li>
</ul>
<h4>Ferramentas de compilação para Windows</h4>
<ul>
  <li><a href="https://nuitka.net/">Nuitka</a></li>
  <li><a href="https://jrsoftware.org/isinfo.php">Inno Setup</a></li>
</ul>'''
        return f'''<h3>OpenScope</h3>
<p>Oscilloscope and engine-signal analyzer built with Qt 5 / PyQt5.</p>
<p><b>Version:</b> {APP_VERSION}<br>
<b>Development:</b> Valdemir<br>
<b>Qt:</b> {QtCore.QT_VERSION_STR}<br>
<b>PyQt5:</b> {QtCore.PYQT_VERSION_STR}<br>
<b>PyQtGraph:</b> {getattr(pg, '__version__', 'unknown')}<br>
<b>NumPy:</b> {np.__version__}<br>
<b>pySerial:</b> {getattr(serial, '__version__', 'unknown')}</p>
<p><b>Project:</b> <a href="{OFFICIAL_PROJECT_URL}">{OFFICIAL_PROJECT_URL}</a></p>
<h4>Projects, libraries and original creators</h4>
<ul>
  <li><a href="https://www.qt.io/">Qt — The Qt Company</a></li>
  <li><a href="https://www.riverbankcomputing.com/software/pyqt/">PyQt5 — Riverbank Computing</a></li>
  <li><a href="https://www.pyqtgraph.org/">PyQtGraph</a></li>
  <li><a href="https://numpy.org/">NumPy</a></li>
  <li><a href="https://pyserial.readthedocs.io/">pySerial</a></li>
  <li><a href="https://www.arduino.cc/">Arduino</a></li>
  <li><a href="https://github.com/speeduino/Ardu-Stim">Ardu-Stim — Speeduino fork</a></li>
  <li><a href="https://gitlab.com/libreems-suite/ardu-stim">Original Ardu-Stim — David Andruczyk / LibreEMS</a></li>
</ul>
<h4>Windows build tooling</h4>
<ul>
  <li><a href="https://nuitka.net/">Nuitka</a></li>
  <li><a href="https://jrsoftware.org/isinfo.php">Inno Setup</a></li>
</ul>'''


class ArduStimTab(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.serial_port: Optional[serial.Serial] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.pattern = QtWidgets.QComboBox()
        self.pattern.addItems(ARDUSTIM_WHEEL_PATTERNS)
        self.output_style = QtWidgets.QComboBox()
        self.output_style.addItems(["Scope", "Gauge"])
        self.rpm_mode = QtWidgets.QComboBox()
        self.rpm_mode.addItems(["Potentiometer", "Fixed RPM", "Ranged sweep"])
        self.fixed_rpm = QtWidgets.QSpinBox()
        self.fixed_rpm.setRange(10, 60000)
        self.fixed_rpm.setSingleStep(100)
        self.fixed_rpm.setValue(4000)
        self.fixed_rpm.setSuffix(" RPM")
        self.sweep_min = QtWidgets.QSpinBox()
        self.sweep_min.setRange(10, 60000)
        self.sweep_min.setSingleStep(100)
        self.sweep_min.setValue(1000)
        self.sweep_min.setSuffix(" RPM")
        self.sweep_max = QtWidgets.QSpinBox()
        self.sweep_max.setRange(10, 60000)
        self.sweep_max.setSingleStep(100)
        self.sweep_max.setValue(5000)
        self.sweep_max.setSuffix(" RPM")
        self.sweep_rate = QtWidgets.QSpinBox()
        self.sweep_rate.setRange(1, 10000)
        self.sweep_rate.setValue(250)
        self.sweep_rate.setSuffix(" RPM/s")

        self.reverse_direction = QtWidgets.QCheckBox("Reverse wheel direction")
        self.invert_primary = QtWidgets.QCheckBox("Invert primary")
        self.invert_secondary = QtWidgets.QCheckBox("Invert secondary")
        self.invert_tertiary = QtWidgets.QCheckBox("Invert tertiary")
        self.compression_waves = QtWidgets.QCheckBox("Enable compression waves")
        self.compression_angle = QtWidgets.QComboBox()
        self.compression_angle.addItems(["Every 180°", "Every 360°", "Every 720°"])
        self.compression_amplitude = QtWidgets.QDoubleSpinBox()
        self.compression_amplitude.setRange(0.0, 100.0)
        self.compression_amplitude.setDecimals(1)
        self.compression_amplitude.setValue(25.0)
        self.compression_amplitude.setSuffix(" %")

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(False)

        connection_page = QtWidgets.QWidget()
        connection_layout = QtWidgets.QVBoxLayout(connection_page)
        connection_layout.setContentsMargins(8, 8, 8, 8)
        connection_layout.setSpacing(8)

        ports_box = QtWidgets.QGroupBox("Serial port")
        ports_layout = QtWidgets.QVBoxLayout(ports_box)
        ports_layout.setContentsMargins(8, 10, 8, 8)
        ports_layout.setSpacing(6)
        self.port_list = QtWidgets.QListWidget()
        self.port_list.setAlternatingRowColors(True)
        self.port_list.setMinimumHeight(150)
        self.port_list.itemDoubleClicked.connect(lambda _item: self.toggle_connection())
        ports_layout.addWidget(self.port_list)
        port_buttons = QtWidgets.QHBoxLayout()
        port_buttons.setSpacing(6)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)
        port_buttons.addWidget(self.refresh_btn)
        port_buttons.addWidget(self.connect_btn)
        ports_layout.addLayout(port_buttons)
        connection_layout.addWidget(ports_box)

        info_box = QtWidgets.QGroupBox("Board")
        info_form = QtWidgets.QFormLayout(info_box)
        info_form.setContentsMargins(8, 10, 8, 8)
        info_form.setVerticalSpacing(6)
        self.board_type = QtWidgets.QComboBox()
        self.board_type.addItems(["Arduino Nano", "Arduino Uno", "Arduino Mega"])
        self.firmware_mode = QtWidgets.QComboBox()
        self.firmware_mode.addItems(["Ready", "Needs upload", "Manual firmware"])
        info_form.addRow("Board", self.board_type)
        info_form.addRow("Firmware", self.firmware_mode)
        connection_layout.addWidget(info_box)
        connection_layout.addStretch(1)

        config_page = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_page)
        config_layout.setContentsMargins(8, 8, 8, 8)
        config_layout.setSpacing(8)
        pattern_box = QtWidgets.QGroupBox("Pattern")
        pattern_form = QtWidgets.QFormLayout(pattern_box)
        pattern_form.setContentsMargins(8, 10, 8, 8)
        pattern_form.setVerticalSpacing(6)
        pattern_form.setHorizontalSpacing(8)
        pattern_form.addRow("Wheel pattern", self.pattern)
        pattern_form.addRow("Output style", self.output_style)
        config_layout.addWidget(pattern_box)

        output_box = QtWidgets.QGroupBox("Outputs")
        output_form = QtWidgets.QFormLayout(output_box)
        output_form.setContentsMargins(8, 10, 8, 8)
        output_form.setVerticalSpacing(6)
        output_form.addRow(self.reverse_direction)
        output_form.addRow(self.invert_primary)
        output_form.addRow(self.invert_secondary)
        output_form.addRow(self.invert_tertiary)
        config_layout.addWidget(output_box)
        config_layout.addStretch(1)

        interaction_page = QtWidgets.QWidget()
        interaction_layout = QtWidgets.QVBoxLayout(interaction_page)
        interaction_layout.setContentsMargins(8, 8, 8, 8)
        interaction_layout.setSpacing(8)

        rpm_box = QtWidgets.QGroupBox("RPM")
        rpm_form = QtWidgets.QFormLayout(rpm_box)
        rpm_form.setContentsMargins(8, 10, 8, 8)
        rpm_form.setVerticalSpacing(6)
        rpm_form.setHorizontalSpacing(8)
        rpm_form.addRow("RPM mode", self.rpm_mode)
        rpm_form.addRow("Fixed RPM", self.fixed_rpm)
        rpm_form.addRow("Sweep min", self.sweep_min)
        rpm_form.addRow("Sweep max", self.sweep_max)
        rpm_form.addRow("Sweep rate", self.sweep_rate)
        interaction_layout.addWidget(rpm_box)

        compression_box = QtWidgets.QGroupBox("Compression waves")
        compression_form = QtWidgets.QFormLayout(compression_box)
        compression_form.setContentsMargins(8, 10, 8, 8)
        compression_form.setVerticalSpacing(6)
        compression_form.addRow(self.compression_waves)
        compression_form.addRow("Angle", self.compression_angle)
        compression_form.addRow("Amplitude", self.compression_amplitude)
        interaction_layout.addWidget(compression_box)
        interaction_layout.addStretch(1)

        tabs.addTab(connection_page, "Connection")
        tabs.addTab(config_page, "Configuration")
        tabs.addTab(interaction_page, "Interaction")
        layout.addWidget(tabs, 1)

        self.status = QtWidgets.QLabel("Selecione uma porta ArduStim e conecte.")
        self.status.setWordWrap(True)
        self.status.setObjectName("mutedLabel")
        layout.addWidget(self.status)

        self.rpm_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self.compression_waves.toggled.connect(self._sync_mode_ui)
        self._sync_mode_ui()
        self.refresh_ports()

    def refresh_ports(self) -> None:
        current = self.port_list.currentItem().data(QtCore.Qt.UserRole) if self.port_list.currentItem() else None
        self.port_list.clear()
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        for p in ports:
            label = format_serial_port_label(p)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, p.device)
            item.setToolTip(label)
            self.port_list.addItem(item)
            if current == p.device:
                self.port_list.setCurrentItem(item)
        if self.port_list.count() and self.port_list.currentRow() < 0:
            self.port_list.setCurrentRow(0)
        if self.port_list.count():
            self.status.setText("Portas atualizadas. Escolha a porta do ArduStim.")
        else:
            self.status.setText("Nenhuma porta serial encontrada para o ArduStim.")

    def toggle_connection(self) -> None:
        if self.serial_port is not None:
            self.close_connection()
            return
        item = self.port_list.currentItem()
        if item is None:
            self.status.setText("Selecione uma porta para conectar.")
            return
        port = str(item.data(QtCore.Qt.UserRole))
        try:
            self.serial_port = serial.Serial(port=port, baudrate=115200, timeout=0.1, write_timeout=0.1)
        except Exception as exc:
            self.serial_port = None
            self.status.setText(f"Falha ao conectar em {port}: {exc}")
            return
        self.connect_btn.setText("Disconnect")
        self.status.setText(f"Conectado ao ArduStim em {port}.")

    def close_connection(self) -> None:
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self.connect_btn.setText("Connect")
        self.status.setText("ArduStim desconectado.")

    def _sync_mode_ui(self) -> None:
        mode = self.rpm_mode.currentText()
        fixed = mode == "Fixed RPM"
        sweep = mode == "Ranged sweep"
        compression = self.compression_waves.isChecked()
        self.fixed_rpm.setEnabled(fixed)
        self.sweep_min.setEnabled(sweep)
        self.sweep_max.setEnabled(sweep)
        self.sweep_rate.setEnabled(sweep)
        self.compression_angle.setEnabled(compression)
        self.compression_amplitude.setEnabled(compression)


class WheelPatternWidget(QtWidgets.QWidget):
    """Render the real Ardu-Stim pattern as a scope trace or radial wheel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._states: List[int] = []
        self._degrees = 360
        self._mode = "Wheel"
        self.setMinimumHeight(150)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_pattern(self, states: List[int], degrees: int) -> None:
        self._states = list(states)
        self._degrees = int(degrees) if degrees in (360, 720) else 360
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = str(mode)
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().color(QtGui.QPalette.Base))
        if not self._states:
            painter.setPen(self.palette().color(QtGui.QPalette.Disabled, QtGui.QPalette.Text))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No wheel pattern loaded")
            return
        if self._mode == "Scope":
            self._paint_scope(painter)
        else:
            self._paint_wheel(painter)

    def _paint_scope(self, painter: QtGui.QPainter) -> None:
        rect = self.rect().adjusted(8, 8, -8, -8)
        channels = ((1, "#ffd34e"), (2, "#51d7ff"), (4, "#ff65c3"))
        n = len(self._states)
        for row, (mask, color) in enumerate(channels):
            if not any(v & mask for v in self._states):
                continue
            y0 = rect.top() + (row + 0.5) * rect.height() / 3.0
            amp = rect.height() / 8.0
            path = QtGui.QPainterPath()
            first = True
            previous = 0
            for i, state in enumerate(self._states + [self._states[0]]):
                x = rect.left() + rect.width() * i / n
                high = 1 if state & mask else 0
                y = y0 - amp if high else y0 + amp
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y0 - amp if previous else y0 + amp)
                    path.lineTo(x, y)
                previous = high
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.5))
            painter.drawPath(path)
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Mid), 1))
        painter.drawRect(rect)

    def _paint_wheel(self, painter: QtGui.QPainter) -> None:
        rect = self.rect().adjusted(8, 8, -8, -8)
        center = rect.center()
        radius = max(18.0, min(rect.width(), rect.height()) * 0.34)
        n = len(self._states)
        # Concentric rings make crank/cam/tertiary outputs readable without
        # pretending the pattern is a linear waveform.
        rings = ((1, radius, "#ffd34e"), (2, radius * 0.72, "#51d7ff"), (4, radius * 0.47, "#ff65c3"))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Mid), 1))
        painter.drawEllipse(center, int(radius * 0.31), int(radius * 0.31))
        for mask, r, color in rings:
            if not any(v & mask for v in self._states):
                continue
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.0))
            inner = r * 0.82
            outer = r
            for i, state in enumerate(self._states):
                if not state & mask:
                    continue
                # A 720-degree pattern is two crank revolutions. Folding it on
                # the same circle intentionally overlays the second revolution;
                # cam/tertiary rings still reveal the phase distinction.
                cycle_states = max(1, n // (2 if self._degrees == 720 else 1))
                angle = -math.pi / 2.0 + (2.0 * math.pi * (i % cycle_states) / cycle_states)
                ca, sa = math.cos(angle), math.sin(angle)
                p1 = QtCore.QPointF(center.x() + inner * ca, center.y() + inner * sa)
                p2 = QtCore.QPointF(center.x() + outer * ca, center.y() + outer * sa)
                painter.drawLine(p1, p2)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))
        painter.drawText(rect.adjusted(4, 4, -4, -4), QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom, f"{len(self._states)} states · {self._degrees}°")


class ArduStimPanel(QtWidgets.QWidget):
    RPM_MODE_LABELS = {0: "Ranged sweep", 1: "Fixed RPM", 2: "Potentiometer"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.protocol = ArduStimProtocol()
        self.config: Optional[ArduStimConfig] = None
        self._loading_config = False
        self._pattern_states: List[int] = []
        self._pattern_degrees = 360

        self.pattern = QtWidgets.QComboBox()
        self.output_style = QtWidgets.QComboBox()
        self.output_style.addItems(["Wheel", "Scope"])
        self.rpm_mode = QtWidgets.QComboBox()
        self.rpm_mode.addItem("Potentiometer", 2)
        self.rpm_mode.addItem("Fixed RPM", 1)
        self.rpm_mode.addItem("Ranged sweep", 0)

        self.fixed_rpm_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fixed_rpm_slider.setRange(0, 20000)
        self.fixed_rpm_slider.setSingleStep(50)
        self.fixed_rpm_slider.setPageStep(250)
        self.fixed_rpm = QtWidgets.QSpinBox()
        self.fixed_rpm.setRange(0, 65535)
        self.fixed_rpm.setSingleStep(50)
        self.fixed_rpm.setSuffix(" RPM")
        self.fixed_rpm_feedback = QtWidgets.QProgressBar()
        self.fixed_rpm_feedback.setRange(0, 20000)
        self.fixed_rpm_feedback.setFormat("Device: %v RPM")

        self.sweep_min = QtWidgets.QSpinBox(); self.sweep_min.setRange(0, 65535); self.sweep_min.setSuffix(" RPM")
        self.sweep_max = QtWidgets.QSpinBox(); self.sweep_max.setRange(0, 65535); self.sweep_max.setSuffix(" RPM")
        self.sweep_interval = QtWidgets.QSpinBox(); self.sweep_interval.setRange(0, 65535); self.sweep_interval.setSuffix(" ms")

        self.compression_waves = QtWidgets.QCheckBox("Enable compression waves")
        self.compression_mode = QtWidgets.QComboBox()
        for i in range(3):
            self.compression_mode.addItem(f"Mode {i}", i)
        self.compression_rpm = QtWidgets.QSpinBox(); self.compression_rpm.setRange(0, 65535); self.compression_rpm.setSuffix(" RPM")
        self.compression_offset = QtWidgets.QSpinBox(); self.compression_offset.setRange(0, 65535)
        self.compression_dynamic = QtWidgets.QCheckBox("Dynamic compression")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setElideMode(QtCore.Qt.ElideRight)

        connection_page = QtWidgets.QWidget()
        connection_layout = QtWidgets.QVBoxLayout(connection_page)
        connection_layout.setContentsMargins(6, 6, 6, 6)
        ports_box = QtWidgets.QGroupBox("Connection")
        ports_layout = QtWidgets.QVBoxLayout(ports_box)
        self.port_list = QtWidgets.QListWidget()
        self.port_list.setAlternatingRowColors(True)
        self.port_list.setUniformItemSizes(True)
        self.port_list.setMinimumHeight(160)
        self.port_list.itemDoubleClicked.connect(lambda _item: self.toggle_connection())
        ports_layout.addWidget(self.port_list)
        row = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)
        row.addWidget(self.refresh_btn); row.addWidget(self.connect_btn)
        ports_layout.addLayout(row)
        connection_layout.addWidget(ports_box)

        device_box = QtWidgets.QGroupBox("Device state")
        device_form = QtWidgets.QFormLayout(device_box)
        self.device_info = QtWidgets.QLabel("Not connected")
        self.device_info.setWordWrap(True); self.device_info.setObjectName("mutedLabel")
        self.device_snapshot = QtWidgets.QPlainTextEdit(); self.device_snapshot.setReadOnly(True); self.device_snapshot.setMaximumHeight(150)
        device_form.addRow("Port", self.device_info)
        device_form.addRow(self.device_snapshot)
        connection_layout.addWidget(device_box)
        connection_layout.addStretch(1)

        wheel_page = QtWidgets.QWidget()
        wheel_layout = QtWidgets.QVBoxLayout(wheel_page); wheel_layout.setContentsMargins(6, 6, 6, 6)
        wheel_box = QtWidgets.QGroupBox("Wheel pattern")
        wheel_form = QtWidgets.QFormLayout(wheel_box)
        self.wheel_example = WheelPatternWidget()
        wheel_form.addRow("Pattern", self.pattern)
        wheel_form.addRow("View", self.output_style)
        wheel_form.addRow("Example", self.wheel_example)
        wheel_layout.addWidget(wheel_box); wheel_layout.addStretch(1)

        rpm_page = QtWidgets.QWidget()
        rpm_page_layout = QtWidgets.QVBoxLayout(rpm_page); rpm_page_layout.setContentsMargins(6, 6, 6, 6)
        rpm_box = QtWidgets.QGroupBox("RPM")
        rpm_layout = QtWidgets.QVBoxLayout(rpm_box)
        rpm_form = QtWidgets.QFormLayout(); rpm_form.addRow("Mode", self.rpm_mode); rpm_layout.addLayout(rpm_form)
        fixed_row = QtWidgets.QHBoxLayout(); fixed_row.addWidget(self.fixed_rpm_slider, 2); fixed_row.addWidget(self.fixed_rpm, 1); rpm_layout.addLayout(fixed_row)
        rpm_layout.addWidget(self.fixed_rpm_feedback)
        sweep_form = QtWidgets.QFormLayout(); sweep_form.addRow("Sweep min", self.sweep_min); sweep_form.addRow("Sweep max", self.sweep_max); sweep_form.addRow("Sweep interval", self.sweep_interval); rpm_layout.addLayout(sweep_form)
        rpm_page_layout.addWidget(rpm_box)
        comp_box = QtWidgets.QGroupBox("Compression waves")
        comp_form = QtWidgets.QFormLayout(comp_box)
        comp_form.addRow(self.compression_waves); comp_form.addRow("Mode", self.compression_mode); comp_form.addRow("RPM threshold", self.compression_rpm); comp_form.addRow("Offset", self.compression_offset); comp_form.addRow(self.compression_dynamic)
        rpm_page_layout.addWidget(comp_box); rpm_page_layout.addStretch(1)

        self.tabs.addTab(connection_page, "Conn")
        self.tabs.addTab(wheel_page, "Wheel")
        self.tabs.addTab(rpm_page, "RPM")
        self.tabs.setCurrentIndex(0)
        self.tabs.currentChanged.connect(lambda _index: self._stabilize_tabs())
        QtCore.QTimer.singleShot(0, self._stabilize_tabs)
        QtCore.QTimer.singleShot(80, self._stabilize_tabs)
        layout.addWidget(self.tabs, 1)
        self.status = QtWidgets.QLabel("Select an Ardu-Stim port and connect.")
        self.status.setWordWrap(True); self.status.setObjectName("mutedLabel")
        layout.addWidget(self.status)

        self.fixed_rpm_slider.valueChanged.connect(self.fixed_rpm.setValue)
        self.fixed_rpm.valueChanged.connect(self.fixed_rpm_slider.setValue)
        self.output_style.currentTextChanged.connect(self.wheel_example.set_mode)
        self.pattern.currentIndexChanged.connect(self._pattern_changed)
        self.rpm_mode.currentIndexChanged.connect(self._controls_changed)
        self.compression_waves.toggled.connect(self._controls_changed)
        self.sweep_min.valueChanged.connect(self._normalize_sweep_range)
        self.sweep_max.valueChanged.connect(self._normalize_sweep_range)
        for widget in (self.fixed_rpm, self.sweep_min, self.sweep_max, self.sweep_interval,
                       self.compression_mode, self.compression_rpm, self.compression_offset,
                       self.compression_dynamic):
            if hasattr(widget, "valueChanged"):
                signal = widget.valueChanged
            elif hasattr(widget, "currentIndexChanged"):
                signal = widget.currentIndexChanged
            else:
                signal = widget.toggled
            signal.connect(self._controls_changed)

        self._control_widgets = [self.pattern, self.rpm_mode, self.fixed_rpm_slider, self.fixed_rpm,
                                 self.sweep_min, self.sweep_max, self.sweep_interval, self.compression_waves,
                                 self.compression_mode, self.compression_rpm, self.compression_offset,
                                 self.compression_dynamic]
        self.send_timer = QtCore.QTimer(self); self.send_timer.setSingleShot(True); self.send_timer.setInterval(140); self.send_timer.timeout.connect(self._send_config)
        self.rpm_timer = QtCore.QTimer(self); self.rpm_timer.setInterval(350); self.rpm_timer.timeout.connect(self._poll_rpm)
        self._set_controls_locked(True)
        self._sync_mode_ui()
        self.refresh_ports()

    def _stabilize_tabs(self) -> None:
        stabilize_tab_widget(self.tabs, "_tabs_initialized")

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._stabilize_tabs)

    @property
    def serial_port(self):
        # Compatibility with the old panel and closeEvent checks.
        return self.protocol.serial

    def _msg(self, english: str, portuguese: str) -> str:
        language = str(QtCore.QSettings("SCP", "STM32Scope").value("language", "pt"))
        return portuguese if language == "pt" else english

    def refresh_ports(self) -> None:
        current = self.port_list.currentItem().data(QtCore.Qt.UserRole) if self.port_list.currentItem() else None
        self.port_list.clear()
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        for p in ports:
            label = format_serial_port_label(p)
            item = QtWidgets.QListWidgetItem(label); item.setData(QtCore.Qt.UserRole, p.device); item.setToolTip(label)
            self.port_list.addItem(item)
            if current == p.device: self.port_list.setCurrentItem(item)
        if self.port_list.count() and self.port_list.currentRow() < 0: self.port_list.setCurrentRow(0)
        self.status.setText(self._msg("Ports refreshed. Choose the Ardu-Stim port.", "Portas atualizadas. Escolha a porta do Ardu-Stim.") if self.port_list.count() else self._msg("No serial ports found for Ardu-Stim.", "Nenhuma porta serial encontrada para o Ardu-Stim."))

    def toggle_connection(self) -> None:
        if self.protocol.connected:
            self.close_connection(); return
        item = self.port_list.currentItem()
        if item is None:
            self.status.setText(self._msg("Select a port before connecting.", "Selecione uma porta antes de conectar.")); return
        port = str(item.data(QtCore.Qt.UserRole))
        self.connect_btn.setEnabled(False)
        self.status.setText(self._msg(f"Connecting to {port} and validating Ardu-Stim firmware…", f"Conectando a {port} e validando o firmware Ardu-Stim…"))
        QtWidgets.QApplication.processEvents()
        try:
            self.protocol.open(port)
            cfg = self.protocol.request_config()
            names = self.protocol.request_pattern_names()
            self.config = cfg
            self._load_pattern_names(names)
            self._apply_config(cfg)
            self._refresh_pattern_from_device()
        except ArduStimFirmwareMismatch as exc:
            self.protocol.close(); self.config = None
            self.status.setText(self._msg(f"Firmware mismatch: {exc}", f"Firmware incompatível: {exc}"))
            self.device_snapshot.setPlainText(str(exc)); self.device_info.setText("Not connected")
            self._set_controls_locked(True); self.connect_btn.setEnabled(True); return
        except Exception as exc:
            self.protocol.close(); self.config = None
            self.status.setText(self._msg(f"Connection rejected: no valid Ardu-Stim protocol response ({exc}).", f"Conexão rejeitada: a porta não respondeu com um protocolo Ardu-Stim válido ({exc})."))
            self.device_snapshot.setPlainText(str(exc)); self.device_info.setText("Not connected")
            self._set_controls_locked(True); self.connect_btn.setEnabled(True); return
        self.device_info.setText(self._msg(f"{port} · Ardu-Stim protocol v{cfg.firmware_version} verified", f"{port} · protocolo Ardu-Stim v{cfg.firmware_version} validado"))
        self.connect_btn.setText(self._msg("Disconnect", "Desconectar")); self.connect_btn.setEnabled(True)
        self._set_controls_locked(False); self._sync_mode_ui(); self._update_device_snapshot()
        self.status.setText(self._msg("Ardu-Stim connected and configuration verified.", "Ardu-Stim conectado e configuração validada."))
        self.rpm_timer.start()

    def close_connection(self) -> None:
        self.rpm_timer.stop(); self.send_timer.stop(); self.protocol.close(); self.config = None
        self.connect_btn.setText(self._msg("Connect", "Conectar")); self.connect_btn.setEnabled(True)
        self.device_info.setText(self._msg("Not connected", "Não conectado")); self.device_snapshot.clear(); self._set_controls_locked(True)
        self.status.setText(self._msg("Ardu-Stim disconnected.", "Ardu-Stim desconectado."))

    def _load_pattern_names(self, names: List[str]) -> None:
        blocker = QtCore.QSignalBlocker(self.pattern); self.pattern.clear()
        for index, name in enumerate(names): self.pattern.addItem(name or f"Pattern {index}", index)
        del blocker

    def _apply_config(self, cfg: ArduStimConfig) -> None:
        self._loading_config = True
        widgets = [self.pattern, self.rpm_mode, self.fixed_rpm, self.sweep_min, self.sweep_max,
                   self.sweep_interval, self.compression_waves, self.compression_mode,
                   self.compression_rpm, self.compression_offset, self.compression_dynamic]
        blockers = [QtCore.QSignalBlocker(w) for w in widgets]
        pidx = self.pattern.findData(cfg.wheel)
        if pidx >= 0: self.pattern.setCurrentIndex(pidx)
        midx = self.rpm_mode.findData(cfg.rpm_mode)
        if midx >= 0: self.rpm_mode.setCurrentIndex(midx)
        self.fixed_rpm.setValue(cfg.fixed_rpm); self.sweep_min.setValue(cfg.sweep_min); self.sweep_max.setValue(cfg.sweep_max)
        self.sweep_interval.setValue(cfg.sweep_interval); self.compression_waves.setChecked(bool(cfg.compression_enabled))
        cidx = self.compression_mode.findData(cfg.compression_mode)
        if cidx >= 0: self.compression_mode.setCurrentIndex(cidx)
        self.compression_rpm.setValue(cfg.compression_rpm); self.compression_offset.setValue(cfg.compression_offset); self.compression_dynamic.setChecked(bool(cfg.compression_dynamic))
        del blockers
        self.fixed_rpm_slider.setValue(min(self.fixed_rpm_slider.maximum(), cfg.fixed_rpm))
        self._loading_config = False; self._sync_mode_ui()

    def _build_config(self) -> Optional[ArduStimConfig]:
        if self.config is None: return None
        wheel = self.pattern.currentData()
        return ArduStimConfig(
            firmware_version=self.config.firmware_version, wheel=int(wheel if wheel is not None else self.config.wheel),
            rpm_mode=int(self.rpm_mode.currentData()), fixed_rpm=self.fixed_rpm.value(),
            sweep_min=self.sweep_min.value(), sweep_max=self.sweep_max.value(), sweep_interval=self.sweep_interval.value(),
            compression_enabled=int(self.compression_waves.isChecked()), compression_mode=int(self.compression_mode.currentData()),
            compression_rpm=self.compression_rpm.value(), compression_offset=self.compression_offset.value(),
            compression_dynamic=int(self.compression_dynamic.isChecked()))

    def _controls_changed(self, *_args) -> None:
        self._sync_mode_ui()
        if not self._loading_config and self.protocol.connected: self.send_timer.start()

    def _send_config(self) -> None:
        cfg = self._build_config()
        if cfg is None or not self.protocol.connected: return
        try:
            self.protocol.send_config(cfg, save=False); self.config = cfg; self._update_device_snapshot()
        except Exception as exc:
            self.status.setText(f"Failed to send Ardu-Stim configuration: {exc}")

    def _pattern_changed(self, _index: int) -> None:
        if self._loading_config or not self.protocol.connected: return
        wheel = self.pattern.currentData()
        if wheel is None: return
        try:
            self.protocol.select_pattern(int(wheel), save=True)
            if self.config is not None: self.config.wheel = int(wheel)
            self._refresh_pattern_from_device(); self._update_device_snapshot()
        except Exception as exc:
            self.status.setText(f"Failed to select wheel pattern: {exc}")

    def _refresh_pattern_from_device(self) -> None:
        states, degrees = self.protocol.request_pattern()
        self._pattern_states, self._pattern_degrees = states, degrees
        self.wheel_example.set_pattern(states, degrees)

    def _poll_rpm(self) -> None:
        if not self.protocol.connected: return
        try:
            rpm = self.protocol.request_rpm()
            if rpm > self.fixed_rpm_feedback.maximum(): self.fixed_rpm_feedback.setMaximum(min(65535, max(20000, rpm)))
            self.fixed_rpm_feedback.setValue(min(rpm, self.fixed_rpm_feedback.maximum()))
        except Exception as exc:
            self.rpm_timer.stop(); self.status.setText(f"Ardu-Stim stopped responding: {exc}")

    def _set_controls_locked(self, locked: bool) -> None:
        for widget in self._control_widgets: widget.setEnabled(not locked)
        self.tabs.setTabEnabled(1, not locked); self.tabs.setTabEnabled(2, not locked)

    def _normalize_sweep_range(self, *_args) -> None:
        if self.sweep_min.value() > self.sweep_max.value():
            blocker = QtCore.QSignalBlocker(self.sweep_max); self.sweep_max.setValue(self.sweep_min.value()); del blocker

    def _sync_mode_ui(self) -> None:
        connected = self.protocol.connected and self.config is not None
        mode = int(self.rpm_mode.currentData() if self.rpm_mode.currentData() is not None else 2)
        fixed, sweep = mode == 1, mode == 0
        self.fixed_rpm_slider.setEnabled(connected and fixed); self.fixed_rpm.setEnabled(connected and fixed)
        self.sweep_min.setEnabled(connected and sweep); self.sweep_max.setEnabled(connected and sweep); self.sweep_interval.setEnabled(connected and sweep)
        comp = connected and self.compression_waves.isChecked()
        self.compression_mode.setEnabled(comp); self.compression_rpm.setEnabled(comp); self.compression_offset.setEnabled(comp); self.compression_dynamic.setEnabled(comp)

    def _update_device_snapshot(self) -> None:
        cfg = self._build_config() or self.config
        if cfg is None: return
        name = self.pattern.currentText() if self.pattern.count() else f"#{cfg.wheel}"
        self.device_snapshot.setPlainText(
            f"Firmware protocol: {cfg.firmware_version}\n"
            f"Wheel: #{cfg.wheel} — {name}\n"
            f"Pattern: {len(self._pattern_states)} states / {self._pattern_degrees}°\n"
            f"RPM mode: {self.RPM_MODE_LABELS.get(cfg.rpm_mode, cfg.rpm_mode)}\n"
            f"Fixed RPM: {cfg.fixed_rpm}\n"
            f"Sweep: {cfg.sweep_min} .. {cfg.sweep_max}, interval {cfg.sweep_interval}\n"
            f"Compression: {'on' if cfg.compression_enabled else 'off'}, mode {cfg.compression_mode}, RPM {cfg.compression_rpm}, offset {cfg.compression_offset}, dynamic {bool(cfg.compression_dynamic)}"
        )


class FFTDialog(QtWidgets.QDialog):
    def __init__(self, capture: Capture, configs: List[ChannelConfig], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FFT / Spectrum")
        self.resize(900, 520)
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.channel = QtWidgets.QComboBox()
        for i in range(capture.channel_count):
            self.channel.addItem(configs[i].name, i)
        top.addWidget(QtWidgets.QLabel("Channel"))
        top.addWidget(self.channel)
        top.addStretch(1)
        layout.addLayout(top)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#080b10")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="V")
        layout.addWidget(self.plot)
        self.capture = capture
        self.configs = configs
        self.curve = self.plot.plot(pen=pg.mkPen("#51d7ff", width=1))
        self.channel.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        ch = self.channel.currentData()
        if ch is None:
            return
        y = self.configs[ch].raw_to_volts(self.capture.raw[:, ch], self.capture.header.adc_bits)
        y = y - np.mean(y)
        if y.size < 2:
            return
        window = np.hanning(y.size)
        spec = np.fft.rfft(y * window)
        scale = max(np.sum(window) / 2.0, 1e-12)
        mag = np.abs(spec) / scale
        freq = np.fft.rfftfreq(y.size, d=1.0 / self.capture.sample_rate)
        self.curve.setData(freq, mag)


class XYDialog(QtWidgets.QDialog):
    def __init__(self, capture: Capture, configs: List[ChannelConfig], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("XY Mode")
        self.resize(900, 520)
        self.capture = capture
        self.configs = configs

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.x_channel = QtWidgets.QComboBox()
        self.y_channel = QtWidgets.QComboBox()
        for i in range(capture.channel_count):
            self.x_channel.addItem(configs[i].name, i)
            self.y_channel.addItem(configs[i].name, i)
        if capture.channel_count > 1:
            self.y_channel.setCurrentIndex(1)
        top.addWidget(QtWidgets.QLabel("X"))
        top.addWidget(self.x_channel)
        top.addWidget(QtWidgets.QLabel("Y"))
        top.addWidget(self.y_channel)
        top.addStretch(1)
        layout.addLayout(top)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#080b10")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "X amplitude", units="V")
        self.plot.setLabel("left", "Y amplitude", units="V")
        layout.addWidget(self.plot)
        self.curve = self.plot.plot(pen=pg.mkPen("#7ee787", width=1))

        self.x_channel.currentIndexChanged.connect(self.refresh)
        self.y_channel.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        x_ch = self.x_channel.currentData()
        y_ch = self.y_channel.currentData()
        if x_ch is None or y_ch is None:
            return
        x = self.configs[x_ch].raw_to_volts(self.capture.raw[:, x_ch], self.capture.header.adc_bits)
        y = self.configs[y_ch].raw_to_volts(self.capture.raw[:, y_ch], self.capture.header.adc_bits)
        size = min(x.size, y.size)
        if size < 2:
            return
        self.curve.setData(x[:size], y[:size])


class RpmFrequencyCalculatorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        language = str(QtCore.QSettings("SCP", "STM32Scope").value("language", "pt"))
        self.is_pt = language == "pt"
        self.setWindowTitle("Calculadora RPM ↔ frequência" if self.is_pt else "RPM ↔ Frequency Calculator")
        apply_optional_window_icon(self)
        self.setMinimumWidth(430)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        info = QtWidgets.QLabel(
            "Converte rotação, frequência de dentes e período para rodas fônicas. "
            "Use a quantidade teórica de posições por volta (por exemplo, 36 em uma roda 36-1)."
            if self.is_pt else
            "Converts RPM, tooth frequency and period for trigger wheels. "
            "Use the theoretical positions per revolution (for example, 36 for a 36-1 wheel)."
        )
        info.setWordWrap(True)
        info.setObjectName("mutedLabel")
        layout.addWidget(info)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.teeth = QtWidgets.QSpinBox()
        self.teeth.setRange(1, 360)
        self.teeth.setValue(36)
        self.rpm = QtWidgets.QDoubleSpinBox()
        self.rpm.setRange(0.0, 200000.0)
        self.rpm.setDecimals(1)
        self.rpm.setValue(2500.0)
        self.rpm.setSuffix(" RPM")
        self.frequency = QtWidgets.QDoubleSpinBox()
        self.frequency.setRange(0.0, 1000000.0)
        self.frequency.setDecimals(3)
        self.frequency.setSuffix(" Hz")

        form.addRow("Posições por volta" if self.is_pt else "Positions per revolution", self.teeth)
        form.addRow("Rotação" if self.is_pt else "RPM", self.rpm)
        form.addRow("Frequência de dentes" if self.is_pt else "Tooth frequency", self.frequency)
        layout.addLayout(form)

        result_box = QtWidgets.QGroupBox("Resultados" if self.is_pt else "Results")
        result_layout = QtWidgets.QFormLayout(result_box)
        self.period_value = QtWidgets.QLabel("—")
        self.angle_value = QtWidgets.QLabel("—")
        self.rev_period_value = QtWidgets.QLabel("—")
        result_layout.addRow("Período entre dentes" if self.is_pt else "Tooth period", self.period_value)
        result_layout.addRow("Ângulo por posição" if self.is_pt else "Angle per position", self.angle_value)
        result_layout.addRow("Período por volta" if self.is_pt else "Revolution period", self.rev_period_value)
        layout.addWidget(result_box)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close_button = buttons.button(QtWidgets.QDialogButtonBox.Close)
        if close_button is not None:
            close_button.setText("Fechar" if self.is_pt else "Close")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._syncing = False
        self.teeth.valueChanged.connect(self._from_rpm)
        self.rpm.valueChanged.connect(self._from_rpm)
        self.frequency.valueChanged.connect(self._from_frequency)
        self._from_rpm()

    def _from_rpm(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            frequency = (self.rpm.value() * self.teeth.value()) / 60.0
            self.frequency.setValue(frequency)
            self._update_results(frequency)
        finally:
            self._syncing = False

    def _from_frequency(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            rpm = (self.frequency.value() * 60.0) / max(1, self.teeth.value())
            self.rpm.setValue(rpm)
            self._update_results(self.frequency.value())
        finally:
            self._syncing = False

    def _update_results(self, frequency: float) -> None:
        teeth = max(1, self.teeth.value())
        rpm = self.rpm.value()
        tooth_period = (1.0 / frequency) if frequency > 0.0 else 0.0
        rev_period = (60.0 / rpm) if rpm > 0.0 else 0.0
        self.period_value.setText(f"{tooth_period * 1e6:,.2f} µs" if tooth_period > 0.0 else "—")
        self.angle_value.setText(f"{360.0 / teeth:.3f}°")
        self.rev_period_value.setText(f"{rev_period * 1000.0:,.3f} ms" if rev_period > 0.0 else "—")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenScope — Professional Oscilloscope · Valdemir")
        apply_optional_window_icon(self)
        self.setMinimumSize(1024, 620)
        self.resize(1440, 860)
        self.setDockOptions(
            QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.AnimatedDocks
        )

        self.worker: Optional[SerialWorker] = None
        self.current_capture: Optional[Capture] = None
        self.run_mode = "run"
        self.device_signature = None
        self.active_channel_count = 0
        self.last_display_history_total = -1
        self.stream_history = RollingHistory()
        self.stream_trigger_abs: Optional[int] = None
        self.stream_last_trigger_value: Optional[float] = None
        self.stream_last_packet_end = 0
        self.stream_profile = PROFILE_AUTO
        self.last_stream_render_ts = 0.0
        self.last_measurement_update_ts = 0.0
        self.last_persistence_update_ts = 0.0
        self.stream_trigger_hold_until = 0.0
        self.trigger_armed_at = time.monotonic()
        self.trigger_ignore_until_abs = 0
        self.trigger_has_lock = False
        self.last_stream_capture: Optional[Capture] = None
        self.quick_control_sync_block = False
        self.plugin_host = PluginHost(self)

        # PC trigger pipeline. Edges are collected continuously while packets
        # arrive; rendering consumes only triggers whose post-trigger window is
        # already present. This removes the old one-trigger-at-a-time 0.5 s
        # stall on long timebases.
        self.pending_trigger_events = deque(maxlen=8192)
        self.last_accepted_trigger_abs: Optional[int] = None
        self.last_rendered_trigger_abs: Optional[int] = None

        # USB decoding and plotting run at independent cadences. The serial
        # worker fills a thread-safe queue; this timer drains it in batches.
        self.io_timer = QtCore.QTimer(self)
        self.io_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.io_timer.setInterval(5)
        self.io_timer.timeout.connect(self._drain_worker)
        self.io_timer.start()

        # Fixed-rate display timer: packet bursts no longer dictate frame rate.
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.render_timer.setInterval(16)
        self.render_timer.timeout.connect(self._display_tick)
        self.render_timer.start()

        self.demo_sequence = 0
        self.demo_sample_offset = 0
        self.demo_last_ts = time.monotonic()
        self.demo_timer = QtCore.QTimer(self)
        self.demo_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.demo_timer.timeout.connect(self._demo_tick)

        self.scope = ScopePlot()
        self.setCentralWidget(self.scope)
        self.scope.cursor_changed.connect(self._cursor_changed)

        self.channel_panels = [ChannelPanel(i) for i in range(MAX_SCOPE_CHANNELS)]
        for panel in self.channel_panels:
            panel.changed.connect(self._settings_changed)

        self._build_menu()
        self._polish_menu()
        self._build_toolbar()
        self._build_channel_dock()
        self._build_acquisition_dock()
        self._build_measurement_dock()
        self._build_view_menu()
        self._build_preferences_menu()
        self._polish_help_menu()
        self._build_status()
        self._sync_channel_count(3)
        self._refresh_ports()
        self._restore_settings()
        settings = QtCore.QSettings("SCP", "STM32Scope")
        self._language = str(settings.value("language", "pt"))
        self._theme = str(settings.value("theme", "dark"))
        translate_ui(self, self._language)
        QtCore.QTimer.singleShot(0, lambda: set_windows_dark_titlebar(self, self._theme == "dark"))

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._stabilize_channel_tabs)
        QtCore.QTimer.singleShot(0, self._stabilize_acquisition_tabs)
        if hasattr(self, "ardustim_tab"):
            QtCore.QTimer.singleShot(0, self.ardustim_tab._stabilize_tabs)

    # ---------- UI ----------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self.act_open = file_menu.addAction("Open capture (.npz)…")
        self.act_save_npz = file_menu.addAction("Save capture (.npz)…")
        self.act_export_csv = file_menu.addAction("Export CSV…")
        self.act_export_png = file_menu.addAction("Export plot PNG…")
        file_menu.addSeparator()
        self.act_save_session = file_menu.addAction("Save settings…")
        self.act_load_session = file_menu.addAction("Load settings…")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        tools = self.menuBar().addMenu("&Tools")
        self.tools_menu = tools
        self.act_fft = tools.addAction("FFT / Spectrum…")
        self.act_xy = tools.addAction("XY Mode...")
        self.act_electrical_network = tools.addAction("Electrical Network Analyzer…")
        self.act_phase_sequence = tools.addAction("Phase Sequence…")
        self.act_rpm_frequency = tools.addAction("RPM / Frequency Calculator…")
        self.act_autoset = tools.addAction("Auto Set")
        self.act_reset_cal = tools.addAction("Reset active channel calibration")
        self.act_reset_y = tools.addAction("Reset vertical view")
        tools.addSeparator()
        self.act_soft_touch = tools.addAction("Soft touch graph")
        self.act_soft_touch.setCheckable(True)
        self.act_tcursor = tools.addAction("Δ Time cursors")
        self.act_tcursor.setCheckable(True)
        self.act_vcursor = tools.addAction("Δ Amplitude cursors")
        self.act_vcursor.setCheckable(True)

        self.act_save_npz.triggered.connect(self._save_npz)
        self.act_open.triggered.connect(self._open_npz)
        self.act_export_csv.triggered.connect(self._export_csv)
        self.act_export_png.triggered.connect(self._export_png)
        self.act_visit_lobby = QtWidgets.QAction("Visit Lobby", self)
        self.menuBar().actions()[0].menu().insertAction(self.act_save_session, self.act_visit_lobby)
        self.act_visit_lobby.triggered.connect(self._visit_lobby)
        self.act_save_session.triggered.connect(self._save_session_dialog)
        self.act_load_session.triggered.connect(self._load_session_dialog)
        self.act_fft.triggered.connect(self._show_fft)
        self.act_xy.triggered.connect(self._show_xy)
        self.act_electrical_network.triggered.connect(self._show_electrical_network)
        self.act_phase_sequence.triggered.connect(self._show_phase_sequence)
        self.act_rpm_frequency.triggered.connect(self._show_rpm_frequency_calculator)
        self.act_autoset.triggered.connect(self._autoset)
        self.act_reset_cal.triggered.connect(self._reset_active_calibration)
        self.act_reset_y.triggered.connect(self.scope.reset_vertical_view)
        self.act_soft_touch.toggled.connect(self.scope.set_soft_touch_enabled)
        self.act_tcursor.toggled.connect(self.scope.set_time_cursors)
        self.act_vcursor.toggled.connect(self.scope.set_voltage_cursors)

        plugins = self.menuBar().addMenu("&Plugins")
        self.act_plugin_manager = plugins.addAction("Plugin manager…")
        self.act_plugin_folder = plugins.addAction("Open plugins folder")
        self.act_plugin_manager.triggered.connect(self._show_plugin_manager)
        self.act_plugin_folder.triggered.connect(self._open_plugin_folder)

    def _build_view_menu(self) -> None:
        view = self.menuBar().addMenu("&View")
        view.addAction(self.channels_dock.toggleViewAction())
        view.addAction(self.acquisition_dock.toggleViewAction())
        view.addAction(self.measurement_dock.toggleViewAction())
        view.addSeparator()

        self.act_compact = view.addAction("Compact HD layout")
        self.act_compact.setCheckable(True)
        self.act_compact.setShortcut("Ctrl+Shift+C")
        self.act_compact.toggled.connect(self._set_compact_mode)

        reset = view.addAction("Reset layout")
        reset.setShortcut("Ctrl+Shift+R")
        reset.triggered.connect(self._reset_layout)

        fullscreen = view.addAction("Full screen")
        fullscreen.setShortcut("F11")
        fullscreen.triggered.connect(
            lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen()
        )

    def _polish_menu(self) -> None:
        file_menu = self.menuBar().actions()[0].menu()
        if file_menu is not None:
            file_menu.removeAction(self.act_visit_lobby)
            file_menu.insertAction(self.act_save_session, self.act_visit_lobby)

        acquisition_menu = self.menuBar().addMenu("&Acquisition")
        self.act_run = acquisition_menu.addAction("Run acquisition")
        self.act_pause = acquisition_menu.addAction("Pause acquisition")
        self.act_stop = acquisition_menu.addAction("Stop acquisition")
        acquisition_menu.addSeparator()
        self.act_rearm_trigger = acquisition_menu.addAction("Re-arm trigger")
        self.act_force_trigger = acquisition_menu.addAction("Force trigger")
        self.act_run.triggered.connect(lambda: self._set_run_mode("run"))
        self.act_pause.triggered.connect(lambda: self._set_run_mode("pause"))
        self.act_stop.triggered.connect(lambda: self._set_run_mode("stop"))
        self.act_rearm_trigger.triggered.connect(self._rearm_trigger)
        self.act_force_trigger.triggered.connect(self._force_trigger)

        tools_menu = None
        for action in self.menuBar().actions():
            menu = action.menu()
            if menu is not None and menu.title() == "&Tools":
                tools_menu = menu
                break
        if tools_menu is not None:
            tools_menu.addSeparator()
            self.act_refresh_ports = tools_menu.addAction("Refresh serial ports")
            self.act_connect_toggle = tools_menu.addAction("Connect / Disconnect serial")
            self.act_refresh_ports.triggered.connect(self._refresh_ports)
            self.act_connect_toggle.triggered.connect(self._toggle_connection)

    def _polish_help_menu(self) -> None:
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.act_about)

    def _build_preferences_menu(self) -> None:
        menu = self.menuBar().addMenu("&Appearance")
        theme_menu = menu.addMenu("Theme")
        self.theme_group = QtWidgets.QActionGroup(self); self.theme_group.setExclusive(True)
        self.act_theme_dark = theme_menu.addAction("Dark"); self.act_theme_dark.setCheckable(True); self.act_theme_dark.setData("dark")
        self.act_theme_legacy = theme_menu.addAction("Legacy PyQt5"); self.act_theme_legacy.setCheckable(True); self.act_theme_legacy.setData("legacy")
        self.theme_group.addAction(self.act_theme_dark); self.theme_group.addAction(self.act_theme_legacy)

        language_menu = menu.addMenu("Language")
        self.language_group = QtWidgets.QActionGroup(self); self.language_group.setExclusive(True)
        self.act_lang_pt = language_menu.addAction("Portuguese"); self.act_lang_pt.setCheckable(True); self.act_lang_pt.setData("pt")
        self.act_lang_en = language_menu.addAction("English"); self.act_lang_en.setCheckable(True); self.act_lang_en.setData("en")
        self.language_group.addAction(self.act_lang_pt); self.language_group.addAction(self.act_lang_en)
        menu.addSeparator()
        self.act_about = menu.addAction("About OpenScope…")
        self.act_about.triggered.connect(self._show_about)

        settings = QtCore.QSettings("SCP", "STM32Scope")
        current_theme = str(settings.value("theme", "dark"))
        current_language = str(settings.value("language", "pt"))
        (self.act_theme_legacy if current_theme == "legacy" else self.act_theme_dark).setChecked(True)
        (self.act_lang_en if current_language == "en" else self.act_lang_pt).setChecked(True)
        self.theme_group.triggered.connect(self._set_theme_preference)
        self.language_group.triggered.connect(self._set_language_preference)

    def _require_capture_for_tool(self, title: str) -> Optional[Capture]:
        if self.current_capture is None:
            QtWidgets.QMessageBox.information(self, title, "Acquire or open a capture first.")
            return None
        return self.current_capture

    def _show_electrical_network(self) -> None:
        capture = self._require_capture_for_tool("Electrical Network Analyzer")
        if capture is None:
            return
        dialog = ElectricalNetworkDialog(capture, self._configs(), self)
        translate_ui(dialog, getattr(self, "_language", "pt"))
        set_windows_dark_titlebar(dialog, getattr(self, "_theme", "dark") == "dark")
        dialog.exec_()

    def _show_phase_sequence(self) -> None:
        capture = self._require_capture_for_tool("Phase Sequence")
        if capture is None:
            return
        dialog = PhaseSequenceDialog(capture, self._configs(), self)
        translate_ui(dialog, getattr(self, "_language", "pt"))
        set_windows_dark_titlebar(dialog, getattr(self, "_theme", "dark") == "dark")
        dialog.exec_()

    def _show_plugin_manager(self) -> None:
        dialog = PluginManagerDialog(self.plugin_host, self)
        translate_ui(dialog, getattr(self, "_language", "pt"))
        set_windows_dark_titlebar(dialog, getattr(self, "_theme", "dark") == "dark")
        dialog.exec_()

    def _open_plugin_folder(self) -> None:
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self.plugin_host.ensure_directory()))
        )

    def _show_about(self) -> None:
        dialog = AboutDialog(self)
        set_windows_dark_titlebar(dialog, getattr(self, "_theme", "dark") == "dark")
        dialog.exec_()

    def _set_theme_preference(self, action: QtWidgets.QAction) -> None:
        self._theme = str(action.data())
        QtCore.QSettings("SCP", "STM32Scope").setValue("theme", self._theme)
        apply_theme(QtWidgets.QApplication.instance(), self._theme)
        # Reapply the density override without introducing dark colors in the
        # legacy theme.
        self._set_compact_mode(self.act_compact.isChecked())
        QtCore.QTimer.singleShot(0, lambda: set_windows_dark_titlebar(self, self._theme == "dark"))

    def _set_language_preference(self, action: QtWidgets.QAction) -> None:
        self._language = str(action.data())
        QtCore.QSettings("SCP", "STM32Scope").setValue("language", self._language)
        translate_ui(self, self._language)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Connection")
        bar.setMovable(False)
        bar.hide()
        self.connection_toolbar = bar
        self.port = QtWidgets.QComboBox()
        self.port.setMinimumWidth(130)
        self.port.setMaximumWidth(250)
        self.baud = QtWidgets.QComboBox()
        self.baud.setFixedWidth(105)
        for b in (115200, 460800, 921600, 1000000, 2000000):
            self.baud.addItem(f"{b:,}", b)
        self.baud.setCurrentIndex(self.baud.count() - 1)
        self.refresh_btn = QtWidgets.QToolButton()
        self.refresh_btn.setText("Refresh")
        self.connect_btn = QtWidgets.QToolButton()
        self.connect_btn.setText("Connect")
        self.connect_btn.setCheckable(False)
        self.demo_btn = QtWidgets.QToolButton()
        self.demo_btn.setText("Demo")
        self.demo_btn.setCheckable(True)

        self.refresh_btn.clicked.connect(self._refresh_ports)
        self.connect_btn.clicked.connect(self._toggle_connection)
        self.demo_btn.toggled.connect(self._toggle_demo)

    def _build_channel_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Channels", self)
        dock.setObjectName("channelsDock")
        dock.setMinimumWidth(220)
        dock.setMaximumWidth(295)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(False)
        tabs.tabBar().setExpanding(True)
        tabs.tabBar().setElideMode(QtCore.Qt.ElideRight)
        self.channel_tab_offset = 1

        graph_page = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(graph_page)
        graph_layout.setContentsMargins(8, 8, 8, 8)
        graph_layout.setSpacing(8)

        graph_quick = QtWidgets.QGroupBox("Graph actions")
        graph_grid = QtWidgets.QGridLayout(graph_quick)
        graph_grid.setContentsMargins(8, 10, 8, 8)
        graph_grid.setHorizontalSpacing(6)
        graph_grid.setVerticalSpacing(4)

        self.quick_x = RotarySpin("View X", -5.0, 5.0, 0.05, suffix=" div")
        self.quick_y = RotarySpin("View Y", -8.0, 8.0, 0.05, suffix=" div")
        self.quick_time = RotaryChoice("Time/div", [si_time(v) for v in TIME_DIVS])
        self.quick_vspan = RotaryChoice("Grid height", [f"{span:g} div" for span in VERTICAL_SPANS])
        graph_grid.addWidget(self.quick_x, 0, 0)
        graph_grid.addWidget(self.quick_y, 0, 1)
        graph_grid.addWidget(self.quick_time, 1, 0)
        graph_grid.addWidget(self.quick_vspan, 1, 1)

        graph_toggles = QtWidgets.QGroupBox("Interaction")
        toggles_layout = QtWidgets.QVBoxLayout(graph_toggles)
        toggles_layout.setContentsMargins(8, 10, 8, 8)
        toggles_layout.setSpacing(4)
        self.soft_touch_toggle = QtWidgets.QCheckBox("Soft touch graph")
        self.time_cursor_toggle = QtWidgets.QCheckBox("Δ Time cursors")
        self.voltage_cursor_toggle = QtWidgets.QCheckBox("Δ Amplitude cursors")
        toggles_layout.addWidget(self.soft_touch_toggle)
        toggles_layout.addWidget(self.time_cursor_toggle)
        toggles_layout.addWidget(self.voltage_cursor_toggle)

        graph_layout.addWidget(graph_quick)
        graph_layout.addWidget(graph_toggles)
        self.lobby_status = QtWidgets.QLabel("Use Arquivo > Ir para o lobby para trocar a conexão do osciloscópio ou abrir o modo Demo.")
        self.lobby_status.setWordWrap(True)
        self.lobby_status.setObjectName("mutedLabel")
        graph_layout.addWidget(self.lobby_status)
        graph_layout.addStretch(1)

        tabs.addTab(graph_page, "Graph")

        for i, panel in enumerate(self.channel_panels):
            panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
            tabs.addTab(panel, f"CH{i + 1}")
        self.ardustim_tab = ArduStimPanel()
        tabs.addTab(self.ardustim_tab, "Stim")
        for i in range(len(self.channel_panels)):
            tabs.tabBar().setTabTextColor(self.channel_tab_offset + i, QtGui.QColor(CHANNEL_COLORS[i]))

        dock.setWidget(tabs)
        self.channels_tabs = tabs
        self.channels_tabs.setCurrentIndex(0)
        self.channels_tabs.currentChanged.connect(self._channel_tab_changed)
        QtCore.QTimer.singleShot(0, self._stabilize_channel_tabs)
        QtCore.QTimer.singleShot(80, self._stabilize_channel_tabs)
        self.quick_x.valueChanged.connect(self._quick_x_changed)
        self.quick_y.valueChanged.connect(self._quick_y_changed)
        self.quick_time.currentIndexChanged.connect(self._quick_time_changed)
        self.quick_vspan.currentIndexChanged.connect(self._quick_vspan_changed)
        self.soft_touch_toggle.toggled.connect(self.act_soft_touch.setChecked)
        self.time_cursor_toggle.toggled.connect(self.act_tcursor.setChecked)
        self.voltage_cursor_toggle.toggled.connect(self.act_vcursor.setChecked)
        self.act_soft_touch.toggled.connect(self.soft_touch_toggle.setChecked)
        self.act_tcursor.toggled.connect(self.time_cursor_toggle.setChecked)
        self.act_vcursor.toggled.connect(self.voltage_cursor_toggle.setChecked)
        self.channels_dock = dock
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self._sync_quick_controls_from_ui()

    def _build_acquisition_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Acquisition / Trigger", self)
        dock.setObjectName("acquisitionDock")
        body = QtWidgets.QWidget()
        body.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(6)
        layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(5)
        self.run_btn = QtWidgets.QPushButton("RUN")
        self.run_btn.setObjectName("runButton")
        self.pause_btn = QtWidgets.QPushButton("PAUSE")
        self.pause_btn.setObjectName("pauseButton")
        self.stop_btn = QtWidgets.QPushButton("STOP")
        self.stop_btn.setObjectName("stopButton")
        for button in (self.run_btn, self.pause_btn, self.stop_btn):
            button.setMinimumWidth(0)
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            buttons.addWidget(button, 1)
        layout.addLayout(buttons)

        acq = QtWidgets.QGroupBox("Timebase")
        form = QtWidgets.QFormLayout(acq)
        form.setContentsMargins(8, 10, 8, 8)
        form.setVerticalSpacing(5)
        form.setHorizontalSpacing(6)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.acq_profile = QtWidgets.QComboBox()
        self.acq_profile.addItems([
            PROFILE_AUTO, PROFILE_STANDARD, PROFILE_HIGH, PROFILE_LONG, PROFILE_MANUAL
        ])
        self.acq_profile.setCurrentText(PROFILE_LONG)
        self.acq_profile.setEnabled(True)
        self.acq_profile.setToolTip("Choose how OpenScope selects the device sample rate. Auto follows the visible timebase; Manual unlocks direct rate entry.")

        self.record_length = QtWidgets.QComboBox()
        for seconds in RECORD_LENGTHS:
            self.record_length.addItem(f"{seconds:g} s", seconds)
        self.record_length.setCurrentIndex(0)
        self.record_length.setToolTip("PC-side rolling history. This does not change Time/div or MCU capture depth.")

        self.time_div = QtWidgets.QComboBox()
        for t in TIME_DIVS:
            self.time_div.addItem(si_time(t), t)
        self.time_div.setCurrentText("100 ms/div")
        self.time_zoom_in = QtWidgets.QToolButton()
        self.time_zoom_in.setText("−")
        self.time_zoom_in.setFixedWidth(26)
        self.time_zoom_in.setToolTip("Zoom in: show less time per division")
        self.time_zoom_out = QtWidgets.QToolButton()
        self.time_zoom_out.setText("+")
        self.time_zoom_out.setFixedWidth(26)
        self.time_zoom_out.setToolTip("Zoom out: show more time per division")
        time_widget = QtWidgets.QWidget()
        time_layout = QtWidgets.QHBoxLayout(time_widget)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(3)
        time_layout.addWidget(self.time_div, 1)
        time_layout.addWidget(self.time_zoom_in)
        time_layout.addWidget(self.time_zoom_out)

        self.vertical_span = QtWidgets.QComboBox()
        for span in VERTICAL_SPANS:
            self.vertical_span.addItem(f"{span:g} div", span)
        self.vertical_span.setCurrentText("8 div")

        self.persistence = QtWidgets.QSpinBox()
        self.persistence.setRange(0, 8)
        self.persistence.setValue(0)
        self.persistence.setSuffix(" captures")
        self.device_rate = QtWidgets.QSpinBox()
        self.device_rate.setRange(10, 850000)
        self.device_rate.setValue(30000)
        self.device_rate.setSuffix(" Sa/s")
        self.device_rate.setToolTip("Effective per-channel sampling rate. Editable in Manual acquisition mode.")

        self.acq_info = QtWidgets.QLabel("Direct stream · PC history · PC trigger")
        self.acq_info.setWordWrap(True)

        form.addRow("Acquisition", self.acq_profile)
        form.addRow("Record/history", self.record_length)
        form.addRow("Time/div", time_widget)
        form.addRow("Grid height", self.vertical_span)
        form.addRow("Persistence", self.persistence)
        form.addRow("Sample rate", self.device_rate)
        layout.addWidget(acq)

        trig = QtWidgets.QGroupBox("Trigger")
        tform = QtWidgets.QFormLayout(trig)
        tform.setContentsMargins(8, 10, 8, 8)
        tform.setVerticalSpacing(5)
        tform.setHorizontalSpacing(6)
        tform.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        tform.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.trigger_enable = QtWidgets.QPushButton("ON")
        self.trigger_enable.setObjectName("triggerButton")
        self.trigger_enable.setCheckable(True)
        self.trigger_enable.setChecked(True)
        self.trigger_enable.setMinimumWidth(0)
        self.trigger_enable.setToolTip("Enable or disable PC-side triggering")
        self.rearm_trigger_btn = QtWidgets.QPushButton("ARM")
        self.rearm_trigger_btn.setMinimumWidth(0)
        self.rearm_trigger_btn.setToolTip("Discard the current trigger and arm again")
        self.force_trigger_btn = QtWidgets.QPushButton("TRIG")
        self.force_trigger_btn.setMinimumWidth(0)
        self.force_trigger_btn.setToolTip("Force a trigger immediately from the current PC history")
        trig_buttons = QtWidgets.QWidget()
        trig_buttons_layout = QtWidgets.QHBoxLayout(trig_buttons)
        trig_buttons_layout.setContentsMargins(0, 0, 0, 0)
        trig_buttons_layout.setSpacing(4)
        trig_buttons_layout.addWidget(self.trigger_enable, 1)
        trig_buttons_layout.addWidget(self.rearm_trigger_btn)
        trig_buttons_layout.addWidget(self.force_trigger_btn)

        self.trigger_source = QtWidgets.QComboBox()
        for i in range(MAX_SCOPE_CHANNELS):
            self.trigger_source.addItem(f"CH{i + 1}", i)
        self.trigger_edge = QtWidgets.QComboBox()
        self.trigger_edge.addItem("Rising ↑", 1)
        self.trigger_edge.addItem("Falling ↓", 0)
        self.trigger_level = QtWidgets.QDoubleSpinBox()
        self.trigger_level.setRange(-1000.0, 1000.0)
        self.trigger_level.setDecimals(4)
        self.trigger_level.setSingleStep(0.1)
        self.trigger_level.setValue(1.65)
        self.trigger_level.setSuffix(" V")
        self.pretrigger = QtWidgets.QSpinBox()
        self.pretrigger.setRange(0, 95)
        self.pretrigger.setValue(50)
        self.pretrigger.setSuffix(" %")
        self.trigger_mode = QtWidgets.QComboBox()
        self.trigger_mode.addItems(["Auto", "Normal"])
        self.trigger_mode.setCurrentText("Normal")
        self.auto_trigger_timeout = QtWidgets.QSpinBox()
        self.auto_trigger_timeout.setRange(50, 5000)
        self.auto_trigger_timeout.setValue(250)
        self.auto_trigger_timeout.setSuffix(" ms")
        self.trigger_holdoff = QtWidgets.QDoubleSpinBox()
        self.trigger_holdoff.setRange(0.0, 5000.0)
        self.trigger_holdoff.setDecimals(1)
        self.trigger_holdoff.setSingleStep(1.0)
        self.trigger_holdoff.setValue(20.0)
        self.trigger_holdoff.setSuffix(" ms")
        tform.addRow(trig_buttons)
        tform.addRow("Source", self.trigger_source)
        tform.addRow("Edge", self.trigger_edge)
        tform.addRow("Level", self.trigger_level)
        tform.addRow("Pre-trigger", self.pretrigger)
        self.auto_trigger_timeout.setPrefix("")
        self.trigger_holdoff.setPrefix("")
        tform.addRow("Mode", self.trigger_mode)
        tform.addRow("Auto timeout", self.auto_trigger_timeout)
        tform.addRow("Holdoff", self.trigger_holdoff)
        layout.addWidget(trig)

        def make_cursor_group() -> tuple[QtWidgets.QGroupBox, QtWidgets.QComboBox, QtWidgets.QLabel]:
            group = QtWidgets.QGroupBox("Cursors")
            cform = QtWidgets.QFormLayout(group)
            cform.setContentsMargins(7, 9, 7, 7)
            cform.setVerticalSpacing(4)
            cform.setHorizontalSpacing(6)
            cform.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
            combo = QtWidgets.QComboBox()
            for channel_index in range(MAX_SCOPE_CHANNELS):
                combo.addItem(f"CH{channel_index + 1}", channel_index)
            readout = QtWidgets.QLabel("Δt —    ΔV —")
            readout.setWordWrap(True)
            time_toggle = QtWidgets.QCheckBox("Δ time")
            voltage_toggle = QtWidgets.QCheckBox("Δ amplitude")
            time_toggle.toggled.connect(self.act_tcursor.setChecked)
            voltage_toggle.toggled.connect(self.act_vcursor.setChecked)
            self.act_tcursor.toggled.connect(time_toggle.setChecked)
            self.act_vcursor.toggled.connect(voltage_toggle.setChecked)
            cform.addRow("Amplitude CH", combo)
            cform.addRow(time_toggle, voltage_toggle)
            cform.addRow(readout)
            return group, combo, readout

        trigger_cursors, self.cursor_channel, self.cursor_readout = make_cursor_group()
        acquisition_cursors, self.cursor_channel_acq, self.cursor_readout_acq = make_cursor_group()
        self.cursor_channel.currentIndexChanged.connect(
            lambda index: self._sync_cursor_channel(index, self.cursor_channel)
        )
        self.cursor_channel_acq.currentIndexChanged.connect(
            lambda index: self._sync_cursor_channel(index, self.cursor_channel_acq)
        )
        right_tabs = QtWidgets.QTabWidget()
        right_tabs.setDocumentMode(True)
        right_tabs.setUsesScrollButtons(False)
        right_tabs.tabBar().setExpanding(True)
        right_tabs.tabBar().setElideMode(QtCore.Qt.ElideRight)
        trigger_page = QtWidgets.QWidget()
        trigger_layout = QtWidgets.QVBoxLayout(trigger_page)
        trigger_layout.setContentsMargins(5, 5, 5, 5)
        trigger_layout.setSpacing(5)
        acquisition_page = QtWidgets.QWidget()
        acquisition_layout = QtWidgets.QVBoxLayout(acquisition_page)
        acquisition_layout.setContentsMargins(5, 5, 5, 5)
        acquisition_layout.setSpacing(5)
        layout.removeWidget(acq)
        layout.removeWidget(trig)
        acq.setParent(acquisition_page)
        trig.setParent(trigger_page)
        acquisition_layout.addWidget(acq)
        acquisition_layout.addWidget(acquisition_cursors)
        acquisition_layout.addWidget(self.acq_info)
        acquisition_layout.addStretch(1)
        trigger_layout.addWidget(trig)
        trigger_layout.addWidget(trigger_cursors)
        trigger_layout.addStretch(1)
        right_tabs.addTab(trigger_page, "Trigger")
        right_tabs.addTab(acquisition_page, "Acquisition")
        right_tabs.setCurrentIndex(0)
        self.acquisition_tabs = right_tabs
        self.acquisition_tabs.currentChanged.connect(lambda _index: self._stabilize_acquisition_tabs())
        QtCore.QTimer.singleShot(0, self._stabilize_acquisition_tabs)
        QtCore.QTimer.singleShot(80, self._stabilize_acquisition_tabs)
        layout.addWidget(right_tabs)

        layout.addStretch(1)
        dock.setMinimumWidth(205)
        dock.setMaximumWidth(292)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        dock.setWidget(scroll)
        self.acquisition_dock = dock
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self.run_btn.clicked.connect(lambda: self._set_run_mode("run"))
        self.pause_btn.clicked.connect(lambda: self._set_run_mode("pause"))
        self.stop_btn.clicked.connect(lambda: self._set_run_mode("stop"))
        self.acq_profile.currentIndexChanged.connect(self._profile_changed)
        self.record_length.currentIndexChanged.connect(self._record_length_changed)
        self.time_div.currentIndexChanged.connect(self._timebase_changed)
        self.time_zoom_in.clicked.connect(lambda: self._step_time_div(-1))
        self.time_zoom_out.clicked.connect(lambda: self._step_time_div(+1))
        self.vertical_span.currentIndexChanged.connect(
            lambda: self.scope.set_vertical_span(float(self.vertical_span.currentData()))
        )
        self.persistence.valueChanged.connect(self.scope.set_persistence)
        self.trigger_enable.toggled.connect(self._trigger_enable_changed)
        self.rearm_trigger_btn.clicked.connect(self._rearm_trigger)
        self.force_trigger_btn.clicked.connect(self._force_trigger)
        self.trigger_source.currentIndexChanged.connect(self._send_trigger_settings)
        self.trigger_level.valueChanged.connect(self._send_trigger_settings)
        self.trigger_edge.currentIndexChanged.connect(self._send_trigger_settings)
        self.pretrigger.valueChanged.connect(self._send_trigger_settings)
        self.trigger_mode.currentIndexChanged.connect(self._send_trigger_settings)
        self.auto_trigger_timeout.valueChanged.connect(self._send_trigger_settings)
        self.trigger_holdoff.valueChanged.connect(self._send_trigger_settings)
        self.device_rate.editingFinished.connect(self._send_rate)
        # cursor comboboxes are synchronized through _sync_cursor_channel

        self.time_zoom_in_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+-"), self)
        self.time_zoom_out_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl++"), self)
        self.time_zoom_in_shortcut.activated.connect(lambda: self._step_time_div(-1))
        self.time_zoom_out_shortcut.activated.connect(lambda: self._step_time_div(+1))

    def _build_measurement_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Measurements", self)
        dock.setObjectName("measurementDock")
        dock.setMinimumHeight(88)
        dock.setMaximumHeight(190)

        labels = ["Min", "Max", "Pk-Pk", "Mean", "RMS", "Freq", "Period", "Duty"]
        headers = ["On", "Color"] + labels
        self.measurements = QtWidgets.QTableWidget(MAX_SCOPE_CHANNELS, len(headers))
        self.measurements.setAlternatingRowColors(True)
        self.measurements.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.measurements.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.measurements.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.measurements.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.measurements.verticalHeader().setVisible(True)
        self.measurements.setVerticalHeaderLabels([f"CH{i + 1}" for i in range(MAX_SCOPE_CHANNELS)])
        self.measurements.setHorizontalHeaderLabels(headers)
        self.measurements.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.measurements.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        for column in range(2, len(headers)):
            self.measurements.horizontalHeader().setSectionResizeMode(column, QtWidgets.QHeaderView.Stretch)
        self.measurements.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.measurements.setMinimumHeight(92)
        self.measurements.setMaximumHeight(132)

        self.measure_channel_checks = []
        self.measure_color_buttons = []
        for ch in range(MAX_SCOPE_CHANNELS):
            toggle = QtWidgets.QCheckBox()
            toggle.setChecked(self.channel_panels[ch].isChecked())
            toggle.toggled.connect(lambda enabled, index=ch: self._measurement_channel_toggled(index, enabled))
            wrap = QtWidgets.QWidget()
            wrap_layout = QtWidgets.QHBoxLayout(wrap)
            wrap_layout.setContentsMargins(0, 0, 0, 0)
            wrap_layout.setAlignment(QtCore.Qt.AlignCenter)
            wrap_layout.addWidget(toggle)
            self.measurements.setCellWidget(ch, 0, wrap)
            self.measure_channel_checks.append(toggle)

            color_btn = QtWidgets.QToolButton()
            color_btn.setText("●")
            color_btn.clicked.connect(lambda _checked=False, index=ch: self._choose_channel_color(index))
            self.measurements.setCellWidget(ch, 1, color_btn)
            self.measure_color_buttons.append(color_btn)

        for ch in range(MAX_SCOPE_CHANNELS):
            self._apply_channel_color(ch)

        dock.setWidget(self.measurements)
        self.measurement_dock = dock
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _build_status(self) -> None:
        self.connection_status = QtWidgets.QLabel("Disconnected")
        self.capture_status = QtWidgets.QLabel("No capture")
        self.packet_status = QtWidgets.QLabel("Packets 0 | CRC 0")
        self.statusBar().addWidget(self.connection_status)
        self.statusBar().addPermanentWidget(self.capture_status)
        self.statusBar().addPermanentWidget(self.packet_status)

    # ---------- Connection ----------
    def _refresh_ports(self) -> None:
        current = self.port.currentData()
        self.port.clear()
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        for p in ports:
            label = p.device
            if p.description and p.description != "n/a":
                label += f" — {p.description}"
            label = format_serial_port_label(p)
            self.port.addItem(label, p.device)
        if current:
            idx = self.port.findData(current)
            if idx >= 0:
                self.port.setCurrentIndex(idx)

    def _toggle_connection(self) -> None:
        if self.worker is not None:
            self._disconnect()
            return
        port = self.port.currentData()
        if not port:
            QtWidgets.QMessageBox.warning(self, "Serial", "No serial port selected.")
            return
        self.demo_btn.setChecked(False)
        worker = SerialWorker(str(port), int(self.baud.currentData()), self)
        worker.connected.connect(self._connected)
        worker.disconnected.connect(self._disconnected)
        worker.error.connect(self._serial_error)
        worker.stats.connect(self._serial_stats)
        self.worker = worker
        self.stream_history.clear()
        self.last_stream_capture = None
        self.current_capture = None
        self._reset_trigger_state()
        worker.start()
        self.connect_btn.setEnabled(False)
        self.connection_status.setText("Opening…")
        if hasattr(self, "lobby_status"):
            self.lobby_status.setText("Opening serial connection…")

    def _disconnect(self) -> None:
        worker = self.worker
        if worker is None:
            return
        self.worker = None
        worker.stop()
        if not worker.wait(1000):
            worker.terminate()
            worker.wait(300)
        self._disconnected()

    def _connected(self, port: str) -> None:
        self.connection_status.setText(f"Connected: {port}")
        if hasattr(self, "lobby_status"):
            self.lobby_status.setText(f"Connected on {port}.")
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        QtCore.QTimer.singleShot(80, self._apply_profile_to_device)
        QtCore.QTimer.singleShot(140, self._send_trigger_settings)

    def _disconnected(self) -> None:
        self.stream_history.clear()
        self.last_stream_capture = None
        self.current_capture = None
        self._reset_trigger_state()
        self.connection_status.setText("Disconnected")
        if hasattr(self, "lobby_status"):
            self.lobby_status.setText("Use Arquivo > Ir para o lobby para trocar a conexão do osciloscópio ou abrir o modo Demo.")
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)

    def _serial_error(self, text: str) -> None:
        self.connection_status.setText("Serial error")
        if hasattr(self, "lobby_status"):
            self.lobby_status.setText("Serial error. Check the port and baud.")
        self.connect_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Serial error", text)

    def _serial_stats(self, good: int, crc: int, fmt_err: int, discarded: int) -> None:
        gaps = self.stream_history.discontinuities
        dropped = self.worker.dropped_transport_packets if self.worker is not None else 0
        self.packet_status.setText(
            f"Packets {good} | CRC {crc} | Format {fmt_err} | "
            f"Resync {discarded} B | Gaps {gaps} | Queue drop {dropped}"
        )

    def _drain_worker(self) -> None:
        worker = self.worker
        if worker is None:
            return
        packets = worker.take_pending(1024)
        if not packets:
            return
        self._capture_received(packets)

    def _send(self, data: bytes) -> None:
        if self.worker is not None:
            self.worker.send(data)

    # ---------- Live device commands ----------
    def _effective_profile(self) -> str:
        return self.acq_profile.currentText()

    def _is_stream_profile(self) -> bool:
        return True

    def _visible_seconds(self) -> float:
        data = self.time_div.currentData()
        try:
            time_div = float(data)
        except (TypeError, ValueError):
            time_div = 0.1
        return max(10.0 * time_div, 1e-6)

    def _history_seconds(self) -> float:
        data = self.record_length.currentData()
        try:
            requested = float(data)
        except (TypeError, ValueError):
            requested = 1.0
        return max(1.0, requested, self._visible_seconds())

    def _device_rate_limit(self) -> int:
        if self.device_signature is None:
            return 160_000
        channels, adc_bits = self.device_signature
        if int(adc_bits) <= 10:
            return 32_000
        if int(channels) >= 4:
            return 120_000
        return 160_000 if int(channels) >= 3 else 240_000

    def _standard_rate(self) -> int:
        if self.device_signature is None:
            return 80_000
        channels, adc_bits = self.device_signature
        if int(adc_bits) <= 10:
            return 25_000
        if int(channels) >= 4:
            return 60_000
        return 80_000 if int(channels) >= 3 else 120_000

    def _high_rate(self) -> int:
        limit = self._device_rate_limit()
        if limit <= 120_000:
            return min(limit, 110_000)
        return min(limit, 150_000 if limit <= 160_000 else 220_000)

    def _long_rate(self) -> int:
        if self.device_signature is not None and int(self.device_signature[1]) <= 10:
            return 8_000
        if self.device_signature is not None and int(self.device_signature[0]) >= 4:
            return 15_000
        return 20_000 if self._device_rate_limit() <= 160_000 else 30_000

    def _auto_rate(self) -> int:
        visible = max(self._visible_seconds(), 1e-6)
        desired = int(round(AUTO_TARGET_POINTS / visible))
        return max(500, min(self._device_rate_limit(), desired))

    def _requested_rate_for_profile(self) -> int:
        profile = self._effective_profile()
        if profile == PROFILE_AUTO:
            return self._auto_rate()
        if profile == PROFILE_HIGH:
            return self._high_rate()
        if profile == PROFILE_LONG:
            return self._long_rate()
        if profile == PROFILE_MANUAL:
            return int(self.device_rate.value())
        return self._standard_rate()

    def _profile_code(self) -> str:
        profile = self._effective_profile()
        if profile == PROFILE_HIGH:
            return "HIGH"
        if profile == PROFILE_LONG:
            return "LONG"
        if profile == PROFILE_MANUAL:
            return "MANUAL"
        if profile == PROFILE_STANDARD:
            # Keep the established wire command so older direct-stream firmware
            # still understands the new user-facing Standard mode.
            return "ENGINE"
        return "MANUAL"

    def _apply_profile_to_device(self) -> None:
        requested = self._requested_rate_for_profile()
        profile = self._effective_profile()
        self.stream_profile = profile
        blocker = QtCore.QSignalBlocker(self.device_rate)
        self.device_rate.setValue(min(requested, self.device_rate.maximum()))
        del blocker

        if profile in (PROFILE_AUTO, PROFILE_MANUAL):
            self._send(make_device_command("RATE", requested))
        else:
            self._send(make_device_command("PROFILE", self._profile_code()))
        self._update_profile_ui()
        self._reset_trigger_state(clear_hold=True)

    def _update_profile_ui(self) -> None:
        manual = self.acq_profile.currentText() == PROFILE_MANUAL
        self.device_rate.setEnabled(manual)
        self.record_length.setEnabled(True)
        self._update_acquisition_info()

    def _update_acquisition_info(self) -> None:
        rate = max(1, int(self.device_rate.value()))
        record_s = self._history_seconds()
        visible_s = self._visible_seconds()
        screen_points = int(round(rate * visible_s))
        dt_us = 1e6 / max(rate, 1)
        samples_per_div = max(1, int(round(screen_points / 10.0)))
        channels = self.active_channel_count or (int(self.device_signature[0]) if self.device_signature else 3)
        payload_mbps = rate * max(1, channels) * 16.0 / 1_000_000.0
        self.acq_info.setText(
            f"{visible_s:g} s visible · ~{screen_points:,} samples/ch · ~{samples_per_div:,} samples/div · "
            f"Δt {dt_us:.2f} µs · {record_s:g} s PC history · payload ≈ {payload_mbps:.2f} Mbit/s"
        )

    def _profile_changed(self) -> None:
        self._apply_profile_to_device()
        self._settings_changed()

    def _record_length_changed(self) -> None:
        self.stream_history.clear()
        self.stream_trigger_abs = None
        self.stream_trigger_hold_until = 0.0
        self._update_acquisition_info()
        self._settings_changed()

    def _step_time_div(self, step: int) -> None:
        index = max(0, min(self.time_div.count() - 1, self.time_div.currentIndex() + int(step)))
        self.time_div.setCurrentIndex(index)

    def _current_channel_index(self) -> int:
        raw_index = self.channels_tabs.currentIndex() - getattr(self, "channel_tab_offset", 0)
        return max(0, min(len(self.channel_panels) - 1, raw_index))

    def _stabilize_channel_tabs(self) -> None:
        if not hasattr(self, "channels_tabs"):
            return
        stabilize_tab_widget(self.channels_tabs, "_channel_tabs_initialized")

    def _stabilize_acquisition_tabs(self) -> None:
        if not hasattr(self, "acquisition_tabs"):
            return
        stabilize_tab_widget(self.acquisition_tabs, "_acquisition_tabs_initialized")

    def _channel_tab_changed(self, _index: int) -> None:
        self._stabilize_channel_tabs()
        self._sync_quick_controls_from_ui()
        self._settings_changed()

    def _sync_quick_controls_from_ui(self) -> None:
        if not hasattr(self, "quick_x") or self.quick_control_sync_block:
            return
        self.quick_control_sync_block = True
        self.quick_x.setValue(self.scope.view_x_offset_div)
        self.quick_y.setValue(self.scope.view_y_offset)
        if hasattr(self, "time_div"):
            self.quick_time.setCurrentIndex(self.time_div.currentIndex())
        if hasattr(self, "vertical_span"):
            self.quick_vspan.setCurrentIndex(self.vertical_span.currentIndex())
        self.quick_control_sync_block = False

    def _quick_x_changed(self, value: float) -> None:
        if self.quick_control_sync_block:
            return
        self.scope.set_view_x_offset_div(value)
        if self.current_capture is not None:
            self._timebase_changed()

    def _quick_y_changed(self, value: float) -> None:
        if self.quick_control_sync_block:
            return
        self.scope.set_view_y_offset(value)
        if self.current_capture is not None:
            self._render_current()

    def _quick_time_changed(self, index: int) -> None:
        if self.quick_control_sync_block:
            return
        self.time_div.setCurrentIndex(index)

    def _quick_vspan_changed(self, index: int) -> None:
        if self.quick_control_sync_block:
            return
        self.vertical_span.setCurrentIndex(index)

    def _timebase_changed(self) -> None:
        # Time/div is a pure Windows-side zoom. Do NOT restart the device or
        # change sample rate here: doing so discarded/reconfigured history on
        # every zoom step and made the trace collapse back into one corner.
        self.scope.clear_persistence_history()
        self._reset_trigger_state(clear_hold=False)
        self._update_acquisition_info()
        if self._is_stream_profile() and self.stream_history.count > 0 and self.last_stream_capture is not None:
            self._render_stream_latest(self.last_stream_capture)
        elif self.current_capture is not None:
            self._render_current()
        self._sync_quick_controls_from_ui()

    def _reset_trigger_state(self, *, clear_hold: bool = True) -> None:
        self.stream_trigger_abs = None
        self.stream_last_trigger_value = None
        self.pending_trigger_events.clear()
        self.last_accepted_trigger_abs = None
        self.last_rendered_trigger_abs = None
        if clear_hold:
            self.stream_trigger_hold_until = 0.0
        self.trigger_armed_at = time.monotonic()
        self.trigger_has_lock = False
        if self.stream_history.count > 0:
            self.trigger_ignore_until_abs = self.stream_history.oldest_abs
        else:
            self.trigger_ignore_until_abs = 0

    def _rearm_trigger(self, checked: bool = False, *, clear_hold: bool = True) -> None:
        # Re-arm only resets the PC trigger detector. Acquisition and history
        # continue uninterrupted.
        self._reset_trigger_state(clear_hold=clear_hold)
        if hasattr(self, "trigger_enable") and not self.trigger_enable.isChecked():
            self.trigger_enable.setChecked(True)
        if self.run_mode == "stop":
            self._set_run_mode("run")
        elif self.run_mode == "pause":
            self._set_run_mode("run")

    def _update_trigger_controls(self) -> None:
        enabled = self.trigger_enable.isChecked()
        self.trigger_enable.setText("ON" if enabled else "OFF")
        for widget in (
            self.rearm_trigger_btn, self.force_trigger_btn, self.trigger_source,
            self.trigger_edge, self.trigger_level, self.pretrigger, self.trigger_mode,
            self.trigger_holdoff,
        ):
            widget.setEnabled(enabled)
        self.auto_trigger_timeout.setEnabled(enabled and self.trigger_mode.currentText() == "Auto")

    def _trigger_enable_changed(self, enabled: bool) -> None:
        self._update_trigger_controls()
        self.scope.set_trigger_visible(enabled, show_time_marker=False)
        self._reset_trigger_state()
        # Rendering is timer-driven. Do not force a stale short packet onto a
        # long timebase here.

    def _send_trigger_settings(self) -> None:
        # Triggering is deliberately 100% PC-side; changing trigger settings
        # must never restart the MCU stream or clear the PC history.
        self._update_trigger_controls()
        self._reset_trigger_state()

    def _send_rate(self) -> None:
        if self.acq_profile.currentText() != PROFILE_MANUAL:
            blockers = QtCore.QSignalBlocker(self.acq_profile)
            self.acq_profile.setCurrentText(PROFILE_MANUAL)
            del blockers
        requested = min(int(self.device_rate.value()), self._device_rate_limit())
        self._send(make_device_command("RATE", requested))
        self.stream_history.clear()
        self._reset_trigger_state()
        self._update_profile_ui()

    def _set_run_mode(self, mode: str) -> None:
        if mode not in ("run", "pause", "stop"):
            mode = "run"
        previous = self.run_mode
        self.run_mode = mode
        if mode == "run":
            self._send(make_device_command("RUN"))
            if self.demo_btn.isChecked() and not self.demo_timer.isActive():
                self.demo_last_ts = time.monotonic()
                self.demo_timer.start(10)
            if previous != "run":
                self._reset_trigger_state()
                if self.last_stream_capture is not None and self.stream_history.count > 0:
                    self._render_stream_latest(self.last_stream_capture)
        elif mode == "pause":
            # PAUSE freezes the Windows display only. The MCU and PC history keep running.
            self._send(make_device_command("RUN"))
        else:
            self._send(make_device_command("STOP"))
            if self.demo_btn.isChecked():
                self.demo_timer.stop()

        self.run_btn.setEnabled(mode != "run")
        self.pause_btn.setEnabled(mode != "pause")
        self.stop_btn.setEnabled(mode != "stop")
        if mode == "pause":
            self.connection_status.setText("Paused display")
        elif mode == "stop":
            self.connection_status.setText("Stopped")
        elif self.demo_btn.isChecked():
            self.connection_status.setText("Demo generator")
        elif self.worker is not None:
            self.connection_status.setText("Connected")

    # ---------- Demo ----------
    def _toggle_demo(self, enabled: bool) -> None:
        if enabled:
            if self.worker is not None:
                self._disconnect()
            self.demo_sequence = 0
            self.demo_sample_offset = 0
            self.stream_history.clear()
            self.last_stream_capture = None
            self.current_capture = None
            self._reset_trigger_state()
            self.run_mode = "run"
            self.run_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.connection_status.setText("Demo generator")
            if hasattr(self, "lobby_status"):
                self.lobby_status.setText("Demo mode active.")
            # Prime one complete visible screen immediately. The old demo started
            # with one tiny packet, which made a 1 s timebase look frozen/empty.
            self._prime_demo_view()
            self.demo_last_ts = time.monotonic()
            self.demo_timer.start(10)
        else:
            self.demo_timer.stop()
            self._reset_trigger_state()
            if self.worker is None:
                self.connection_status.setText("Disconnected")
                if hasattr(self, "lobby_status"):
                    self.lobby_status.setText("Use Arquivo > Ir para o lobby para trocar a conexão do osciloscópio ou abrir o modo Demo.")

    def _demo_stream_rate(self) -> int:
        return self._requested_rate_for_profile()

    def _prime_demo_view(self) -> None:
        if not self._is_stream_profile():
            return
        rate = self._demo_stream_rate()
        # Prime at most one second synchronously so clicking Demo never blocks
        # for several seconds on very long timebases. The timer then fills the
        # remaining history at real-time speed.
        prime_seconds = min(self._visible_seconds(), 1.0)
        frames = max(64, int(math.ceil(rate * prime_seconds)))
        # Keep each synthetic packet bounded. _stream_received() is deliberately
        # called repeatedly so the exact same history/trigger path is exercised.
        remaining = frames
        last_capture = None
        while remaining > 0:
            chunk = min(8192, remaining)
            capture = demo_capture(
                sample_rate=rate, frames=chunk, channels=3,
                packet_type=PACKET_STREAM, sequence=self.demo_sequence,
                sample_offset=self.demo_sample_offset,
            )
            self.demo_sequence = (self.demo_sequence + 1) & 0xFFFFFFFF
            self.demo_sample_offset += chunk
            self._capture_received(capture)
            last_capture = capture
            remaining -= chunk
        if last_capture is not None:
            self._render_stream_latest(last_capture)

    def _demo_tick(self) -> None:
        # Generate the exact amount of data corresponding to elapsed wall time.
        # The old demo emitted 50 ms of samples every 70 ms, creating visible
        # packet bursts even with no hardware connected.
        now = time.monotonic()
        elapsed = max(0.0, min(0.050, now - self.demo_last_ts))
        self.demo_last_ts = now
        channels = 3
        rate = self._demo_stream_rate()
        frames = max(1, int(round(rate * elapsed)))
        capture = demo_capture(
            sample_rate=rate, frames=frames, channels=channels,
            packet_type=PACKET_STREAM, sequence=self.demo_sequence,
            sample_offset=self.demo_sample_offset,
        )
        self.demo_sequence = (self.demo_sequence + 1) & 0xFFFFFFFF
        self.demo_sample_offset += frames
        self._capture_received(capture)

    # ---------- Capture processing ----------
    def _configs(self) -> List[ChannelConfig]:
        return [panel.config() for panel in self.channel_panels]

    def _local_capture(self, raw: np.ndarray, sample_rate: int, adc_bits: int,
                       trigger_frame: int, source_header: CaptureHeader) -> Capture:
        frames, channels = raw.shape
        header = CaptureHeader(
            MAGIC_U32, 4, 38, int(sample_rate), int(frames), int(trigger_frame),
            int(self._trigger_level_raw(adc_bits)), int(channels),
            int(self.trigger_source.currentIndex() + 1), int(self.trigger_edge.currentData()),
            int(adc_bits), int(frames * channels * 2), 0,
            PACKET_CAPTURE, 0, int(source_header.sequence),
        )
        return Capture(header, np.asarray(raw, dtype=np.uint16).copy())

    def _trigger_level_raw(self, adc_bits: int) -> int:
        source = max(0, min(len(self.channel_panels) - 1, self.trigger_source.currentIndex()))
        return self.channel_panels[source].config().volts_scalar_to_raw(
            float(self.trigger_level.value()), adc_bits
        )

    def _trigger_window_geometry(self, sample_rate: int) -> tuple[int, int, int]:
        total_frames = max(2, int(round(self._visible_seconds() * max(sample_rate, 1))))
        pre_frames = int(total_frames * self.pretrigger.value() / 100.0)
        pre_frames = min(max(0, pre_frames), total_frames - 1)
        post_frames = total_frames - pre_frames - 1
        return total_frames, pre_frames, post_frames

    def _collect_stream_triggers(
        self, capture: Capture, start_abs: int, *, discontinuity: bool = False
    ) -> None:
        if not self.trigger_enable.isChecked():
            self.stream_last_trigger_value = None
            return

        source = max(0, min(capture.channel_count - 1, self.trigger_source.currentIndex()))
        cfg = self.channel_panels[source].config()
        values = cfg.raw_to_volts(capture.raw[:, source], capture.header.adc_bits, apply_ac=False)

        if discontinuity:
            # Never create an artificial edge across a missing USB packet.
            self.stream_last_trigger_value = None

        total_frames, pre_frames, _post_frames = self._trigger_window_geometry(capture.sample_rate)
        min_abs = max(
            self.stream_history.oldest_abs + pre_frames,
            self.trigger_ignore_until_abs,
        )
        crossings, previous = find_crossings(
            values,
            start_abs=start_abs,
            level=float(self.trigger_level.value()),
            rising=bool(self.trigger_edge.currentData()),
            previous=self.stream_last_trigger_value,
            min_abs=min_abs,
        )
        self.stream_last_trigger_value = previous

        if crossings.size == 0:
            return

        holdoff_frames = max(
            0,
            int(round(capture.sample_rate * float(self.trigger_holdoff.value()) / 1000.0)),
        )
        last = self.last_accepted_trigger_abs
        for value in crossings.tolist():
            absolute = int(value)
            if last is not None and absolute - last < holdoff_frames:
                continue
            self.pending_trigger_events.append(absolute)
            last = absolute
        self.last_accepted_trigger_abs = last

    def _render_trigger_at(self, trigger_abs: int, source_capture: Capture) -> bool:
        total_frames, pre_frames, _post_frames = self._trigger_window_geometry(source_capture.sample_rate)
        start = int(trigger_abs) - pre_frames
        raw = self.stream_history.extract(start, total_frames)
        if raw is None:
            return False
        self.current_capture = self._local_capture(
            raw,
            source_capture.sample_rate,
            source_capture.header.adc_bits,
            pre_frames,
            source_capture.header,
        )
        self.last_rendered_trigger_abs = int(trigger_abs)
        self.trigger_has_lock = True
        self._render_current(rolling=False)
        return True

    def _force_trigger(self) -> None:
        if not self.trigger_enable.isChecked():
            self.trigger_enable.setChecked(True)
        source_capture = self.last_stream_capture
        if source_capture is None or self.stream_history.count <= 1:
            return
        total_frames, pre_frames, post_frames = self._trigger_window_geometry(source_capture.sample_rate)
        if self.stream_history.count < total_frames:
            # Not enough data for a complete requested screen yet: show the
            # live history rather than pretending a complete trigger exists.
            self._render_stream_latest(source_capture)
            return
        trigger_abs = self.stream_history.total - post_frames - 1
        earliest = self.stream_history.oldest_abs + pre_frames
        trigger_abs = max(int(earliest), int(trigger_abs))
        if self._render_trigger_at(trigger_abs, source_capture):
            self.trigger_armed_at = time.monotonic()

    def _render_stream_latest(self, capture: Capture) -> None:
        visible_s = self._visible_seconds()
        requested = max(2, int(round(visible_s * capture.sample_rate)))
        raw = self.stream_history.latest(requested)
        if raw is None or raw.size == 0:
            return
        warmup = raw.shape[0] < requested
        trig = max(0, raw.shape[0] - 1)
        self.current_capture = self._local_capture(
            raw, capture.sample_rate, capture.header.adc_bits, trig, capture.header
        )
        self._render_current(rolling=True, warmup=warmup)

    def _stream_batch_received(self, captures) -> None:
        if not captures:
            return

        # Process contiguous groups with identical stream geometry. A rate or
        # device change legitimately starts a new PC history.
        group = []
        signature = None
        for capture in captures:
            sig = (capture.sample_rate, capture.channel_count, capture.header.adc_bits)
            if group and sig != signature:
                self._ingest_stream_group(group)
                group = []
            signature = sig
            group.append(capture)
        if group:
            self._ingest_stream_group(group)

    def _ingest_stream_group(self, captures) -> None:
        if not captures:
            return
        last_capture = captures[-1]
        history_seconds = self._history_seconds()

        old_geometry = (
            self.stream_history.sample_rate,
            self.stream_history.channels,
            self.stream_history.adc_bits,
        )
        new_geometry = (
            int(last_capture.sample_rate),
            int(last_capture.channel_count),
            int(last_capture.header.adc_bits),
        )
        self.stream_history.configure(
            last_capture.sample_rate,
            last_capture.channel_count,
            history_seconds,
            last_capture.header.adc_bits,
        )
        if old_geometry[0] > 0 and old_geometry != new_geometry:
            self._reset_trigger_state()

        for capture in captures:
            expected = None
            if self.stream_history.sequence is not None:
                expected = (self.stream_history.sequence + 1) & 0xFFFFFFFF
            discontinuity = bool(capture.header.flags & FLAG_DISCONTINUITY)
            if expected is not None and capture.header.sequence != expected:
                discontinuity = True

            start_abs, _end_abs = self.stream_history.append(capture)
            # A transport gap no longer clears the whole history. Clearing was
            # the main reason the waveform repeatedly collapsed into the right
            # corner. We only break trigger continuity at the gap.
            self._collect_stream_triggers(capture, start_abs, discontinuity=discontinuity)

        self.last_stream_capture = last_capture
        if int(self.device_rate.value()) != int(last_capture.sample_rate):
            blocker = QtCore.QSignalBlocker(self.device_rate)
            self.device_rate.setValue(
                min(int(last_capture.sample_rate), int(self.device_rate.maximum()))
            )
            del blocker

    def _display_tick(self) -> None:
        if self.run_mode in ("pause", "stop"):
            return
        capture = self.last_stream_capture
        if capture is None or self.stream_history.count <= 0:
            return

        # Trigger OFF = pure rolling monitor.
        if not self.trigger_enable.isChecked():
            self._render_stream_latest(capture)
            return

        total_frames, pre_frames, post_frames = self._trigger_window_geometry(capture.sample_rate)
        matured_limit = self.stream_history.total - post_frames - 1
        newest_matured = None

        # Consume every trigger that has enough post-trigger data and display
        # only the newest one. Periodic waveforms then update at the display
        # cadence instead of one trigger every post-trigger window.
        earliest_valid = self.stream_history.oldest_abs + pre_frames
        while self.pending_trigger_events and self.pending_trigger_events[0] <= matured_limit:
            candidate = self.pending_trigger_events.popleft()
            if candidate >= earliest_valid:
                newest_matured = candidate

        if newest_matured is not None:
            if self._render_trigger_at(int(newest_matured), capture):
                self.trigger_armed_at = time.monotonic()
                return

        now = time.monotonic()
        if (
            self.trigger_mode.currentText() == "Auto"
            and self.stream_history.count >= total_frames
            and (now - self.trigger_armed_at) * 1000.0 >= self.auto_trigger_timeout.value()
        ):
            forced = int(matured_limit)
            earliest = self.stream_history.oldest_abs + pre_frames
            if forced >= earliest and self._render_trigger_at(forced, capture):
                self.trigger_armed_at = now
                return

        # Before the first valid trigger, provide a live preview. Once Normal
        # has a lock, keep that stable trace until a newer matured trigger exists.
        if self.last_rendered_trigger_abs is None:
            self._render_stream_latest(capture)

    def _stream_received(self, capture: Capture) -> None:
        self._stream_batch_received([capture])

    def _capture_received(self, capture) -> None:
        if self.run_mode == "stop":
            return

        if isinstance(capture, (list, tuple)):
            stream_group = []
            for item in capture:
                if item.is_stream:
                    stream_group.append(item)
                    continue
                if stream_group:
                    last = stream_group[-1]
                    self._sync_device_profile(last)
                    self._sync_channel_count(last.channel_count)
                    self._stream_batch_received(stream_group)
                    stream_group = []
                self._sync_device_profile(item)
                self._sync_channel_count(item.channel_count)
                self.current_capture = item
                self._render_current()
            if stream_group:
                last = stream_group[-1]
                self._sync_device_profile(last)
                self._sync_channel_count(last.channel_count)
                self._stream_batch_received(stream_group)
            return

        self._sync_device_profile(capture)
        self._sync_channel_count(capture.channel_count)
        if capture.is_stream:
            self._stream_received(capture)
            return
        self.current_capture = capture
        self._render_current()

    def _sync_device_profile(self, capture: Capture) -> None:
        signature = (capture.channel_count, capture.header.adc_bits)
        changed = False

        if self.device_signature != signature:
            first_device = self.device_signature is None
            for panel in self.channel_panels:
                panel.set_adc_bits(capture.header.adc_bits)

            expected_vref = 5.0 if capture.header.adc_bits <= 10 else 3.3
            for panel in self.channel_panels:
                cfg = panel.config()
                factory_like = (
                    cfg.input_mode == "Direct"
                    and abs(cfg.probe_factor - 1.0) < 1e-9
                    and abs(cfg.bias_voltage) < 1e-12
                    and abs(cfg.fine_gain - 1.0) < 1e-9
                    and abs(cfg.calibration_offset) < 1e-12
                    and (abs(cfg.full_scale - 3.3) < 1e-6 or abs(cfg.full_scale - 5.0) < 1e-6)
                )
                if factory_like and abs(cfg.full_scale - expected_vref) > 1e-6:
                    blocker = QtCore.QSignalBlocker(panel.full_scale)
                    panel.full_scale.setValue(expected_vref)
                    del blocker
                    panel._update_calibration_summary()

            self.device_signature = signature
            rate_max = self._device_rate_limit()
            blocker = QtCore.QSignalBlocker(self.device_rate)
            self.device_rate.setRange(10, rate_max)
            self.device_rate.setValue(min(int(capture.sample_rate), rate_max))
            del blocker

            if first_device:
                source = max(0, min(capture.channel_count - 1, self.trigger_source.currentIndex()))
                mid_count = ((1 << int(capture.header.adc_bits)) - 1) * 0.5
                default_level = self.channel_panels[source].config().raw_scalar_to_volts(
                    mid_count, capture.header.adc_bits
                )
                blocker = QtCore.QSignalBlocker(self.trigger_level)
                self.trigger_level.setValue(default_level)
                del blocker

            self._reset_trigger_state()
            if self.acq_profile.currentText() == PROFILE_AUTO:
                QtCore.QTimer.singleShot(0, self._apply_profile_to_device)
            changed = True
        elif int(self.device_rate.value()) != int(capture.sample_rate):
            blocker = QtCore.QSignalBlocker(self.device_rate)
            self.device_rate.setValue(
                min(int(capture.sample_rate), int(self.device_rate.maximum()))
            )
            del blocker
            changed = True

        if changed:
            self._update_acquisition_info()

    def _sync_channel_count(self, count: int) -> None:
        count = max(1, min(MAX_SCOPE_CHANNELS, int(count)))
        if count == self.active_channel_count:
            return
        self.active_channel_count = count
        for i, panel in enumerate(self.channel_panels):
            visible = i < count
            panel.setVisible(visible)
            tab_index = i + getattr(self, "channel_tab_offset", 0)
            if hasattr(self.channels_tabs, "setTabVisible"):
                self.channels_tabs.setTabVisible(tab_index, visible)
            else:
                self.channels_tabs.setTabEnabled(tab_index, visible)
            self.measurements.setRowHidden(i, not visible)
            self.trigger_source.model().item(i).setEnabled(visible)
            self.cursor_channel.model().item(i).setEnabled(visible)
            if hasattr(self, "cursor_channel_acq"):
                self.cursor_channel_acq.model().item(i).setEnabled(visible)
        if self.trigger_source.currentIndex() >= count:
            self.trigger_source.setCurrentIndex(0)
        if self.cursor_channel.currentIndex() >= count:
            self.cursor_channel.setCurrentIndex(0)
        if hasattr(self, "cursor_channel_acq") and self.cursor_channel_acq.currentIndex() >= count:
            self.cursor_channel_acq.setCurrentIndex(0)

    def _render_current(self, *, rolling: bool = False, warmup: bool = False) -> None:
        capture = self.current_capture
        if capture is None:
            return
        configs = self._configs()
        try:
            time_div = float(self.time_div.currentData())
        except (TypeError, ValueError):
            time_div = 0.1
        axis_ch = max(0, min(capture.channel_count - 1, self._current_channel_index()))
        self.scope.set_voltage_reference(configs[axis_ch])
        now = time.monotonic()
        update_persistence = (
            (not rolling)
            or (now - self.last_persistence_update_ts >= 0.20)
        )
        if update_persistence:
            self.last_persistence_update_ts = now
        self.scope.render_capture(
            capture, configs, time_div, rolling=rolling, warmup=warmup,
            update_persistence=update_persistence,
        )
        self.scope.set_trigger_visible(self.trigger_enable.isChecked(), show_time_marker=not rolling)
        self._update_trigger_line()
        if (not rolling) or (now - self.last_measurement_update_ts >= 0.20):
            self.last_measurement_update_ts = now
            self._update_measurements()
        self._cursor_changed()
        duration = capture.frame_count / max(capture.sample_rate, 1)
        if self._is_stream_profile() and self.stream_history.sample_rate > 0:
            filled_s = self.stream_history.count / max(self.stream_history.sample_rate, 1)
            history_s = self._history_seconds()
            self.capture_status.setText(
                f"{capture.channel_count} CH | {capture.sample_rate:,} Sa/s/ch | "
                f"Direct | Live {self._visible_seconds():g} s | PC history {filled_s:.2f}/{history_s:g} s | "
                f"{capture.header.adc_bits}-bit"
            )
        else:
            self.capture_status.setText(
                f"{capture.channel_count} CH | {capture.sample_rate:,} Sa/s/ch | "
                f"{capture.frame_count:,} frames | {duration * 1000:.1f} ms | "
                f"{capture.header.adc_bits}-bit"
            )

    def _settings_changed(self) -> None:
        if hasattr(self, "measure_channel_checks"):
            for index, checkbox in enumerate(self.measure_channel_checks):
                blocker = QtCore.QSignalBlocker(checkbox)
                checkbox.setChecked(self.channel_panels[index].isChecked())
                del blocker
        self.scope.clear_persistence_history()
        if self.current_capture is not None:
            self._render_current()
        self._send_trigger_settings_guarded()
        self._sync_quick_controls_from_ui()

    def _send_trigger_settings_guarded(self) -> None:
        if hasattr(self, "trigger_source"):
            self._update_trigger_line()

    def _measurement_channel_toggled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self.channel_panels):
            self.channel_panels[index].setChecked(enabled)

    def _choose_channel_color(self, index: int) -> None:
        if not (0 <= index < len(CHANNEL_COLORS)):
            return
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(CHANNEL_COLORS[index]), self, f"Color CH{index + 1}"
        )
        if not color.isValid():
            return
        CHANNEL_COLORS[index] = color.name()
        self._apply_channel_color(index)
        if self.current_capture is not None:
            self._render_current()
            self._update_measurements()

    def _apply_channel_color(self, index: int) -> None:
        if not (0 <= index < len(CHANNEL_COLORS)):
            return
        color = CHANNEL_COLORS[index]
        self.channel_panels[index].setStyleSheet(
            f"QGroupBox{{border-top:2px solid {color};}}"
        )
        if hasattr(self, "channels_tabs"):
            tab_index = index + getattr(self, "channel_tab_offset", 0)
            self.channels_tabs.tabBar().setTabTextColor(tab_index, QtGui.QColor(color))
        if hasattr(self, "measurements"):
            item = self.measurements.verticalHeaderItem(index)
            if item is not None:
                item.setForeground(QtGui.QColor(color))
        if hasattr(self, "measure_color_buttons") and index < len(self.measure_color_buttons):
            self.measure_color_buttons[index].setStyleSheet(
                f"color: {color}; font-size: 16px;"
            )
        if hasattr(self.scope, "curves") and index < len(self.scope.curves):
            self.scope.curves[index].setPen(pg.mkPen(color, width=1))
            for history_curve in self.scope.history_curves[index]:
                hc = pg.mkColor(color)
                hc.setAlpha(55)
                history_curve.setPen(pg.mkPen(hc, width=1))

    def _update_trigger_line(self) -> None:
        capture = self.current_capture
        if capture is None or not self.trigger_enable.isChecked():
            return
        ch = self.trigger_source.currentIndex()
        if ch >= capture.channel_count:
            return
        cfg = self.channel_panels[ch].config()
        y_div = float(self.trigger_level.value()) / max(cfg.v_div, 1e-12) + cfg.position
        self.scope.set_trigger_level(y_div)

    def _update_measurements(self) -> None:
        capture = self.current_capture
        if capture is None:
            return
        configs = self._configs()
        for ch in range(MAX_SCOPE_CHANNELS):
            values = ["—"] * 8
            if ch < capture.channel_count:
                volts = configs[ch].raw_to_volts(capture.raw[:, ch], capture.header.adc_bits)
                m = analyze(volts, capture.sample_rate)
                values = [
                    fmt(m.minimum, "V"), fmt(m.maximum, "V"), fmt(m.vpp, "V"),
                    fmt(m.mean, "V"), fmt(m.rms, "V"), fmt(m.frequency, "Hz"),
                    fmt(m.period, "s"), fmt(m.duty, "%"),
                ]
            for metric, text in enumerate(values, start=2):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if ch < len(CHANNEL_COLORS):
                    item.setForeground(QtGui.QColor(CHANNEL_COLORS[ch]))
                self.measurements.setItem(ch, metric, item)

    # ---------- Cursors ----------
    def _sync_cursor_channel(self, index: int, source: QtWidgets.QComboBox) -> None:
        for combo in (self.cursor_channel, getattr(self, "cursor_channel_acq", None)):
            if combo is None or combo is source or combo.currentIndex() == index:
                continue
            blocker = QtCore.QSignalBlocker(combo)
            combo.setCurrentIndex(index)
            del blocker
        self._cursor_changed()

    def _cursor_changed(self, *_) -> None:
        dt = abs(self.scope.t2.value() - self.scope.t1.value())
        inv = 1.0 / dt if dt > 0 else math.nan
        ch = self.cursor_channel.currentIndex()
        dv = math.nan
        if 0 <= ch < len(self.channel_panels):
            cfg = self.channel_panels[ch].config()
            dv = abs(self.scope.v2.value() - self.scope.v1.value()) * cfg.v_div
        text = f"Δt {fmt(dt, 's')}    1/Δt {fmt(inv, 'Hz')}    ΔV {fmt(dv, 'V')}"
        self.cursor_readout.setText(text)
        if hasattr(self, "cursor_readout_acq"):
            self.cursor_readout_acq.setText(text)
        self.scope.set_delta_text(text)

    def _reset_active_calibration(self) -> None:
        index = self._current_channel_index()
        if not (0 <= index < len(self.channel_panels)):
            return
        bits = self.current_capture.header.adc_bits if self.current_capture is not None else 12
        self.channel_panels[index].reset_for_device(bits)

    # ---------- Autoset ----------
    def _autoset(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Auto Set",
            "Aplicar o Auto Set agora? Isso vai reajustar escala, posicao e time/div.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        capture = self.current_capture
        if capture is None:
            return
        configs = self._configs()
        for ch in range(capture.channel_count):
            if not configs[ch].enabled:
                continue
            volts = configs[ch].raw_to_volts(capture.raw[:, ch], capture.header.adc_bits)
            span = max(float(np.ptp(volts)), 1e-6)
            target = span / 5.5
            idx = min(range(len(V_DIVS)), key=lambda i: abs(math.log(V_DIVS[i] / target)))
            self.channel_panels[ch].vdiv.setCurrentIndex(idx)
            center_v = float((np.max(volts) + np.min(volts)) * 0.5)
            self.channel_panels[ch].position.setValue(-center_v / V_DIVS[idx])

        duration = capture.frame_count / max(capture.sample_rate, 1)
        target_td = duration / 10.0
        idx = min(range(len(TIME_DIVS)), key=lambda i: abs(math.log(TIME_DIVS[i] / target_td)))
        self.time_div.setCurrentIndex(idx)
        self._render_current()

    # ---------- Files ----------
    def _save_npz(self) -> None:
        if self.current_capture is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save capture", "capture.npz", "NumPy capture (*.npz)")
        if not path:
            return
        capture = self.current_capture
        meta = {
            "sample_rate": capture.sample_rate,
            "pretrigger_frames": capture.pretrigger_frames,
            "trigger_level": capture.header.trigger_level,
            "trigger_channel": capture.header.trigger_channel,
            "trigger_edge": capture.header.trigger_edge,
            "adc_bits": capture.header.adc_bits,
            "channels": [asdict(c) for c in self._configs()[:capture.channel_count]],
        }
        np.savez_compressed(path, raw=capture.raw, meta=json.dumps(meta))

    def _open_npz(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open capture", "", "NumPy capture (*.npz)")
        if not path:
            return
        try:
            from .protocol import CaptureHeader, MAGIC_U32
            data = np.load(path, allow_pickle=False)
            raw = np.asarray(data["raw"], dtype=np.uint16)
            meta_value = data["meta"].item() if getattr(data["meta"], "shape", None) == () else str(data["meta"])
            meta = json.loads(str(meta_value))
            frames, channels = raw.shape
            header = CaptureHeader(
                MAGIC_U32, 2, 30, int(meta["sample_rate"]), frames,
                int(meta.get("pretrigger_frames", frames // 2)),
                int(meta.get("trigger_level", 2048)),
                int(channels), int(meta.get("trigger_channel", 1)),
                int(meta.get("trigger_edge", 1)), int(meta.get("adc_bits", 12)),
                frames * channels * 2, 0,
            )
            self.current_capture = Capture(header, raw.copy())
            self._sync_device_profile(self.current_capture)
            for i, cfg in enumerate(meta.get("channels", [])):
                if i < len(self.channel_panels):
                    self.channel_panels[i].set_config(cfg)
            self._sync_channel_count(channels)
            self._render_current()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open capture", str(exc))

    def _export_csv(self) -> None:
        capture = self.current_capture
        if capture is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", "capture.csv", "CSV (*.csv)")
        if not path:
            return
        configs = self._configs()
        cols = [capture.time_axis()]
        names = ["time_s"]
        for ch in range(capture.channel_count):
            cols.append(configs[ch].raw_to_volts(capture.raw[:, ch], capture.header.adc_bits))
            names.append(configs[ch].name + "_V")
        array = np.column_stack(cols)
        np.savetxt(path, array, delimiter=",", header=",".join(names), comments="", fmt="%.10g")

    def _export_png(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", "OpenScope.png", "PNG image (*.png)")
        if path:
            self.scope.export_png(path)

    def _session_data(self) -> dict:
        return {
            "channels": [asdict(c) for c in self._configs()],
            "acq_profile": self.acq_profile.currentText(),
            "record_length": float(self.record_length.currentData()),
            "time_div": float(self.time_div.currentData()),
            "vertical_span": float(self.vertical_span.currentData()),
            "persistence": self.persistence.value(),
            "trigger_enabled": self.trigger_enable.isChecked(),
            "trigger_source": self.trigger_source.currentIndex(),
            "trigger_edge": int(self.trigger_edge.currentData()),
            "trigger_level_v": float(self.trigger_level.value()),
            "pretrigger": self.pretrigger.value(),
            "trigger_mode": self.trigger_mode.currentText(),
            "auto_trigger_timeout_ms": self.auto_trigger_timeout.value(),
            "trigger_holdoff_ms": self.trigger_holdoff.value(),
            "device_rate": self.device_rate.value(),
        }

    def _apply_session(self, data: dict) -> None:
        for i, cfg in enumerate(data.get("channels", [])):
            if i < len(self.channel_panels):
                self.channel_panels[i].set_config(cfg)
        saved_profile = str(data.get("acq_profile", PROFILE_LONG))
        if self.acq_profile.findText(saved_profile) >= 0:
            self.acq_profile.setCurrentText(saved_profile)
        else:
            self.acq_profile.setCurrentText(PROFILE_LONG)

        record = float(data.get("record_length", 1.0))
        ridx = min(range(len(RECORD_LENGTHS)), key=lambda i: abs(RECORD_LENGTHS[i] - record))
        self.record_length.setCurrentIndex(ridx)

        td = float(data.get("time_div", 0.1))
        idx = min(range(len(TIME_DIVS)), key=lambda i: abs(TIME_DIVS[i] - td))
        self.time_div.setCurrentIndex(idx)

        span = float(data.get("vertical_span", 8.0))
        sidx = min(range(len(VERTICAL_SPANS)), key=lambda i: abs(VERTICAL_SPANS[i] - span))
        self.vertical_span.setCurrentIndex(sidx)
        self.scope.set_vertical_span(float(self.vertical_span.currentData()))
        self.persistence.setValue(int(data.get("persistence", 0)))
        self.trigger_enable.setChecked(bool(data.get("trigger_enabled", True)))
        self.trigger_source.setCurrentIndex(int(data.get("trigger_source", 0)))
        edge = int(data.get("trigger_edge", 1))
        self.trigger_edge.setCurrentIndex(max(0, self.trigger_edge.findData(edge)))
        if "trigger_level_v" in data:
            self.trigger_level.setValue(float(data.get("trigger_level_v", 1.65)))
        self.pretrigger.setValue(int(data.get("pretrigger", 50)))
        mode = str(data.get("trigger_mode", "Normal"))
        if mode == "Single":
            mode = "Normal"
        mode_index = self.trigger_mode.findText(mode, QtCore.Qt.MatchFixedString)
        if mode_index >= 0:
            self.trigger_mode.setCurrentIndex(mode_index)
        self.auto_trigger_timeout.setValue(int(data.get("auto_trigger_timeout_ms", 250)))
        self.trigger_holdoff.setValue(float(data.get("trigger_holdoff_ms", 20.0)))
        self.device_rate.setValue(int(data.get("device_rate", 30000)))
        self._update_profile_ui()
        self._settings_changed()

    def _save_session_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save settings", "openscope_settings.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self._session_data(), indent=2), encoding="utf-8")

    def _load_session_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if path:
            try:
                self._apply_session(json.loads(Path(path).read_text(encoding="utf-8")))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Load settings", str(exc))

    def _show_fft(self) -> None:
        if self.current_capture is None:
            return
        dlg = FFTDialog(self.current_capture, self._configs(), self)
        dlg.exec_()

    def _show_xy(self) -> None:
        if self.current_capture is None or self.current_capture.channel_count < 2:
            return
        dlg = XYDialog(self.current_capture, self._configs(), self)
        dlg.exec_()

    def _show_rpm_frequency_calculator(self) -> None:
        dialog = RpmFrequencyCalculatorDialog(self)
        set_windows_dark_titlebar(dialog, getattr(self, "_theme", "dark") == "dark")
        dialog.exec_()

    def _visit_lobby(self) -> None:
        lobby = LobbyDialog(self)
        translate_ui(lobby, getattr(self, "_language", "pt"))
        QtCore.QTimer.singleShot(0, lambda: set_windows_dark_titlebar(lobby, getattr(self, "_theme", "dark") == "dark"))
        if lobby.exec_() != QtWidgets.QDialog.Accepted:
            return
        if self.worker is not None:
            self._disconnect()
        if self.demo_btn.isChecked():
            self.demo_btn.setChecked(False)
        if lobby.use_demo:
            self.start_demo_session()
        elif lobby.selected_port:
            self.start_serial_session(lobby.selected_port, lobby.selected_baud)

    # ---------- QSettings ----------
    def _restore_settings(self) -> None:
        s = QtCore.QSettings("SCP", "STM32Scope")
        layout_version = int(s.value("layoutVersion", 0) or 0)

        # Version 4 resets older tab/dock geometry that could stack pages on
        # first start after an update. Older geometry is intentionally ignored.
        if layout_version >= 4:
            geometry = s.value("geometry")
            state = s.value("windowState")
            if geometry is not None:
                self.restoreGeometry(geometry)
            if state is not None:
                self.restoreState(state)
        else:
            self._apply_default_layout()

        acquisition_version = int(s.value("acquisitionVersion", 0) or 0)
        session = s.value("session")
        if session:
            try:
                self._apply_session(json.loads(str(session)))
            except Exception:
                pass

        # v2 separates long PC-side history from the visible Time/div zoom.
        # Preserve channel calibration/layout from older builds, but start with
        # a sane 1-second engine record instead of resurrecting a 2 ms window.
        if acquisition_version < 5:
            blockers = [
                QtCore.QSignalBlocker(self.acq_profile),
                QtCore.QSignalBlocker(self.record_length),
                QtCore.QSignalBlocker(self.time_div),
                QtCore.QSignalBlocker(self.vertical_span),
                QtCore.QSignalBlocker(self.persistence),
            ]
            self.acq_profile.setCurrentText(PROFILE_LONG)
            self.record_length.setCurrentIndex(0)
            self.time_div.setCurrentText("100 ms/div")
            self.vertical_span.setCurrentText("8 div")
            self.persistence.setValue(0)
            del blockers
            self.scope.set_vertical_span(8.0)
            self.scope.set_persistence(0)

        self.scope.set_view_x_offset_div(float(s.value("viewXOffsetDiv", 0.0) or 0.0))
        self.scope.set_view_y_offset(float(s.value("viewYOffsetDiv", 0.0) or 0.0))
        if hasattr(self, "act_soft_touch"):
            soft_touch = str(s.value("softTouchEnabled", "false")).lower() in ("1", "true", "yes")
            self.act_soft_touch.setChecked(soft_touch)

        self._update_profile_ui()
        self._update_trigger_controls()
        self._sync_quick_controls_from_ui()

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            if available.height() <= 800 or available.width() <= 1366:
                self.act_compact.setChecked(True)

    def _apply_default_layout(self) -> None:
        self.channels_dock.show()
        self.acquisition_dock.show()
        self.measurement_dock.show()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.channels_dock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.acquisition_dock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.measurement_dock)
        QtCore.QTimer.singleShot(0, self._size_default_docks)

    def _size_default_docks(self) -> None:
        self.resizeDocks(
            [self.channels_dock, self.acquisition_dock],
            [235, 252],
            QtCore.Qt.Horizontal,
        )
        self.resizeDocks(
            [self.measurement_dock],
            [108],
            QtCore.Qt.Vertical,
        )

    def _reset_layout(self) -> None:
        self._apply_default_layout()
        self.act_compact.setChecked(self.height() <= 800 or self.width() <= 1366)

    def _set_compact_mode(self, enabled: bool) -> None:
        self.setProperty("compactMode", enabled)
        self.setStyleSheet(COMPACT_QSS if enabled else "")
        if enabled:
            self.channels_dock.setMaximumWidth(235)
            self.acquisition_dock.setMaximumWidth(270)
            self.measurement_dock.setMaximumHeight(135)
        else:
            self.channels_dock.setMaximumWidth(295)
            self.acquisition_dock.setMaximumWidth(292)
            self.measurement_dock.setMaximumHeight(210)

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
        QtCore.QTimer.singleShot(0, self._size_default_docks if enabled else lambda: None)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None:
            self._disconnect()
        if hasattr(self, "ardustim_tab"):
            self.ardustim_tab.close_connection()
        s = QtCore.QSettings("SCP", "STM32Scope")
        s.setValue("layoutVersion", 4)
        s.setValue("acquisitionVersion", 5)
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        s.setValue("session", json.dumps(self._session_data()))
        s.setValue("viewXOffsetDiv", self.scope.view_x_offset_div)
        s.setValue("viewYOffsetDiv", self.scope.view_y_offset)
        s.setValue("softTouchEnabled", self.act_soft_touch.isChecked() if hasattr(self, "act_soft_touch") else False)
        super().closeEvent(event)

    def start_serial_session(self, port: str, baud: int) -> None:
        self._refresh_ports()
        port_index = self.port.findData(port)
        if port_index >= 0:
            self.port.setCurrentIndex(port_index)
        baud_index = self.baud.findData(int(baud))
        if baud_index >= 0:
            self.baud.setCurrentIndex(baud_index)
        self._toggle_connection()

    def start_demo_session(self) -> None:
        self.demo_btn.setChecked(True)
        self._rearm_trigger()


def run(app: Optional[QtWidgets.QApplication] = None) -> int:
    import sys
    if app is None and QtWidgets.QApplication.instance() is None:
        if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = app or QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setOrganizationName(APP_ORGANIZATION_NAME)
    icon_path = resource_path("OpenScope.ico")
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    settings = QtCore.QSettings("SCP", "STM32Scope")
    theme = str(settings.value("theme", "dark"))
    language = str(settings.value("language", "pt"))
    apply_theme(app, theme)
    lobby = LobbyDialog()
    translate_ui(lobby, language)
    QtCore.QTimer.singleShot(0, lambda: set_windows_dark_titlebar(lobby, theme == "dark"))
    if lobby.exec_() != QtWidgets.QDialog.Accepted:
        return 0
    window = MainWindow()
    window.show()
    QtCore.QTimer.singleShot(0, lambda: set_windows_dark_titlebar(window, theme == "dark"))
    if lobby.use_demo:
        QtCore.QTimer.singleShot(0, window.start_demo_session)
    elif lobby.selected_port:
        QtCore.QTimer.singleShot(
            0,
            lambda: window.start_serial_session(lobby.selected_port, lobby.selected_baud),
        )
    return app.exec_()
