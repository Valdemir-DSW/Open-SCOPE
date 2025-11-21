import sys
import struct
import json
import numpy as np
import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import os
from PyQt5.QtGui import QIcon
CONFIG_FILE = "config.json"

# unidades
VOLT_UNITS = ["pV", "nV", "µV", "mV", "V", "kV", "MV"]
VOLT_FACTORS = {"pV": 1e-12, "nV": 1e-9, "µV": 1e-6, "mV": 1e-3, "V": 1.0, "kV": 1e3, "MV": 1e6}
TIME_UNITS = ["ps", "ns", "µs", "ms", "s"]
TIME_FACTORS = {"ps": 1e-12, "ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}


# ---------- util: frequência robusta ----------
def compute_frequency(data, sps):
    """Estima frequência usando cruzamentos ascendentes com interpolação linear,
    retornando Hz. Usa sps (samples/segundo) para converter amostras em tempo real.
    Usa apenas janela recente (até 2s).
    """
    try:
        sps = max(1, int(sps))
        if data is None or data.size < 16:
            return 0.0
        vals = np.asarray(data)
        # use at most last 2 seconds (ou buffer length)
        max_samples = min(vals.size, max(int(sps * 2), 128))
        seg_all = vals[-max_samples:]
        if np.allclose(seg_all, seg_all[0], atol=1e-12):
            return 0.0
        # detect active region inside seg_all
        diffs = np.abs(np.diff(seg_all))
        thr = max(np.std(seg_all) * 0.02, 1e-6)
        good = np.where(diffs > thr)[0]
        if good.size >= 2:
            first = max(0, good[0] - 2)
            last = min(seg_all.size - 1, good[-1] + 2)
            seg = seg_all[first:last + 1]
            offset = (vals.size - max_samples) + first
        else:
            seg = seg_all
            offset = vals.size - seg.size
        if seg.size < 4:
            return 0.0
        mean = seg.mean()
        v0 = seg[:-1] - mean
        v1 = seg[1:] - mean
        mask = (v0 < 0) & (v1 >= 0)
        idxs = np.nonzero(mask)[0]
        if idxs.size < 2:
            return 0.0
        crossings = []
        for i in idxs:
            denom = (v1[i] - v0[i])
            frac = 0.0 if denom == 0 else (-v0[i] / denom)
            sample_pos = offset + i + frac
            crossings.append(sample_pos / float(sps))
        if len(crossings) < 2:
            return 0.0
        periods = np.diff(np.array(crossings, dtype=float))
        periods = periods[periods > 0]
        if periods.size == 0:
            return 0.0
        period = float(np.median(periods))
        if period <= 0:
            return 0.0
        return 1.0 / period
    except Exception:
        return 0.0


class SerialReader(QtCore.QThread):
    # fixo: dois canais apenas. Emite raw ADC (inteiros) para permitir calibração no GUI.
    data_ready = QtCore.pyqtSignal(int, int)
    connection_lost = QtCore.pyqtSignal()

    def __init__(self, port, baudrate=115200, sps=1000, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.ser = None
        self.sps = sps
        self.frame_bytes = 4  # 2 canais x 2 bytes (uint16 little-endian)

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
        except Exception:
            self.connection_lost.emit()
            return

        self.running = True
        buffer = bytearray()
        while self.running:
            try:
                data = self.ser.read(4096)
                if data:
                    buffer.extend(data)
                    if len(buffer) > 65536:
                        buffer = buffer[-65536:]
                    # parse frames de 4 bytes (2x uint16)
                    while len(buffer) >= self.frame_bytes:
                        raw = buffer[:self.frame_bytes]
                        buffer = buffer[self.frame_bytes:]
                        try:
                            raw1, raw2 = struct.unpack('<HH', raw)
                            # emit raw ADC reading (expected 0..max_adc per calibration)
                            raw1 = int(raw1)
                            raw2 = int(raw2)
                            self.data_ready.emit(raw1, raw2)
                        except Exception:
                            # tentar realinhar
                            buffer = buffer[1:]
                            continue
                else:
                    QtCore.QThread.yieldCurrentThread()
            except Exception:
                self.connection_lost.emit()
                break

        if self.ser and self.ser.is_open:
            self.ser.close()

    def stop(self):
        self.running = False
        self.wait(2000)


class CalibrationDialog(QtWidgets.QDialog):
    """Diálogo para calibrar mapeamento ADC->Volts por canal, bits, probe 10x, biases e salvar PNG."""
    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.setWindowTitle("Configurações / Calibração")
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)

        cfg = cfg or {}
        c = cfg.get("calibration", {})
        ch1_0 = c.get("ch1_0", 0.0)
        ch1_fs = c.get("ch1_1023", 5.0)
        ch2_0 = c.get("ch2_0", 0.0)
        ch2_fs = c.get("ch2_1023", 5.0)
        bits = cfg.get("adc_bits", 10)
        probe10x = cfg.get("probe_10x", False)
        time_bias = cfg.get("time_bias", 6.0)
        freq_bias = cfg.get("freq_bias", 6.0)

        self.ch1_0_box = QtWidgets.QDoubleSpinBox()
        self.ch1_0_box.setRange(-1e6, 1e6)
        self.ch1_0_box.setDecimals(6)
        self.ch1_0_box.setValue(ch1_0)

        self.ch1_fs_box = QtWidgets.QDoubleSpinBox()
        self.ch1_fs_box.setRange(-1e6, 1e6)
        self.ch1_fs_box.setDecimals(6)
        self.ch1_fs_box.setValue(ch1_fs)

        self.ch2_0_box = QtWidgets.QDoubleSpinBox()
        self.ch2_0_box.setRange(-1e6, 1e6)
        self.ch2_0_box.setDecimals(6)
        self.ch2_0_box.setValue(ch2_0)

        self.ch2_fs_box = QtWidgets.QDoubleSpinBox()
        self.ch2_fs_box.setRange(-1e6, 1e6)
        self.ch2_fs_box.setDecimals(6)
        self.ch2_fs_box.setValue(ch2_fs)

        self.bits_box = QtWidgets.QComboBox()
        self.bits_box.addItems(["8", "10", "12", "14", "16"])
        self.bits_box.setCurrentText(str(bits))

        self.probe10x_box = QtWidgets.QCheckBox("Ponta 10x (ativa na GUI)")
        self.probe10x_box.setChecked(bool(probe10x))

        # bias de tempo / frequência
        self.time_bias_box = QtWidgets.QDoubleSpinBox()
        self.time_bias_box.setRange(0.0001, 1e6)
        self.time_bias_box.setDecimals(6)
        self.time_bias_box.setValue(float(time_bias))
        self.freq_bias_box = QtWidgets.QDoubleSpinBox()
        self.freq_bias_box.setRange(0.000001, 1e6)
        self.freq_bias_box.setDecimals(6)
        self.freq_bias_box.setValue(float(freq_bias))

        layout.addRow("CH1 -> ADC=0 (V):", self.ch1_0_box)
        layout.addRow("CH1 -> ADC=FS (V):", self.ch1_fs_box)
        layout.addRow("CH2 -> ADC=0 (V):", self.ch2_0_box)
        layout.addRow("CH2 -> ADC=FS (V):", self.ch2_fs_box)
        layout.addRow("ADC bits:", self.bits_box)
        layout.addRow(self.probe10x_box)
        layout.addRow("Bias tempo (multiplicador):", self.time_bias_box)
        layout.addRow("Bias frequência (divisor):", self.freq_bias_box)

        # botões e salvar PNG rápido + GitHub link + restaurar padrões
        btns_row = QtWidgets.QHBoxLayout()
        self.btn_save_png = QtWidgets.QPushButton("Salvar PNG do gráfico")
        btns_row.addWidget(self.btn_save_png)
        self.github_btn = QtWidgets.QPushButton("GitHub")
        btns_row.addWidget(self.github_btn)
        self.btn_restore = QtWidgets.QPushButton("Restaurar padrões")
        btns_row.addWidget(self.btn_restore)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns_row.addWidget(btns)
        layout.addRow(btns_row)

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.btn_save_png.clicked.connect(self.on_save_png)
        self.github_btn.clicked.connect(self.open_github)
        self.btn_restore.clicked.connect(self.restore_defaults)

    def on_save_png(self):
        p = self.parent()
        if p and hasattr(p, "save_plot_png"):
            p.save_plot_png()

    def open_github(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://github.com/Valdemir-DSW/Open-SCOPE"))

    def restore_defaults(self):
        try:
            self.ch1_0_box.setValue(0.0)
            self.ch1_fs_box.setValue(5.0)
            self.ch2_0_box.setValue(0.0)
            self.ch2_fs_box.setValue(5.0)
            self.bits_box.setCurrentText("10")
            self.probe10x_box.setChecked(False)
            self.time_bias_box.setValue(5.0)
            self.freq_bias_box.setValue(5.0)
        except Exception:
            pass

    def values(self):
        return {
            "ch1_0": float(self.ch1_0_box.value()),
            "ch1_1023": float(self.ch1_fs_box.value()),
            "ch2_0": float(self.ch2_0_box.value()),
            "ch2_1023": float(self.ch2_fs_box.value()),
            "adc_bits": int(self.bits_box.currentText()),
            "probe_10x": bool(self.probe10x_box.isChecked()),
            "time_bias": float(self.time_bias_box.value()),
            "freq_bias": float(self.freq_bias_box.value())
        }


class Oscilloscope(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open Oscilloscope - Osciloscópio 2CH - By Valdemir")
        base = os.path.dirname(__file__)

        icon_path = os.path.join(base, "ico.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.resize(1400, 720)

        # estado
        self.reader = None
        self.paused = False

        # parâmetros persistentes / defaults
        self.scale_min = -5.0
        self.scale_max = 5.0
        self.sps = 1000
        self.channels_expected = 2  # fixo dois canais
        self.autoscale_y = False
        self.auto_scroll = True  # se True, XRange é forçado para janela atual

        # calibração ADC->V (permanente)
        self.calibration = {
            "ch1_0": 0.0, "ch1_1023": 5.0,
            "ch2_0": 0.0, "ch2_1023": 5.0
        }
        # adc bits and probe
        self.adc_bits = 10
        self.probe_10x = False

        # tempo / frequência bias (padrões)
        self.time_bias = 5.0
        self.freq_bias = 5.0

        # unidades selecionadas (por padrão)
        self.volt_unit = "V"
        self.time_unit = "ms"

        # buffers
        self.buffer_len = 4000  # aumentar para estabilidade da frequência
        self.data1 = np.zeros(self.buffer_len)
        self.data2 = np.zeros(self.buffer_len)
        self.vmax1 = 0.0
        self.vmax2 = 0.0

        # display type: Voltage / RMS / PMPO / Frequency
        self.display_types = ["Voltage", "RMS", "PMPO", "Frequency"]

        # carregar configuração (pode sobrescrever defaults acima)
        self.load_config()

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_l = QtWidgets.QVBoxLayout(central)

        # top: controls (serial + SPS + autoscale toggle) in a horizontal toolbar-like row
        top_row = QtWidgets.QHBoxLayout()
        main_l.addLayout(top_row)

        top_row.addWidget(QtWidgets.QLabel("Porta:"))
        self.port_box = QtWidgets.QComboBox()
        top_row.addWidget(self.port_box)
        self.refresh_ports()
        self.btn_refresh = QtWidgets.QPushButton("Atualizar")
        top_row.addWidget(self.btn_refresh)
        self.btn_connect = QtWidgets.QPushButton("Conectar")
        self.btn_disconnect = QtWidgets.QPushButton("Desconectar")
        self.btn_disconnect.setEnabled(False)
        top_row.addWidget(self.btn_connect)
        top_row.addWidget(self.btn_disconnect)

        top_row.addSpacing(10)
        top_row.addWidget(QtWidgets.QLabel("SPS:"))
        self.sps_box = QtWidgets.QSpinBox()
        self.sps_box.setRange(1, 20000)
        self.sps_box.setValue(self.sps)
        top_row.addWidget(self.sps_box)

        self.chk_autoscale = QtWidgets.QCheckBox("Auto-scale Y")
        self.chk_autoscale.setChecked(self.autoscale_y)
        top_row.addWidget(self.chk_autoscale)

        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll tempo")
        self.chk_autoscroll.setChecked(self.auto_scroll)
        top_row.addWidget(self.chk_autoscroll)

        top_row.addSpacing(10)
        self.btn_calibrate = QtWidgets.QPushButton("Hardware")
        top_row.addWidget(self.btn_calibrate)

        top_row.addSpacing(10)
        self.btn_fit = QtWidgets.QPushButton("Enquadrar")
        top_row.addWidget(self.btn_fit)

        top_row.addStretch()

        # middle: plot
        self.plot_widget = pg.PlotWidget()
        main_l.addWidget(self.plot_widget, 1)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()

        self.curve1 = self.plot_widget.plot(pen='y', name="CH1")
        self.curve2 = self.plot_widget.plot(pen='c', name="CH2")

        # create two movable vertical cursors (hidden until paused)
        self.cursor1 = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen('w', width=1))
        self.cursor2 = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen('r', width=1))
        self.cursor1.setVisible(False)
        self.cursor2.setVisible(False)
        self.plot_widget.addItem(self.cursor1)
        self.plot_widget.addItem(self.cursor2)
        self.cursor1.sigPositionChanged.connect(lambda: self.update_cursor_info())
        self.cursor2.sigPositionChanged.connect(lambda: self.update_cursor_info())

        # make viewbox allow mouse interactions; capture range-changed to detect user zoom/pan
        vb = self.plot_widget.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        try:
            vb.setMouseMode(pg.ViewBox.RectMode)
            vb.sigRangeChanged.connect(self.on_view_range_changed)
        except Exception:
            pass

        # label de informação de escala/performance abaixo do gráfico
        self.scale_info_label = QtWidgets.QLabel("")
        main_l.addWidget(self.scale_info_label)

        # refresh rate control row (slider para FPS)
        refresh_row = QtWidgets.QHBoxLayout()
        main_l.addLayout(refresh_row)
        
        refresh_row.addWidget(QtWidgets.QLabel("Taxa de atualização:"))
        self.refresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.refresh_slider.setRange(10, 1000)  # 10ms (100 FPS) até 1000ms (1 FPS)
        self.refresh_slider.setValue(25)  # padrão: 25ms = 40 FPS
        self.refresh_slider.setSingleStep(5)
        self.refresh_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.refresh_slider.setTickInterval(50)
        self.refresh_slider.setMaximumWidth(300)
        refresh_row.addWidget(self.refresh_slider)
        
        self.refresh_label = QtWidgets.QLabel("25ms (40 FPS)")
        refresh_row.addWidget(self.refresh_label)
        
        self.refresh_full_btn = QtWidgets.QPushButton("FULL")
        self.refresh_full_btn.setMaximumWidth(60)
        refresh_row.addWidget(self.refresh_full_btn)
        
        refresh_row.addStretch()

        # bottom: controls under plot (units, display type, autoscale button, value labels)
        bottom_row = QtWidgets.QHBoxLayout()
        main_l.addLayout(bottom_row)

        # display type
        bottom_row.addWidget(QtWidgets.QLabel("Exibir:"))
        self.display_combo = QtWidgets.QComboBox()
        self.display_combo.addItems(self.display_types)
        bottom_row.addWidget(self.display_combo)

        # channel on/off
        self.ch1_enable = QtWidgets.QCheckBox("CH1")
        self.ch1_enable.setChecked(True)
        self.ch2_enable = QtWidgets.QCheckBox("CH2")
        self.ch2_enable.setChecked(True)
        bottom_row.addWidget(self.ch1_enable)
        bottom_row.addWidget(self.ch2_enable)
        self.ch1_enable.stateChanged.connect(lambda s: self.set_channel_visible(1, bool(s)))
        self.ch2_enable.stateChanged.connect(lambda s: self.set_channel_visible(2, bool(s)))

        bottom_row.addSpacing(10)
        # voltage unit
        bottom_row.addWidget(QtWidgets.QLabel("Voltagem:"))
        self.volt_unit_box = QtWidgets.QComboBox()
        self.volt_unit_box.addItems(VOLT_UNITS)
        self.volt_unit_box.setCurrentText(self.volt_unit if self.volt_unit in VOLT_UNITS else "V")
        bottom_row.addWidget(self.volt_unit_box)

        # time unit
        bottom_row.addWidget(QtWidgets.QLabel("Tempo:"))
        self.time_unit_box = QtWidgets.QComboBox()
        self.time_unit_box.addItems(TIME_UNITS)
        self.time_unit_box.setCurrentText(self.time_unit if self.time_unit in TIME_UNITS else "ms")
        bottom_row.addWidget(self.time_unit_box)

        bottom_row.addSpacing(10)
        # autoscale Y now a button to trigger immediate autoscale
        self.btn_autoscale = QtWidgets.QPushButton("Auto-scale Y (ajustar)")
        bottom_row.addWidget(self.btn_autoscale)

        bottom_row.addStretch()

        # values area under plot (single-line per channel) + cursor info
        values_layout = QtWidgets.QHBoxLayout()
        main_l.addLayout(values_layout)
        self.value_label_ch1 = QtWidgets.QLabel("CH1: -")
        self.value_label_ch2 = QtWidgets.QLabel("CH2: -")
        self.cursor_info_label = QtWidgets.QLabel("")  # Δt info
        values_layout.addWidget(self.value_label_ch1)
        values_layout.addSpacing(20)
        values_layout.addWidget(self.value_label_ch2)
        values_layout.addSpacing(40)
        values_layout.addWidget(self.cursor_info_label)
        values_layout.addStretch()

        # status row
        status_row = QtWidgets.QHBoxLayout()
        main_l.addLayout(status_row)
        self.status_label = QtWidgets.QLabel("Desconectado")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        # pause/save/reset buttons on right
        self.btn_pause = QtWidgets.QPushButton("Pausar")
        self.btn_save = QtWidgets.QPushButton("Salvar CSV")
        self.btn_reset = QtWidgets.QPushButton("Zerar")
        # add Save PNG beside Save CSV as requested
        self.btn_save_png = QtWidgets.QPushButton("Salvar PNG")
        status_row.addWidget(self.btn_pause)
        status_row.addWidget(self.btn_save)
        status_row.addWidget(self.btn_save_png)
        status_row.addWidget(self.btn_reset)

        # timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(25)

        # conexões
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.connect_serial)
        self.btn_disconnect.clicked.connect(self.disconnect_serial)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_save.clicked.connect(self.save_csv)
        self.btn_save_png.clicked.connect(self.save_plot_png)
        self.btn_reset.clicked.connect(self.reset_measurements)
        self.sps_box.valueChanged.connect(self.update_sps)
        self.chk_autoscroll.stateChanged.connect(self.on_autoscroll_changed)
        self.chk_autoscale.stateChanged.connect(lambda s: None)  # keep checkbox state; use button to trigger
        self.btn_autoscale.clicked.connect(self.auto_scale_y)
        self.btn_calibrate.clicked.connect(self.open_calibration)
        self.btn_fit.clicked.connect(self.fit_view)
        
        # refresh rate connections
        self.refresh_slider.valueChanged.connect(self.on_refresh_rate_changed)
        self.refresh_full_btn.clicked.connect(self.set_refresh_full)

        self.volt_unit_box.currentIndexChanged.connect(self.update_scale_info_label)
        self.time_unit_box.currentIndexChanged.connect(self.update_scale_info_label)
        self.display_combo.currentIndexChanged.connect(self.update_scale_info_label)

        # initial UI labels and restore view
        self.plot_widget.setLabel('left', 'Tensão (V)')
        self.plot_widget.setLabel('bottom', 'Tempo (s)')
        QtCore.QTimer.singleShot(50, self.apply_saved_view_ranges)
        self.update_scale_info_label()

    # ---------- helpers ----------
    def on_refresh_rate_changed(self, value):
        """Atualizar intervalo do timer quando slider muda."""
        interval_ms = int(value)
        fps = 1000.0 / max(1, interval_ms)
        self.timer.setInterval(interval_ms)
        self.refresh_label.setText(f"{interval_ms}ms ({fps:.1f} FPS)")
        self.save_config()

    def set_refresh_full(self):
        """Botão FULL: atualizar a cada data packet (10ms padrão, ajustável)."""
        # usar mínimo do slider (10ms) para máxima taxa
        self.refresh_slider.setValue(10)

    def refresh_ports(self):
        self.port_box.clear()
        for p in serial.tools.list_ports.comports():
            self.port_box.addItem(p.device)

    def connect_serial(self):
        port = self.port_box.currentText()
        if not port:
            self.status_label.setText("Nenhuma porta selecionada")
            return
        self.reader = SerialReader(port, baudrate=115200, sps=self.sps_box.value())
        self.reader.data_ready.connect(self.on_data)
        self.reader.connection_lost.connect(self.on_disconnect)
        self.reader.start()
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self.status_label.setText(f"Conectado a {port}")

    def disconnect_serial(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.status_label.setText("Desconectado")

    def on_disconnect(self):
        self.disconnect_serial()
        self.status_label.setText("Conexão perdida.")

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("Retomar" if self.paused else "Pausar")
        # enable/disable cursors
        if self.paused:
            self.cursor1.setVisible(True)
            self.cursor2.setVisible(True)
            # pg.InfiniteLine uses setMovable only in some versions; use setMovable alias if present
            try:
                self.cursor1.setMovable(True)
                self.cursor2.setMovable(True)
            except Exception:
                try:
                    self.cursor1.setMovable = True
                    self.cursor2.setMovable = True
                except Exception:
                    pass
            # place cursors near ends by default
            try:
                vb = self.plot_widget.getViewBox()
                xr = vb.viewRange()[0]
                self.cursor1.setValue(xr[0] + (xr[1] - xr[0]) * 0.25)
                self.cursor2.setValue(xr[0] + (xr[1] - xr[0]) * 0.75)
            except Exception:
                pass
            self.update_cursor_info()
        else:
            self.cursor1.setVisible(False)
            self.cursor2.setVisible(False)
            try:
                self.cursor1.setMovable(False)
                self.cursor2.setMovable(False)
            except Exception:
                pass
            self.cursor_info_label.setText("")

    def reset_measurements(self):
        self.data1[:] = 0.0
        self.data2[:] = 0.0
        self.vmax1 = 0.0
        self.vmax2 = 0.0

    def update_sps(self):
        self.sps = int(self.sps_box.value())
        if self.reader:
            self.reader.sps = self.sps
        self.save_config()
        self.update_scale_info_label()

    def on_autoscroll_changed(self, state):
        self.auto_scroll = bool(state)

    # ---------- calibration UI ----------
    def open_calibration(self):
        # passar estado atual para o diálogo (evita inconsistências entre memória e arquivo)
        cfg = {
            "calibration": dict(self.calibration),
            "adc_bits": int(getattr(self, "adc_bits", 10)),
            "probe_10x": bool(getattr(self, "probe_10x", False)),
            "time_bias": float(getattr(self, "time_bias", 5.0)),
            "freq_bias": float(getattr(self, "freq_bias", 5.0))
        }
        dlg = CalibrationDialog(self, cfg)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            vals = dlg.values()
            self.calibration["ch1_0"] = vals["ch1_0"]
            self.calibration["ch1_1023"] = vals["ch1_1023"]
            self.calibration["ch2_0"] = vals["ch2_0"]
            self.calibration["ch2_1023"] = vals["ch2_1023"]
            self.adc_bits = int(vals.get("adc_bits", self.adc_bits))
            self.probe_10x = bool(vals.get("probe_10x", self.probe_10x))
            # carregar bias de tempo/freq vindos do diálogo
            self.time_bias = float(vals.get("time_bias", getattr(self, "time_bias", 5.0)))
            self.freq_bias = float(vals.get("freq_bias", getattr(self, "freq_bias", 5.0)))
            self.save_config()

    # ---------- channel visibility ----------
    def set_channel_visible(self, ch, visible):
        try:
            if ch == 1:
                self.curve1.setVisible(bool(visible))
            elif ch == 2:
                self.curve2.setVisible(bool(visible))
            # persist state
            self.save_config()
        except Exception:
            pass

    # ---------- data handling (recebe raw ADC ints) ----------
    def on_data(self, raw1, raw2):
        if self.paused:
            return
        try:
            # Map ADC->volts using calibration values and adc_bits and probe_10x
            bits = int(self.adc_bits if hasattr(self, "adc_bits") else 10)
            max_adc = max(1, (1 << bits) - 1)
            r1 = int(min(max(raw1, 0), max_adc))
            r2 = int(min(max(raw2, 0), max_adc))
            v1 = float(np.interp(r1, [0, max_adc], [self.calibration.get("ch1_0", 0.0), self.calibration.get("ch1_1023", 5.0)]))
            v2 = float(np.interp(r2, [0, max_adc], [self.calibration.get("ch2_0", 0.0), self.calibration.get("ch2_1023", 5.0)]))
            if getattr(self, "probe_10x", False):
                v1 *= 10.0
                v2 *= 10.0
            # shift buffers and append latest sample (now in volts)
            self.data1 = np.roll(self.data1, -1)
            self.data2 = np.roll(self.data2, -1)
            self.data1[-1] = v1
            self.data2[-1] = v2
            self.vmax1 = max(self.vmax1, float(v1))
            self.vmax2 = max(self.vmax2, float(v2))
        except Exception:
            pass

    # ---------- plot/update ----------
    def update_plot(self):
        try:
            if not self.paused:
                sps = max(1, int(self.sps_box.value()))
                N = self.buffer_len
                # X axis in seconds (from -window .. 0) — aplicar time_bias
                t0 = -float(N - 1) / float(sps)
                x_seconds = np.linspace(t0, 0.0, N) * float(getattr(self, "time_bias", 1.0))
                # scale x to selected time unit for display
                t_unit = self.time_unit_box.currentText()
                tfactor = TIME_FACTORS.get(t_unit, 1.0)
                x_display = x_seconds / tfactor
                # apply voltage unit
                v_unit = self.volt_unit_box.currentText()
                vfactor = VOLT_FACTORS.get(v_unit, 1.0)
                self.curve1.setData(x_display, self.data1 / vfactor)
                self.curve2.setData(x_display, self.data2 / vfactor)
                if self.auto_scroll:
                    # only force X range when auto-scroll enabled
                    self.plot_widget.setXRange(float(x_display[0]), float(x_display[-1]), padding=0.0)
        except Exception:
            pass

        # compute measures
        sps = max(1, int(self.sps_box.value()))
        f1 = compute_frequency(self.data1, sps)
        f2 = compute_frequency(self.data2, sps)
        # aplicar correção de frequência baseada em bias configurado
        try:
            fb = float(getattr(self, "freq_bias", 1.0))
            if fb != 0:
                f1 = f1 / fb
                f2 = f2 / fb
        except Exception:
            pass

        r1 = float(np.sqrt(np.mean(np.square(self.data1)))) if self.data1.size else 0.0
        r2 = float(np.sqrt(np.mean(np.square(self.data2)))) if self.data2.size else 0.0
        p1 = float(np.max(self.data1) - np.min(self.data1)) if self.data1.size else 0.0
        p2 = float(np.max(self.data2) - np.min(self.data2)) if self.data2.size else 0.0

        # display according to units
        v_unit = self.volt_unit_box.currentText()
        t_unit = self.time_unit_box.currentText()
        vfactor = VOLT_FACTORS.get(v_unit, 1.0)
        tfactor = TIME_FACTORS.get(t_unit, 1.0)

        dtype = self.display_combo.currentText()
        if dtype == "Voltage":
            last1 = self.data1[-1] / vfactor
            last2 = self.data2[-1] / vfactor
            self.value_label_ch1.setText(f"CH1: {last1:.3f} {v_unit}")
            self.value_label_ch2.setText(f"CH2: {last2:.3f} {v_unit}")
        elif dtype == "RMS":
            self.value_label_ch1.setText(f"CH1 RMS: {(r1 / vfactor):.3f} {v_unit}")
            self.value_label_ch2.setText(f"CH2 RMS: {(r2 / vfactor):.3f} {v_unit}")
        elif dtype == "PMPO":
            self.value_label_ch1.setText(f"CH1 Vpp: {(p1 / vfactor):.3f} {v_unit}")
            self.value_label_ch2.setText(f"CH2 Vpp: {(p2 / vfactor):.3f} {v_unit}")
        elif dtype == "Frequency":
            self.value_label_ch1.setText(f"CH1 Freq: {f1:.2f} Hz")
            self.value_label_ch2.setText(f"CH2 Freq: {f2:.2f} Hz")

        # scale info under plot: show window in chosen time unit
        window_s = float(self.buffer_len) / max(1, sps) * float(getattr(self, "time_bias", 1.0))
        window_display = window_s / tfactor
        self.update_scale_info_label()

    def fit_view(self):
        """Enquadrar: ajustar X para janela e Y para dados, but keep user able to change afterwards."""
        try:
            sps = max(1, int(self.sps_box.value()))
            N = self.buffer_len
            t0 = -float(N - 1) / float(sps) * float(getattr(self, "time_bias", 1.0))
            t_unit = self.time_unit_box.currentText()
            tfactor = TIME_FACTORS.get(t_unit, 1.0)
            x0 = t0 / tfactor
            x1 = 0.0 / tfactor
            # y auto-fit using visible channels
            yvals = []
            if self.ch1_enable.isChecked():
                yvals.append(self.data1)
            if self.ch2_enable.isChecked():
                yvals.append(self.data2)
            if yvals:
                yy = np.hstack(yvals)
                vmin = np.min(yy)
                vmax = np.max(yy)
                margin = max(1e-6, (vmax - vmin) * 0.1)
                # convert to display unit
                vfactor = VOLT_FACTORS.get(self.volt_unit_box.currentText(), 1.0)
                self.plot_widget.setYRange((vmin - margin) / vfactor, (vmax + margin) / vfactor)
            # set X
            self.plot_widget.setXRange(float(x0), float(x1))
            # persist view
            self.save_config()
        except Exception:
            pass

    def on_view_range_changed(self, vb, ranges):
        # user changed view: disable auto-scroll so GUI doesn't force XRange
        try:
            if self.auto_scroll:
                # if user interaction occurred, turn off auto_scroll to allow manual zoom/pan
                self.chk_autoscroll.setChecked(False)
                self.auto_scroll = False
                self.save_config()
        except Exception:
            pass

    def auto_scale_y(self):
        """Ajusta vertical (Y) para visualizar melhor os canais visíveis."""
        try:
            # combine visible channel data
            vals = []
            if getattr(self, "ch1_enable", None) and self.ch1_enable.isChecked():
                vals.append(self.data1)
            if getattr(self, "ch2_enable", None) and self.ch2_enable.isChecked():
                vals.append(self.data2)
            if not vals:
                return
            yy = np.hstack(vals)
            vmax = float(np.max(yy))
            vmin = float(np.min(yy))
            if not np.isfinite(vmax) or not np.isfinite(vmin):
                return
            # add margin
            span = max(1e-12, vmax - vmin)
            margin = span * 0.15 if span > 0 else max(abs(vmax), abs(vmin), 1.0) * 0.1
            top = vmax + margin
            bottom = vmin - margin
            # convert to display unit
            vfactor = VOLT_FACTORS.get(self.volt_unit_box.currentText() if hasattr(self, "volt_unit_box") else self.volt_unit, 1.0)
            self.plot_widget.setYRange(bottom / vfactor, top / vfactor)
            # persist view
            self.save_config()
        except Exception:
            pass

    def update_cursor_info(self):
        try:
            if not self.cursor1.isVisible() or not self.cursor2.isVisible():
                self.cursor_info_label.setText("")
                return
            p1 = float(self.cursor1.value())
            p2 = float(self.cursor2.value())
            # current X axis is displayed in chosen time unit; convert to seconds (incl. time bias)
            t_unit = self.time_unit_box.currentText()
            tfactor = TIME_FACTORS.get(t_unit, 1.0)
            # displayed X already includes the configured time_bias.
            # To show Δt consistent with the axis, convert display units back to seconds:
            # delta_seconds_biased = delta_display * tfactor
            delta_display = abs(p2 - p1)
            delta_seconds = delta_display * tfactor
            # format nicely
            if delta_seconds >= 1.0:
                txt = f"Δt = {delta_seconds:.6f} s"
            elif delta_seconds >= 1e-3:
                txt = f"Δt = {delta_seconds*1e3:.6f} ms"
            elif delta_seconds >= 1e-6:
                txt = f"Δt = {delta_seconds*1e6:.6f} µs"
            else:
                txt = f"Δt = {delta_seconds:.9f} s"
            self.cursor_info_label.setText(txt)
        except Exception:
            pass

    def apply_saved_view_ranges(self):
        """Aplicar ranges X/Y salvos no config (se existirem)."""
        try:
            if not os.path.exists(CONFIG_FILE):
                return
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            xr = cfg.get("xrange", None)
            yr = cfg.get("yrange", None)
            ch1_vis = cfg.get("ch1_visible", True)
            ch2_vis = cfg.get("ch2_visible", True)
            # apply channel visibility
            if hasattr(self, "ch1_enable"):
                self.ch1_enable.setChecked(bool(ch1_vis))
                self.curve1.setVisible(bool(ch1_vis))
            if hasattr(self, "ch2_enable"):
                self.ch2_enable.setChecked(bool(ch2_vis))
                self.curve2.setVisible(bool(ch2_vis))
            # apply ranges if present
            if xr and isinstance(xr, (list, tuple)) and len(xr) == 2:
                try:
                    self.plot_widget.setXRange(float(xr[0]), float(xr[1]))
                except Exception:
                    pass
            if yr and isinstance(yr, (list, tuple)) and len(yr) == 2:
                try:
                    self.plot_widget.setYRange(float(yr[0]), float(yr[1]))
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, ev):
        """Salvar estado (view ranges e visibilidade de canais) ao fechar."""
        try:
            vb = self.plot_widget.getViewBox()
            xr = vb.viewRange()[0]
            yr = vb.viewRange()[1]
            # persist into config then call original save_config
            try:
                if os.path.exists(CONFIG_FILE):
                    with open(CONFIG_FILE, 'r') as f:
                        cfg = json.load(f)
                else:
                    cfg = {}
                cfg.update({
                    "xrange": [float(xr[0]), float(xr[1])],
                    "yrange": [float(yr[0]), float(yr[1])],
                    "ch1_visible": bool(self.ch1_enable.isChecked()) if hasattr(self, "ch1_enable") else True,
                    "ch2_visible": bool(self.ch2_enable.isChecked()) if hasattr(self, "ch2_enable") else True,
                    "calibration": self.calibration,
                    "adc_bits": int(self.adc_bits),
                    "probe_10x": bool(self.probe_10x),
                    "time_bias": float(getattr(self, "time_bias", 5.0)),
                    "freq_bias": float(getattr(self, "freq_bias", 5.0))
                })
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(cfg, f, indent=4)
            except Exception:
                pass
        except Exception:
            pass
        # call super to close
        try:
            super().closeEvent(ev)
        except Exception:
            ev.accept()

    # ---------- CSV ----------
    def save_csv(self):
        try:
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Salvar CSV", "", "CSV Files (*.csv)")
            if fname:
                import csv
                with open(fname, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["time_s", "CH1", "CH2"])
                    sps = max(1, int(self.sps_box.value()))
                    N = self.buffer_len
                    t0 = -float(N - 1) / float(sps)
                    times = np.linspace(t0, 0.0, N) * float(getattr(self, "time_bias", 1.0))
                    for t, v1, v2 in zip(times, self.data1, self.data2):
                        writer.writerow([f"{t:.6f}", f"{v1:.6f}", f"{v2:.6f}"])
        except Exception:
            pass

    # ---------- Save PNG ----------
    def save_plot_png(self):
        try:
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Salvar PNG", "", "PNG Files (*.png)")
            if not fname:
                return
            # use pyqtgraph exporter
            try:
                from pyqtgraph.exporters import ImageExporter
                exporter = ImageExporter(self.plot_widget.plotItem)
                # try to set reasonable width
                exporter.parameters()['width'] = 1600
                exporter.export(fname)
            except Exception:
                # fallback: grab screenshot of widget
                pix = self.plot_widget.grab()
                pix.save(fname, "PNG")
        except Exception:
            pass

    # ---------- config ----------
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    cfg = json.load(f)
                self.scale_min = cfg.get("scale_min", self.scale_min)
                self.scale_max = cfg.get("scale_max", self.scale_max)
                self.sps = cfg.get("sps", self.sps)
                self.auto_scroll = cfg.get("auto_scroll", self.auto_scroll)
                self.autoscale_y = cfg.get("autoscale_y", self.autoscale_y)
                self.volt_unit = cfg.get("volt_unit", self.volt_unit)
                self.time_unit = cfg.get("time_unit", self.time_unit)
                self.calibration.update(cfg.get("calibration", {}))
                self.adc_bits = int(cfg.get("adc_bits", self.adc_bits))
                self.probe_10x = bool(cfg.get("probe_10x", self.probe_10x))
                # carregar biases de tempo/frequência se existirem
                self.time_bias = float(cfg.get("time_bias", getattr(self, "time_bias", 5.0)))
                self.freq_bias = float(cfg.get("freq_bias", getattr(self, "freq_bias", 5.0)))
            except Exception:
                pass
        # ensure UI boxes reflect loaded values after init
        try:
            if hasattr(self, "sps_box"):
                self.sps_box.setValue(self.sps)
            if hasattr(self, "chk_autoscroll"):
                self.chk_autoscroll.setChecked(self.auto_scroll)
            if hasattr(self, "chk_autoscale"):
                self.chk_autoscale.setChecked(self.autoscale_y)
            if hasattr(self, "volt_unit_box"):
                if self.volt_unit in VOLT_UNITS:
                    self.volt_unit_box.setCurrentText(self.volt_unit)
            if hasattr(self, "time_unit_box"):
                if self.time_unit in TIME_UNITS:
                    self.time_unit_box.setCurrentText(self.time_unit)
        except Exception:
            pass

    def save_config(self):
        try:
            cfg = {}
            # try to preserve existing config entries
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, 'r') as f:
                        cfg = json.load(f)
                except Exception:
                    cfg = {}
            cfg.update({
                "scale_min": self.scale_min,
                "scale_max": self.scale_max,
                "sps": int(self.sps_box.value()) if hasattr(self, "sps_box") else int(self.sps),
                "auto_scroll": bool(self.auto_scroll),
                "autoscale_y": bool(self.autoscale_y),
                "volt_unit": self.volt_unit_box.currentText() if hasattr(self, "volt_unit_box") else self.volt_unit,
                "time_unit": self.time_unit_box.currentText() if hasattr(self, "time_unit_box") else self.time_unit,
                "ch1_visible": bool(self.ch1_enable.isChecked()) if hasattr(self, "ch1_enable") else True,
                "ch2_visible": bool(self.ch2_enable.isChecked()) if hasattr(self, "ch2_enable") else True,
                "calibration": self.calibration,
                "adc_bits": int(self.adc_bits),
                "probe_10x": bool(self.probe_10x),
                "time_bias": float(getattr(self, "time_bias", 5.0)),
                "freq_bias": float(getattr(self, "freq_bias", 5.0))
            })
            # also save current view ranges
            try:
                vb = self.plot_widget.getViewBox()
                xr = vb.viewRange()[0]
                yr = vb.viewRange()[1]
                cfg["xrange"] = [float(xr[0]), float(xr[1])]
                cfg["yrange"] = [float(yr[0]), float(yr[1])]
            except Exception:
                pass
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception:
            pass

    def update_scale_info_label(self):
        # update bottom info and unit labels
        try:
            sps = int(self.sps_box.value()) if hasattr(self, "sps_box") else int(self.sps)
            v_unit = self.volt_unit_box.currentText() if hasattr(self, "volt_unit_box") else self.volt_unit
            t_unit = self.time_unit_box.currentText() if hasattr(self, "time_unit_box") else self.time_unit
            dtype = self.display_combo.currentText() if hasattr(self, "display_combo") else "Voltage"
            window_s = float(self.buffer_len) / max(1, sps) * float(getattr(self, "time_bias", 1.0))
            # show window in chosen time unit
            tfactor = TIME_FACTORS.get(t_unit, 1.0)
            window_display = window_s / tfactor
            self.scale_info_label.setText(f"SPS: {sps} | Janela: {window_display:.3g} {t_unit} | Exibindo: {dtype} | V unit: {v_unit}")
            # persist minimal config
            self.save_config()
            # update axis labels to match units
            try:
                self.plot_widget.setLabel('left', f'Tensão ({v_unit})')
                self.plot_widget.setLabel('bottom', f'Tempo ({t_unit})')
            except Exception:
                pass
        except Exception:
            pass


def apply_dark_palette(app):
    """Aplica paleta escura simples sem dependências externas."""
    try:
        p = QtGui.QPalette()
        dark = QtGui.QColor(45, 45, 45)
        mid = QtGui.QColor(60, 60, 60)
        light = QtGui.QColor(200, 200, 200)
        disabled = QtGui.QColor(127, 127, 127)
        p.setColor(QtGui.QPalette.Window, dark)
        p.setColor(QtGui.QPalette.WindowText, light)
        p.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        p.setColor(QtGui.QPalette.AlternateBase, mid)
        p.setColor(QtGui.QPalette.ToolTipBase, light)
        p.setColor(QtGui.QPalette.ToolTipText, light)
        p.setColor(QtGui.QPalette.Text, light)
        p.setColor(QtGui.QPalette.Button, mid)
        p.setColor(QtGui.QPalette.ButtonText, light)
        p.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        p.setColor(QtGui.QPalette.Highlight, QtGui.QColor(40, 80, 160))
        p.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled)
        app.setPalette(p)
    except Exception:
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # preferir tema pronto (qdarkstyle) se disponível, senão usar paleta interna
    try:
        import qdarkstyle  # pip install qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except Exception:
        apply_dark_palette(app)
    win = Oscilloscope()
    win.show()
    sys.exit(app.exec_())
