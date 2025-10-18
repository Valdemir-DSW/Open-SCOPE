import sys
import struct
import json
import numpy as np
import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtWidgets, QtGui 
import pyqtgraph as pg
import os

CONFIG_FILE = "config.json"

class SerialReader(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    connection_lost = QtCore.pyqtSignal()

    def __init__(self, port, baudrate=115200, sps=1000, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.ser = None
        self.sps = sps

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        except Exception:
            self.connection_lost.emit()
            return

        self.running = True
        buffer = bytearray()
        # delay in ms between loop iterations (approx)
        delay = max(1, int(1000.0 / max(1, self.sps)))

        while self.running:
            try:
                data = self.ser.read(1024)
                if data:
                    buffer.extend(data)
                    while len(buffer) >= 4:
                        raw1, raw2 = struct.unpack('<HH', buffer[:4])
                        buffer = buffer[4:]
                        ch1 = raw1 / 1023.0 * 5.0
                        ch2 = raw2 / 1023.0 * 5.0
                        self.data_ready.emit(np.array([ch1]), np.array([ch2]))
            except Exception:
                self.connection_lost.emit()
                break
            QtCore.QThread.msleep(delay)

        if self.ser and self.ser.is_open:
            self.ser.close()

    def stop(self):
        self.running = False
        self.wait(2000)

class Oscilloscope(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Osciloscópio - By Valdemir")
        # Ativar ícone se existir (procura icon.png ou icon.ico ao lado do script)
        base = os.path.dirname(__file__)
        icon_path = os.path.join(base, "icon.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(base, "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.resize(1400, 720)

        # estado básico
        self.reader = None
        self.paused = False

        # escalas e parâmetros iniciais
        self.scale_min = -5.0
        self.scale_max = 5.0
        self.volts_div = 1.0     # V/div internal
        self.time_div_s = 0.01   # seconds per division
        self.sps = 1000
        self.channels_expected = 2
        self.autoscale_y = False

        # buffers e medidores
        self.buffer_len = 1000
        self.data1 = np.zeros(self.buffer_len)
        self.data2 = np.zeros(self.buffer_len)
        self.vmax1 = 0.0
        self.vmax2 = 0.0

        # bloqueios para evitar loops de sinais
        self.block_volt_signals = False
        self.block_time_signals = False

        self.load_config()

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_l = QtWidgets.QHBoxLayout(central)
        sidebar = QtWidgets.QVBoxLayout()
        main_l.addLayout(sidebar, 0)
        sidebar.setSpacing(8)

        # porta serial
        sidebar.addWidget(QtWidgets.QLabel("Porta serial:"))
        self.port_box = QtWidgets.QComboBox()
        sidebar.addWidget(self.port_box)
        self.refresh_ports()
        self.btn_refresh = QtWidgets.QPushButton("Atualizar portas")
        sidebar.addWidget(self.btn_refresh)
        self.btn_connect = QtWidgets.QPushButton("Conectar")
        self.btn_disconnect = QtWidgets.QPushButton("Desconectar")
        self.btn_disconnect.setEnabled(False)
        sidebar.addWidget(self.btn_connect)
        sidebar.addWidget(self.btn_disconnect)

        # leitura escala
        sidebar.addWidget(QtWidgets.QLabel("Faixa de leitura (±):"))
        self.min_box = QtWidgets.QDoubleSpinBox()
        self.min_box.setRange(-10000, 0)
        self.max_box = QtWidgets.QDoubleSpinBox()
        self.max_box.setRange(0, 10000)
        self.min_box.setValue(self.scale_min)
        self.max_box.setValue(self.scale_max)
        sidebar.addWidget(self.min_box)
        sidebar.addWidget(self.max_box)

        # volts/div controls
        sidebar.addWidget(QtWidgets.QLabel("Volts por divisão:"))
        h_volt = QtWidgets.QHBoxLayout()
        self.volts_display = QtWidgets.QDoubleSpinBox()
        self.volts_display.setRange(1e-12, 1e12)
        self.volts_display.setDecimals(6)
        self.volts_unit_box = QtWidgets.QComboBox()
        # unit list: pico, nano, micro, milli, normal, kilo, mega, tera
        self.volts_unit_box.addItems(["pV/div", "nV/div", "µV/div", "mV/div", "V/div", "kV/div", "MV/div", "TV/div"])
        h_volt.addWidget(self.volts_display)
        h_volt.addWidget(self.volts_unit_box)
        sidebar.addLayout(h_volt)
        self.volts_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # slider stores value as integer representing volts * 1e6 (microvolts)
        # QSlider espera inteiros 32-bit; usar um máximo seguro (2_000_000_000 = 2000 V em µV)
        self.volts_slider.setRange(1, 2_000_000_000)
        sidebar.addWidget(self.volts_slider)

        # time/div controls
        sidebar.addWidget(QtWidgets.QLabel("Tempo por divisão:"))
        h_time = QtWidgets.QHBoxLayout()
        self.time_display = QtWidgets.QDoubleSpinBox()
        self.time_display.setRange(1e-12, 1e12)
        self.time_display.setDecimals(6)
        self.time_unit_box = QtWidgets.QComboBox()
        self.time_unit_box.addItems(["ps/div", "ns/div", "µs/div", "ms/div", "s/div", "ks/div", "Ms/div", "Gs/div", "Ts/div"])
        h_time.addWidget(self.time_display)
        h_time.addWidget(self.time_unit_box)
        sidebar.addLayout(h_time)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # slider stores value as integer representing microseconds (µs)
        self.time_slider.setRange(1, 10**9)  # up to 1000 seconds approx, clamped later
        sidebar.addWidget(self.time_slider)

        # samples per second
        sidebar.addWidget(QtWidgets.QLabel("Amostras por segundo (SPS máximo 20000):"))
        self.sps_box = QtWidgets.QSpinBox()
        self.sps_box.setRange(1, 20000)
        self.sps_box.setValue(self.sps)
        sidebar.addWidget(self.sps_box)

        # canais esperado e autoscale
        sidebar.addWidget(QtWidgets.QLabel("Número de canais esperado:"))
        self.channels_box = QtWidgets.QSpinBox()
        self.channels_box.setRange(1, 8)
        self.channels_box.setValue(self.channels_expected)
        sidebar.addWidget(self.channels_box)
        self.autoscale_y_box = QtWidgets.QCheckBox("Auto-scale Y ao conectar")
        self.autoscale_y_box.setChecked(self.autoscale_y)
        sidebar.addWidget(self.autoscale_y_box)

        # botões utilitários
        self.btn_pause = QtWidgets.QPushButton("Pausar gráfico")
        sidebar.addWidget(self.btn_pause)
        self.btn_save_csv = QtWidgets.QPushButton("Salvar CSV")
        sidebar.addWidget(self.btn_save_csv)
        self.btn_reset = QtWidgets.QPushButton("Zerar medição")
        sidebar.addWidget(self.btn_reset)

        # medidores
        self.freq_label = QtWidgets.QLabel("Freq CH1: 0Hz | CH2: 0Hz")
        self.rms_label = QtWidgets.QLabel("RMS CH1: 0 | CH2: 0")
        self.pmpo_label = QtWidgets.QLabel("PMPO CH1: 0 | CH2: 0")
        self.vmax_label = QtWidgets.QLabel("Vmax CH1: 0 | CH2: 0")
        sidebar.addWidget(self.freq_label)
        sidebar.addWidget(self.rms_label)
        sidebar.addWidget(self.pmpo_label)
        sidebar.addWidget(self.vmax_label)

        self.status_label = QtWidgets.QLabel("Desconectado")
        sidebar.addWidget(self.status_label)
        sidebar.addStretch()

        # plot area
        self.plot_widget = pg.PlotWidget()
        main_l.addWidget(self.plot_widget, 1)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.curve1 = self.plot_widget.plot(pen='y', name="Canal 1")
        self.curve2 = self.plot_widget.plot(pen='c', name="Canal 2")

        # timer para atualizar plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)

        # conexões dos widgets
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.connect_serial)
        self.btn_disconnect.clicked.connect(self.disconnect_serial)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_save_csv.clicked.connect(self.save_csv)
        self.btn_reset.clicked.connect(self.reset_measurements)
        self.min_box.valueChanged.connect(self.update_scale)
        self.max_box.valueChanged.connect(self.update_scale)

        # volts handlers
        self.volts_display.valueChanged.connect(self.on_volts_display_changed)
        self.volts_slider.valueChanged.connect(self.on_volts_slider_changed)
        self.volts_unit_box.currentIndexChanged.connect(self.on_volts_unit_changed)

        # time handlers
        self.time_display.valueChanged.connect(self.on_time_display_changed)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.time_unit_box.currentIndexChanged.connect(self.on_time_unit_changed)

        # sps and config
        self.sps_box.valueChanged.connect(self.update_sps)
        self.channels_box.valueChanged.connect(self.on_channels_changed)
        self.autoscale_y_box.stateChanged.connect(self.on_autoscale_changed)

        # conectar sinal de alteração do viewbox (x e y) para sincronizar controles
        vb = self.plot_widget.getViewBox()
        try:
            vb.sigRangeChanged.connect(self.on_view_range_changed)
        except Exception:
            # fallback: não crítico
            pass

        # inicializar widgets a partir das configurações carregadas
        self.set_volts_div(self.volts_div, update_widgets=True)
        self.set_time_div(self.time_div_s, update_widgets=True)

    # ---------- UI helpers ----------
    def refresh_ports(self):
        self.port_box.clear()
        for p in serial.tools.list_ports.comports():
            self.port_box.addItem(p.device)

    def connect_serial(self):
        port = self.port_box.currentText()
        if not port:
            self.status_label.setText("Nenhuma porta selecionada")
            return
        self.reader = SerialReader(port, sps=self.sps_box.value())
        self.reader.data_ready.connect(self.on_data)
        self.reader.connection_lost.connect(self.on_disconnect)
        self.reader.start()
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self.status_label.setText(f"Conectado a {port}")
        # aplicar autoscale Y se selecionado
        if self.autoscale_y_box.isChecked():
            self.auto_scale_y()

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
        self.btn_pause.setText("Retomar gráfico" if self.paused else "Pausar gráfico")

    def reset_measurements(self):
        self.data1[:] = 0
        self.data2[:] = 0
        self.vmax1 = 0.0
        self.vmax2 = 0.0

    def update_scale(self):
        self.scale_min = float(self.min_box.value())
        self.scale_max = float(self.max_box.value())
        self.save_config()

    def update_sps(self):
        self.sps = int(self.sps_box.value())
        if self.reader:
            self.reader.sps = self.sps
        self.save_config()

    def on_channels_changed(self, val):
        self.channels_expected = int(val)
        self.save_config()

    def on_autoscale_changed(self, state):
        self.autoscale_y = bool(state)
        self.save_config()

    # ---------- Volts/div handling ----------
    def volts_unit_factor(self, unit_text):
        # returns multiplier to convert display units to volts
        m = {
            "pV/div": 1e-12, "nV/div": 1e-9, "µV/div": 1e-6, "mV/div": 1e-3,
            "V/div": 1.0, "kV/div": 1e3, "MV/div": 1e6, "TV/div": 1e12
        }
        return m.get(unit_text, 1.0)

    def on_volts_display_changed(self, val):
        if self.block_volt_signals:
            return
        unit = self.volts_unit_box.currentText()
        volts = float(val) * self.volts_unit_factor(unit)
        self.set_volts_div(volts, update_widgets=True)
        self.save_config()

    def on_volts_slider_changed(self, sval):
        if self.block_volt_signals:
            return
        # slider stores microvolts (1 = 1 µV)
        volts = float(sval) / 1e6
        self.set_volts_div(volts, update_widgets=True)
        self.save_config()

    def on_volts_unit_changed(self, idx):
        # update display formatting without changing internal volts_div
        self.set_volts_div(self.volts_div, update_widgets=True)

    def set_volts_div(self, volts, update_widgets=False):
        self.volts_div = max(1e-12, float(volts))
        y_min = -5 * self.volts_div
        y_max = 5 * self.volts_div
        try:
            self.plot_widget.setYRange(y_min, y_max)
        except Exception:
            pass
        if update_widgets:
            try:
                self.block_volt_signals = True
                # map volts to slider (microvolts)
                slider_val = int(round(self.volts_div * 1e6))
                slider_val = max(1, min(self.volts_slider.maximum(), slider_val))
                self.volts_slider.setValue(slider_val)
                unit = self.volts_unit_box.currentText()
                factor = 1.0 / self.volts_unit_factor(unit)  # display = volts * factor
                display_val = self.volts_div * factor
                # adjust decimals for small units
                if unit in ("pV/div", "nV/div", "µV/div"):
                    self.volts_display.setDecimals(0)
                elif unit == "mV/div":
                    self.volts_display.setDecimals(3)
                else:
                    self.volts_display.setDecimals(6)
                self.volts_display.setValue(display_val)
            finally:
                self.block_volt_signals = False

    # ---------- Time/div handling ----------
    def time_unit_factor(self, unit_text):
        m = {
            "ps/div": 1e-12, "ns/div": 1e-9, "µs/div": 1e-6, "ms/div": 1e-3,
            "s/div": 1.0, "ks/div": 1e3, "Ms/div": 1e6, "Gs/div": 1e9, "Ts/div": 1e12
        }
        return m.get(unit_text, 1.0)

    def on_time_display_changed(self, val):
        if self.block_time_signals:
            return
        unit = self.time_unit_box.currentText()
        secs = float(val) * self.time_unit_factor(unit)
        self.set_time_div(secs, update_widgets=True)
        self.save_config()

    def on_time_slider_changed(self, sval):
        if self.block_time_signals:
            return
        # slider stores microseconds
        secs = float(sval) / 1e6
        self.set_time_div(secs, update_widgets=True)
        self.save_config()

    def on_time_unit_changed(self, idx):
        self.set_time_div(self.time_div_s, update_widgets=True)

    def set_time_div(self, seconds, update_widgets=False):
        self.time_div_s = max(1e-12, float(seconds))
        total_time = 10.0 * self.time_div_s
        try:
            self.plot_widget.setXRange(0, total_time)
        except Exception:
            pass
        if update_widgets:
            try:
                self.block_time_signals = True
                slider_val = int(round(self.time_div_s * 1e6))
                slider_val = max(1, min(self.time_slider.maximum(), slider_val))
                self.time_slider.setValue(slider_val)
                unit = self.time_unit_box.currentText()
                factor = 1.0 / self.time_unit_factor(unit)
                display_val = self.time_div_s * factor
                # decimals
                if unit in ("ps/div", "ns/div", "µs/div"):
                    self.time_display.setDecimals(0)
                else:
                    self.time_display.setDecimals(6)
                self.time_display.setValue(display_val)
            finally:
                self.block_time_signals = False

    # ---------- View range changes (sync sliders/fields on zoom/pan) ----------
    def on_view_range_changed(self, vb, ranges):
        try:
            (x_min, x_max), (y_min, y_max) = ranges
        except Exception:
            vr = vb.viewRange()
            x_min, x_max = vr[0]
            y_min, y_max = vr[1]

        # update volts/div from Y-range (use +/-5 divisions)
        try:
            self.block_volt_signals = True
            volts = max(abs(y_min), abs(y_max)) / 5.0
            self.volts_div = max(1e-12, volts)
            slider_val = int(round(self.volts_div * 1e6))
            slider_val = max(1, min(self.volts_slider.maximum(), slider_val))
            self.volts_slider.setValue(slider_val)
            unit = self.volts_unit_box.currentText()
            factor = 1.0 / self.volts_unit_factor(unit)
            self.volts_display.setValue(self.volts_div * factor)
        finally:
            self.block_volt_signals = False

        # update time/div from X-range (10 divisions)
        try:
            self.block_time_signals = True
            width = abs(x_max - x_min)
            if width <= 0:
                width = 10.0 * self.time_div_s
            secs = width / 10.0
            self.time_div_s = max(1e-12, secs)
            slider_val = int(round(self.time_div_s * 1e6))
            slider_val = max(1, min(self.time_slider.maximum(), slider_val))
            self.time_slider.setValue(slider_val)
            unit = self.time_unit_box.currentText()
            factor = 1.0 / self.time_unit_factor(unit)
            self.time_display.setValue(self.time_div_s * factor)
        finally:
            self.block_time_signals = False

        self.save_config()

    # ---------- Data handling ----------
    def on_data(self, ch1, ch2):
        if self.paused:
            return
        # ensure buffers length (optionally adapt to time_div and sps)
        try:
            # roll and append new values
            self.data1 = np.roll(self.data1, -1)
            self.data2 = np.roll(self.data2, -1)
            v1 = np.interp(ch1, [0, 5], [self.scale_min, self.scale_max])[0]
            v2 = np.interp(ch2, [0, 5], [self.scale_min, self.scale_max])[0]
            self.data1[-1] = v1
            self.data2[-1] = v2
            self.vmax1 = max(self.vmax1, float(v1))
            self.vmax2 = max(self.vmax2, float(v2))
        except Exception:
            pass

    # ---------- Plot updating and measurements ----------
    def update_plot(self):
        try:
            if not self.paused:
                self.curve1.setData(self.data1)
                self.curve2.setData(self.data2)
        except Exception:
            pass
        # update measurements
        f1 = self.calc_freq(self.data1)
        f2 = self.calc_freq(self.data2)
        r1 = self.calc_rms(self.data1)
        r2 = self.calc_rms(self.data2)
        p1 = self.calc_pmpo(self.data1)
        p2 = self.calc_pmpo(self.data2)
        self.freq_label.setText(f"Freq CH1: {f1:.2f}Hz | CH2: {f2:.2f}Hz")
        self.rms_label.setText(f"RMS CH1: {r1:.3f} | CH2: {r2:.3f}")
        self.pmpo_label.setText(f"PMPO CH1: {p1:.3f} | CH2: {p2:.3f}")
        self.vmax_label.setText(f"Vmax CH1: {self.vmax1:.3f} | CH2: {self.vmax2:.3f}")

    def calc_rms(self, data):
        try:
            return float(np.sqrt(np.mean(np.square(data))))
        except Exception:
            return 0.0

    def calc_pmpo(self, data):
        try:
            return float(np.max(data) - np.min(data))
        except Exception:
            return 0.0

    def calc_freq(self, data):
        try:
            sps = max(1, int(self.sps))
            if len(data) < 3:
                return 0.0
            mean = np.mean(data)
            signs = np.sign(data - mean)
            signs[signs == 0] = -1
            crossings = np.where((signs[:-1] < 0) & (signs[1:] > 0))[0]
            count = len(crossings)
            if count > 0:
                total_time = len(data) / float(sps)
                return float(count) / total_time
            return 0.0
        except Exception:
            return 0.0

    # ---------- CSV ----------
    def save_csv(self):
        try:
            import csv
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Salvar CSV", "", "CSV Files (*.csv)")
            if fname:
                with open(fname, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["CH1", "CH2"])
                    for v1, v2 in zip(self.data1, self.data2):
                        writer.writerow([v1, v2])
        except Exception:
            pass

    # ---------- Autoscale ----------
    def auto_scale_y(self):
        try:
            maxv = max(np.max(np.abs(self.data1)), np.max(np.abs(self.data2)), 1e-6)
            # choose volts_div so that 5 divisions cover maxv
            volts_div = maxv / 5.0
            self.set_volts_div(volts_div, update_widgets=True)
        except Exception:
            pass

    # ---------- Config file ----------
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    cfg = json.load(f)
                self.scale_min = cfg.get("scale_min", self.scale_min)
                self.scale_max = cfg.get("scale_max", self.scale_max)
                self.volts_div = cfg.get("volts_div", self.volts_div)
                # compatibility ms_div in ms -> seconds
                ms_div = cfg.get("ms_div", None)
                if ms_div is not None:
                    self.time_div_s = float(ms_div) / 1000.0
                self.time_div_s = cfg.get("time_div_s", self.time_div_s)
                self.sps = cfg.get("sps", self.sps)
                self.channels_expected = cfg.get("channels_expected", self.channels_expected)
                self.autoscale_y = cfg.get("autoscale_y", self.autoscale_y)
            except Exception:
                pass

    def save_config(self):
        try:
            cfg = {
                "scale_min": self.scale_min,
                "scale_max": self.scale_max,
                "volts_div": self.volts_div,
                "time_div_s": self.time_div_s,
                "ms_div": int(self.time_div_s * 1000),
                "sps": self.sps,
                "channels_expected": self.channels_expected,
                "autoscale_y": self.autoscale_y
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception:
            pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Oscilloscope()
    win.show()
    sys.exit(app.exec_())
