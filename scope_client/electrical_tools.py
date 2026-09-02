from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .measurements import analyze, fmt
from .protocol import Capture


MAX_ANALYSIS_SAMPLES = 200_000


def _wrap_phase(degrees: float) -> float:
    return (float(degrees) + 180.0) % 360.0 - 180.0


def _analysis_slice(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= MAX_ANALYSIS_SAMPLES:
        return values
    return values[-MAX_ANALYSIS_SAMPLES:]


def _dominant_frequency(values: np.ndarray, sample_rate: float) -> float:
    values = _analysis_slice(values)
    if values.size < 16 or sample_rate <= 0:
        return math.nan
    centered = values - float(np.mean(values))
    if float(np.ptp(centered)) <= 1e-12:
        return math.nan

    # A Hann window keeps mains-frequency estimates stable when the visible
    # record does not contain an integer number of cycles.
    windowed = centered * np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    if spectrum.size <= 1:
        return math.nan
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    if index <= 0:
        return math.nan

    # Parabolic peak interpolation reduces the coarse-bin error without
    # requiring zero-padding or a much larger FFT.
    refined = float(index)
    if 1 <= index < spectrum.size - 1:
        a, b, c = spectrum[index - 1:index + 2]
        denom = a - 2.0 * b + c
        if abs(float(denom)) > 1e-18:
            refined += 0.5 * float(a - c) / float(denom)
    return refined * float(sample_rate) / float(values.size)


def _frequency(values: np.ndarray, sample_rate: float) -> float:
    measured = analyze(_analysis_slice(values), sample_rate).frequency
    if np.isfinite(measured) and measured > 0.0:
        return float(measured)
    return _dominant_frequency(values, sample_rate)


def _phasor(values: np.ndarray, sample_rate: float, frequency: float) -> complex:
    values = _analysis_slice(values)
    if values.size < 4 or sample_rate <= 0 or not np.isfinite(frequency) or frequency <= 0:
        return complex(math.nan, math.nan)
    centered = values - float(np.mean(values))
    n = np.arange(centered.size, dtype=np.float64)
    angle = -2.0 * np.pi * float(frequency) * n / float(sample_rate)
    return complex(np.dot(centered, np.exp(1j * angle)))


def channel_values(capture: Capture, configs: Sequence[object]) -> List[np.ndarray]:
    values: List[np.ndarray] = []
    for channel in range(capture.channel_count):
        values.append(
            configs[channel].raw_to_volts(
                capture.raw[:, channel], capture.header.adc_bits, apply_ac=False
            )
        )
    return values


class ElectricalNetworkDialog(QtWidgets.QDialog):
    """General 1–4 channel mains/network inspection without assuming probe topology."""

    def __init__(self, capture: Capture, configs: Sequence[object], parent=None) -> None:
        super().__init__(parent)
        self.capture = capture
        self.configs = configs
        self.setWindowTitle("Electrical Network Analyzer")
        self.resize(900, 430)

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Analyzes the calibrated waveform already captured by OpenScope. "
            "It supports up to four channels and does not assume that a channel is safe for direct mains connection."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Phase reference"))
        self.reference = QtWidgets.QComboBox()
        for i in range(capture.channel_count):
            self.reference.addItem(configs[i].name or f"CH{i + 1}", i)
        controls.addWidget(self.reference)
        controls.addStretch(1)
        layout.addLayout(controls)

        headers = ["Channel", "RMS", "Peak", "Pk-Pk", "DC", "Crest", "Frequency", "Phase vs ref"]
        self.table = QtWidgets.QTableWidget(capture.channel_count, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.reference.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        values = channel_values(self.capture, self.configs)
        ref_index = max(0, min(len(values) - 1, int(self.reference.currentData() or 0)))
        ref_frequency = _frequency(values[ref_index], self.capture.sample_rate)
        if not np.isfinite(ref_frequency) or ref_frequency <= 0:
            frequencies = [_frequency(v, self.capture.sample_rate) for v in values]
            valid = [f for f in frequencies if np.isfinite(f) and f > 0]
            ref_frequency = float(np.median(valid)) if valid else math.nan
        ref_phasor = _phasor(values[ref_index], self.capture.sample_rate, ref_frequency)
        ref_angle = math.degrees(math.atan2(ref_phasor.imag, ref_phasor.real)) if np.isfinite(ref_phasor.real) else math.nan

        valid_freqs = []
        for ch, waveform in enumerate(values):
            m = analyze(_analysis_slice(waveform), self.capture.sample_rate)
            peak = max(abs(m.minimum), abs(m.maximum)) if np.isfinite(m.minimum) and np.isfinite(m.maximum) else math.nan
            crest = peak / m.rms if np.isfinite(peak) and np.isfinite(m.rms) and m.rms > 1e-15 else math.nan
            frequency = _frequency(waveform, self.capture.sample_rate)
            if np.isfinite(frequency):
                valid_freqs.append(frequency)
            ph = _phasor(waveform, self.capture.sample_rate, ref_frequency)
            angle = math.degrees(math.atan2(ph.imag, ph.real)) if np.isfinite(ph.real) else math.nan
            phase = _wrap_phase(angle - ref_angle) if np.isfinite(angle) and np.isfinite(ref_angle) else math.nan

            row = [
                self.configs[ch].name or f"CH{ch + 1}",
                fmt(m.rms, "V"),
                fmt(peak, "V"),
                fmt(m.vpp, "V"),
                fmt(m.mean, "V"),
                f"{crest:.3f}" if np.isfinite(crest) else "—",
                fmt(frequency, "Hz"),
                f"{phase:+.1f}°" if np.isfinite(phase) else "—",
            ]
            for col, text in enumerate(row):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(ch, col, item)

        if valid_freqs:
            median = float(np.median(valid_freqs))
            spread = float(max(valid_freqs) - min(valid_freqs)) if len(valid_freqs) > 1 else 0.0
            self.summary.setText(
                f"Common frequency ≈ {median:.3f} Hz · channel spread {spread:.3f} Hz · "
                f"phase reference: {self.configs[ref_index].name or f'CH{ref_index + 1}'}"
            )
        else:
            self.summary.setText("No stable periodic fundamental was detected in this capture.")


class PhaseSequenceDialog(QtWidgets.QDialog):
    """Three-phase sequence tool using any three of the available 3–4 channels."""

    def __init__(self, capture: Capture, configs: Sequence[object], parent=None) -> None:
        super().__init__(parent)
        self.capture = capture
        self.configs = configs
        self.setWindowTitle("Phase Sequence")
        self.resize(620, 360)

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Select the three phase channels. OpenScope compares their fundamental phasors and reports positive (ABC) or reverse (ACB) sequence."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        selectors = QtWidgets.QFormLayout()
        self.phase_boxes: List[QtWidgets.QComboBox] = []
        for label in ("Phase A / L1", "Phase B / L2", "Phase C / L3"):
            combo = QtWidgets.QComboBox()
            for i in range(capture.channel_count):
                combo.addItem(configs[i].name or f"CH{i + 1}", i)
            selectors.addRow(label, combo)
            self.phase_boxes.append(combo)
        if capture.channel_count >= 3:
            self.phase_boxes[0].setCurrentIndex(0)
            self.phase_boxes[1].setCurrentIndex(1)
            self.phase_boxes[2].setCurrentIndex(2)
        layout.addLayout(selectors)

        self.result = QtWidgets.QLabel()
        self.result.setObjectName("heroTitle")
        self.result.setAlignment(QtCore.Qt.AlignCenter)
        self.result.setWordWrap(True)
        layout.addWidget(self.result)

        self.details = QtWidgets.QLabel()
        self.details.setAlignment(QtCore.Qt.AlignCenter)
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        layout.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for combo in self.phase_boxes:
            combo.currentIndexChanged.connect(self.refresh)
        self.refresh()

    @staticmethod
    def _sequence_error(phase_b: float, phase_c: float, target_b: float, target_c: float) -> float:
        return abs(_wrap_phase(phase_b - target_b)) + abs(_wrap_phase(phase_c - target_c))

    def refresh(self) -> None:
        if self.capture.channel_count < 3:
            self.result.setText("Three channels are required")
            self.details.setText("The current capture does not contain three phase channels.")
            return

        selected = [int(combo.currentData()) for combo in self.phase_boxes]
        if len(set(selected)) != 3:
            self.result.setText("Select three different channels")
            self.details.setText("A, B and C must point to distinct channels.")
            return

        values = channel_values(self.capture, self.configs)
        frequencies = [_frequency(values[index], self.capture.sample_rate) for index in selected]
        valid = [f for f in frequencies if np.isfinite(f) and f > 0]
        if len(valid) != 3:
            self.result.setText("Sequence unavailable")
            self.details.setText("A stable periodic fundamental could not be detected on all three selected channels.")
            return

        common_frequency = float(np.median(valid))
        phasors = [_phasor(values[index], self.capture.sample_rate, common_frequency) for index in selected]
        angles = [math.degrees(math.atan2(p.imag, p.real)) for p in phasors]
        phase_b = _wrap_phase(angles[1] - angles[0])
        phase_c = _wrap_phase(angles[2] - angles[0])

        abc_error = self._sequence_error(phase_b, phase_c, -120.0, 120.0)
        acb_error = self._sequence_error(phase_b, phase_c, 120.0, -120.0)
        best_error = min(abc_error, acb_error)
        freq_spread = max(valid) - min(valid)

        # Large phase error generally means the selected signals are not a
        # balanced three-phase set, so avoid presenting a confident sequence.
        if best_error > 90.0 or freq_spread > max(1.0, common_frequency * 0.05):
            sequence = "Indeterminate sequence"
        elif abc_error <= acb_error:
            sequence = "ABC · positive sequence"
        else:
            sequence = "ACB · reverse sequence"

        self.result.setText(sequence)
        self.details.setText(
            f"Fundamental ≈ {common_frequency:.3f} Hz · B relative to A {phase_b:+.1f}° · "
            f"C relative to A {phase_c:+.1f}°"
        )
