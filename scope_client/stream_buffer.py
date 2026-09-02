from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .protocol import Capture, FLAG_DISCONTINUITY


class RollingHistory:
    """Fixed-capacity chronological ring buffer owned entirely by the PC.

    The MCU never needs to know record length. Changing Time/div only changes
    how much of this history is rendered; it does not resize or destroy history.
    """

    def __init__(self) -> None:
        self.raw = np.empty((0, 0), dtype=np.uint16)
        self.capacity = 0
        self.channels = 0
        self.sample_rate = 0
        self.adc_bits = 12
        self.write = 0
        self.count = 0
        self.total = 0
        self.sequence = None
        self.discontinuities = 0

    def configure(self, sample_rate: int, channels: int, seconds: float, adc_bits: int) -> None:
        sample_rate = max(1, int(sample_rate))
        channels = max(1, int(channels))
        # 5% margin prevents a packet crossing the nominal record edge from
        # evicting data while the GUI is slicing a trigger window.
        requested = max(1, int(math.ceil(max(float(seconds), 1.0) * sample_rate)))
        capacity = requested + max(256, requested // 20)
        if (
            capacity == self.capacity
            and channels == self.channels
            and sample_rate == self.sample_rate
            and int(adc_bits) == self.adc_bits
        ):
            return
        self.capacity = capacity
        self.channels = channels
        self.sample_rate = sample_rate
        self.adc_bits = int(adc_bits)
        self.raw = np.empty((capacity, channels), dtype=np.uint16)
        self.write = 0
        self.count = 0
        self.total = 0
        self.sequence = None
        self.discontinuities = 0

    def clear(self, reset_stats: bool = True) -> None:
        self.write = 0
        self.count = 0
        self.total = 0
        self.sequence = None
        if reset_stats:
            self.discontinuities = 0

    @property
    def oldest_abs(self) -> int:
        return self.total - self.count

    def append(self, capture: Capture) -> tuple[int, int]:
        data = np.asarray(capture.raw, dtype=np.uint16)
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError(
                f"stream shape changed: expected (*,{self.channels}), got {data.shape}"
            )
        n_original = int(data.shape[0])
        if n_original <= 0:
            return self.total, self.total

        flagged_gap = bool(capture.header.flags & FLAG_DISCONTINUITY)
        sequence_gap = (
            self.sequence is not None
            and capture.header.sequence != ((self.sequence + 1) & 0xFFFFFFFF)
        )
        if flagged_gap or sequence_gap:
            self.discontinuities += 1
        self.sequence = capture.header.sequence

        start_abs = self.total
        data_to_store = data
        if n_original >= self.capacity:
            data_to_store = data[-self.capacity :]
        n = int(data_to_store.shape[0])

        first = min(n, self.capacity - self.write)
        self.raw[self.write : self.write + first] = data_to_store[:first]
        rest = n - first
        if rest:
            self.raw[:rest] = data_to_store[first:]
        self.write = (self.write + n) % self.capacity
        self.count = min(self.capacity, self.count + n)
        self.total += n_original
        return start_abs, self.total

    def extract(self, start_abs: int, count: int) -> Optional[np.ndarray]:
        start_abs = int(start_abs)
        count = int(count)
        if count <= 0:
            return np.empty((0, self.channels), dtype=np.uint16)
        if start_abs < self.oldest_abs or start_abs + count > self.total:
            return None

        offset = start_abs - self.oldest_abs
        oldest_index = (self.write - self.count) % self.capacity
        index = (oldest_index + offset) % self.capacity
        out = np.empty((count, self.channels), dtype=np.uint16)
        first = min(count, self.capacity - index)
        out[:first] = self.raw[index : index + first]
        if first < count:
            out[first:] = self.raw[: count - first]
        return out

    def latest(self, count: int) -> Optional[np.ndarray]:
        if self.count <= 0:
            return None
        count = min(max(1, int(count)), self.count)
        return self.extract(self.total - count, count)
