from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Measurements:
    minimum: float = math.nan
    maximum: float = math.nan
    vpp: float = math.nan
    mean: float = math.nan
    rms: float = math.nan
    frequency: float = math.nan
    period: float = math.nan
    duty: float = math.nan


def analyze(y: np.ndarray, sample_rate: float) -> Measurements:
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2:
        return Measurements()

    minimum = float(np.min(y))
    maximum = float(np.max(y))
    mean = float(np.mean(y))
    rms = float(np.sqrt(np.mean(np.square(y))))
    vpp = maximum - minimum

    frequency = math.nan
    period = math.nan
    duty = math.nan

    if sample_rate > 0 and vpp > max(abs(mean) * 1e-6, 1e-12):
        low = minimum + 0.35 * vpp
        high = minimum + 0.65 * vpp
        state = bool(y[0] >= high)
        rises = []
        high_samples = 0

        for i, value in enumerate(y):
            if state:
                high_samples += 1
                if value <= low:
                    state = False
            elif value >= high:
                state = True
                rises.append(i)
                high_samples += 1

        if len(rises) >= 2:
            periods = np.diff(np.asarray(rises, dtype=np.float64)) / sample_rate
            good = periods[periods > 0]
            if good.size:
                period = float(np.median(good))
                frequency = 1.0 / period

        duty = 100.0 * high_samples / float(y.size)

    return Measurements(
        minimum=minimum,
        maximum=maximum,
        vpp=vpp,
        mean=mean,
        rms=rms,
        frequency=frequency,
        period=period,
        duty=duty,
    )


def fmt(value: float, unit: str = "", digits: int = 4) -> str:
    if not np.isfinite(value):
        return "—"
    av = abs(value)
    prefixes = [
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "µ"),
        (1e-9, "n"),
    ]
    for scale, prefix in prefixes:
        if av >= scale or scale == 1e-9:
            return f"{value / scale:.{digits}g} {prefix}{unit}".strip()
    return f"{value:.{digits}g} {unit}".strip()
