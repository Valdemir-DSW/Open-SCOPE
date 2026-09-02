from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def find_crossings(
    values: np.ndarray,
    *,
    start_abs: int,
    level: float,
    rising: bool,
    previous: Optional[float],
    min_abs: int,
) -> Tuple[np.ndarray, Optional[float]]:
    """Vectorized PC-side edge detector.

    Returns *all* valid absolute crossing indices in the supplied block. Keeping
    every edge is important for a periodic waveform: the renderer can then use
    the newest trigger whose post-trigger window is already available instead
    of waiting for one trigger, rendering it, and only then starting to search
    for the next one.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.int64), previous

    hits = []
    first = float(values[0])
    if previous is not None:
        crossed = previous < level <= first if rising else previous > level >= first
        if crossed and int(start_abs) >= int(min_abs):
            hits.append(int(start_abs))

    if values.size > 1:
        a = values[:-1]
        b = values[1:]
        if rising:
            idx = np.flatnonzero((a < level) & (b >= level))
        else:
            idx = np.flatnonzero((a > level) & (b <= level))
        if idx.size:
            absolute = int(start_abs) + idx.astype(np.int64) + 1
            absolute = absolute[absolute >= int(min_abs)]
            if absolute.size:
                hits.extend(absolute.tolist())

    return np.asarray(hits, dtype=np.int64), float(values[-1])


def find_first_crossing(
    values: np.ndarray,
    *,
    start_abs: int,
    level: float,
    rising: bool,
    previous: Optional[float],
    min_abs: int,
) -> Tuple[Optional[int], Optional[float]]:
    """Compatibility helper used by older tests."""
    crossings, previous = find_crossings(
        values,
        start_abs=start_abs,
        level=level,
        rising=rising,
        previous=previous,
        min_abs=min_abs,
    )
    return (int(crossings[0]) if crossings.size else None), previous
