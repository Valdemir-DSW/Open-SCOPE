from __future__ import annotations

import zlib
import numpy as np

from .protocol import (
    Capture, CaptureHeader, MAGIC_U32, PACKET_CAPTURE, PACKET_STREAM, V4_HEADER_SIZE,
)


def demo_capture(
    sample_rate: int = 80_000,
    frames: int = 4096,
    channels: int = 3,
    *,
    packet_type: int = PACKET_STREAM,
    sequence: int = 0,
    sample_offset: int = 0,
) -> Capture:
    """Realistic engine-position demo: 60-2 at 4000 RPM.

    CH1: 60-2 digital wheel, 58 pulses/revolution.
    CH2: cam/phase square pulse once every two crank revolutions.
    CH3: analog-ish crank pickup waveform with the same missing-tooth gap.
    Absolute sample time keeps packet boundaries perfectly continuous.
    """
    rate = max(int(sample_rate), 1)
    n = np.arange(frames, dtype=np.float64) + float(sample_offset)
    t = n / float(rate)

    rpm = 4000.0
    rev_hz = rpm / 60.0
    tooth_slots_hz = rev_hz * 60.0  # 4000 slots/s

    slot_phase = (t * tooth_slots_hz) % 60.0
    slot_index = np.floor(slot_phase).astype(np.int32)
    intra = slot_phase - np.floor(slot_phase)
    missing = slot_index >= 58

    # Hall-like 60-2: 42 % high pulse for each real tooth.
    wheel_high = (intra < 0.42) & (~missing)
    ch1 = np.where(wheel_high, 3300.0, 80.0)

    # Cam pulse: one 90-degree-ish window every 720 crank degrees.
    engine720 = (t * rev_hz / 2.0) % 1.0
    phase_high = (engine720 >= 0.08) & (engine720 < 0.20)
    ch2 = np.where(phase_high, 3000.0, 120.0)

    # VR-like conditioned waveform for visual testing. The missing two teeth
    # naturally produce a long gap. Kept within 0..4095 because this is ADC data.
    carrier = np.sin(2.0 * np.pi * tooth_slots_hz * t)
    envelope = np.where(missing, 0.06, 1.0)
    ch3 = 2048.0 + 1450.0 * carrier * envelope

    values = [ch1, ch2, ch3]
    raw = np.column_stack(values[:channels])
    raw = np.clip(raw, 0, 4095).astype('<u2')
    payload = raw.tobytes(order='C')

    header = CaptureHeader(
        magic=MAGIC_U32,
        version=5,
        header_size=V4_HEADER_SIZE,
        sample_rate=rate,
        frame_count=frames,
        pretrigger_frames=frames // 2 if packet_type == PACKET_CAPTURE else 0,
        trigger_level=2048,
        channel_count=channels,
        trigger_channel=1,
        trigger_edge=1,
        adc_bits=12,
        payload_bytes=len(payload),
        crc32=zlib.crc32(payload) & 0xFFFFFFFF,
        packet_type=packet_type,
        flags=0x02 if packet_type == PACKET_STREAM else 0,
        sequence=sequence,
    )
    return Capture(header=header, raw=raw)
