from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib
from typing import List

import numpy as np

MAGIC = b"SCP1"
MAGIC_U32 = 0x31504353
HEADER = struct.Struct("<IHHIHHHBBBBII")
EXT_V4 = struct.Struct("<BBHI")  # packet_type, flags, reserved, sequence
MIN_HEADER_SIZE = HEADER.size
V4_HEADER_SIZE = HEADER.size + EXT_V4.size
MAX_PAYLOAD = 16 * 1024 * 1024

PACKET_CAPTURE = 0
PACKET_STREAM = 1
FLAG_DISCONTINUITY = 0x01
FLAG_DIRECT = 0x02


@dataclass(frozen=True)
class CaptureHeader:
    magic: int
    version: int
    header_size: int
    sample_rate: int
    frame_count: int
    pretrigger_frames: int
    trigger_level: int
    channel_count: int
    trigger_channel: int
    trigger_edge: int
    adc_bits: int
    payload_bytes: int
    crc32: int
    packet_type: int = PACKET_CAPTURE
    flags: int = 0
    sequence: int = 0


@dataclass
class Capture:
    header: CaptureHeader
    raw: np.ndarray

    @property
    def sample_rate(self) -> int:
        return self.header.sample_rate

    @property
    def frame_count(self) -> int:
        return self.header.frame_count

    @property
    def channel_count(self) -> int:
        return self.header.channel_count

    @property
    def pretrigger_frames(self) -> int:
        return self.header.pretrigger_frames

    @property
    def is_stream(self) -> bool:
        return self.header.packet_type == PACKET_STREAM

    def time_axis(self) -> np.ndarray:
        if self.sample_rate <= 0:
            return np.arange(self.frame_count, dtype=np.float64)
        return (
            np.arange(self.frame_count, dtype=np.float64) - self.pretrigger_frames
        ) / float(self.sample_rate)


class CaptureDecoder:
    """Streaming decoder for legacy SCP1 captures and v4/v5 direct-stream packets."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.good_packets = 0
        self.crc_errors = 0
        self.format_errors = 0
        self.discarded_bytes = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.good_packets = 0
        self.crc_errors = 0
        self.format_errors = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> List[Capture]:
        if data:
            self.buffer.extend(data)
        captures: List[Capture] = []

        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
                if len(self.buffer) > 3:
                    keep = self.buffer[-3:]
                    self.discarded_bytes += len(self.buffer) - len(keep)
                    self.buffer[:] = keep
                break

            if start:
                del self.buffer[:start]
                self.discarded_bytes += start

            if len(self.buffer) < MIN_HEADER_SIZE:
                break

            fields = HEADER.unpack_from(self.buffer, 0)
            base = CaptureHeader(*fields)

            if (
                base.magic != MAGIC_U32
                or base.header_size < MIN_HEADER_SIZE
                or base.header_size > 4096
                or base.channel_count < 1
                or base.channel_count > 3
                or base.adc_bits < 8
                or base.adc_bits > 16
                or base.frame_count < 1
                or base.frame_count > 65535
                or base.payload_bytes < 2
                or base.payload_bytes > MAX_PAYLOAD
            ):
                del self.buffer[0]
                self.format_errors += 1
                continue

            expected = base.frame_count * base.channel_count * 2
            if base.payload_bytes != expected:
                del self.buffer[0]
                self.format_errors += 1
                continue

            if len(self.buffer) < base.header_size:
                break

            packet_type = PACKET_CAPTURE
            flags = 0
            sequence = 0
            if base.version >= 4 and base.header_size >= V4_HEADER_SIZE:
                packet_type, flags, _reserved, sequence = EXT_V4.unpack_from(
                    self.buffer, HEADER.size
                )
                if packet_type not in (PACKET_CAPTURE, PACKET_STREAM):
                    del self.buffer[0]
                    self.format_errors += 1
                    continue

            header = CaptureHeader(
                base.magic,
                base.version,
                base.header_size,
                base.sample_rate,
                base.frame_count,
                base.pretrigger_frames,
                base.trigger_level,
                base.channel_count,
                base.trigger_channel,
                base.trigger_edge,
                base.adc_bits,
                base.payload_bytes,
                base.crc32,
                packet_type,
                flags,
                sequence,
            )

            packet_size = header.header_size + header.payload_bytes
            if len(self.buffer) < packet_size:
                break

            payload = bytes(
                self.buffer[header.header_size : header.header_size + header.payload_bytes]
            )
            del self.buffer[:packet_size]

            if (zlib.crc32(payload) & 0xFFFFFFFF) != header.crc32:
                self.crc_errors += 1
                continue

            raw = np.frombuffer(payload, dtype="<u2").reshape(
                header.frame_count, header.channel_count
            ).copy()
            captures.append(Capture(header=header, raw=raw))
            self.good_packets += 1

        return captures


def make_device_command(command: str, *args: object) -> bytes:
    command = command.strip().upper().replace(" ", "_")
    suffix = " ".join(str(arg) for arg in args)
    line = f"@SCP {command}"
    if suffix:
        line += " " + suffix
    return (line + "\n").encode("ascii", "strict")
