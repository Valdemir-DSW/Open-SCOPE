from __future__ import annotations

from dataclasses import dataclass
import struct
import time
from typing import List, Optional, Tuple

import serial


CONFIG_SIZE = 18
SUPPORTED_FW_VERSION = 2


class ArduStimError(RuntimeError):
    pass


class ArduStimFirmwareMismatch(ArduStimError):
    pass


@dataclass
class ArduStimConfig:
    firmware_version: int = SUPPORTED_FW_VERSION
    wheel: int = 0
    rpm_mode: int = 2  # 0=sweep, 1=fixed, 2=potentiometer
    fixed_rpm: int = 4000
    sweep_min: int = 1000
    sweep_max: int = 5000
    sweep_interval: int = 250
    compression_enabled: int = 0
    compression_mode: int = 0
    compression_rpm: int = 0
    compression_offset: int = 0
    compression_dynamic: int = 0

    @classmethod
    def from_bytes(cls, data: bytes) -> "ArduStimConfig":
        if len(data) != CONFIG_SIZE:
            raise ArduStimError(f"invalid configuration length: {len(data)} (expected {CONFIG_SIZE})")
        fw = data[0]
        if fw != SUPPORTED_FW_VERSION:
            raise ArduStimFirmwareMismatch(
                f"unsupported Ardu-Stim firmware protocol {fw}; expected {SUPPORTED_FW_VERSION}"
            )
        mode = data[2]
        if mode not in (0, 1, 2):
            raise ArduStimError(f"invalid RPM mode returned by device: {mode}")
        return cls(
            firmware_version=fw,
            wheel=data[1],
            rpm_mode=mode,
            fixed_rpm=struct.unpack_from("<H", data, 3)[0],
            sweep_min=struct.unpack_from("<H", data, 5)[0],
            sweep_max=struct.unpack_from("<H", data, 7)[0],
            sweep_interval=struct.unpack_from("<H", data, 9)[0],
            compression_enabled=data[11],
            compression_mode=data[12],
            compression_rpm=struct.unpack_from("<H", data, 13)[0],
            compression_offset=struct.unpack_from("<H", data, 15)[0],
            compression_dynamic=data[17],
        )

    def command_bytes(self) -> bytes:
        # Lower-case c is the Ardu-Stim "receive full config" command. The
        # following 17 bytes match the firmware's packed configuration fields.
        payload = bytearray(CONFIG_SIZE)
        payload[0] = ord("c")
        payload[1] = self.wheel & 0xFF
        payload[2] = self.rpm_mode & 0xFF
        struct.pack_into("<H", payload, 3, max(0, min(65535, int(self.fixed_rpm))))
        struct.pack_into("<H", payload, 5, max(0, min(65535, int(self.sweep_min))))
        struct.pack_into("<H", payload, 7, max(0, min(65535, int(self.sweep_max))))
        struct.pack_into("<H", payload, 9, max(0, min(65535, int(self.sweep_interval))))
        payload[11] = 1 if self.compression_enabled else 0
        payload[12] = self.compression_mode & 0xFF
        struct.pack_into("<H", payload, 13, max(0, min(65535, int(self.compression_rpm))))
        struct.pack_into("<H", payload, 15, max(0, min(65535, int(self.compression_offset))))
        payload[17] = 1 if self.compression_dynamic else 0
        return bytes(payload)


class ArduStimProtocol:
    """Small, strict wrapper around the official Ardu-Stim serial protocol."""

    def __init__(self) -> None:
        self.serial: Optional[serial.Serial] = None
        self.port = ""

    @property
    def connected(self) -> bool:
        return self.serial is not None and bool(self.serial.is_open)

    def open(self, port: str) -> None:
        self.close()
        self.serial = serial.Serial(port=port, baudrate=115200, timeout=0.20, write_timeout=0.20)
        self.port = port
        # Most Uno/Nano boards reset when DTR is asserted. Do not trust any
        # bytes until the boot loader has had time to leave the serial line.
        time.sleep(1.65)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def close(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None
        self.port = ""

    def _require(self) -> serial.Serial:
        if not self.connected:
            raise ArduStimError("Ardu-Stim is not connected")
        assert self.serial is not None
        return self.serial

    def _flush_input(self) -> None:
        ser = self._require()
        ser.reset_input_buffer()

    def _read_exact(self, size: int, timeout: float = 0.8) -> bytes:
        ser = self._require()
        deadline = time.monotonic() + timeout
        data = bytearray()
        while len(data) < size and time.monotonic() < deadline:
            chunk = ser.read(size - len(data))
            if chunk:
                data.extend(chunk)
        return bytes(data)

    def _readline(self, timeout: float = 0.8) -> bytes:
        ser = self._require()
        deadline = time.monotonic() + timeout
        data = bytearray()
        while time.monotonic() < deadline:
            ch = ser.read(1)
            if not ch:
                continue
            data.extend(ch)
            if ch == b"\n":
                break
        return bytes(data).strip(b"\r\n")

    def request_config(self, retries: int = 3) -> ArduStimConfig:
        ser = self._require()
        last_error: Optional[Exception] = None
        for _ in range(max(1, retries)):
            try:
                self._flush_input()
                ser.write(b"C")
                ser.flush()
                raw = self._read_exact(CONFIG_SIZE, 0.9)
                return ArduStimConfig.from_bytes(raw)
            except ArduStimFirmwareMismatch:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(0.12)
        raise ArduStimError(f"device did not return a valid {CONFIG_SIZE}-byte configuration: {last_error}")

    def send_config(self, config: ArduStimConfig, save: bool = False) -> None:
        ser = self._require()
        ser.write(config.command_bytes())
        if save:
            ser.write(b"s")
        ser.flush()

    def save(self) -> None:
        ser = self._require()
        ser.write(b"s")
        ser.flush()

    def request_pattern_names(self) -> List[str]:
        ser = self._require()
        self._flush_input()
        ser.write(b"n")
        ser.flush()
        count_raw = self._readline(0.8)
        try:
            count = int(count_raw.decode("ascii", errors="strict").strip())
        except Exception as exc:
            raise ArduStimError(f"invalid wheel count returned by device: {count_raw!r}") from exc
        if count <= 0 or count > 255:
            raise ArduStimError(f"invalid wheel count returned by device: {count}")
        ser.write(b"L")
        ser.flush()
        names: List[str] = []
        for index in range(count):
            line = self._readline(0.8)
            if not line:
                raise ArduStimError(f"wheel list stopped at item {index + 1}/{count}")
            names.append(line.decode("utf-8", errors="replace").strip())
        return names

    def select_pattern(self, index: int, save: bool = True) -> None:
        if not 0 <= int(index) <= 255:
            raise ValueError("wheel index outside byte range")
        ser = self._require()
        ser.write(bytes((ord("S"), int(index))))
        if save:
            ser.write(b"s")
        ser.flush()

    def request_pattern(self) -> Tuple[List[int], int]:
        ser = self._require()
        self._flush_input()
        ser.write(b"P")
        ser.flush()
        pattern_line = self._readline(1.0).decode("ascii", errors="strict").strip()
        degrees_line = self._readline(0.8).decode("ascii", errors="strict").strip()
        try:
            states = [int(token.strip()) for token in pattern_line.split(",") if token.strip()]
            degrees = int(degrees_line)
        except Exception as exc:
            raise ArduStimError("invalid wheel pattern returned by device") from exc
        if not states or len(states) > 4096 or any(v < 0 or v > 7 for v in states):
            raise ArduStimError("wheel pattern payload is outside expected limits")
        if degrees not in (360, 720):
            raise ArduStimError(f"invalid wheel cycle returned by device: {degrees}")
        return states, degrees

    def request_rpm(self) -> int:
        ser = self._require()
        self._flush_input()
        ser.write(b"R")
        ser.flush()
        raw = self._readline(0.45)
        try:
            return max(0, int(raw.decode("ascii", errors="strict").strip()))
        except Exception as exc:
            raise ArduStimError(f"invalid RPM returned by device: {raw!r}") from exc
