from __future__ import annotations

from collections import deque
from queue import Empty, Queue
from threading import Lock
import time

from PyQt5 import QtCore
import serial

from .protocol import CaptureDecoder


class SerialWorker(QtCore.QThread):
    """Low-latency serial reader.

    Decoding happens entirely in this worker thread. Completed SCP packets are
    placed in a bounded thread-safe queue and the Qt GUI *pulls* them at its own
    cadence. This prevents hundreds of queued Qt signals from building up when
    USB CDC delivers data in bursts.
    """

    connected = QtCore.pyqtSignal(str)
    disconnected = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    stats = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, port: str, baudrate: int, parent=None) -> None:
        super().__init__(parent)
        self.port = port
        self.baudrate = int(baudrate)
        self._running = True
        self._tx: Queue[bytes] = Queue()
        self._serial = None

        self._rx_lock = Lock()
        self._rx = deque()
        self._rx_packets = 0
        self._rx_dropped = 0
        # This is a safety cap only. With normal GUI draining the queue remains
        # very small. If the GUI stalls, drop oldest transport packets rather
        # than allowing unbounded memory growth and seconds of stale latency.
        self._rx_max_packets = 2048

    def stop(self) -> None:
        self._running = False
        self.requestInterruption()

    def send(self, data: bytes) -> None:
        if data:
            self._tx.put(bytes(data))

    def take_pending(self, max_packets: int = 512):
        """Return the oldest decoded packets without blocking the GUI."""
        out = []
        limit = max(1, int(max_packets))
        with self._rx_lock:
            while self._rx and len(out) < limit:
                out.append(self._rx.popleft())
        return out

    @property
    def dropped_transport_packets(self) -> int:
        with self._rx_lock:
            return int(self._rx_dropped)

    def _queue_captures(self, captures) -> None:
        if not captures:
            return
        with self._rx_lock:
            for capture in captures:
                if len(self._rx) >= self._rx_max_packets:
                    self._rx.popleft()
                    self._rx_dropped += 1
                self._rx.append(capture)
                self._rx_packets += 1

    def run(self) -> None:
        decoder = CaptureDecoder()
        last_stats = 0.0
        try:
            # Non-blocking reads are important on Windows USB CDC. A 30 ms
            # timeout combined with packet coalescing made the old client feel
            # bursty even when the device was sending continuously.
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0,
                write_timeout=0.05,
            )
            # Do not let pyserial insert software flow control.
            self._serial.xonxoff = False
            self._serial.rtscts = False
            self._serial.dsrdtr = False
            self.connected.emit(self.port)

            while self._running and not self.isInterruptionRequested():
                try:
                    while True:
                        packet = self._tx.get_nowait()
                        self._serial.write(packet)
                except Empty:
                    pass

                waiting = int(self._serial.in_waiting)
                if waiting > 0:
                    # Read everything currently queued by the OS in one syscall.
                    data = self._serial.read(min(waiting, 65536))
                    self._queue_captures(decoder.feed(data))
                else:
                    # 1 ms idle sleep: low CPU, but no 30 ms polling latency.
                    self.msleep(1)

                now = time.monotonic()
                if now - last_stats >= 0.10:
                    self.stats.emit(
                        decoder.good_packets,
                        decoder.crc_errors,
                        decoder.format_errors,
                        decoder.discarded_bytes,
                    )
                    last_stats = now
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self._serial = None
            self.disconnected.emit()
