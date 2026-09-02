"""Protocol, direct-stream history and PC-trigger tests. No PyQt5 required."""
import zlib
import numpy as np

from scope_client.demo import demo_capture
from scope_client.pc_trigger import find_first_crossing, find_crossings
from scope_client.protocol import (
    CaptureDecoder, CaptureHeader, HEADER, EXT_V4, MAGIC_U32,
    PACKET_CAPTURE, PACKET_STREAM, V4_HEADER_SIZE,
)
from scope_client.stream_buffer import RollingHistory


def make_packet(channels, bits, frames, rate, *, version=5, packet_type=PACKET_STREAM, sequence=0, flags=0):
    max_count = (1 << bits) - 1
    x = np.arange(frames * channels, dtype=np.uint32).reshape(frames, channels)
    raw = ((x * 37 + 11) & max_count).astype('<u2')
    payload = raw.tobytes(order='C')
    header_size = V4_HEADER_SIZE if version >= 4 else HEADER.size
    pre = frames // 2 if packet_type == PACKET_CAPTURE else 0
    header = CaptureHeader(
        MAGIC_U32, version, header_size, rate, frames, pre,
        (max_count + 1) // 2, channels, 1, 1, bits,
        len(payload), zlib.crc32(payload) & 0xFFFFFFFF,
        packet_type, flags, sequence,
    )
    packet = HEADER.pack(
        header.magic, header.version, header.header_size, header.sample_rate,
        header.frame_count, header.pretrigger_frames, header.trigger_level,
        header.channel_count, header.trigger_channel, header.trigger_edge,
        header.adc_bits, header.payload_bytes, header.crc32,
    )
    if version >= 4:
        packet += EXT_V4.pack(packet_type, flags, 0, sequence)
    return packet + payload


def decode_fragmented(packet: bytes):
    decoder = CaptureDecoder()
    captures = []
    for offset in range(0, len(packet), 37):
        captures.extend(decoder.feed(packet[offset:offset + 37]))
    assert len(captures) == 1
    assert decoder.crc_errors == 0
    assert decoder.format_errors == 0
    return captures[0]


def test_direct_history_and_trigger():
    rate = 80_000
    chunk = 64
    history = RollingHistory()
    history.configure(rate, 3, 1.0, 12)
    previous = None
    first_trigger = None
    total = rate  # one full second
    offset = 0
    seq = 0
    while offset < total:
        frames = min(chunk, total - offset)
        cap = demo_capture(rate, frames, 3, packet_type=PACKET_STREAM, sequence=seq, sample_offset=offset)
        start, _ = history.append(cap)
        trig, previous = find_first_crossing(
            cap.raw[:, 0], start_abs=start, level=2048, rising=True,
            previous=previous, min_abs=rate // 2,
        )
        if first_trigger is None and trig is not None:
            first_trigger = trig
        offset += frames
        seq += 1

    assert history.count >= rate
    latest = history.latest(rate)
    assert latest is not None and latest.shape == (rate, 3)
    assert first_trigger is not None and first_trigger >= rate // 2

    volts = np.array([0.2, 0.8, 1.4, 1.7, 2.2], dtype=np.float64)
    trig_v, prev_v = find_first_crossing(
        volts, start_abs=100, level=1.65, rising=True, previous=None, min_abs=100
    )
    assert trig_v == 103
    assert abs(float(prev_v) - 2.2) < 1e-12

    periodic = np.tile(np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float64), 100)
    edges, _ = find_crossings(
        periodic, start_abs=0, level=1.0, rising=True, previous=None, min_abs=0
    )
    assert edges.size == 100
    assert int(edges[0]) == 2 and int(edges[-1]) == 398


def main():
    direct = decode_fragmented(make_packet(3, 12, 64, 80_000, sequence=123, flags=0x02))
    four = decode_fragmented(make_packet(4, 12, 64, 60_000, sequence=124, flags=0x02))
    leo = decode_fragmented(make_packet(2, 10, 32, 25_000, sequence=9, flags=0x02))
    old = decode_fragmented(make_packet(2, 10, 64, 20_000, version=3, packet_type=PACKET_CAPTURE))

    assert direct.is_stream and direct.raw.shape == (64, 3)
    assert four.is_stream and four.raw.shape == (64, 4)
    assert leo.is_stream and leo.raw.shape == (32, 2)
    assert int(leo.raw.max()) <= 1023
    assert not old.is_stream
    test_direct_history_and_trigger()

    print('OpenScope direct-stream v7 self-test: PASS')
    print('Header:', HEADER.size, 'bytes | extended:', V4_HEADER_SIZE, 'bytes')
    print('STM direct:', direct.raw.shape, direct.sample_rate, 'Sa/s/ch')
    print('4-channel protocol:', four.raw.shape, four.sample_rate, 'Sa/s/ch')
    print('Leonardo direct:', leo.raw.shape, leo.sample_rate, 'Sa/s/ch')
    print('PC history: 1.000 s continuous | voltage trigger: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
