from __future__ import annotations

import numpy as np

from src.audio.microphone import MicrophoneAudioStream


def test_chunks_returns_immediately_when_stream_is_closed() -> None:
    stream = MicrophoneAudioStream()

    assert list(stream.chunks()) == []


def test_chunks_yields_queued_audio_then_stops_on_close() -> None:
    stream = MicrophoneAudioStream(max_queue_chunks=2)
    stream._closed = False
    stream._queue.put_nowait(np.array([1.0], dtype=np.float32))
    stream.close()

    chunks = list(stream.chunks())

    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], np.array([1.0], dtype=np.float32))


def test_close_wakes_empty_consumer() -> None:
    stream = MicrophoneAudioStream(max_queue_chunks=1)
    stream._closed = False
    stream.close()

    chunks = list(stream.chunks())

    assert chunks == []


def test_close_wakes_consumer_when_queue_is_full() -> None:
    stream = MicrophoneAudioStream(max_queue_chunks=1)
    stream._closed = False
    stream._queue.put_nowait(np.array([1.0], dtype=np.float32))
    stream.close()

    chunks = list(stream.chunks())

    assert chunks == []


def test_callback_ignores_audio_after_close() -> None:
    stream = MicrophoneAudioStream(max_queue_chunks=2)
    stream._closed = False
    stream.close()

    stream._callback(
        np.array([[1.0]], dtype=np.float32),
        _frames=1,
        _time_info=None,
        status=None,
    )

    assert list(stream.chunks()) == []


def test_callback_counts_chunks_dropped_when_queue_is_full() -> None:
    stream = MicrophoneAudioStream(max_queue_chunks=1)
    stream._closed = False

    for value in (1.0, 2.0, 3.0):
        stream._callback(np.array([[value]], dtype=np.float32), _frames=1, _time_info=None, status=None)

    assert stream.dropped_chunks == 2
    np.testing.assert_array_equal(stream._queue.get_nowait(), np.array([3.0], dtype=np.float32))
