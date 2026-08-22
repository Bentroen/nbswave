import math
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import samplerate as sr
import soundfile as sf

from . import effects

_PCM_FORMATS = frozenset({"WAV", "AIFF", "FLAC", "WAVEX", "W64", "RF64", "CAF"})
_BIT_DEPTH_SUBTYPES = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",
}


def _subtype_for(filename: str, bit_depth: int) -> str | None:
    """Map a user-facing bit depth to a libsndfile subtype, or `None` for lossy formats."""

    ext = os.path.splitext(filename)[1][1:].upper()
    if ext == "AIF":
        ext = "AIFF"
    if ext not in _PCM_FORMATS:
        return None

    try:
        subtype = _BIT_DEPTH_SUBTYPES[bit_depth]
    except KeyError:
        raise ValueError(f"bit_depth must be 16, 24, or 32, got {bit_depth}") from None

    if ext == "FLAC" and subtype == "FLOAT":
        raise ValueError("FLAC does not support 32-bit float; use bit_depth=16 or 24")

    return subtype


def key_to_pitch(key: float) -> float:
    return 2.0 ** ((key) / 12)


def vol_to_gain(vol: float) -> float:
    if vol == 0.0:
        return -float("inf")
    return math.log10(vol) * 20.0


def gain_to_vol(gain: float) -> float:
    return 10.0 ** (gain / 20.0)


def panning_to_vol(panning: float) -> tuple[float, float]:
    # Simplified panning algorithm from pydub to operate on numpy arrays
    # https://github.com/jiaaro/pydub/blob/0c26b10619ee6e31c2b0ae26a8e99f461f694e5f/pydub/effects.py#L284

    max_boost_db = vol_to_gain(2.0)
    boost_db = abs(panning) * max_boost_db

    boost_factor = gain_to_vol(boost_db)
    reduce_factor = gain_to_vol(max_boost_db) - boost_factor

    reduce_db = vol_to_gain(reduce_factor)
    boost_db /= 2.0

    if panning < 0:
        return gain_to_vol(boost_db), gain_to_vol(reduce_db)
    else:
        return gain_to_vol(reduce_db), gain_to_vol(boost_db)


@dataclass
class OverlayOperation:
    position: int
    volume: float
    panning: float


class AudioSegment:
    # Largely inspired by pydub.AudioSegment:
    # https://github.com/jiaaro/pydub/blob/v0.25.1/pydub/audio_segment.py

    def __init__(self, data: np.ndarray, frame_rate: int, channels: int):
        self.data = data
        self.frame_rate = frame_rate
        self.channels = channels

    def _spawn(self, data: np.ndarray, overrides: dict[str, int] | None = None):
        metadata = {
            "frame_rate": self.frame_rate,
            "channels": self.channels,
        }
        if overrides:
            metadata.update(overrides)
        return self.__class__(data=data.copy(), **metadata)

    def set_frame_rate(self, frame_rate: int):
        if frame_rate == self.frame_rate:
            return self

        ratio = frame_rate / self.frame_rate
        # https://libsndfile.github.io/libsamplerate/api_misc.html#converters
        new_data = sr.resample(self.data, ratio, "sinc_best")
        return self._spawn(new_data, {"frame_rate": frame_rate})

    def set_channels(self, channels: int):
        if channels == self.channels:
            return self

        if channels == 1 and self.channels == 2:
            new_data = np.mean(self.data, axis=1)
        elif channels == 2 and self.channels == 1:
            new_data = np.repeat(self.data, 2, axis=1)
        else:
            raise ValueError("Unsupported channel conversion")

        return self._spawn(new_data, {"channels": channels})

    @property
    def duration_seconds(self):
        return len(self.data) / (self.frame_rate * self.channels)

    @property
    def raw_data(self):
        return self.data

    def __len__(self):
        return round(self.duration_seconds * 1000)

    def set_speed(
        self, speed: float = 1.0, frame_rate: int | None = None
    ) -> "AudioSegment":
        if frame_rate is not None and frame_rate != self.frame_rate:
            speed *= self.frame_rate / frame_rate

        if speed == 1.0:
            return self

        new = self._spawn(
            self.raw_data, overrides={"frame_rate": round(self.frame_rate * speed)}
        )
        return new.set_frame_rate(self.frame_rate)

    def set_volume(self, volume: float) -> "AudioSegment":
        return self._spawn(self.raw_data * volume, {})

    def apply_volume_stereo(self, left_vol: float, right_vol: float) -> "AudioSegment":
        left = self.data[:, 0] * left_vol
        right = self.data[:, 1] * right_vol

        return self._spawn(np.stack([left, right], axis=1), {})

    def set_panning(self, panning: float) -> "AudioSegment":
        # Simplified panning algorithm from pydub to operate on numpy arrays
        # https://github.com/jiaaro/pydub/blob/0c26b10619ee6e31c2b0ae26a8e99f461f694e5f/pydub/effects.py#L284

        if panning == 0:
            return self

        left_vol, right_vol = panning_to_vol(panning)
        return self.apply_volume_stereo(left_vol, right_vol)


def load_sound(path: str) -> AudioSegment:
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        raise FileNotFoundError(f"Could not load sound file {path}: {e}")

    channels = data.shape[1]

    # TODO: remove channel count coercion
    if channels == 1:
        data = np.repeat(data, 2, axis=1)
        channels = 2

    return AudioSegment(data, sample_rate, channels)


class Mixer:
    def __init__(
        self,
        frame_rate: int = 44100,
        channels: int = 2,
        length: float = 0,
        max_workers: int = 8,
    ):
        self.frame_rate = frame_rate
        self.channels = channels
        self.output = np.zeros(
            (self._get_array_size(length), self.channels), dtype="float32"
        )
        self.max_workers = max_workers

    def _get_array_size(self, length_in_ms: float) -> int:
        frame_count = length_in_ms * (self.frame_rate / 1000.0)
        return int(frame_count)

    def overlay(self, sound: AudioSegment, position_ms: float = 0):
        samples = sound.raw_data

        frame_offset = int(self.frame_rate * position_ms / 1000.0)

        start = frame_offset
        end = start + len(samples)

        output_size = len(self.output)
        if end > output_size:
            pad_length = self._get_array_size(end - output_size)
            self.output = np.pad(
                self.output, ((0, pad_length), (0, 0)), mode="constant"
            )
            print(f"Padded from {output_size} to {end} (added {pad_length} entries)")

        self.output[start:end] += samples

        return self

    def batch_resample(self, tasks: Iterable[tuple[AudioSegment, float, Any]]):
        """Resample multiple AudioSegments in parallel using ThreadPoolExecutor."""

        def set_speed_with_context(segment: AudioSegment, speed: float, context: Any):
            return segment.set_speed(speed, self.frame_rate), context

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(set_speed_with_context, segment, speed, context)
                for segment, speed, context in tasks
            ]
            for future in as_completed(futures):
                yield future.result()

    @property
    def duration_ms(self) -> float:
        return len(self.output) / (self.frame_rate * self.channels) * 1000

    def append(self, sound: AudioSegment):
        self.overlay(sound, position_ms=self.duration_ms)

    def to_audio_segment(self) -> "Track":
        output_segment = AudioSegment(
            self.output,
            frame_rate=self.frame_rate,
            channels=self.channels,
        )

        return Track.from_audio_segment(output_segment)


class Track(AudioSegment):
    """Rendered track with optional post-processing effects."""

    @classmethod
    def from_audio_segment(cls, segment: AudioSegment) -> "Track":
        return cls(
            segment.raw_data,
            frame_rate=segment.frame_rate,
            channels=segment.channels,
        )

    def _with_data(self, data: np.ndarray) -> "Track":
        return self.__class__(
            data,
            frame_rate=self.frame_rate,
            channels=self.channels,
        )

    def clip_guard(self, target_db: float = 0.0) -> "Track":
        """Scale the track down only if its peak sample is greater than `target_db` dBFS. This is useful
        for preventing clipping when exporting to common audio formats, such as MP3 or 16-bit WAV files.

        Unlike :meth:`Track.normalize`, this only scales the signal down. Quiet mixes are left unchanged.

        Args:
            `target_db`: The dBFS level of the peak sample.
        """
        return self._with_data(effects.clip_guard(self.raw_data, target_db=target_db))

    def normalize(self, target_db: float = -1.0) -> "Track":
        """Scale the track so its peak sample is at `target_db` dBFS.
        All other samples are scaled to maintain the same relative intensity.

        Unlike :meth:`Track.clip_guard`, this always scales the signal (up or down).

        Args:
            `target_db`: The dBFS level of the peak sample.
        """
        return self._with_data(effects.normalize(self.raw_data, target_db=target_db))

    def compress(
        self,
        threshold_db: float = -24,
        ratio: float = 12,
        attack_ms: float = 3,
        release_ms: float = 250,
    ) -> "Track":
        """Apply soft dynamic-range compression, pushing only the loudest samples down to
        `threshold_db`. Requires `pedalboard`.

        Args:
            threshold_db: The level, in decibels, below which the signal is not compressed.
            ratio: The intensity of the compression as a ratio of the original signal, e.g., a ratio of 3 means a 3:1 compression.
            attack_ms: How quickly the signal is compressed when it exceeds the threshold.
            release_ms: How quickly compression is released when the signal falls below the threshold.
        """
        return self._with_data(
            effects.compress(
                self.raw_data,
                self.frame_rate,
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
        )

    def limiter(
        self,
        threshold_db: float = -3,
        release_ms: float = 100,
    ) -> "Track":
        """Apply a brick-wall peak limiter. Requires `pedalboard`.

        Args:
            threshold_db: The threshold below which the signal is not clipped.
            release_ms: The release time in milliseconds.
        """
        return self._with_data(
            effects.limiter(
                self.raw_data,
                self.frame_rate,
                threshold_db=threshold_db,
                release_ms=release_ms,
            )
        )

    def loudness(self, target_lufs: float = -14.0) -> "Track":
        """Normalize loudness to `target_lufs`. The signal is scaled down or up uniformly to
        achieve the target LUFS level.

        LUFS (Loudness Units relative to Full Scale) is a standard unit used to measure how
        loud a sound feels to human ears. Unlike raw volume or peak meters, LUFS matches how
        the human brain perceives different frequencies and loudness over time.

        It is useful for ensuring consistent loudness across different audio content (e.g.,
        multiple tracks in an album). Many streaming and media services also use LUFS to
        normalize audio content. You can use this method to ensure your exported track is
        compatible with these services.

        Requires `pyloudnorm`.

        Args:
            target_lufs: The target LUFS level of the signal.

        """
        return self._with_data(
            effects.loudness(self.raw_data, self.frame_rate, target_lufs=target_lufs)
        )

    def save(self, filename: str, bit_depth: int = 16):
        sf.write(
            filename,
            self.raw_data,
            samplerate=self.frame_rate,
            subtype=_subtype_for(filename, bit_depth),
        )
