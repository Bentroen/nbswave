import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import samplerate as sr
import soundfile as sf


def key_to_pitch(key: int) -> float:
    return 2 ** ((key) / 12)


def vol_to_gain(vol: float) -> float:
    if vol == 0:
        return -float("inf")
    return math.log(vol, 10) * 20


def gain_to_vol(gain: float) -> float:
    return 10 ** (gain / 20)


def panning_to_vol(panning: float) -> tuple[float, float]:
    # Simplified panning algorithm from pydub to operate on numpy arrays
    # https://github.com/jiaaro/pydub/blob/0c26b10619ee6e31c2b0ae26a8e99f461f694e5f/pydub/effects.py#L284

    max_boost_db = gain_to_vol(2.0)
    boost_db = abs(panning) * max_boost_db

    boost_factor = gain_to_vol(boost_db)
    reduce_factor = gain_to_vol(max_boost_db) - boost_factor

    boost_factor /= 2.0

    if panning < 0:
        return boost_factor, reduce_factor
    else:
        return reduce_factor, boost_factor


@dataclass
class OverlayOperation:
    position: int
    volume: float
    panning: float


class AudioSegment:
    # Largely inspired by pydub.AudioSegment:
    # https://github.com/jiaaro/pydub/blob/v0.25.1/pydub/audio_segment.py

    def __init__(
        self, data: np.ndarray, frame_rate: int, sample_width: int, channels: int
    ):
        self.data = data
        self.frame_rate = frame_rate
        self.sample_width = sample_width
        self.channels = channels

    def _spawn(self, data: np.ndarray, overrides: Dict[str, int]):
        metadata = {
            "sample_width": self.sample_width,
            "frame_rate": self.frame_rate,
            "channels": self.channels,
        }
        metadata.update(overrides)
        return self.__class__(data=data.copy(), **metadata)

    def set_sample_width(self, sample_width: int):
        if sample_width == self.sample_width:
            return self

        new_data = self.data.astype(f"int{sample_width * 8}")
        return self._spawn(new_data, {"sample_width": sample_width})

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

    return AudioSegment(data, sample_rate, 2, channels)


class Mixer:
    def __init__(
        self,
        sample_width: int = 2,
        frame_rate: int = 44100,
        channels: int = 2,
        length: float = 0,
        max_workers: int = 8,
    ):
        self.sample_width = sample_width
        self.frame_rate = frame_rate
        self.channels = channels
        self.output = np.zeros(
            (self._get_array_size(length), self.channels), dtype="float32"
        )
        self.max_workers = max_workers

    def _get_array_size(self, length_in_ms: float) -> int:
        frame_count = length_in_ms * (self.frame_rate / 1000.0)
        return int(frame_count)

    def overlay(self, sound: AudioSegment, position: int = 0):
        samples = sound.raw_data

        frame_offset = int(self.frame_rate * position / 1000.0)

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

    def __len__(self):
        return len(self.output) / ((self.frame_rate / 1000.0) * self.channels)

    def append(self, sound: AudioSegment):
        self.overlay(sound, position=len(self))

    def to_audio_segment(self):
        peak = np.abs(self.output).max()
        clipping_factor = peak / 1.0

        if clipping_factor > 1:
            print(
                f"The output is clipping by {clipping_factor:.2f}x. Normalizing to 0dBFS"
            )
            normalized_signal = self.output / clipping_factor
        else:
            normalized_signal = self.output

        output_segment = AudioSegment(
            normalized_signal,
            frame_rate=self.frame_rate,
            sample_width=self.sample_width,
            channels=self.channels,
        )

        return Track.from_audio_segment(output_segment)


class Track(AudioSegment):
    """A subclass of `pydub.AudioSegment` for applying post-rendering
    effects to rendered tracks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_audio_segment(cls, segment: AudioSegment):
        return cls(
            segment.raw_data,
            sample_width=segment.sample_width,
            frame_rate=segment.frame_rate,
            channels=segment.channels,
        )

    def save(
        self,
        filename: str,
        format: str = "wav",
        sample_width: int = 2,
        frame_rate: int = 44100,
        channels: int = 2,
        target_bitrate: int = 320,
        target_size: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ):

        seconds = self.duration_seconds

        if target_size:
            bitrate = (target_size / seconds) * 8
            bitrate = min(bitrate, target_bitrate)
        else:
            bitrate = target_bitrate

        output_segment = self  # sync(self, channels, frame_rate, sample_width)

        sf.write(filename, output_segment.raw_data, samplerate=frame_rate)
