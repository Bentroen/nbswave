from __future__ import annotations

from collections.abc import Iterable, Iterator

import pynbs


def sorted_notes(notes: Iterable[Note]) -> list[Note]:
    """Return a list of notes sorted by pitch, instrument, velocity, and
    panning."""
    return sorted(notes, key=lambda x: (x.pitch, x.instrument, x.velocity, x.panning))


class Note:
    """Extends `pynbs.Note` with extra functionality to calculate
    the compensated pitch, volume and panning values."""

    def __init__(
        self,
        tick: int,
        layer: int,
        instrument: int,
        key: float,
        velocity: float,
        panning: float,
        pitch: float,
    ):
        self.tick = tick
        self.layer = layer
        self.instrument = instrument
        self.key = key
        self.velocity = velocity
        self.panning = panning
        self.pitch = pitch

    @classmethod
    def from_note(cls, note: pynbs.Note | Note):
        return cls(
            note.tick,
            note.layer,
            note.instrument,
            note.key,
            note.velocity,
            note.panning,
            note.pitch,
        )

    def move(self, offset: int) -> Note:
        """Return a new Note object with its tick moved by a certain amount of ticks."""
        return self.__class__(
            self.tick + offset,
            self.layer,
            self.instrument,
            self.key,
            self.velocity,
            self.panning,
            self.pitch,
        )

    def apply_layer_weight(
        self, layer: pynbs.Layer, custom_instrument: pynbs.Instrument | None = None
    ) -> Note:
        """Return a new Note object with compensated pitch, volume and panning."""
        pitch = self._get_pitch(custom_instrument)
        volume = self._get_volume(layer)
        panning = self._get_panning(layer)
        return self.__class__(
            self.tick, self.layer, self.instrument, pitch, volume, panning, 0
        )

    def _get_pitch(self, custom_instrument: pynbs.Instrument | None = None) -> float:
        """Return the detune-aware pitch of this note."""
        if custom_instrument is not None:
            instrument_key = (45 - custom_instrument.pitch) + 45
        else:
            instrument_key = 45  # This assumes all default instruments are pitched F#4
        key = self.key - instrument_key
        detune = self.pitch / 100
        pitch = key + detune
        return pitch

    def _get_volume(self, layer: pynbs.Layer) -> float:
        """Return the layer-aware volume of this note."""
        layer_vol = layer.volume / 100
        note_vol = self.velocity / 100
        vol = layer_vol * note_vol
        return vol

    def _get_panning(self, layer: pynbs.Layer) -> float:
        """Return the layer-aware panning of this note."""
        layer_pan = layer.panning / 100
        note_pan = self.panning / 100
        if layer_pan == 0:
            pan = note_pan
        else:
            pan = (layer_pan + note_pan) / 2
        return pan


class Song(pynbs.File):
    """Extends the `pynbs.File` class with extra functionality."""

    def __init__(self, song: pynbs.File):
        super().__init__(song.header, song.notes, song.layers, song.instruments)
        self.notes = [Note.from_note(note) for note in self.notes]

    def __len__(self) -> int:
        """Return the length of the song, in ticks."""
        if self.header.version in (1, 2):
            # Length isn't correct in version 1 and 2 songs, so we need this workaround
            length = max(note.tick for note in self.notes)
        else:
            length = self.header.song_length
        return length

    def __getitem__(self, key: int | slice) -> list[Note]:
        """Return the notes in a certain section (vertical slice) of the song."""
        if isinstance(key, int):
            section = [note for note in self.notes if note.tick == key]
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            section = [
                note for note in self.notes if note.tick > start and note.tick < stop
            ]
        else:
            raise TypeError("Index must be an integer")
        return list(section)

    @property
    def duration(self) -> int:
        """The duration of the song, in seconds."""
        return self._duration

    @duration.getter
    def duration(self) -> None:
        self._duration = len(self) / self.header.tempo * 1000

    @property
    def tempo_changer_ids(self) -> list[int]:
        """
        Return a list of all instruments which act as tempo changers.
        This is a hidden NBS feature.
        """
        return [
            ins.id + self.header.default_instruments
            for ins in self.instruments
            if ins.name == "Tempo Changer"
        ]

    @property
    def has_tempo_changers(self) -> bool:
        """Return true if this song contains any tempo changes."""
        tc_ids = self.tempo_changer_ids
        return tc_ids != [] and any(note.instrument in tc_ids for note in self.notes)

    @property
    def tempo_segments(self) -> list[float]:
        """
        Return a list with the same length as the number of ticks in the song,
        where each value is the point in milliseconds where that tick is played.
        """
        tc_ids = self.tempo_changer_ids
        tempo_change_blocks = [note for note in self.notes if note.instrument in tc_ids]
        tempo_change_blocks.sort(key=lambda x: x.tick)
        current_tick = 0
        current_tempo = self.header.tempo
        tempo_segments = []
        millis = 0
        for note in tempo_change_blocks:
            # Edge case: if there are multiple tempo changers in the same tick,
            # the following will be a no-op, so only the first is considered
            for tick in range(current_tick, note.tick):
                millis += 1 / current_tempo * 1000
                tempo_segments.append(millis)
            current_tick = note.tick
            current_tempo = (
                note.pitch / 15
            )  # The note pitch is the new BPM of the song (t/s = BPM / 15)

        # Fill the remainder of the song (after the last tempo changer)
        for tick in range(current_tick, len(self) + 1):
            millis += 1 / current_tempo * 1000
            tempo_segments.append(millis)

        return tempo_segments

    def get_layer(self, id: int) -> pynbs.Layer:
        """Return the layer with the given ID."""
        # The song may not have a layer object for every layer with a note block
        try:
            return self.layers[id]
        except IndexError:
            return pynbs.Layer(id=id)

    def weighted_notes(self) -> Iterator[Note]:
        """Return all notes in this song with their layer velocity and panning applied."""
        for note in self.notes:
            layer = self.get_layer(note.layer)
            custom_instrument_id = note.instrument - self.header.default_instruments
            if custom_instrument_id >= 0:
                instrument = self.instruments[custom_instrument_id]
            else:
                instrument = None
            yield note.apply_layer_weight(layer, instrument)

    def layer_groups(self) -> dict[str, pynbs.Layer]:
        """Return a dict containing each unique layer name in this song and a list
        of all layers with that name."""
        groups = {}
        for layer in self.layers:
            name = layer.name
            if name not in groups:
                groups[name] = []
            else:
                groups[name].append(layer.id)
        return groups

    def notes_by_layer(self, group_by_name: bool = False) -> dict[str, list[Note]]:
        """Return a dict of lists containing the weighted notes in each non-empty layer of the
        song. If `group_by_name` is true, notes in layers with identical names will be grouped.
        """
        groups = {}
        for note in self.weighted_notes():
            layer = self.get_layer(note.layer)
            group_name = layer.name if group_by_name else f"{layer.id} {layer.name}"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(note)
        return groups

    def loop(self, count: int, start: int | None = None) -> Song:
        """Return this song looped `count` times with an optional loop start tick (`start`).
        If `start` is not provided, defaults to the start tick defined in the song)."""
        if start is None:
            start = int(self.header.loop_start_tick)
        notes = self[start:]
        new_song = self
        for i in range(1, count):
            offset = (len(self) - start) * i
            notes = (note.move(offset) for note in self.notes)
            new_song.notes.extend(notes)
        return new_song

    def get_locked_layers(self) -> list[int]:
        """Return a list of the layer IDs of all locked layers in the song."""
        return [layer.id for layer in self.layers if layer.lock]

    def get_unlocked_notes(self) -> Iterator[Note]:
        """Return all notes in this song whose layers are not locked."""
        locked_layers = self.get_locked_layers()
        return (
            note for note in self.weighted_notes() if note.layer not in locked_layers
        )

    def sorted_notes(self) -> list[Note]:
        """Return the notes in this song sorted by pitch, instrument, velocity, and
        panning."""
        return sorted_notes(self.notes)
