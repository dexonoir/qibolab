"""PulseSequence class."""

from collections import UserList
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import core_schema

from qibolab.components import ChannelId

from .pulses import Align, Delay, Pulse, PulseLike

__all__ = ["PulseSequence"]

_Element = tuple[ChannelId, PulseLike]


def _synchronize(sequence: "PulseSequence", channels: Iterable[ChannelId]) -> None:
    """Helper for ``concatenate`` and ``align_to_delays``.

    Modifies given ``sequence`` in-place!
    """
    durations = {ch: sequence.channel_duration(ch) for ch in channels}
    max_duration = max(durations.values(), default=0.0)
    for ch, duration in durations.items():
        delay = max_duration - duration
        if delay > 0:
            sequence.append((ch, Delay(duration=delay)))


class PulseSequence(UserList[_Element]):
    """Synchronized sequence of control instructions across multiple channels.

    The sequence is a linear stream of instructions, which may be
    executed in parallel over multiple channels.

    Each instruction is composed by the pulse-like object representing
    the action, and the channel on which it should be performed.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        schema = handler(list[_Element])
        return core_schema.no_info_after_validator_function(
            cls._validate,
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=False
            ),
        )

    @classmethod
    def _validate(cls, value):
        return cls(value)

    @staticmethod
    def _serialize(value):
        return TypeAdapter(list[_Element]).dump_python(list(value))

    @property
    def duration(self) -> float:
        """Duration of the entire sequence."""
        return max((self.channel_duration(ch) for ch in self.channels), default=0.0)

    @property
    def channels(self) -> set[ChannelId]:
        """Channels involved in the sequence."""
        return {ch for (ch, _) in self}

    def channel(self, channel: ChannelId) -> Iterable[PulseLike]:
        """Isolate pulses on a given channel."""
        return (pulse for (ch, pulse) in self if ch == channel)

    def channel_duration(self, channel: ChannelId) -> float:
        """Duration of the given channel."""
        return sum(pulse.duration for pulse in self.channel(channel))

    def pulse_channels(self, pulse_id: int) -> ChannelId:
        """Find channels on which a pulse with a given id plays."""
        channels = [channel for channel, pulse in self if pulse.id == pulse_id]
        if len(channels) == 0:
            raise ValueError(
                f"Pulse with id {pulse_id} does not exist in the sequence."
            )
        return channels

    def concatenate(self, other: "PulseSequence") -> None:
        """Juxtapose two sequences.

        Appends ``other`` in-place such that the result is:
            - ``self``
            - necessary delays to synchronize channels
            - ``other``
        """
        _synchronize(self, other.channels)
        self.extend(other)

    def align(self, channels: list[ChannelId]) -> Align:
        """Introduce align commands to the sequence."""
        align = Align()
        for channel in channels:
            self.append((channel, align))
        return align

    def align_to_delays(self) -> "PulseSequence":
        """Compile align commands to delays."""

        # keep track of ``Align`` command that were already played
        # because the same ``Align`` will appear on multiple channels
        # in the sequence
        processed_aligns = set()

        new = type(self)()
        for channel, pulse in self:
            if isinstance(pulse, Align):
                if pulse.id not in processed_aligns:
                    channels = self.pulse_channels(pulse.id)
                    _synchronize(new, channels)
                    processed_aligns.add(pulse.id)
            else:
                new.append((channel, pulse))
        return new

    def trim(self) -> "PulseSequence":
        """Drop final delays.

        The operation is not in place, and does not modify the original
        sequence.
        """
        terminated = set()
        new = []
        for ch, pulse in reversed(self):
            if ch not in terminated:
                if isinstance(pulse, Delay):
                    continue
                terminated.add(ch)
            new.append((ch, pulse))
        return type(self)(reversed(new))

    @property
    def probe_pulses(self) -> list[Pulse]:
        """Return list of the readout pulses in this sequence."""
        # pulse filter needed to exclude delays
        return [
            pulse for (ch, pulse) in self if isinstance(pulse, Pulse) if "probe" in ch
        ]
