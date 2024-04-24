"""A platform for executing quantum algorithms."""

from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import networkx as nx
from qibo.config import log, raise_error

from .channel_config import ChannelConfig
from .couplers import Coupler
from .execution_parameters import ExecutionParameters
from .instruments.abstract import Controller, Instrument, InstrumentId
from .pulses import Delay, Drag, PulseSequence, PulseType
from .qubits import Qubit, QubitId, QubitPair, QubitPairId
from .serialize_ import replace
from .sweeper import Sweeper
from .unrolling import batch

InstrumentMap = Dict[InstrumentId, Instrument]
QubitMap = Dict[QubitId, Qubit]
CouplerMap = Dict[QubitId, Coupler]
QubitPairMap = Dict[QubitPairId, QubitPair]

NS_TO_SEC = 1e-9


def unroll_sequences(
    sequences: List[PulseSequence], relaxation_time: int
) -> Tuple[PulseSequence, dict[str, list[str]]]:
    """Unrolls a list of pulse sequences to a single pulse sequence with
    multiple measurements.

    Args:
        sequences (list): List of pulse sequences to unroll.
        relaxation_time (int): Time in ns to wait for the qubit to relax between
            playing different sequences.

    Returns:
        total_sequence (:class:`qibolab.pulses.PulseSequence`): Unrolled pulse sequence containing
            multiple measurements.
        readout_map (dict): Map from original readout pulse serials to the unrolled readout pulse
            serials. Required to construct the results dictionary that is returned after execution.
    """
    total_sequence = PulseSequence()
    readout_map = defaultdict(list)
    channels = {pulse.channel for sequence in sequences for pulse in sequence}
    for sequence in sequences:
        total_sequence.extend(sequence)
        # TODO: Fix unrolling results
        for pulse in sequence:
            if pulse.type is PulseType.READOUT:
                readout_map[pulse.id].append(pulse.id)

        length = sequence.duration + relaxation_time
        pulses_per_channel = sequence.pulses_per_channel
        for channel in channels:
            delay = length - pulses_per_channel[channel].duration
            total_sequence.append(Delay(duration=delay, channel=channel))

    return total_sequence, readout_map


@dataclass
class Settings:
    """Default execution settings read from the runcard."""

    nshots: int = 1024
    """Default number of repetitions when executing a pulse sequence."""
    relaxation_time: int = int(1e5)
    """Time in ns to wait for the qubit to relax to its ground state between
    shots."""

    def fill(self, options: ExecutionParameters):
        """Use default values for missing execution options."""
        if options.nshots is None:
            options = replace(options, nshots=self.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.relaxation_time)

        return options


@dataclass
class Platform:
    """Platform for controlling quantum devices."""

    name: str
    """Name of the platform."""
    qubits: QubitMap
    """Dictionary mapping qubit names to :class:`qibolab.qubits.Qubit`
    objects."""
    pairs: QubitPairMap
    """Dictionary mapping tuples of qubit names to
    :class:`qibolab.qubits.QubitPair` objects."""
    instruments: InstrumentMap
    """Dictionary mapping instrument names to
    :class:`qibolab.instruments.abstract.Instrument` objects."""

    settings: Settings = field(default_factory=Settings)
    """Container with default execution settings."""
    resonator_type: Optional[str] = None
    """Type of resonator (2D or 3D) in the used QPU.

    Default is 3D for single-qubit chips and 2D for multi-qubit.
    """

    couplers: CouplerMap = field(default_factory=dict)
    """Dictionary mapping coupler names to :class:`qibolab.couplers.Coupler`
    objects."""

    is_connected: bool = False
    """Flag for whether we are connected to the physical instruments."""

    topology: nx.Graph = field(default_factory=nx.Graph)
    """Graph representing the qubit connectivity in the quantum chip."""

    def __post_init__(self):
        log.info("Loading platform %s", self.name)
        if self.resonator_type is None:
            self.resonator_type = "3D" if self.nqubits == 1 else "2D"

        self.topology.add_nodes_from(self.qubits.keys())
        self.topology.add_edges_from(
            [(pair.qubit1.name, pair.qubit2.name) for pair in self.pairs.values()]
        )
        self._set_channels_to_single_qubit_gates()
        self._set_channels_to_two_qubit_gates()

    def _set_channels_to_single_qubit_gates(self):
        """Set channels to pulses that implement single-qubit gates.

        This function should be removed when the duplication caused by
        (``pulse.qubit``, ``pulse.type``) -> ``pulse.channel``
        is resolved. For now it just makes sure that the channels of
        native pulses are consistent in order to test the rest of the code.
        """
        for qubit in self.qubits.values():
            gates = qubit.native_gates
            for fld in fields(gates):
                pulse = getattr(gates, fld.name)
                if pulse is not None:
                    channel = getattr(qubit, pulse.type.name.lower()).name
                    setattr(gates, fld.name, replace(pulse, channel=channel))
        for coupler in self.couplers.values():
            if gates.CP is not None:
                gates.CP = replace(gates.CP, channel=coupler.flux.name)

    def _set_channels_to_two_qubit_gates(self):
        """Set channels to pulses that implement single-qubit gates.

        This function should be removed when the duplication caused by
        (``pulse.qubit``, ``pulse.type``) -> ``pulse.channel``
        is resolved. For now it just makes sure that the channels of
        native pulses are consistent in order to test the rest of the code.
        """
        for pair in self.pairs.values():
            gates = pair.native_gates
            for fld in fields(gates):
                sequence = getattr(gates, fld.name)
                if len(sequence) > 0:
                    new_sequence = PulseSequence()
                    for pulse in sequence:
                        if pulse.type is PulseType.VIRTUALZ:
                            channel = self.qubits[pulse.qubit].drive.name
                        elif pulse.type is PulseType.COUPLERFLUX:
                            channel = self.couplers[pulse.qubit].flux.name
                        else:
                            channel = getattr(
                                self.qubits[pulse.qubit], pulse.type.name.lower()
                            ).name
                        new_sequence.append(replace(pulse, channel=channel))
                    setattr(gates, fld.name, new_sequence)

    def __str__(self):
        return self.name

    @property
    def nqubits(self) -> int:
        """Total number of usable qubits in the QPU."""
        return len(self.qubits)

    @property
    def ordered_pairs(self):
        """List of qubit pairs that are connected in the QPU."""
        return sorted({tuple(sorted(pair)) for pair in self.pairs})

    @property
    def sampling_rate(self):
        """Sampling rate of control electronics in giga samples per second
        (GSps)."""
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                return instrument.sampling_rate

    def connect(self):
        """Connect to all instruments."""
        if not self.is_connected:
            for instrument in self.instruments.values():
                try:
                    log.info(f"Connecting to instrument {instrument}.")
                    instrument.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {instrument} instruments. Error captured: '{exception}'",
                    )
        self.is_connected = True

    def disconnect(self):
        """Disconnects from instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.disconnect()
        self.is_connected = False

    def _execute(self, sequence, options, **kwargs):
        """Executes sequence on the controllers."""
        result = {}

        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.play(
                    self.qubits, self.couplers, sequence, options
                )
                if isinstance(new_result, dict):
                    result.update(new_result)

        return result

    def execute_pulse_sequence(
        self,
        sequence: PulseSequence,
        channel_config: dict[str, ChannelConfig],
        options: ExecutionParameters,
        **kwargs,
    ):
        """
        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequences to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.
        """
        options = self.settings.fill(options)

        time = (
            (sequence.duration + options.relaxation_time) * options.nshots * NS_TO_SEC
        )
        log.info(f"Minimal execution time (sequence): {time}")

        return self._execute(sequence, options, **kwargs)

    @property
    def _controller(self):
        """Controller instrument used for splitting the unrolled sequences to
        batches.

        Used only by :meth:`qibolab.platform.Platform.execute_pulse_sequences` (unrolling).
        This method does not support platforms with more than one controller instruments.
        """
        controllers = [
            instr
            for instr in self.instruments.values()
            if isinstance(instr, Controller)
        ]
        assert len(controllers) == 1
        return controllers[0]

    def execute_pulse_sequences(
        self,
        sequences: List[PulseSequence],
        channel_config: dict[str, ChannelConfig],
        options: ExecutionParameters,
        **kwargs,
    ):
        """
        Args:
            sequence (List[:class:`qibolab.pulses.PulseSequence`]): Pulse sequences to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.
        """
        options = self.settings.fill(options)

        duration = sum(seq.duration for seq in sequences)
        time = (
            (duration + len(sequences) * options.relaxation_time)
            * options.nshots
            * NS_TO_SEC
        )
        log.info(f"Minimal execution time (unrolling): {time}")

        # find readout pulses
        ro_pulses = {
            pulse.id: pulse.qubit
            for sequence in sequences
            for pulse in sequence.ro_pulses
        }

        results = defaultdict(list)
        bounds = kwargs.get("bounds", self._controller.bounds)
        for b in batch(sequences, bounds):
            sequence, readouts = unroll_sequences(b, options.relaxation_time)
            result = self._execute(sequence, options, **kwargs)
            for serial, new_serials in readouts.items():
                results[serial].extend(result[ser] for ser in new_serials)

        for serial, qubit in ro_pulses.items():
            results[qubit] = results[serial]

        return results

    def sweep(
        self,
        sequence: PulseSequence,
        channel_config: dict[str, ChannelConfig],
        options: ExecutionParameters,
        *sweepers: Sweeper,
    ):
        """Executes a pulse sequence for different values of sweeped
        parameters.

        Useful for performing chip characterization.

        Example:
            .. testcode::

                import numpy as np
                from qibolab.dummy import create_dummy
                from qibolab.sweeper import Sweeper, Parameter
                from qibolab.pulses import PulseSequence
                from qibolab.execution_parameters import ExecutionParameters


                platform = create_dummy()
                sequence = PulseSequence()
                parameter = Parameter.frequency
                pulse = platform.create_qubit_readout_pulse(qubit=0)
                sequence.append(pulse)
                parameter_range = np.random.randint(10, size=10)
                sweeper = Sweeper(parameter, parameter_range, [pulse])
                platform.sweep(sequence, ExecutionParameters(), sweeper)

        Returns:
            Readout results acquired by after execution.
        """
        if options.nshots is None:
            options = replace(options, nshots=self.settings.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.settings.relaxation_time)

        time = (
            (sequence.duration + options.relaxation_time) * options.nshots * NS_TO_SEC
        )
        for sweep in sweepers:
            time *= len(sweep.values)
        log.info(f"Minimal execution time (sweep): {time}")

        result = {}
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.sweep(
                    self.qubits, self.couplers, sequence, options, *sweepers
                )
                if isinstance(new_result, dict):
                    result.update(new_result)
        return result

    def __call__(self, sequence, options):
        return self.execute_pulse_sequence(sequence, options)

    def get_qubit(self, qubit):
        """Return the name of the physical qubit corresponding to a logical
        qubit.

        Temporary fix for the compiler to work for platforms where the
        qubits are not named as 0, 1, 2, ...
        """
        try:
            return self.qubits[qubit]
        except KeyError:
            return list(self.qubits.values())[qubit]

    def get_coupler(self, coupler):
        """Return the name of the physical coupler corresponding to a logical
        coupler.

        Temporary fix for the compiler to work for platforms where the
        couplers are not named as 0, 1, 2, ...
        """
        try:
            return self.couplers[coupler]
        except KeyError:
            return list(self.couplers.values())[coupler]

    def create_RX90_pulse(self, qubit, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return replace(
            qubit.native_gates.RX90,
            relative_phase=relative_phase,
            channel=qubit.drive.name,
        )

    def create_RX_pulse(self, qubit, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return replace(
            qubit.native_gates.RX,
            relative_phase=relative_phase,
            channel=qubit.drive.name,
        )

    def create_RX12_pulse(self, qubit, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return replace(
            qubit.native_gates.RX12,
            relative_phase=relative_phase,
            channel=qubit.drive.name,
        )

    def create_CZ_pulse_sequence(self, qubits):
        pair = tuple(self.get_qubit(q).name for q in qubits)
        if pair not in self.pairs or len(self.pairs[pair].native_gates.CZ) == 0:
            raise_error(
                ValueError,
                f"Calibration for CZ gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.CZ

    def create_iSWAP_pulse_sequence(self, qubits):
        pair = tuple(self.get_qubit(q).name for q in qubits)
        if pair not in self.pairs or len(self.pairs[pair].native_gates.iSWAP) == 0:
            raise_error(
                ValueError,
                f"Calibration for iSWAP gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.iSWAP

    def create_CNOT_pulse_sequence(self, qubits):
        pair = tuple(self.get_qubit(q).name for q in qubits)
        if pair not in self.pairs or len(self.pairs[pair].native_gates.CNOT) == 0:
            raise_error(
                ValueError,
                f"Calibration for CNOT gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.CNOT

    def create_MZ_pulse(self, qubit):
        qubit = self.get_qubit(qubit)
        return replace(qubit.native_gates.MZ, channel=qubit.readout.name)

    def create_qubit_drive_pulse(self, qubit, duration, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return replace(
            qubit.native_gates.RX,
            duration=duration,
            relative_phase=relative_phase,
            channel=qubit.drive.name,
        )

    def create_qubit_readout_pulse(self, qubit):
        return self.create_MZ_pulse(qubit)

    def create_coupler_pulse(self, coupler, duration=None, amplitude=None):
        coupler = self.get_coupler(coupler)
        pulse = coupler.native_gates.CP
        if duration is not None:
            pulse = replace(pulse, duration=duration)
        if amplitude is not None:
            pulse = replace(pulse, amplitude=amplitude)
        return replace(pulse, channel=coupler.flux.name)

    # TODO Remove RX90_drag_pulse and RX_drag_pulse, replace them with create_qubit_drive_pulse
    # TODO Add RY90 and RY pulses

    def create_RX90_drag_pulse(self, qubit, beta, relative_phase=0):
        """Create native RX90 pulse with Drag shape."""
        qubit = self.get_qubit(qubit)
        pulse = qubit.native_gates.RX90
        return replace(
            pulse,
            relative_phase=relative_phase,
            envelope=Drag(rel_sigma=pulse.envelope.rel_sigma, beta=beta),
            channel=qubit.drive.name,
        )

    def create_RX_drag_pulse(self, qubit, beta, relative_phase=0):
        """Create native RX pulse with Drag shape."""
        qubit = self.get_qubit(qubit)
        pulse = qubit.native_gates.RX
        return replace(
            pulse,
            relative_phase=relative_phase,
            envelope=Drag(rel_sigma=pulse.envelope.rel_sigma, beta=beta),
            channel=qubit.drive.name,
        )
