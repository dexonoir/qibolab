"""A platform for executing quantum algorithms."""

import copy
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from qibo.config import log, raise_error

from qibolab.components import Config
from qibolab.couplers import Coupler
from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.abstract import Controller, Instrument, InstrumentId
from qibolab.pulses import Delay, PulseSequence
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId
from qibolab.serialize_ import replace
from qibolab.sweeper import Sweeper
from qibolab.unrolling import batch

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
    for sequence in sequences:
        total_sequence.extend(sequence)
        # TODO: Fix unrolling results
        for pulse in sequence.ro_pulses:
            readout_map[pulse.id].append(pulse.id)

        length = sequence.duration + relaxation_time
        for channel in sequence.keys():
            delay = length - sequence.channel_duration(channel)
            total_sequence[channel].append(Delay(duration=delay))

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
    component_configs: dict[str, Config]
    """Maps name of component to its default config."""
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

    @property
    def components(self) -> set[str]:
        """Names of all components available in the platform."""
        return set(self.component_configs.keys())

    def config(self, name: str) -> Config:
        """Returns configuration of given component."""
        return self.component_configs[name]

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

    def _apply_config_updates(
        self, updates: list[dict[str, Config]]
    ) -> dict[str, Config]:
        """Apply given list of config updates to the default configuration and
        return the updated config dict.

        Args:
            updates: list of updates, where each entry is a dict mapping component name to new config. Later entries
                     in the list override earlier entries (if they happen to update the same thing).
        """
        components = self.component_configs.copy()
        for update in updates:
            for name, cfg in update.items():
                if name not in components:
                    raise ValueError(
                        f"Cannot update configuration for unknown component {name}"
                    )
                if type(cfg) is not type(components[name]):
                    raise ValueError(
                        f"Configuration of component {name} with type {type(components[name])} cannot be updated with configuration of type {type(cfg)}"
                    )
                components[name] = cfg
        return components

    def _execute(self, sequences, options, **kwargs):
        """Executes sequence on the controllers."""
        result = {}

        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.play(
                    options.component_configs, sequences, options, {}
                )
                if isinstance(new_result, dict):
                    result.update(new_result)

        return result

    def execute_pulse_sequence(
        self,
        sequence: PulseSequence,
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

        return self._execute([sequence], options, **kwargs)

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
        options: ExecutionParameters,
        **kwargs,
    ):
        """
        Args:
            sequences: Pulse sequences to execute.
            options: Object holding the execution options.
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

        results = defaultdict(list)
        bounds = kwargs.get("bounds", self._controller.bounds)
        for seq_batch in batch(sequences, bounds):
            result = self._execute(seq_batch, options, **kwargs)
            for serial, data in result.items():
                results[serial].append(data)

        return results

    def sweep(
        self,
        sequence: PulseSequence,
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
                qubit = platform.qubits[0]
                sequence = qubit.native_gates.MZ.create_sequence()
                parameter = Parameter.frequency
                parameter_range = np.random.randint(10, size=10)
                sweeper = Sweeper(parameter, parameter_range, channels=[qubit.measure.name])
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

        configs = self._apply_config_updates(options.component_configs)

        # for components that represent aux external instruments (e.g. lo) to the main control instrument
        # set the config directly
        for name, cfg in configs.items():
            if name in self.instruments:
                self.instruments[name].setup(**asdict(cfg))

        # maps acquisition channel name to corresponding kernel and iq_angle
        # FIXME: this is temporary solution to deliver the information to drivers
        # until we make acquisition channels first class citizens in the sequences
        # so that each acquisition command carries the info with it.
        integration_setup: dict[str, tuple[np.ndarray, float]] = {}
        for qubit in self.qubits.values():
            integration_setup[qubit.acquisition.name] = (qubit.kernel, qubit.iq_angle)

        result = {}
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.sweep(
                    configs,
                    [sequence],
                    options,
                    integration_setup,
                    *sweepers,
                )
                if isinstance(new_result, dict):
                    result.update(new_result)
        return result

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
