"""Qibolab driver for Keysight QCS instrument set."""

import keysight.qcs as qcs  # pylint: disable=E0401

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.instruments.abstract import Controller
from qibolab.instruments.port import Port
from qibolab.pulses import (
    Drag,
    Envelope,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseType,
    Rectangular,
)
from qibolab.qubits import QubitId
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedRawWaveformResults,
    AveragedSampleResults,
    IntegratedResults,
    RawWaveformResults,
    SampleResults,
)
from qibolab.sweeper import Parameter, Sweeper


class QCSPort(Port):
    output_channel: qcs.Channels
    input_channel: qcs.Channels = None


def generate_envelope_qcs(shape: Envelope):
    """Converts a Qibolab pulse envelope to a QCS Envelope object."""
    if isinstance(shape, Rectangular):
        return qcs.ConstantEnvelope()

    elif isinstance(shape, (Gaussian, Drag)):
        return qcs.GaussianEnvelope(shape.rel_sigma)

    else:
        raise Exception("QCS pulse shape not supported")


class KeysightQCS(Controller):
    """Interaction for interacting with QCS main server."""

    PortType = QCSPort

    def __init__(
        self,
        name,
        address,
        channel_mapper: qcs.ChannelMapper,
        qubit_readout_channels: dict[QubitId, qcs.Channels],
    ):
        super().__init__(name, address)
        self.mapper = channel_mapper
        self.qubit_readout_map = qubit_readout_channels

    def connect(self):
        if not self.is_connected:
            self.backend = qcs.HclBackend(
                self.mapper, hw_demod=True, address=self.address
            )
            self.backend.is_system_ready()
            self.is_connected = True

    def play(
        self, sequence: PulseSequence, options: ExecutionParameters, *sweepers: Sweeper
    ):

        program = qcs.Program()

        # Sweeper management
        scount = 0
        sweeper_pulse_map: dict[Pulse, tuple[str, qcs.Scalar]] = {}
        for sweeper in sweepers:
            if sweeper.parameter in [
                Parameter.attenuation,
                Parameter.bias,
                Parameter.gain,
                Parameter.lo_frequency,
            ]:
                raise ValueError("Sweeper parameter not supported")

            sweeper_arrays = []
            sweeper_variables = []

            for pulse in sweeper.pulses:
                sweeper_name = f"s{scount}"
                parameter = sweeper.parameter.name
                qcs_var = qcs.Scalar(sweeper_name, dtype=float)
                sweeper_variables.append(qcs_var)
                sweeper_arrays.append(
                    qcs.Array(
                        sweeper_name,
                        value=sweeper.get_values(
                            getattr(pulse, parameter), dtype=float
                        ),
                    )
                )
                sweeper_pulse_map[pulse] = (parameter, qcs_var)
                scount += 1

            # For the same sweeper, the variables can be swept simultaneously
            program.sweep(sweeper_arrays, sweeper_variables)

        # Map of virtual Z rotations to qubits for phase tracking
        vz_map = {}

        # Iterate over the pulses in the sequence and add them to the program in order
        for pulse in sequence:
            envelope = generate_envelope_qcs(pulse.envelope)

            if pulse.type is PulseType.FLUX or pulse.type is PulseType.COUPLERFLUX:
                qcs_pulse = qcs.DCWaveform(
                    duration=pulse.duration * 1e-9,
                    envelope=envelope,
                    amplitude=pulse.amplitude,
                )
            elif pulse.type is PulseType.DELAY:
                qcs_pulse = qcs.Delay(pulse.duration * 1e-9)
            elif pulse.type is PulseType.VIRTUALZ:
                # While QCS supports a PhaseIncrement instruction, in our case,
                # the phase is relative to the qubit and not the channel
                vz_map[pulse.qubit] = vz_map.get(pulse.qubit, 0) + pulse.phase
                continue
            else:
                qcs_pulse = qcs.RFWaveform(
                    duration=pulse.duration * 1e-9,
                    envelope=envelope,
                    amplitude=pulse.amplitude,
                    frequency=pulse.frequency,
                    instantaneous_phase=pulse.relative_phase
                    + vz_map.get(pulse.qubit, 0),
                )
                if pulse.type is PulseType.READOUT:
                    program.add_acquisition(
                        pulse.duration * 1e-9,
                        self.ports(pulse.channel).input_channel,
                    )

                if isinstance(pulse.envelope, Drag):
                    qcs_pulse = qcs_pulse.drag(pulse.shape.beta)

            # If this pulse is part of a sweeper, set the variable
            if pulse in sweeper_pulse_map:
                parameter, qcs_var = sweeper_pulse_map[pulse]
                setattr(qcs_pulse, parameter, qcs_var)
            program.add_waveform(qcs_pulse, self.ports(pulse.channel).output_channel)

        # Set the number of shots
        program.n_shots(options.nshots)
        # Run the program on the backend
        self.backend.apply(program)

        qubits_to_measure = [pulse.qubit for pulse in sequence.ro_pulses]
        results = {}
        for qubit in qubits_to_measure:
            if options.acquisition_type is AcquisitionType.RAW:
                res = program.get_trace(
                    channels=self.qubit_readout_map[qubit],
                    avg=options.averaging_mode is not AveragingMode.SINGLESHOT,
                )
                results[qubit] = (
                    RawWaveformResults(res)
                    if AveragingMode.SINGLESHOT
                    else AveragedRawWaveformResults(res)
                )

            elif options.acquisition_type is AcquisitionType.INTEGRATION:
                res = program.get_iq(
                    channels=self.qubit_readout_map[qubit],
                    avg=options.averaging_mode is not AveragingMode.SINGLESHOT,
                )
                results[qubit] = (
                    IntegratedResults(res)
                    if AveragingMode.SINGLESHOT
                    else AveragedIntegratedResults(res)
                )

            elif options.acquisition_type is AcquisitionType.DISCRIMINATION:
                res = program.get_classified(
                    channels=self.qubit_readout_map[qubit],
                    avg=options.averaging_mode is not AveragingMode.SINGLESHOT,
                )
                results[qubit] = (
                    SampleResults(res)
                    if AveragingMode.SINGLESHOT
                    else AveragedSampleResults(res)
                )

        return results
