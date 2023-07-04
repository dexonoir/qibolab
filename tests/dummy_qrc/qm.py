import pathlib

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.oscillator import LocalOscillator
from qibolab.instruments.qm import QMSim
from qibolab.platform import Platform

RUNCARD = pathlib.Path(__file__).parent / "qm.yml"


def create(runcard=RUNCARD):
    """Dummy platform using Quantum Machines (QM) OPXs and Rohde Schwarz local oscillators.

    Used in ``test_instruments_qm.py`` and ``test_instruments_qmsim.py``
    """
    controller = QMSim("qmopx", "0.0.0.0:0", simulation_duration=1000, cloud=False, time_of_flight=280)

    # Create channel objects and map controllers to channels
    channels = ChannelMap()
    # readout
    channels |= Channel("L3-25_a", port=controller[(("con1", 10), ("con1", 9))])
    channels |= Channel("L3-25_b", port=controller[(("con2", 10), ("con2", 9))])
    # feedback
    channels |= Channel("L2-5_a", port=controller[(("con1", 2), ("con1", 1))])
    channels |= Channel("L2-5_b", port=controller[(("con2", 2), ("con2", 1))])
    # drive
    channels |= (Channel(f"L3-1{i}", port=controller[(("con1", 2 * i), ("con1", 2 * i - 1))]) for i in range(1, 5))
    channels |= Channel("L3-15", port=controller[(("con3", 2), ("con3", 1))])
    # flux
    channels |= (Channel(f"L4-{i}", port=controller[(("con2", i),)]) for i in range(1, 6))
    # TWPA
    channels |= "L4-26"

    # Instantiate local oscillators (HARDCODED)
    local_oscillators = [
        LocalOscillator("lo_readout_a", "192.168.0.39"),
        LocalOscillator("lo_readout_b", "192.168.0.31"),
        LocalOscillator("lo_drive_low", "192.168.0.32"),
        LocalOscillator("lo_drive_mid", "192.168.0.33"),
        LocalOscillator("lo_drive_high", "192.168.0.34"),
        LocalOscillator("twpa_a", "192.168.0.35"),
    ]
    # Set LO parameters
    local_oscillators[0].frequency = 7_300_000_000
    local_oscillators[1].frequency = 7_900_000_000
    local_oscillators[2].frequency = 4_700_000_000
    local_oscillators[3].frequency = 5_600_000_000
    local_oscillators[4].frequency = 6_500_000_000
    local_oscillators[0].power = 18.0
    local_oscillators[1].power = 15.0
    for i in range(2, 5):
        local_oscillators[i].power = 16.0
    # Set TWPA parameters
    local_oscillators[5].frequency = 6_511_000_000
    local_oscillators[5].power = 4.5
    # Map LOs to channels
    channels["L3-25_a"].local_oscillator = local_oscillators[0]
    channels["L3-25_b"].local_oscillator = local_oscillators[1]
    channels["L3-15"].local_oscillator = local_oscillators[2]
    channels["L3-11"].local_oscillator = local_oscillators[2]
    channels["L3-12"].local_oscillator = local_oscillators[3]
    channels["L3-13"].local_oscillator = local_oscillators[4]
    channels["L3-14"].local_oscillator = local_oscillators[4]
    channels["L4-26"].local_oscillator = local_oscillators[5]

    instruments = [controller] + local_oscillators
    platform = Platform("qw5q_gold", runcard, instruments, channels)

    # assign channels to qubits
    qubits = platform.qubits
    for q in [0, 1, 5]:
        qubits[q].readout = channels["L3-25_a"]
        qubits[q].feedback = channels["L2-5_a"]
    for q in [2, 3, 4]:
        qubits[q].readout = channels["L3-25_b"]
        qubits[q].feedback = channels["L2-5_b"]

    qubits[0].drive = channels["L3-15"]
    qubits[0].flux = channels["L4-5"]
    for q in range(1, 5):
        qubits[q].drive = channels[f"L3-{10 + q}"]
        qubits[q].flux = channels[f"L4-{q}"]

    # set filter for flux channel
    qubits[2].flux.filters = {"feedforward": [1.0684635881381783, -1.0163217174522334], "feedback": [0.947858129314055]}

    # set maximum allowed bias values to protect amplifier
    # relevant only for qubits where an amplifier is used
    for q in range(5):
        platform.qubits[q].flux.max_bias = 0.2

    return platform
