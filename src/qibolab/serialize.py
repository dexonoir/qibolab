"""Helper methods for loading and saving to runcards.

The format of runcards in the ``qiboteam/qibolab_platforms_qrc``
repository is assumed here. See :ref:`Using runcards <using_runcards>`
example for more details.
"""

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Tuple

from pydantic import ConfigDict, TypeAdapter

from qibolab.couplers import Coupler
from qibolab.kernels import Kernels
from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.platform.platform import (
    CouplerMap,
    InstrumentMap,
    Platform,
    QubitMap,
    QubitPairMap,
    Settings,
)
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses.pulse import PulseLike
from qibolab.qubits import Qubit, QubitId, QubitPair

RUNCARD = "parameters.json"
PLATFORM = "platform.py"


def load_runcard(path: Path) -> dict:
    """Load runcard JSON to a dictionary."""
    return json.loads((path / RUNCARD).read_text())


def load_settings(runcard: dict) -> Settings:
    """Load platform settings section from the runcard."""
    return Settings(**runcard["settings"])


def load_qubit_name(name: str) -> QubitId:
    """Convert qubit name from string to integer or string."""
    try:
        return int(name)
    except ValueError:
        return name


def load_qubits(
    runcard: dict, kernels: Kernels = None
) -> Tuple[QubitMap, CouplerMap, QubitPairMap]:
    """Load qubits and pairs from the runcard.

    Uses the native gate and characterization sections of the runcard to
    parse the
    :class: `qibolab.qubits.Qubit` and
    :class: `qibolab.qubits.QubitPair`
    objects.
    """
    qubits = {}
    for q, char in runcard["characterization"]["single_qubit"].items():
        raw_qubit = Qubit(load_qubit_name(q), **char)
        raw_qubit.crosstalk_matrix = {
            load_qubit_name(key): value
            for key, value in raw_qubit.crosstalk_matrix.items()
        }
        qubits[load_qubit_name(q)] = raw_qubit

    if kernels is not None:
        for q in kernels:
            qubits[q].kernel = kernels[q]

    couplers = {}
    pairs = {}
    two_qubit_characterization = runcard["characterization"].get("two_qubit", {})
    if "coupler" in runcard["characterization"]:
        couplers = {
            load_qubit_name(c): Coupler(load_qubit_name(c), **char)
            for c, char in runcard["characterization"]["coupler"].items()
        }
        for c, pair in runcard["topology"].items():
            q0, q1 = pair
            char = two_qubit_characterization.get(str(q0) + "-" + str(q1), {})
            pairs[(q0, q1)] = pairs[(q1, q0)] = QubitPair(
                qubits[q0], qubits[q1], **char, coupler=couplers[load_qubit_name(c)]
            )
    else:
        for pair in runcard["topology"]:
            q0, q1 = pair
            char = two_qubit_characterization.get(str(q0) + "-" + str(q1), {})
            pairs[(q0, q1)] = pairs[(q1, q0)] = QubitPair(
                qubits[q0], qubits[q1], **char, coupler=None
            )

    qubits, pairs, couplers = register_gates(runcard, qubits, pairs, couplers)

    return qubits, couplers, pairs


_PulseLike = TypeAdapter(PulseLike, config=ConfigDict(extra="ignore"))
"""Parse a pulse-like object.

.. note::

    Extra arguments are ignored, in order to standardize the qubit handling, since the
    :cls:`Delay` object has no `qubit` field.
    This will be removed once there won't be any need for dedicated couplers handling.
"""


def _load_pulse(pulse_kwargs: dict, qubit: Qubit):
    coupler = "coupler" in pulse_kwargs
    pulse_kwargs["qubit"] = pulse_kwargs.pop(
        "coupler" if coupler else "qubit", qubit.name
    )

    return _PulseLike.validate_python(pulse_kwargs)


def _load_single_qubit_natives(qubit, gates) -> SingleQubitNatives:
    """Parse native gates of the qubit from the runcard.

    Args:
        qubit (:class:`qibolab.qubits.Qubit`): Qubit object that the
            native gates are acting on.
        gates (dict): Dictionary with native gate pulse parameters as loaded
            from the runcard.
    """
    return SingleQubitNatives(
        **{name: _load_pulse(kwargs, qubit) for name, kwargs in gates.items()}
    )


def _load_two_qubit_natives(qubits, couplers, gates) -> TwoQubitNatives:
    sequences = {}
    for name, seq_kwargs in gates.items():
        if isinstance(seq_kwargs, dict):
            seq_kwargs = [seq_kwargs]

        sequence = PulseSequence()
        for kwargs in seq_kwargs:
            if "coupler" in kwargs:
                qubit = couplers[kwargs["coupler"]]
            else:
                qubit = qubits[kwargs["qubit"]]
            sequence.append(_load_pulse(kwargs, qubit))
        sequences[name] = sequence

    return TwoQubitNatives(**sequences)


def register_gates(
    runcard: dict, qubits: QubitMap, pairs: QubitPairMap, couplers: CouplerMap = None
) -> Tuple[QubitMap, QubitPairMap]:
    """Register single qubit native gates to ``Qubit`` objects from the
    runcard.

    Uses the native gate and characterization sections of the runcard
    to parse the :class:`qibolab.qubits.Qubit` and :class:`qibolab.qubits.QubitPair`
    objects.
    """

    native_gates = runcard.get("native_gates", {})
    for q, gates in native_gates.get("single_qubit", {}).items():
        qubit = qubits[load_qubit_name(q)]
        qubit.native_gates = _load_single_qubit_natives(qubit, gates)

    for c, gates in native_gates.get("coupler", {}).items():
        coupler = couplers[load_qubit_name(c)]
        coupler.native_gates = _load_single_qubit_natives(coupler, gates)

    # register two-qubit native gates to ``QubitPair`` objects
    for pair, gatedict in native_gates.get("two_qubit", {}).items():
        q0, q1 = tuple(int(q) if q.isdigit() else q for q in pair.split("-"))
        native_gates = _load_two_qubit_natives(qubits, couplers, gatedict)
        coupler = pairs[(q0, q1)].coupler
        pairs[(q0, q1)] = QubitPair(qubits[q0], qubits[q1], coupler, native_gates)
        if native_gates.symmetric:
            pairs[(q1, q0)] = pairs[(q0, q1)]

    return qubits, pairs, couplers


def load_instrument_settings(
    runcard: dict, instruments: InstrumentMap
) -> InstrumentMap:
    """Setup instruments according to the settings given in the runcard."""
    for name, settings in runcard.get("instruments", {}).items():
        instruments[name].setup(**settings)
    return instruments


def dump_qubit_name(name: QubitId) -> str:
    """Convert qubit name from integer or string to string."""
    if isinstance(name, int):
        return str(name)
    return name


def _dump_pulse(pulse: Pulse):
    data = pulse.model_dump()
    data["type"] = data["type"].value
    if "channel" in data:
        del data["channel"]
    if "relative_phase" in data:
        del data["relative_phase"]
    return data


def _dump_single_qubit_natives(natives: SingleQubitNatives):
    data = {}
    for fld in fields(natives):
        pulse = getattr(natives, fld.name)
        if pulse is not None:
            data[fld.name] = _dump_pulse(pulse)
            del data[fld.name]["qubit"]
    return data


def _dump_two_qubit_natives(natives: TwoQubitNatives):
    data = {}
    for fld in fields(natives):
        sequence = getattr(natives, fld.name)
        if len(sequence) > 0:
            data[fld.name] = []
            for pulse in sequence:
                pulse_serial = _dump_pulse(pulse)
                if pulse.type == PulseType.COUPLERFLUX:
                    pulse_serial["coupler"] = pulse_serial.pop("qubit")
                data[fld.name].append(pulse_serial)
    return data


def dump_native_gates(
    qubits: QubitMap, pairs: QubitPairMap, couplers: CouplerMap = None
) -> dict:
    """Dump native gates section to dictionary following the runcard format,
    using qubit and pair objects."""
    # single-qubit native gates
    native_gates = {
        "single_qubit": {
            dump_qubit_name(q): _dump_single_qubit_natives(qubit.native_gates)
            for q, qubit in qubits.items()
        }
    }

    if couplers:
        native_gates["coupler"] = {
            dump_qubit_name(c): _dump_single_qubit_natives(coupler.native_gates)
            for c, coupler in couplers.items()
        }

    # two-qubit native gates
    native_gates["two_qubit"] = {}
    for pair in pairs.values():
        natives = _dump_two_qubit_natives(pair.native_gates)
        if len(natives) > 0:
            pair_name = f"{pair.qubit1.name}-{pair.qubit2.name}"
            native_gates["two_qubit"][pair_name] = natives

    return native_gates


def dump_characterization(
    qubits: QubitMap, pairs: QubitPairMap = None, couplers: CouplerMap = None
) -> dict:
    """Dump qubit characterization section to dictionary following the runcard
    format, using qubit and pair objects."""
    single_qubit = {}
    for q, qubit in qubits.items():
        char = qubit.characterization
        char["crosstalk_matrix"] = {
            dump_qubit_name(q): c for q, c in qubit.crosstalk_matrix.items()
        }
        single_qubit[dump_qubit_name(q)] = char

    characterization = {"single_qubit": single_qubit}
    if len(pairs) > 0:
        characterization["two_qubit"] = {
            f"{q1}-{q2}": pair.characterization for (q1, q2), pair in pairs.items()
        }

    if couplers:
        characterization["coupler"] = {
            dump_qubit_name(c.name): {"sweetspot": c.sweetspot}
            for c in couplers.values()
        }
    return characterization


def dump_instruments(instruments: InstrumentMap) -> dict:
    """Dump instrument settings to a dictionary following the runcard
    format."""
    # Qblox modules settings are dictionaries and not dataclasses
    data = {}
    for name, instrument in instruments.items():
        try:
            # TODO: Migrate all instruments to this approach
            # (I think it is also useful for qblox)
            settings = instrument.dump()
            if len(settings) > 0:
                data[name] = settings
        except AttributeError:
            settings = instrument.settings
            if settings is not None:
                if isinstance(settings, dict):
                    data[name] = settings
                else:
                    data[name] = settings.dump()

    return data


def dump_runcard(platform: Platform, path: Path):
    """Serializes the platform and saves it as a json runcard file.

    The file saved follows the format explained in :ref:`Using runcards <using_runcards>`.

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path that the json file will be saved.
    """

    settings = {
        "nqubits": platform.nqubits,
        "settings": asdict(platform.settings),
        "qubits": list(platform.qubits),
        "topology": [list(pair) for pair in platform.ordered_pairs],
        "instruments": dump_instruments(platform.instruments),
    }

    if platform.couplers:
        settings["couplers"] = list(platform.couplers)
        settings["topology"] = {
            platform.pairs[pair].coupler.name: list(pair)
            for pair in platform.ordered_pairs
        }

    settings["native_gates"] = dump_native_gates(
        platform.qubits, platform.pairs, platform.couplers
    )

    settings["characterization"] = dump_characterization(
        platform.qubits, platform.pairs, platform.couplers
    )

    (path / RUNCARD).write_text(json.dumps(settings, sort_keys=False, indent=4))


def dump_kernels(platform: Platform, path: Path):
    """Creates Kernels instance from platform and dumps as npz.

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path that the kernels file will be saved.
    """

    # create kernels
    kernels = Kernels()
    for qubit in platform.qubits.values():
        if qubit.kernel is not None:
            kernels[qubit.name] = qubit.kernel

    # dump only if not None
    if kernels:
        kernels.dump(path)


def dump_platform(platform: Platform, path: Path):
    """Platform serialization as runcard (json) and kernels (npz).

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path where json and npz will be dumped.
    """

    dump_kernels(platform=platform, path=path)
    dump_runcard(platform=platform, path=path)
