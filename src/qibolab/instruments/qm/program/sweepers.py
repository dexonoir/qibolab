import math
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibo.config import raise_error
from qm import qua
from qm.qua import declare, fixed
from qm.qua._dsl import _Variable  # for type declaration only

from qibolab.components import Channel, Config
from qibolab.pulses import Pulse
from qibolab.sweeper import Parameter

from ..config import operation
from .arguments import ExecutionArguments

MAX_OFFSET = 0.5
"""Maximum voltage supported by Quantum Machines OPX+ instrument in volts."""


def maximum_sweep_value(values: npt.NDArray, value0: npt.NDArray) -> float:
    """Calculates maximum value that is reached during a sweep.

    Useful to check whether a sweep exceeds the range of allowed values.
    Note that both the array of values we sweep and the center value can
    be negative, so we need to make sure that the maximum absolute value
    is within range.

    Args:
        values (np.ndarray): Array of values we will sweep over.
        value0 (float, int): Center value of the sweep.
    """
    return max(abs(min(values) + value0), abs(max(values) + value0))


def check_max_offset(offset: Optional[float], max_offset: float = MAX_OFFSET):
    """Checks if a given offset value exceeds the maximum supported offset.

    This is to avoid sending high currents that could damage lab
    equipment such as amplifiers.
    """
    if max_offset is not None and abs(offset) > max_offset:
        raise_error(
            ValueError, f"{offset} exceeds the maximum allowed offset {max_offset}."
        )


def _frequency(
    channels: list[Channel],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for channel in channels:
        name = str(channel.name)
        lo_frequency = configs[channel.lo].frequency
        # convert to IF frequency for readout and drive pulses
        f0 = math.floor(configs[name].frequency - lo_frequency)
        # check if sweep is within the supported bandwidth [-400, 400] MHz
        max_freq = maximum_sweep_value(values, f0)
        if max_freq > 4e8:
            raise_error(
                ValueError,
                f"Frequency {max_freq} for channel {name} is beyond instrument bandwidth.",
            )
        qua.update_frequency(name, variable + f0)


def _amplitude(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    # TODO: Consider sweeping amplitude without multiplication
    if min(values) < -2:
        raise_error(
            ValueError, "Amplitude sweep values are <-2 which is not supported."
        )
    if max(values) > 2:
        raise_error(ValueError, "Amplitude sweep values are >2 which is not supported.")

    for pulse in pulses:
        args.parameters[operation(pulse)].amplitude = qua.amp(variable)


def _relative_phase(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for pulse in pulses:
        args.parameters[operation(pulse)].phase = variable


def _bias(
    channels: list[Channel],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for channel in channels:
        name = str(channel.name)
        offset = configs[name].offset
        max_value = maximum_sweep_value(values, offset)
        check_max_offset(max_value, MAX_OFFSET)
        b0 = declare(fixed, value=offset)
        with qua.if_((variable + b0) >= 0.49):
            qua.set_dc_offset(f"flux{name}", "single", 0.49)
        with qua.elif_((variable + b0) <= -0.49):
            qua.set_dc_offset(f"flux{name}", "single", -0.49)
        with qua.else_():
            qua.set_dc_offset(f"flux{name}", "single", (variable + b0))


def _duration(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for pulse in pulses:
        args.parameters[operation(pulse)].duration = variable


def _duration_interpolated(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for pulse in pulses:
        params = args.parameters[operation(pulse)]
        params.duration = variable
        params.interpolated = True


def normalize_phase(values):
    """Normalize phase from [0, 2pi] to [0, 1]."""
    return values / (2 * np.pi)


def normalize_duration(values):
    """Convert duration from ns to clock cycles (clock cycle = 4ns)."""
    if not all(values % 4 == 0):
        raise ValueError(
            "Cannot use interpolated duration sweeper for durations that are not multiple of 4ns. Please use normal duration sweeper."
        )
    return (values // 4).astype(int)


INT_TYPE = {Parameter.frequency, Parameter.duration, Parameter.duration_interpolated}
"""Sweeper parameters for which we need ``int`` variable type.

The rest parameters need ``fixed`` type.
"""

NORMALIZERS = {
    Parameter.relative_phase: normalize_phase,
    Parameter.duration_interpolated: normalize_duration,
}
"""Functions to normalize sweeper values.

The rest parameters do not need normalization (identity function).
"""

SWEEPER_METHODS = {
    Parameter.frequency: _frequency,
    Parameter.amplitude: _amplitude,
    Parameter.duration: _duration,
    Parameter.duration_interpolated: _duration_interpolated,
    Parameter.relative_phase: _relative_phase,
    Parameter.bias: _bias,
}
"""Methods that return part of QUA program to be used inside the loop."""
