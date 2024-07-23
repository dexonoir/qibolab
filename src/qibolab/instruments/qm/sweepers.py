import math

import numpy as np
from qibo.config import raise_error
from qm import qua
from qm.qua import declare, fixed, for_
from qualang_tools.loops import from_array

from qibolab.components import Config
from qibolab.sweeper import Sweeper

from .config import operation
from .program import ExecutionArguments, play

MAX_OFFSET = 0.5
"""Maximum voltage supported by Quantum Machines OPX+ instrument in volts."""


def maximum_sweep_value(values, value0):
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


def check_max_offset(offset, max_offset=MAX_OFFSET):
    """Checks if a given offset value exceeds the maximum supported offset.

    This is to avoid sending high currents that could damage lab
    equipment such as amplifiers.
    """
    if max_offset is not None and abs(offset) > max_offset:
        raise_error(
            ValueError, f"{offset} exceeds the maximum allowed offset {max_offset}."
        )


# def _update_baked_pulses(sweeper, qmsequence, config):
#    """Updates baked pulse if duration sweeper is used."""
#    qmpulse = qmsequence.pulse_to_qmpulse[sweeper.pulses[0].id]
#    is_baked = isinstance(qmpulse, BakedPulse)
#    for pulse in sweeper.pulses:
#        qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
#        if isinstance(qmpulse, BakedPulse):
#            if not is_baked:
#                raise_error(
#                    TypeError,
#                    "Duration sweeper cannot contain both baked and not baked pulses.",
#                )
#            values = np.array(sweeper.values).astype(int)
#            qmpulse.bake(config, values)


def sweep(
    sweepers: list[Sweeper], configs: dict[str, Config], args: ExecutionArguments
):
    """Public sweep function that is called by the driver."""
    # for sweeper in sweepers:
    #    if sweeper.parameter is Parameter.duration:
    #        _update_baked_pulses(sweeper, instructions, config)
    _sweep_recursion(sweepers, configs, args)


def _sweep_recursion(sweepers, configs, args):
    """Unrolls a list of qibolab sweepers to the corresponding QUA for loops
    using recursion."""
    if len(sweepers) > 0:
        parameter = sweepers[0].parameter.name
        func_name = f"_sweep_{parameter}"
        if func_name in globals():
            globals()[func_name](sweepers, configs, args)
        else:
            raise_error(
                NotImplementedError, f"Sweeper for {parameter} is not implemented."
            )
    else:
        play(args)


def _sweep_frequency(sweepers, configs, args):
    sweeper = sweepers[0]
    freqs0 = []
    for channel in sweeper.channels:
        lo_frequency = configs[channel.lo].frequency
        # convert to IF frequency for readout and drive pulses
        f0 = math.floor(configs[channel.name].frequency - lo_frequency)
        freqs0.append(declare(int, value=f0))
        # check if sweep is within the supported bandwidth [-400, 400] MHz
        max_freq = maximum_sweep_value(sweeper.values, f0)
        if max_freq > 4e8:
            raise_error(
                ValueError,
                f"Frequency {max_freq} for channel {channel.name} is beyond instrument bandwidth.",
            )

    # is it fine to have this declaration inside the ``nshots`` QUA loop?
    f = declare(int)
    with for_(*from_array(f, sweeper.values.astype(int))):
        for channel, f0 in zip(sweeper.channels, freqs0):
            qua.update_frequency(channel.name, f + f0)

        _sweep_recursion(sweepers[1:], configs, args)


def _sweep_amplitude(sweepers, configs, args):
    sweeper = sweepers[0]
    # TODO: Consider sweeping amplitude without multiplication
    if min(sweeper.values) < -2:
        raise_error(
            ValueError, "Amplitude sweep values are <-2 which is not supported."
        )
    if max(sweeper.values) > 2:
        raise_error(ValueError, "Amplitude sweep values are >2 which is not supported.")

    a = declare(fixed)
    with for_(*from_array(a, sweeper.values)):
        for pulse in sweeper.pulses:
            # if isinstance(instruction, Bake):
            #    instructions.update_kwargs(instruction, amplitude=a)
            # else:
            args.parameters[operation(pulse)].amplitude = qua.amp(a)

        _sweep_recursion(sweepers[1:], configs, args)


def _sweep_relative_phase(sweepers, configs, args):
    sweeper = sweepers[0]
    relphase = declare(fixed)
    with for_(*from_array(relphase, sweeper.values / (2 * np.pi))):
        for pulse in sweeper.pulses:
            args.parameters[operation(pulse)].phase = relphase

        _sweep_recursion(sweepers[1:], configs, args)


def _sweep_bias(sweepers, configs, args):
    sweeper = sweepers[0]
    offset0 = []
    for channel in sweeper.channels:
        b0 = configs[channel.name].offset
        max_value = maximum_sweep_value(sweeper.values, b0)
        check_max_offset(max_value, MAX_OFFSET)
        offset0.append(declare(fixed, value=b0))
    b = declare(fixed)
    with for_(*from_array(b, sweeper.values)):
        for channel, b0 in zip(sweeper.channels, offset0):
            with qua.if_((b + b0) >= 0.49):
                qua.set_dc_offset(f"flux{channel.name}", "single", 0.49)
            with qua.elif_((b + b0) <= -0.49):
                qua.set_dc_offset(f"flux{channel.name}", "single", -0.49)
            with qua.else_():
                qua.set_dc_offset(f"flux{channel.name}", "single", (b + b0))

        _sweep_recursion(sweepers[1:], configs, args)


def _sweep_duration(sweepers, configs, args):
    # TODO: Handle baked pulses
    sweeper = sweepers[0]
    dur = declare(int)
    with for_(*from_array(dur, (sweeper.values // 4).astype(int))):
        for pulse in sweeper.pulses:
            args.parameters[operation(pulse)].duration = dur

        _sweep_recursion(sweepers[1:], configs, args)
