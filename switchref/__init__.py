"""Switching-reference voltage control library.

Module layout:
    config.py       constants (seed, data paths)
    env.py          linear voltage env (Voltage class)
    case33.py       IEEE 33-bus loader + per-bus droop gain
    utils.py        seed, droop action, disturbance shaping, palette, metrics
    controllers.py  switching-reference controller
    plotting.py     plot helpers
    runners.py      single-DC + two-DC scenario setup, trace loaders, closed-loop
"""
from .config import (
    SEED, ROOT, DATA_DIR, TRACE_DIR, CASE_DIR,
    DT, STRIDE_CTRL,
    ALPHA, BETA, GAMMA, C_RATE, ETA_B,
    K_SCALE, DEADBAND,
    T_CYCLE_DGX_H100, T_CYCLE_RTX8000, T_CYCLE_L40S, T_CYCLE_H200,
)
from .env import Voltage, linear_V
from .case33 import load_case33, load_linear_gain
from .utils import (
    set_seed,
    action_linear, action_fractional, ACTION_REGISTRY,
    rescale_diff, make_dP_last_from_data, make_dP_last_multi_square,
    make_bus_colors,
    compute_metrics,
)
from .controllers import RingSeries, BaseAdaController, AdaRefVBiasAmpMaxDV
from .runners import (
    setup_common, setup_two_bus,
    load_h200, load_l40s, load_rtx8000, load_dgx_h100, load_h200_regulated,
    load_two_bus,
    closed_loop, closed_loop_two_bus,
    TRACES,
)

__all__ = [
    "SEED", "ROOT", "DATA_DIR", "TRACE_DIR", "CASE_DIR",
    "DT", "STRIDE_CTRL",
    "ALPHA", "BETA", "GAMMA", "C_RATE", "ETA_B",
    "K_SCALE", "DEADBAND",
    "T_CYCLE_DGX_H100", "T_CYCLE_RTX8000", "T_CYCLE_L40S", "T_CYCLE_H200",
    "TRACES",
    "Voltage", "linear_V",
    "load_case33", "load_linear_gain",
    "set_seed",
    "action_linear", "action_fractional", "ACTION_REGISTRY",
    "rescale_diff", "make_dP_last_from_data", "make_dP_last_multi_square",
    "make_bus_colors", "compute_metrics",
    "RingSeries", "BaseAdaController", "AdaRefVBiasAmpMaxDV",
    "setup_common", "setup_two_bus",
    "load_h200", "load_l40s", "load_rtx8000", "load_dgx_h100", "load_h200_regulated",
    "load_two_bus",
    "closed_loop", "closed_loop_two_bus",
]
