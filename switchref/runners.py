"""Scenario setup, trace loaders, and closed-loop rollouts."""
import pickle
from typing import Callable

import numpy as np
import pandas as pd

from .config import (
    SEED, TRACE_DIR,
    T_CYCLE_DGX_H100, T_CYCLE_RTX8000, T_CYCLE_L40S, T_CYCLE_H200,
)
from .case33 import load_case33, load_linear_gain
from .env import Voltage
from .utils import (
    set_seed,
    make_bus_colors,
    rescale_diff,
    make_dP_last_from_data,
)


# ---------------------------------------------------------------------------
# Single-DC scenario
# ---------------------------------------------------------------------------

def setup_common(*, dc_bus: int | None = None) -> dict:
    """Build the network + env + droop gain + palette + DC-bus choice
    for a single-DC scenario. dc_bus defaults to 20 (network label 22).
    """
    case = load_case33()
    R, X = case["R"], case["X"]
    P0, Q0 = case["P0"], case["Q0"]
    num_buses = case["num_buses"]

    dt_sys = 0.1
    stride_pwr = 1
    env = Voltage(R, X, P0=P0, Q0=Q0, dt=dt_sys)

    set_seed(SEED, verbose=False)
    P0_INIT = np.zeros((num_buses, 1))

    K = load_linear_gain(droop_scale=0.5)
    bus_colors = make_bus_colors(num_buses)

    dc_buses = [20, 23]
    if dc_bus is None:
        dc_bus = dc_buses[0]

    R_dc_self = R[dc_bus, dc_bus]
    R_dc_row = R[dc_bus, :]

    return dict(
        R=R, X=X, P0=P0, Q0=Q0,
        num_buses=num_buses,
        dt_sys=dt_sys, stride_pwr=stride_pwr, dt_pwr=stride_pwr * dt_sys,
        env=env, P0_INIT=P0_INIT,
        K=K,
        bus_colors=bus_colors,
        dc_buses=dc_buses, dc_bus=dc_bus,
        R_dc_self=R_dc_self, R_dc_row=R_dc_row,
    )


# ---------------------------------------------------------------------------
# Two-DC scenario
# ---------------------------------------------------------------------------

def setup_two_bus() -> dict:
    """Build context for the two-DC scenario; same network, two DC buses."""
    case = load_case33()
    R, X, P0, Q0 = case["R"], case["X"], case["P0"], case["Q0"]
    num_buses = case["num_buses"]
    dt_sys = 0.1
    env = Voltage(R, X, P0=P0, Q0=Q0, dt=dt_sys)
    set_seed(SEED, verbose=False)
    P0_INIT = np.zeros((num_buses, 1))
    K = load_linear_gain(droop_scale=0.5)
    bus_colors = make_bus_colors(num_buses)
    dc_buses = [20, 23]
    return dict(
        R=R, X=X, P0=P0, Q0=Q0,
        num_buses=num_buses,
        dt_sys=dt_sys, env=env, P0_INIT=P0_INIT,
        K=K, bus_colors=bus_colors,
        dc_buses=dc_buses, dc_bus=dc_buses[0],
    )


# ---------------------------------------------------------------------------
# Trace -> simulator-input dict, shared shape
# ---------------------------------------------------------------------------

def _build_realistic(t_sys: np.ndarray, pwr_sys: np.ndarray, ctx: dict, amp: float) -> dict:
    """Shape a raw power array into the dict expected by `closed_loop`."""
    stride_pwr = ctx["stride_pwr"]
    R_dc_row = ctx["R_dc_row"]
    t_pwr = t_sys[::stride_pwr]
    pwr_pwr = -pwr_sys[::stride_pwr]

    dP_sys, scale = rescale_diff(pwr_sys, amp, return_scale=True)
    dP_last, T_sys = make_dP_last_from_data(dP_sys)
    v_sens = scale * R_dc_row
    return {
        "amp": amp,
        "t_pwr": t_pwr, "pwr_pwr": pwr_pwr,
        "dP_sys": dP_sys, "scale": scale, "v_sens": v_sens,
        "dP_last": dP_last, "SimulationLength": T_sys,
    }


def _read_pwr_csv(path, slice_obj, dt_sys, *, interpolate: bool):
    df = pd.read_csv(path)
    t_raw = df["t"][slice_obj].to_numpy(np.float64)
    pwr_raw = df["pwr"][slice_obj].to_numpy(np.float32)
    if interpolate:
        t_sys = np.arange(t_raw[0], t_raw[-1] + 1e-9, dt_sys, dtype=np.float64)
        pwr_sys = np.interp(t_sys, t_raw, pwr_raw).astype(np.float32)
    else:
        t_sys = t_raw
        pwr_sys = pwr_raw
    return t_sys, pwr_sys


# ---------------------------------------------------------------------------
# Trace loaders
# ---------------------------------------------------------------------------

def load_h200(ctx: dict) -> dict:
    """4xH200 trace (LLaMA-2-70B-chat QLoRA). Cycle ~132 s."""
    amp = 0.10 * 0.95 / ctx["R_dc_self"]
    t_sys, pwr_sys = _read_pwr_csv(
        TRACE_DIR / "h200_4x.csv",
        slice(1500, 28500), ctx["dt_sys"], interpolate=False,
    )
    return _build_realistic(t_sys, pwr_sys, ctx, amp)


def load_l40s(ctx: dict) -> dict:
    """4xL40S trace (LLaMA-3.3-70B-Instruct QLoRA). Cycle ~107 s."""
    amp = 0.10 * 0.95 / ctx["R_dc_self"]
    t_sys, pwr_sys = _read_pwr_csv(
        TRACE_DIR / "l40s_4x.csv",
        slice(1620, 22200), ctx["dt_sys"], interpolate=True,
    )
    return _build_realistic(t_sys, pwr_sys, ctx, amp)


def load_rtx8000(ctx: dict) -> dict:
    """4xRTX8000 trace (LLaMA-3.3-70B QLoRA). Cycle ~82 s."""
    amp = 0.10 * 0.95 / ctx["R_dc_self"]
    t_sys, pwr_sys = _read_pwr_csv(
        TRACE_DIR / "rtx8000_4x.csv",
        slice(3000, 28500), ctx["dt_sys"], interpolate=True,
    )
    return _build_realistic(t_sys, pwr_sys, ctx, amp)


def load_dgx_h100(ctx: dict, target_seconds: float = 4260.0) -> dict:
    """DGX-H100 single-rack trace (Choukse 2025, arXiv:2508.14318).
    The recording is 152 s long; tiled ~28x so the band-excess RMS metric
    (max-type) reaches convergence. Cycle ~17 s.
    """
    dt_sys = ctx["dt_sys"]
    amp = 0.10 * 0.95 / ctx["R_dc_self"]

    df = pd.read_csv(TRACE_DIR / "dgx_h100_choukse.csv")
    df = df.rename(columns={"time_s": "t", "gpu_power_norm": "pwr"})
    df["t"] = df["t"] - df["t"].iloc[0]
    df = df.sort_values("t").drop_duplicates(subset=["t"])

    t1 = np.arange(df["t"].min(), df["t"].max() + 1e-12, dt_sys)
    p1 = np.interp(t1, df["t"].to_numpy(), df["pwr"].to_numpy()).astype(np.float32)
    cyc_steps = int(round(17.0 / dt_sys))
    p1 = np.roll(p1, -int(np.argmin(p1[: cyc_steps + cyc_steps // 3])))

    per = t1[-1] - t1[0]
    n = int(np.ceil(target_seconds / per))
    pwr_sys = np.tile(p1, n)
    t_sys = np.arange(len(pwr_sys)) * dt_sys
    return _build_realistic(t_sys, pwr_sys, ctx, amp)


# ---------------------------------------------------------------------------
# Regulated-tail trace
# ---------------------------------------------------------------------------

def load_h200_regulated(ctx: dict) -> dict:
    """Compose the 4xH200 unregulated head with the storage-regulated
    tail from `data/traces_regulated.pkl`. Amplitudes preserve the
    regulated-tail / unregulated-baseline ratio so the smooth tail
    looks visibly attenuated.
    """
    R_dc_self = ctx["R_dc_self"]
    amp = 0.10 * 0.95 / R_dc_self

    df = pd.read_csv(TRACE_DIR / "h200_4x.csv")
    pwr_unreg = df["pwr"][slice(1500, 15001)].to_numpy(np.float32)
    dP_unreg, _ = rescale_diff(pwr_unreg, amp, return_scale=True)
    xmin_unreg, xmax_unreg = float(np.min(pwr_unreg)), float(np.max(pwr_unreg))
    p0_unreg = (pwr_unreg[0] - xmin_unreg) / (xmax_unreg - xmin_unreg) * amp
    pwr_unreg_scale = np.concatenate([[p0_unreg], p0_unreg + np.cumsum(dP_unreg)])

    with open(TRACE_DIR.parent / "traces_regulated.pkl", "rb") as f:
        results = pickle.load(f)
    tail = slice(None, 20300)
    P_base = np.asarray(results["unregulated"]["Pinj_dc"][tail])
    P_net = np.asarray(results["regulated"]["Pinj_dc"][tail])

    xmin_base, xmax_base = float(np.min(P_base)), float(np.max(P_base))
    range_base = xmax_base - xmin_base
    range_net = float(np.max(P_net)) - float(np.min(P_net))
    amp_net = amp * range_net / range_base
    dP_net_scale = rescale_diff(P_net, amp_net, return_scale=False)

    p0_base = float(pwr_unreg_scale[-1])
    p0_net = (float(P_net[0]) - xmin_base) / range_base * amp + p0_base
    P_net_scale = np.concatenate([[p0_net], p0_net + np.cumsum(dP_net_scale)])

    P_combined = np.concatenate(
        [pwr_unreg_scale, P_net_scale[1:]]
    ).astype(np.float32)
    pwr_sys = P_combined
    t_sys = np.arange(len(P_combined)) * 0.1
    return _build_realistic(t_sys, pwr_sys, ctx, amp)


# ---------------------------------------------------------------------------
# Two-DC trace pair
# ---------------------------------------------------------------------------

def load_two_bus(ctx: dict) -> dict:
    """Two heterogeneous H200 traces, one per data-center bus
    (b16x2_seq2048 at bus 20, b8x2_seq2048 at bus 23)."""
    R = ctx["R"]
    dt_sys = ctx["dt_sys"]
    dc_buses = ctx["dc_buses"]
    files = {
        20: TRACE_DIR / "h200_4x.csv",
        23: TRACE_DIR / "h200_4x_b8x2.csv",
    }
    dPs = {}
    Ts = []
    for b in dc_buses:
        amp = 0.10 * 0.95 / R[b, b]
        t_sys, pwr_sys = _read_pwr_csv(
            files[b], slice(2300, 28500), dt_sys, interpolate=True,
        )
        dP_sys, _ = rescale_diff(pwr_sys, amp, return_scale=True)
        dP_last, T_sys = make_dP_last_from_data(dP_sys)
        dPs[b] = dP_last
        Ts.append(T_sys)
    return {
        "dc_buses": dc_buses,
        "dP_last_0": dPs[dc_buses[0]],
        "dP_last_1": dPs[dc_buses[1]],
        "SimulationLength": min(Ts),
    }


# ---------------------------------------------------------------------------
# Closed-loop rollouts
# ---------------------------------------------------------------------------

def closed_loop(
    *,
    ctx: dict,
    ds: dict,
    ctrl_step: Callable[[int, np.ndarray], np.ndarray],
    stride_ctrl: int = 1,
):
    """Single-DC closed-loop rollout.

    `ctrl_step(k_sys, V) -> dQ` runs every `stride_ctrl` steps;
    in-between dQ = 0. Returns (record_dQ, record_V) as float32 arrays.
    """
    env = ctx["env"]
    P0 = ctx["P0"]
    num_buses = ctx["num_buses"]
    dc_bus = ctx["dc_bus"]
    P0_INIT = ctx["P0_INIT"]
    dP_last = ds["dP_last"]
    T_sys = ds["SimulationLength"]

    set_seed(SEED, verbose=False)
    V = env.reset(P=P0_INIT, balance_Q=True)

    record_dQ = []
    record_V = [V.squeeze().copy()]

    for k_sys in range(T_sys):
        dP_t = np.zeros_like(P0)
        dP_t[dc_bus, 0] = dP_last(k_sys)

        dQ = np.zeros((num_buses, 1), dtype=np.float32)
        if k_sys % stride_ctrl == 0:
            dQ = ctrl_step(k_sys, V).reshape(num_buses, 1).astype(np.float32)

        V = env.step(dQ, dP=dP_t)
        record_dQ.append(dQ.squeeze().copy())
        record_V.append(V.squeeze().copy())

    return (
        np.asarray(record_dQ, dtype=np.float32),
        np.asarray(record_V, dtype=np.float32),
    )


def closed_loop_two_bus(
    *,
    ctx: dict,
    ds: dict,
    ctrl_step: Callable[[int, np.ndarray], np.ndarray],
    stride_ctrl: int = 1,
    vref_getter: Callable[[], np.ndarray] | None = None,
):
    """Two-DC closed-loop rollout. Disturbances act at BOTH dc_buses each
    step; optional `vref_getter` logs the controller's V_ref per step."""
    env = ctx["env"]
    P0 = ctx["P0"]
    num_buses = ctx["num_buses"]
    P0_INIT = ctx["P0_INIT"]
    dc_buses = ds["dc_buses"]
    dP_last_0 = ds["dP_last_0"]
    dP_last_1 = ds["dP_last_1"]
    T_sys = ds["SimulationLength"]

    set_seed(SEED, verbose=False)
    V = env.reset(P=P0_INIT, balance_Q=True)
    record_dQ = []
    record_V = [V.squeeze().copy()]
    record_Vref = []

    for k_sys in range(T_sys):
        dP_t = np.zeros_like(P0)
        dP_t[dc_buses[0], 0] = dP_last_0(k_sys)
        dP_t[dc_buses[1], 0] = dP_last_1(k_sys)

        dQ = np.zeros((num_buses, 1), dtype=np.float32)
        if k_sys % stride_ctrl == 0:
            dQ = ctrl_step(k_sys, V).reshape(num_buses, 1).astype(np.float32)

        V = env.step(dQ, dP=dP_t)
        record_dQ.append(dQ.squeeze().copy())
        record_V.append(V.squeeze().copy())
        if vref_getter is not None:
            record_Vref.append(vref_getter().squeeze().copy())

    res = {
        "record_dQ": np.asarray(record_dQ, dtype=np.float32),
        "record_V": np.asarray(record_V, dtype=np.float32),
    }
    if vref_getter is not None:
        res["record_Vref"] = np.asarray(record_Vref, dtype=np.float32)
    return res


# Trace registry: (name, cycle_s, loader). Order = Table I row order.
TRACES = (
    ("DGX-H100",  T_CYCLE_DGX_H100, load_dgx_h100),
    ("4xRTX8000", T_CYCLE_RTX8000,  load_rtx8000),
    ("4xL40S",    T_CYCLE_L40S,     load_l40s),
    ("4xH200",    T_CYCLE_H200,     load_h200),
)
