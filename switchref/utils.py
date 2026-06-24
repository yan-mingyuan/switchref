"""Numerical utilities: seed, droop action, disturbance shaping,
network-palette helper, and the steady-state evaluation metric."""
import random

import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42, verbose: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    if verbose:
        print(f"[Seed set]: {seed}")


# ---------------------------------------------------------------------------
# Droop control action
# ---------------------------------------------------------------------------

def action_linear(V, K, V_ref, deadband=0.0, u_max=0.01, **kwargs):
    """Linear droop with symmetric deadband and per-update saturation.
    Used by both the fixed-reference (V_ref = 0) and switching-reference
    (V_ref updated online by AdaRefVBiasAmpMaxDV) control loops.
    """
    e = V - V_ref
    g = np.sign(e) * np.maximum(np.abs(e) - deadband, 0.0)
    u = -K * g
    if u_max is not None:
        u = np.clip(u, -u_max, u_max)
    return u


def action_fractional(V, K, V_ref, alpha=0.5, eps=1e-6, **kwargs):
    """Smoothed signed-power droop. Kept for API compatibility with the
    action registry; not used by the shipped notebooks."""
    e = V - V_ref
    return -K * e / (np.abs(e) + eps) * (np.abs(e) ** alpha)


ACTION_REGISTRY = {
    "linear": action_linear,
    "fractional": action_fractional,
}


# ---------------------------------------------------------------------------
# Disturbance shaping (real AI-training traces -> simulator input)
# ---------------------------------------------------------------------------

def rescale_diff(x, amp, return_scale=False, eps=1e-12):
    """Rescale `x` to [0, amp] and return its first difference.

    Used to turn a raw power trace into the per-step disturbance dP that
    drives the data-center bus in the linear voltage env.
    """
    x = np.asarray(x)
    xmin, xmax = x.min(), x.max()
    d = xmax - xmin

    if d < eps:
        diff = np.zeros(len(x) - 1, dtype=np.float32)
        scale = 0.0
        return (diff, scale) if return_scale else diff

    y = (x - xmin) / d * amp
    diff = np.diff(y).astype(np.float32)
    scale = amp / d
    return (diff, scale) if return_scale else diff


def make_dP_last_from_data(dP):
    """Build the step-indexed disturbance callable `dP_last(t)` from a
    pre-computed difference array. Returns (callable, total_length)."""
    def dP_last(t):
        if 0 <= t < len(dP):
            return float(-dP[t])
        return 0.0
    return dP_last, len(dP)


def make_dP_last_multi_square(start, stages):
    """Build an idealized square-wave disturbance train: a `+amp` impulse at
    every low->high (communication->compute) edge and a `-amp` impulse at every
    high->low edge. Used for the Section III motivation figure, where a clean
    two-level load isolates the power->voltage step relation from the intra-phase
    fluctuations present in the measured traces.

    Parameters
    ----------
    start : int
        Step index of the first rising edge.
    stages : list of dict
        Each stage has keys ``amp`` (step magnitude), ``T_on`` (compute-phase
        length, steps), ``T_off`` (communication-phase length, steps), and
        ``cycles`` (number of on/off cycles in the stage).

    Returns
    -------
    (callable, int)
        ``dP_last(t)`` returning the impulse at step ``t`` (0 elsewhere), and the
        total length in steps.
    """
    impulse = {}
    t = start
    for st in stages:
        amp = float(st["amp"]); T_on = int(st["T_on"])
        T_off = int(st["T_off"]); cycles = int(st["cycles"])
        for _ in range(cycles):
            impulse[t] = +amp; t += T_on
            impulse[t] = -amp; t += T_off

    def dP_last(t_now):
        return impulse.get(t_now, 0.0)
    return dP_last, t


# ---------------------------------------------------------------------------
# Visualization palette
# ---------------------------------------------------------------------------

def make_bus_colors(num_buses: int):
    """Per-bus color palette for the network plots.

    tab20 with stride 7, bus 0 forced to black, and the swap
    {6, 18, 20, 23} that the original notebook used to keep adjacent
    buses visually distinguishable.
    """
    palette = list(plt.get_cmap("tab20").colors)
    stride, offset = 7, 0
    bus_colors = [palette[(offset + stride * i) % len(palette)] for i in range(num_buses)]
    bus_colors[0] = (0, 0, 0, 1)
    if num_buses > 23:
        bus_colors[6], bus_colors[18], bus_colors[20], bus_colors[23] = (
            bus_colors[23], bus_colors[6], bus_colors[18], bus_colors[20]
        )
    return bus_colors


# ---------------------------------------------------------------------------
# Steady-state metrics for closed-loop runs
# ---------------------------------------------------------------------------

def compute_metrics(
    dQ, V,
    *,
    dt: float = 0.1,
    warmup_secs: float = 300.0,
    sat_limit: float = 0.01,
    v_band: float = 0.05,
) -> dict:
    """Steady-state metrics for a closed-loop run, excluding the initial
    warmup window. The two headline metrics are:

        rms_excess : RMS of max(|v - 1| - v_band, 0) over (t, i).
                     Band-violation magnitude.
        rms_dq     : RMS of |Delta q| over (t, i). Reactive control effort.

    Extra fields (max/mean variants, raw |v-1| stats, saturation rate)
    are computed cheaply and kept for diagnostics.
    """
    dQ = np.asarray(dQ, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    warmup_steps = int(round(warmup_secs / dt))
    dQ_ss = dQ[warmup_steps:]
    dV_ss = V[warmup_steps:]

    eps_sat = 1e-9
    excess = np.maximum(np.abs(dV_ss) - v_band, 0.0)
    return {
        "violation_rate": float(np.mean(np.any(np.abs(dV_ss) > v_band, axis=1))),
        "max_excess":     float(np.max(excess)),
        "rms_excess":     float(np.sqrt(np.mean(excess ** 2))),
        "mean_excess":    float(np.mean(excess)),
        "max_dv":         float(np.max(np.abs(dV_ss))),
        "rms_dv":         float(np.sqrt(np.mean(dV_ss ** 2))),
        "max_dq":          float(np.max(np.abs(dQ_ss))),
        "rms_dq":          float(np.sqrt(np.mean(dQ_ss ** 2))),
        "mean_abs_dq":     float(np.mean(np.abs(dQ_ss))),
        "saturation_rate": float(np.mean(np.abs(dQ_ss) >= sat_limit - eps_sat)),
        "warmup_steps":    warmup_steps,
        "T_ss":            int(dQ_ss.shape[0]),
    }
