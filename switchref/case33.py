"""IEEE 33-bus test case loader + pre-computed per-bus droop gain.

The .mat file holds the sensitivity matrices R, X of the linearized
distribution model, plus the reference operating point (P_ref, Q_ref,
V_ref). The notebook scaling (R *= 0.1, X *= 0.1, P, Q *= 10) is
applied here so callers receive the matrices in the simulator units.

`linear_gain.pkl` is the pre-computed per-bus droop vector K used by
both the fixed-reference and switching-reference controllers.
"""
import pickle
from pathlib import Path

import numpy as np
from mat4py import loadmat

from .config import CASE_DIR


def load_case33():
    """Load 33-bus matrices and the linearization reference."""
    data = loadmat(str(CASE_DIR / "TestCase33.mat"))["TestCase"]

    R = np.asarray(data["R"], dtype=np.float32) * 0.1
    X = np.asarray(data["X"], dtype=np.float32) * 0.1
    P0 = np.asarray(data["P_ref"], dtype=np.float32) * 10.0
    Q0 = np.asarray(data["Q_ref"], dtype=np.float32) * 10.0
    V_ref = np.asarray(data["V_ref"], dtype=np.float32)
    ones = np.asarray(data["ones"], dtype=np.float32)

    return {
        "R": R, "X": X, "V_ref": V_ref, "P0": P0, "Q0": Q0, "ones": ones,
        "num_buses": R.shape[0], "bus_idx": np.arange(R.shape[0]),
    }


def load_linear_gain(droop_scale: float = 0.5):
    """Load the pre-computed (32, 1) per-bus droop gain K.

    The unscaled vector solves a Lyapunov-style PSD bound for closed-loop
    stability; droop_scale rescales it (default 0.5).
    """
    with open(CASE_DIR / "linear_gain.pkl", "rb") as fp:
        return pickle.load(fp)[0].reshape(-1, 1) * float(droop_scale)
