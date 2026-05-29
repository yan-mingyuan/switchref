"""Constants and paths."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRACE_DIR = DATA_DIR / "traces"
CASE_DIR = DATA_DIR / "case33"

# Simulation timing
DT = 0.1                  # system time step (s)
STRIDE_CTRL = 10          # control period = STRIDE_CTRL * DT = 1 s

# Unified design rule: alpha = beta = 1, gamma = 1/8, c = 1/4
ALPHA = 1.0
BETA  = 1.0
GAMMA = 1.0 / 8.0
C_RATE = 0.25
ETA_B = C_RATE * GAMMA    # = 1/32, identical for every trace

# Controller scalars
K_SCALE = 25              # multiplier on the per-bus droop gain vector
DEADBAND = 0.02

# Workload cycle period per trace (seconds)
T_CYCLE_DGX_H100 = 17
T_CYCLE_RTX8000  = 82
T_CYCLE_L40S     = 107
T_CYCLE_H200     = 132
