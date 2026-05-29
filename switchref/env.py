"""Linear voltage env."""
import numpy as np
import gymnasium as gym


def linear_V(R, X, P, Q):
    """Voltage deviation from 1.0 p.u.: dV = R @ P + X @ Q."""
    return R @ P + X @ Q


class Voltage(gym.Env):
    """
    Linear voltage model (deviation form):
        dV = R @ P + X @ Q

    Stored state (float32, shape (n,1)):
      - P, Q, V, V_prev
    """

    def __init__(self, R, X, P0=None, Q0=None, dt=0.1):
        self.R = np.asarray(R, dtype=np.float32)
        self.X = np.asarray(X, dtype=np.float32)
        self.num_buses = self.R.shape[0]
        self.dt = float(dt)

        self.P0 = np.zeros((self.num_buses, 1), dtype=np.float32) if P0 is None \
            else np.asarray(P0, dtype=np.float32).reshape(self.num_buses, 1)
        self.Q0 = np.zeros((self.num_buses, 1), dtype=np.float32) if Q0 is None \
            else np.asarray(Q0, dtype=np.float32).reshape(self.num_buses, 1)

        self.V_ref = np.zeros((self.num_buses, 1), dtype=np.float32)

        self.reset()

    def reset(self, P=None, Q=None, balance_Q=False):
        """
        Set (P,Q) and update V.

        Typical use: call once at the beginning of each rollout.
        Debug use: you may call it mid-rollout to force a state change.

        - If P/Q are None, use P0/Q0.
        - If balance_Q=True, ignore Q and solve Q to make V = 0.
        """
        self.P = self.P0.copy() if P is None \
            else np.asarray(P, dtype=np.float32).reshape(self.num_buses, 1)
        self.P0 = self.P.copy()

        if balance_Q:
            self.Q = -np.linalg.solve(self.X, self.R @ self.P).astype(np.float32)
        else:
            self.Q = self.Q0.copy() if Q is None \
                else np.asarray(Q, dtype=np.float32).reshape(self.num_buses, 1)
        self.Q0 = self.Q.copy()

        self.V = linear_V(self.R, self.X, self.P, self.Q).astype(np.float32)
        self.V_prev = self.V.copy()
        return self.V

    def step(self, dQ, dP=None):
        """Apply deltaQ control and optional deltaP disturbance, then return updated V."""
        self.V_prev = self.V.copy()

        if dP is not None:
            self.P += np.asarray(dP, dtype=np.float32).reshape(self.num_buses, 1)

        self.Q += np.asarray(dQ, dtype=np.float32).reshape(self.num_buses, 1)
        self.V = linear_V(self.R, self.X, self.P, self.Q).astype(np.float32)
        return self.V
