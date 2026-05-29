"""Switching-reference voltage controller.

Three classes are exposed:

    RingSeries           : fixed-length ring buffer with optional EMA.
    BaseAdaController    : shared state machine (V_bias, V_amp_half,
                           V_sign, V_ref, cooldown, action).
    AdaRefVBiasAmpMaxDV  : per-bus sign selection + per-bus amplitude
                           update from the rolling max-|dV| window.
"""
import numpy as np

from .utils import ACTION_REGISTRY


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class RingSeries:
    """Fixed-length ring buffer for vector time series, with optional EMA.

    EMA definition:
        ema_t = (1 - beta) * ema_{t-1} + beta * x_t,  0 <= beta <= 1
        beta=None disables EMA tracking.
    """

    def __init__(self, length, shape, dtype=np.float32, beta=None, fill_value=np.nan):
        self.cap = int(length)
        if self.cap <= 0:
            raise ValueError(f"length must be positive, got {length}")

        self.shape = tuple(shape)
        self.dtype = dtype
        self.fill_value = fill_value

        if beta is None:
            self.beta = None
        else:
            b = float(beta)
            if not (0.0 <= b <= 1.0):
                raise ValueError(f"beta must be in [0,1], got {beta}")
            self.beta = b

        self.buf = np.full((self.cap, *self.shape),
                           np.array(self.fill_value, dtype=self.dtype),
                           dtype=self.dtype)
        self.ptr = 0
        self.count = 0
        self.step = 0
        self.ema = None

    def reset(self):
        self.buf[...] = np.array(self.fill_value, dtype=self.dtype)
        self.ptr = 0
        self.count = 0
        self.step = 0
        self.ema = None

    def push(self, x):
        x_arr = np.asarray(x, dtype=self.dtype).reshape(self.shape)
        self.buf[self.ptr, ...] = x_arr
        self.ptr = (self.ptr + 1) % self.cap
        self.count = min(self.count + 1, self.cap)
        self.step += 1

        if self.beta is not None:
            if self.ema is None:
                self.ema = x_arr.copy()
            else:
                b = self.beta
                self.ema[...] = (1.0 - b) * self.ema + b * x_arr

        return x_arr

    def get_last(self, k=1, oldest_first=True, require_full=False):
        if require_full and (self.count < self.cap):
            raise ValueError(f"buffer not full: count={self.count}, cap={self.cap}")

        if self.count == 0:
            return np.empty((0, *self.shape), dtype=self.dtype)

        k = int(k)
        if k <= 0:
            return np.empty((0, *self.shape), dtype=self.dtype)
        k = min(k, self.count)

        end = self.ptr
        start = (end - k) % self.cap

        if start < end:
            out = self.buf[start:end, ...].copy()
        else:
            out = np.concatenate((self.buf[start:, ...], self.buf[:end, ...]), axis=0).copy()

        if not oldest_first:
            out = out[::-1].copy()
        return out

    @property
    def full(self):
        return self.count == self.cap


# ---------------------------------------------------------------------------
# Shared adaptive-reference controller state
# ---------------------------------------------------------------------------

class BaseAdaController:
    """Shared state, ring histories, bias update, cooldown, and droop action."""

    def __init__(
        self, n, K,
        *,
        action_type="linear", action_kwargs=None,
        eps=1e-8,
        # bias-update block (all disabled when bias_len=0)
        bias_len=0, bias_per=None, bias_skip=0, bias_frac=0.0,
        v_hist_len=None,        # None -> auto: matches bias_len when bias enabled, else 0
        # dV history (used by subclasses for amp updates)
        dv_hist_len=0, dv_beta=None,
        cooldown_steps=0,
    ):
        self.n = int(n)
        self.K = K
        self.eps = float(eps)
        self.dtype = np.float32

        self.action_kwargs = action_kwargs or {}
        if action_type not in ACTION_REGISTRY:
            raise ValueError(f"Unknown action_type={action_type}. Choose from {list(ACTION_REGISTRY.keys())}")
        self.action_fn = ACTION_REGISTRY[action_type]

        self.V_prev = None
        self.V_bias = np.zeros((self.n, 1), dtype=self.dtype)
        self.V_amp_half = np.zeros((self.n, 1), dtype=self.dtype)
        self.V_sign = np.zeros((self.n, 1), dtype=np.int8)
        self.V_ref = np.zeros((self.n, 1), dtype=self.dtype)

        # Bias window + update period.
        self.bias_len = int(bias_len)
        self.bias_skip = int(bias_skip)
        self.bias_frac = float(bias_frac)

        if self.bias_len < 0:
            raise ValueError("bias_len must be >= 0")
        if self.bias_skip < 0:
            raise ValueError("bias_skip must be >= 0")

        if bias_per is None:
            bias_per = self.bias_len if self.bias_len > 0 else 0
        self.off_per = int(bias_per)

        if self.off_per < 0:
            raise ValueError("bias_per must be >= 0")
        if self.bias_len > 0 and self.off_per <= 0:
            raise ValueError("bias_per must be > 0 when bias is enabled")

        # V-history (used by the bias update; decoupled from bias_len).
        if v_hist_len is None:
            v_hist_len = self.bias_len if self.bias_len > 0 else 0
        self.v_hist_len = int(v_hist_len)

        if self.v_hist_len < 0:
            raise ValueError("v_hist_len must be >= 0")

        if self.bias_len > 0:
            if not (0 <= self.bias_skip < self.bias_len):
                raise ValueError("bias_skip must be in [0, bias_len)")
            if not (0.0 <= self.bias_frac <= 1.0):
                raise ValueError("bias_frac must be in [0, 1]")
            if self.v_hist_len <= 0:
                raise ValueError("v_hist_len must be > 0 when bias is enabled")
            if self.v_hist_len < self.bias_len:
                raise ValueError(f"v_hist_len ({self.v_hist_len}) must be >= bias_len ({self.bias_len})")
            self.V_hist = RingSeries(self.v_hist_len, shape=(self.n,), dtype=self.dtype, beta=None)
        else:
            self.V_hist = None

        # dV history (+ optional EMA), required by the amplitude update.
        self.dv_hist_len = int(dv_hist_len)
        if self.dv_hist_len < 0:
            raise ValueError("dv_hist_len must be >= 0")

        if self.dv_hist_len > 0:
            if dv_beta is not None:
                dv_beta = float(dv_beta)
                if not (0.0 < dv_beta <= 1.0):
                    raise ValueError("dv_beta must be in (0, 1] or None")
            self.dV_hist = RingSeries(self.dv_hist_len, shape=(self.n, 1), dtype=self.dtype, beta=dv_beta)
        else:
            self.dV_hist = None

        self.cooldown_steps = int(cooldown_steps)
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
        self.cooldown = np.zeros((self.n, 1), dtype=np.int32) if self.cooldown_steps > 0 else None

    def reset(self):
        self.V_prev = None
        self.V_bias[...] = 0
        self.V_amp_half[...] = 0
        self.V_sign[...] = 0
        self.V_ref[...] = 0
        if self.cooldown is not None:
            self.cooldown[...] = 0
        if self.V_hist is not None:
            self.V_hist.reset()
        if self.dV_hist is not None:
            self.dV_hist.reset()

    def _col(self, x):
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        if x.shape[0] != self.n:
            raise ValueError(f"dimension mismatch: got {x.shape[0]}, expected {self.n}")
        return x

    def safe_range(self):
        return np.maximum(2.0 * self.V_amp_half, self.eps)

    def update_histories(self, V):
        """Return (dV_raw, dV_eff). dV_eff is EMA if enabled, else dV_raw."""
        dV = np.zeros_like(V) if self.V_prev is None else (V - self.V_prev)
        if self.V_hist is not None:
            self.V_hist.push(V.reshape(-1))
        if self.dV_hist is not None:
            self.dV_hist.push(dV)
            if self.dV_hist.beta is not None:
                return dV, self.dV_hist.ema
        return dV, dV

    def maybe_update_offset(self):
        """Periodic bias update: V_bias <- V_bias - bias_frac * center."""
        if self.V_hist is None or self.bias_len <= 0:
            return
        if (self.V_hist.count < self.bias_len) or (self.off_per <= 0):
            return
        if self.V_hist.step % self.off_per != 0:
            return

        W = self.V_hist.get_last(self.bias_len, oldest_first=True)
        if self.bias_skip > 0:
            W = W[self.bias_skip:, :]
        center = 0.5 * (W.max(axis=0) + W.min(axis=0))
        self.V_bias[:, 0] -= (self.bias_frac * center).astype(self.dtype, copy=False)

    def cooldown_free(self, mask=None):
        if self.cooldown is None:
            if mask is None:
                return np.ones((self.n, 1), dtype=bool)
            out = np.zeros((self.n, 1), dtype=bool)
            out[mask] = True
            return out
        if mask is None:
            return self.cooldown <= 0
        out = np.zeros((self.n, 1), dtype=bool)
        out[mask] = (self.cooldown[mask] <= 0)
        return out

    def cooldown_tick(self, mask=None):
        if self.cooldown is None:
            return
        if mask is None:
            m = self.cooldown > 0
            self.cooldown[m] -= 1
        else:
            m = (self.cooldown > 0) & mask
            self.cooldown[m] -= 1

    def cooldown_start(self, mask):
        if self.cooldown is None:
            return
        self.cooldown[mask] = self.cooldown_steps

    def set_vref(self, mask=None):
        """V_ref = V_bias + V_sign * V_amp_half."""
        if mask is None:
            self.V_ref[...] = self.V_bias + self.V_sign.astype(self.dtype, copy=False) * self.V_amp_half
        else:
            self.V_ref[mask] = self.V_bias[mask] + self.V_sign[mask].astype(self.dtype, copy=False) * self.V_amp_half[mask]

    def act(self, V):
        return self.action_fn(V, self.K, self.V_ref, **self.action_kwargs).astype(self.dtype, copy=False)


# ---------------------------------------------------------------------------
# Switching-reference controller: AdaRef-VBias-AmpMaxDV
# ---------------------------------------------------------------------------

class AdaRefVBiasAmpMaxDV(BaseAdaController):
    """Per-bus sign selection + per-bus amplitude update from rolling max-|dV|.

    Sign rule (per bus):  +1 when V > V_bias, -1 otherwise.
    Amp rule (per bus):   V_amp_half <- amp_gain * max_{t in last amp_win_len} |dV_t|,
                          updated every amp_per steps (default 1).
    """

    def __init__(
        self,
        n, K,
        *,
        amp_win_len,                # required: rolling window length for amp update
        action_type="linear", action_kwargs=None,
        bias_len=0, bias_per=None, bias_skip=0, bias_frac=0.0,
        dv_hist_len=None,           # None -> auto: equal to amp_win_len
        dv_beta=None,
        cooldown_steps=0,
        eps=1e-8,
        amp_gain=0.5,               # empirical Algorithm 1 default
        amp_per=1, amp_min=None,
    ):
        if dv_hist_len is None:
            dv_hist_len = amp_win_len
        super().__init__(
            n, K,
            action_type=action_type, action_kwargs=action_kwargs,
            eps=eps,
            bias_len=bias_len, bias_per=bias_per, bias_skip=bias_skip, bias_frac=bias_frac,
            dv_hist_len=dv_hist_len, dv_beta=dv_beta,
            cooldown_steps=cooldown_steps,
        )

        self.amp_gain = float(amp_gain)
        if self.amp_gain < 0:
            raise ValueError("amp_gain must be >= 0")

        self.amp_win_len = int(amp_win_len)
        if self.amp_win_len <= 0:
            raise ValueError("amp_win_len must be positive")

        self.amp_per = int(amp_per)
        if self.amp_per <= 0:
            raise ValueError("amp_per must be positive (>= 1)")

        if self.dV_hist is None:
            raise ValueError("dv_hist_len must be > 0 to enable amp update")
        if self.dv_hist_len < self.amp_win_len:
            raise ValueError(f"dv_hist_len ({self.dv_hist_len}) must be >= amp_win_len ({self.amp_win_len})")

        if amp_min is None:
            self.amp_min = None
        else:
            mn = np.asarray(amp_min, dtype=self.dtype).reshape(-1)
            if mn.size == 1:
                mn = np.full(self.n, mn.item(), dtype=self.dtype)
            if mn.size != self.n:
                raise ValueError(f"amp_min must be scalar or length n={self.n}")
            if np.any(mn < 0):
                raise ValueError("amp_min must be >= 0")
            self.amp_min = mn.reshape(-1, 1)

        self.amp_maxabs_dv = np.full((self.n, 1), np.nan, dtype=self.dtype)

    def reset(self):
        super().reset()
        self.amp_maxabs_dv[...] = np.nan

    def _dv_window_maxabs(self):
        """Per-bus max |dV| over the last amp_win_len samples."""
        W = self.dV_hist.get_last(self.amp_win_len, oldest_first=True)
        if W.size == 0:
            return None, False
        W = W[:, :, 0]
        if not np.isfinite(W).all():
            return None, False
        m = np.max(np.abs(W), axis=0).astype(self.dtype, copy=False).reshape(-1, 1)
        return m, True

    def _maybe_update_amp_maxdv(self):
        updated = np.zeros((self.n, 1), dtype=bool)
        if (self.dV_hist.step % self.amp_per) != 0:
            return updated
        m, ok = self._dv_window_maxabs()
        if not ok:
            return updated
        self.amp_maxabs_dv[...] = m
        new_amp = (self.amp_gain * m).astype(self.dtype, copy=False)
        if self.amp_min is not None:
            new_amp = np.maximum(new_amp, self.amp_min)
        self.V_amp_half[...] = new_amp
        updated[...] = True
        return updated

    def step(self, V):
        V = self._col(V)
        if self.V_prev is None:
            self.V_prev = V.copy()
            self.set_vref()
            return np.zeros_like(V)

        self.update_histories(V)
        self.maybe_update_offset()
        self._maybe_update_amp_maxdv()

        self.cooldown_tick()
        free = self.cooldown_free()

        desired = np.where(V > self.V_bias, +1, -1).astype(np.int8).reshape(-1, 1)
        did_switch = free & (desired != self.V_sign)
        if np.any(did_switch):
            self.V_sign[did_switch] = desired[did_switch]
        self.cooldown_start(did_switch)

        self.set_vref()
        out = self.act(V)
        self.V_prev = V.copy()
        return out
