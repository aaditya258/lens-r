"""Drift-aware LENS (LENS-DA) and the simple sliding-baseline fix.

All settings are fixed a priori and every learned quantity comes from normal data only.

Per sensor i, at update time t (1 Hz data):
  short window  S   = the last S_LEN samples  (t-S_LEN, t]
  reference     Ref = the REF_LEN samples before the short window
  Simple fix    D_sw = |x_i(t) - mean_Ref(x_i)| / sigma_i            (sigma from normal data)
  Level change  C    = |mean_S(x_i) - mean_Ref(x_i)| / s_C,i          (s_C = normal variability of that difference)
  Relational ch RC   = |mean_S(e_i) - mean_Ref(e_i)| / s_RC,i         (e = residual of predicting x_i from all others)
  Drift state   d_i  = |mean_Ref(x_i) - mu_i| / sigma_i               (how far the recent baseline sits from normal)
  Gated abs dev D_g  = D_abs * exp(-max(d_i - Z_D, 0))                 (drifting sensors lose absolute-deviation weight)
  Entropy       H    = windowed entropy (as in LENS)
LENS-DA score = equal-weight fusion of min-max normalised (H, D_g, C, RC), normalised to sum to 1.
Sensor state: 'drifting' if d_i > Z_D; 'abrupt' if C > Z_D or RC > Z_D; else 'stable'.
"""
import numpy as np
from lens import CAP, EPS, minmax

S_LEN = 10        # seconds
REF_LEN = 1800    # seconds (30 min)
Z_D = 3.0
MIN_REF = 60


def rolling_mean_at(A, end_excl, length):
    """Mean of rows [end_excl-length, end_excl) for each end index, via cumulative sums (shrinks at the start)."""
    c = np.cumsum(np.vstack([np.zeros((1, A.shape[1])), A]), 0)
    lo = np.maximum(end_excl - length, 0)
    n = np.maximum(end_excl - lo, 1)[:, None]
    return (c[end_excl] - c[lo]) / n


def residuals(X, nm):
    Z = (X.astype(np.float64) - nm.mu) / nm.sd
    return (Z @ nm.P) / np.diag(nm.P)


def _robust_sd(A):
    med = np.median(A, 0)
    return 1.4826 * np.median(np.abs(A - med), 0)


def two_timescale(A, upd):
    """(short mean, reference mean) at each update position."""
    end = upd + 1
    return rolling_mean_at(A, end, S_LEN), rolling_mean_at(A, np.maximum(end - S_LEN, MIN_REF), REF_LEN)


class DriftModel:
    def fit(self, Xn, nm, step=7):
        """Normal variability of the short-minus-reference differences (robust scale, with floors)."""
        self.nm = nm
        upd = np.arange(REF_LEN + S_LEN, len(Xn), step)
        s, r = two_timescale(Xn.astype(np.float64), upd)
        E = residuals(Xn, nm)
        se, re = two_timescale(E, upd)
        self.sC = np.maximum(_robust_sd(s - r), 0.05 * nm.sd)
        self.sRC = np.maximum(_robust_sd(se - re), 0.05)
        return self

    def features(self, X, upd):
        nm = self.nm
        Xd = X.astype(np.float64)
        s, r = two_timescale(Xd, upd)
        E = residuals(X, nm)
        se, re = two_timescale(E, upd)
        xu = Xd[upd]
        D_abs = np.minimum(np.abs(xu - nm.mu) / nm.sd, CAP)
        drift = np.minimum(np.abs(r - nm.mu) / nm.sd, CAP)
        return dict(
            D_sw=np.minimum(np.abs(xu - r) / nm.sd, CAP).astype(np.float32),
            C=np.minimum(np.abs(s - r) / self.sC, CAP).astype(np.float32),
            RC=np.minimum(np.abs(se - re) / self.sRC, CAP).astype(np.float32),
            drift=drift.astype(np.float32),
            D_g=(D_abs * np.exp(-np.maximum(drift - Z_D, 0))).astype(np.float32),
        )


def lens_da(H, D_g, C, RC, use=("H", "D_g", "C", "RC")):
    parts = dict(H=minmax(H), D_g=minmax(D_g), C=minmax(C), RC=minmax(RC))
    s = sum(parts[k] for k in use) / len(use)
    return s / (s.sum(-1, keepdims=True) + EPS)


def sensor_state(drift, C, RC):
    st = np.where(drift > Z_D, 1, 0)                   # 1 drifting
    st = np.where((C > Z_D) | (RC > Z_D), 2, st)       # 2 abrupt (takes precedence)
    return st
