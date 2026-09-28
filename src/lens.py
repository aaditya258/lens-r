"""LENS: Lightweight, state-aware sensor prioritisation.

Four per-sensor indicators, computed every DT samples:
  H  windowed entropy over the last W samples (K fixed bins from normal data)
  D  baseline deviation |x - mu| / sigma (normal data), capped
  R  relevance: supervised = |Kendall tau| with the attack label (static)
                label-free (LENS-U) = residual of a regularised linear prediction of the
                sensor from all other sensors, fitted on normal data only
  B  recursive belief with forgetting: supervised = attack vs normal bin likelihoods;
                label-free = normal likelihood vs a uniform off-normal alternative
Each indicator is min-max normalised across sensors at every update, fused with the
current mode's coefficients, and normalised to sum to one. The layer never changes the
detector's input (Option A).
"""
import numpy as np
from scipy.stats import kendalltau

EPS = 1e-9
CAP = 50.0


class Config:
    W = 50          # entropy window (samples)
    K = 10          # entropy / likelihood bins
    DT = 3          # update interval (samples = seconds)
    Z_TH = 3.0      # deviation threshold for the plant-wide risk r(t)
    RHO = 0.95      # belief forgetting factor
    WM = 20         # detector-alarm window for f(t), in updates
    RIDGE = 1e-2    # covariance regulariser for the label-free relevance
    F1, F2 = 0.05, 0.20   # alarm-rate thresholds (NORMAL / STRESSED)
    R_Q1, R_Q2 = 0.95, 0.999  # r thresholds = these quantiles of r(t) on normal data

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class NormalModel:
    """Everything LENS-U learns, from normal data only."""

    def fit(self, Xn, cfg):
        self.cfg = cfg
        X = Xn.astype(np.float64)
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.maximum(sd, 1e-4 * np.maximum(1.0, np.abs(self.mu)))
        self.lo, self.hi = X.min(0), X.max(0)
        B = self.bins(Xn)
        self.p0 = np.stack([np.bincount(B[:, i], minlength=cfg.K) for i in range(X.shape[1])]).astype(float)
        self.p0 = (self.p0 + 1) / (self.p0.sum(1, keepdims=True) + cfg.K)          # Laplace smoothing
        Z = (X - self.mu) / self.sd
        C = Z.T @ Z / len(Z) + cfg.RIDGE * np.eye(Z.shape[1])
        self.P = np.linalg.inv(C)
        e = self._resid(Z[:: max(1, len(Z) // 200000)])
        self.res_sd = np.maximum(e.std(0), 1e-3)
        return self

    def bins(self, X):
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        b = np.floor((X - self.lo) / span * self.cfg.K).astype(np.int64)
        return np.clip(b, 0, self.cfg.K - 1)

    def deviation(self, X):
        return np.minimum(np.abs((X - self.mu) / self.sd), CAP)

    def _resid(self, Z):
        # residual of predicting each standardised variable from all others (precision-matrix form)
        return (Z @ self.P) / np.diag(self.P)

    def relevance_u(self, X):
        Z = (X.astype(np.float64) - self.mu) / self.sd
        return np.minimum(np.abs(self._resid(Z)) / self.res_sd, CAP)


class SupervisedModel:
    """What supervised LENS adds, from the training partition (normal + training attacks)."""

    def fit(self, nm, Xtr, ytr, cfg, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ytr), size=min(40000, len(ytr)), replace=False)
        tau = []
        for i in range(Xtr.shape[1]):
            x = Xtr[idx, i]
            t = kendalltau(x, ytr[idx]).statistic if x.std() > 0 else 0.0
            tau.append(0.0 if np.isnan(t) else abs(t))
        self.R = np.array(tau)
        B = nm.bins(Xtr[ytr == 1])
        p1 = np.stack([np.bincount(B[:, i], minlength=cfg.K) for i in range(Xtr.shape[1])]).astype(float)
        self.p1 = (p1 + 1) / (p1.sum(1, keepdims=True) + cfg.K)
        return self


def rolling_entropy(Bidx, K, W, upd):
    """Entropy of the last W bin indices at each update position (vectorised per sensor)."""
    T, N = Bidx.shape
    H = np.zeros((len(upd), N), np.float32)
    lo = np.maximum(upd - W + 1, 0)
    for i in range(N):
        oh = np.zeros((T + 1, K), np.int32)
        oh[np.arange(1, T + 1), Bidx[:, i]] = 1
        c = np.cumsum(oh, 0)
        cnt = (c[upd + 1] - c[lo]).astype(np.float32)
        p = cnt / cnt.sum(1, keepdims=True)
        H[:, i] = -(p * np.log2(p + EPS)).sum(1)
    return H


def belief(Bidx_upd, L1, L0, rho):
    """Recursive Bayesian belief with forgetting toward 0.5, over successive updates."""
    U, N = Bidx_upd.shape
    cols = np.arange(N)
    out = np.empty((U, N), np.float32)
    b = np.full(N, 0.5)
    for u in range(U):
        k = Bidx_upd[u]
        l1, l0 = L1[cols, k], L0[cols, k]
        bt = rho * b + (1 - rho) * 0.5
        b = l1 * bt / (l1 * bt + l0 * (1 - bt))
        b = np.clip(b, 1e-6, 1 - 1e-6)
        out[u] = b
    return out


def indicators(stream_X, nm, sm, cfg, start=None):
    """All raw indicators at every update position of a stream."""
    T = len(stream_X)
    upd = np.arange(cfg.W - 1 if start is None else start, T, cfg.DT)
    Bidx = nm.bins(stream_X)
    Xu = stream_X[upd]
    out = dict(upd=upd,
               H=rolling_entropy(Bidx, cfg.K, cfg.W, upd),
               D=nm.deviation(Xu).astype(np.float32),
               RU=nm.relevance_u(Xu).astype(np.float32))
    unif = np.full_like(nm.p0, 1.0 / cfg.K)
    out["BU"] = belief(Bidx[upd], unif, nm.p0, cfg.RHO)
    if sm is not None:
        out["BS"] = belief(Bidx[upd], sm.p1, nm.p0, cfg.RHO)
    out["r"] = (out["D"] > cfg.Z_TH).mean(1).astype(np.float32)
    return out


def minmax(A):
    lo, hi = A.min(-1, keepdims=True), A.max(-1, keepdims=True)
    return np.where(hi > lo, (A - lo) / np.maximum(hi - lo, EPS), 0.0)


MODES = ("NORMAL", "STRESSED", "ATTACK")


def modes(r, f, th):
    """0 NORMAL, 1 STRESSED, 2 ATTACK (AND logic in both conditions)."""
    m = np.full(len(r), 2, np.int8)
    m[(f < th["f2"]) & (r < th["r2"])] = 1
    m[(f < th["f1"]) & (r < th["r1"])] = 0
    return m


def fuse(Hn, Rn, Bn, Dn, mode, coef):
    """coef: array (3 modes, 4) over (H, R, B, D); rows sum to 1."""
    c = coef[mode]                                     # (U, 4)
    s = c[:, :1] * Hn + c[:, 1:2] * Rn + c[:, 2:3] * Bn + c[:, 3:4] * Dn
    return s / (s.sum(1, keepdims=True) + EPS)
