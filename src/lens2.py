"""LENS v2 and the two new baselines, exactly as fixed in PREREG.md.

1. Context-conditioned residual: ridge prediction of z_i(t) from all other tags at t plus the tag's own
   level over [t-600 s, t-300 s).
2. Online new-normal adaptation of each tag's residual bias and scale (EWMA, tau = 1800 s),
   frozen while the tag looks abnormal (u > 3) for at most 30 min.
3. Onset precedence: per-tag CUSUM of the adapted score (k = 2).
Baselines: conditional reference (CondAttr-style kNN) and linear forecast residual (GDN-style).
"""
import numpy as np

LAG_FROM, LAG_TO = 600, 300      # own-level window [t-600, t-300)
RIDGE = 1.0
TAU = 1800.0
DT = 3
GATE, FREEZE_MAX = 3.0, 600      # freeze while u > 3, at most 600 updates (30 min)
K_CUSUM, CAP_U, CAP_S = 2.0, 50.0, 500.0
FLOOR = 0.05
KNN_K, KNN_N, KNN_HELD = 20, 30000, 2000
FC_LEN = 10


def _cum(Z):
    return np.cumsum(np.vstack([np.zeros((1, Z.shape[1])), Z.astype(np.float64)]), 0)


def _window_mean(C, lo, hi):
    lo = np.maximum(lo, 0); hi = np.maximum(hi, lo + 1)
    return (C[hi] - C[lo]) / (hi - lo)[:, None]


def own_level(C, t):
    return _window_mean(C, t - LAG_FROM, t - LAG_TO)


def _mad(E):
    med = np.median(E, 0)
    return np.maximum(1.4826 * np.median(np.abs(E - med), 0), FLOOR)


def _valid_idx(T, bounds, lookback, step, seed=0):
    b = [0, T] if bounds is None else list(bounds)
    idx = np.concatenate([np.arange(s + lookback, e, step) for s, e in zip(b[:-1], b[1:])])
    return idx


class ContextModel:
    def fit(self, Xn, nm, bounds=None, own_history=True):
        self.nm, self.own = nm, own_history
        Z = ((Xn - nm.mu) / nm.sd).astype(np.float32)
        N = Z.shape[1]
        idx = _valid_idx(len(Z), bounds, LAG_FROM, 5)
        C = _cum(Z)
        Zi = Z[idx].astype(np.float64)
        L = own_level(C, idx)
        M = np.hstack([Zi, L])
        G = M.T @ M
        c = M.T @ Zi
        self.W = np.zeros((N, N)); self.v = np.zeros(N)
        for i in range(N):
            F = [j for j in range(N) if j != i] + ([N + i] if own_history else [])
            w = np.linalg.solve(G[np.ix_(F, F)] + RIDGE * np.eye(len(F)), c[F, i])
            self.W[[j for j in range(N) if j != i], i] = w[: N - 1]
            if own_history:
                self.v[i] = w[-1]
        E = Zi - self.predict(Zi, L)
        self.sigma0 = _mad(E)
        return self

    def predict(self, Zu, Lu):
        return Zu @ self.W + Lu * self.v

    def residual_stream(self, X, upd):
        Z = ((X - self.nm.mu) / self.nm.sd)
        C = _cum(Z)
        Zu = Z[upd].astype(np.float64)
        return Zu - self.predict(Zu, own_level(C, upd))


def adapt(E, sigma0):
    """Online new-normal adaptation; returns adapted scores u (U, N)."""
    U, N = E.shape
    a = DT / TAU
    b = np.zeros(N); s = sigma0.copy(); frozen = np.zeros(N, int)
    out = np.empty((U, N), np.float32)
    for t in range(U):
        e = E[t]
        u = np.minimum(np.abs(e - b) / np.maximum(s, 0.5 * sigma0), CAP_U)
        out[t] = u
        upd = (u <= GATE) | (frozen >= FREEZE_MAX)
        frozen = np.where(upd, 0, frozen + 1)
        b = np.where(upd, (1 - a) * b + a * e, b)
        s = np.where(upd, (1 - a) * s + a * 1.2533 * np.abs(e - b), s)
    return out


def static_score(E, sigma0):
    return np.minimum(np.abs(E) / sigma0, CAP_U).astype(np.float32)


def cusum(Uu):
    S = np.zeros(Uu.shape[1]); out = np.empty_like(Uu)
    for t in range(len(Uu)):
        S = np.minimum(np.maximum(0.0, S + Uu[t] - K_CUSUM), CAP_S)
        out[t] = S
    return out


def all_variants(X, upd, cm1, cm0):
    """Per-update scores for LENS v2 and its ablations over a whole stream."""
    E1 = cm1.residual_stream(X, upd)
    E0 = cm0.residual_stream(X, upd)
    ua1, ua0 = adapt(E1, cm1.sigma0), adapt(E0, cm0.sigma0)
    us1, us0 = static_score(E1, cm1.sigma0), static_score(E0, cm0.sigma0)
    return {
        "LENS v2 (full)": cusum(ua1),
        "LENS v2 without own-history term": cusum(ua0),
        "LENS v2 without adaptation": cusum(us1),
        "LENS v2 without CUSUM": ua1,
        "Static context residual (no history, no adaptation, no CUSUM)": us0,
    }


class KNNReference:
    """Conditional reference, CondAttr-style simplification (raw standardised space, target excluded)."""

    def fit(self, Xn, nm, seed=0):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xn))
        self.nm = nm
        self.Zk = ((Xn[perm[:KNN_N]] - nm.mu) / nm.sd).astype(np.float32)
        held = ((Xn[perm[KNN_N:KNN_N + KNN_HELD]] - nm.mu) / nm.sd).astype(np.float32)
        E = np.stack([self._resid(q) for q in held])
        self.scale = _mad(E)
        return self

    def _resid(self, q):
        D2 = (self.Zk - q) ** 2
        dm = D2.sum(1, keepdims=True) - D2                      # distance excluding each target tag
        nb = np.argpartition(dm, KNN_K, axis=0)[:KNN_K]          # (K, N)
        pred = self.Zk[nb, np.arange(self.Zk.shape[1])].mean(0)
        return q - pred

    def score(self, Xu):
        Z = ((Xu - self.nm.mu) / self.nm.sd).astype(np.float32)
        return np.stack([np.minimum(np.abs(self._resid(q)) / self.scale, CAP_U) for q in Z])


class ForecastResidual:
    """Linear forecast residual, GDN-style: predict z(t) from the mean of all tags over [t-10, t-1]."""

    def fit(self, Xn, nm, bounds=None):
        self.nm = nm
        Z = ((Xn - nm.mu) / nm.sd).astype(np.float32)
        idx = _valid_idx(len(Z), bounds, FC_LEN, 5)
        C = _cum(Z)
        F = _window_mean(C, idx - FC_LEN, idx)
        Y = Z[idx].astype(np.float64)
        self.Wf = np.linalg.solve(F.T @ F + RIDGE * np.eye(F.shape[1]), F.T @ Y)
        E = Y - F @ self.Wf
        self.med = np.median(E, 0)
        q75, q25 = np.percentile(E, [75, 25], 0)
        self.iqr = np.maximum(q75 - q25, FLOOR)
        return self

    def score_stream(self, X, upd):
        Z = ((X - self.nm.mu) / self.nm.sd)
        C = _cum(Z)
        F = _window_mean(C, upd - FC_LEN, upd)
        E = Z[upd] - F @ self.Wf
        return np.minimum(np.abs(E - self.med) / self.iqr, CAP_U).astype(np.float32)
