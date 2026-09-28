"""LENS-R: sparse, two-timescale reconstruction for attack localisation.

Consistency model (normal data only), on standardised tags z:
  J(z) = z^T M z, with M either
    'T2'  : regularised precision matrix of normal data (Mahalanobis / Hotelling T^2)
    'SPE' : residual projector I - V V^T of a PCA keeping 95% of normal variance (relational breaks only)
Reconstructing a set R of tags (replacing them by the values that best restore consistency) gives
  J_R(z) = J(z) - b_R^T (M_RR)^-1 b_R,  b = M z,  and fault magnitudes f_R = (M_RR)^-1 b_R.
For a single tag this is the classical reconstruction-based contribution (Alcala & Qin, 2009).

Per update t:
  slow stage : z_slow = mean z over [t-1800 s, t-300 s). Greedy sparse repair while J > normal 99.9th percentile
               (at most Q_MAX tags) -> drifting set Q with offsets f_Q.
  correction : z_c = z_fast - offsets on Q, where z_fast = mean z over (t-10 s, t]   (drift quarantine)
  fast stage : greedy sparse repair of z_c for K_FAST steps; score = consistency gain when each tag was repaired,
               and for the rest their marginal gain after the repaired tags are removed.
Also returned: expected value of each repaired tag (reading minus reconstructed fault) in engineering units.
"""
import numpy as np

SLOW_FROM, SLOW_TO, FAST = 1800, 300, 10
Q_MAX, K_FAST = 10, 5
PCA_VAR = 0.95
EPS = 1e-6


def _cum(Z):
    return np.cumsum(np.vstack([np.zeros((1, Z.shape[1])), Z.astype(np.float64)]), 0)


def _wmean(C, lo, hi):
    lo = np.maximum(lo, 0); hi = np.maximum(hi, lo + 1)
    return (C[hi] - C[lo]) / (hi - lo)[:, None]


def repair_cost(M, b, J0, S):
    A = M[np.ix_(S, S)] + EPS * np.eye(len(S))
    f = np.linalg.solve(A, b[S])
    return J0 - b[S] @ f, f


def greedy(z, M, steps, thr=None):
    """Greedy sparse repair. Returns gains per tag, repaired list, fault magnitudes, final J."""
    N = len(z)
    b = M @ z; J0 = float(z @ b)
    R, gains, Jcur, f = [], np.zeros(N), J0, np.zeros(0)
    for _ in range(steps):
        if thr is not None and Jcur <= thr:
            break
        best = (np.inf, None, None)
        for j in range(N):
            if j in R:
                continue
            J, fj = repair_cost(M, b, J0, R + [j])
            if J < best[0]:
                best = (J, j, fj)
        J, j, f = best
        gains[j] = Jcur - J; Jcur = J; R.append(j)
    for j in range(N):
        if j not in R:
            gains[j] = Jcur - repair_cost(M, b, J0, R + [j])[0]
    return np.maximum(gains, 0), R, f, Jcur


class ReconModel:
    def fit(self, Xn, nm, kind="T2", bounds=None, step=11):
        self.nm, self.kind = nm, kind
        Z = ((Xn - nm.mu) / nm.sd)
        if kind == "T2":
            self.M = nm.P.copy()
        else:
            C = np.cov(Z[::step].T)
            w, V = np.linalg.eigh(C)
            o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
            k = int(np.searchsorted(np.cumsum(w) / w.sum(), PCA_VAR) + 1)
            self.k = k
            self.M = np.eye(C.shape[0]) - V[:, :k] @ V[:, :k].T
        # 99.9th percentile of J for slow-averaged normal data (threshold for drift repair)
        Cn = _cum(Z)
        b = [0, len(Z)] if bounds is None else list(bounds)
        idx = np.concatenate([np.arange(s + SLOW_FROM, e, 61) for s, e in zip(b[:-1], b[1:])])
        zs = _wmean(Cn, idx - SLOW_FROM, idx - SLOW_TO)
        self.thr_slow = float(np.quantile(np.einsum("ij,jk,ik->i", zs, self.M, zs), 0.999))
        # localisability index: how well each tag is reconstructed from the others (1 - 1/(P_jj * var_j))
        P = nm.P
        self.localisability = np.clip(1 - 1 / np.maximum(np.diag(P), EPS), 0, 1)
        return self

    def window(self, X, upd, drift_correct=True, sparse=True):
        """Scores (len(upd), N), drifting sets, and expected values for the given update positions."""
        nm = self.nm
        Z = (X - nm.mu) / nm.sd
        C = _cum(Z)
        zf = _wmean(C, upd + 1 - FAST, upd + 1)
        zs = _wmean(C, upd - SLOW_FROM, upd - SLOW_TO)
        out, Qs, expv = [], [], []
        for t in range(len(upd)):
            z = zf[t].copy()
            Q = []
            if drift_correct:
                _, Q, fQ, _ = greedy(zs[t], self.M, Q_MAX, thr=self.thr_slow)
                if len(Q):
                    z[Q] -= fQ
            g, R, fR, _ = greedy(z, self.M, K_FAST if sparse else 0)
            out.append(g); Qs.append(Q)
            ev = {}
            for j, fj in zip(R, fR):
                ev[j] = float(nm.mu[j] + (z[j] - fj) * nm.sd[j])       # expected reading in engineering units
            expv.append(ev)
        return np.array(out), Qs, expv
