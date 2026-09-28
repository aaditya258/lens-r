"""One-at-a-time settings sweep for LENS-R on SWaT and HAI (development data only; WADI is not touched).
recon.py is not modified: module-level settings are overridden at run time and restored afterwards."""
import pickle, time
import numpy as np
import pandas as pd
import data as Dm
import recon
from recon import ReconModel, _cum, _wmean
from lens import EPS
from evaluate import rank_eval

BASE = dict(SLOW_FROM=1800, SLOW_TO=300, FAST=10, Q=0.999, LAM=0.01, K_FAST=5, Q_MAX=10, N=10)
GRID = {"SLOW_FROM": [900, 3600], "SLOW_TO": [120, 600], "FAST": [5, 30], "Q": [0.99, 0.9999],
        "LAM": [0.001, 0.1], "K_FAST": [1, 3], "Q_MAX": [5, 20], "N": [5, 20]}
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)


def threshold(Z, M, bounds, frm, to, q):
    b = [0, len(Z)] if bounds is None else list(bounds)
    C = _cum(Z)
    idx = np.concatenate([np.arange(s + frm, e, 61) for s, e in zip(b[:-1], b[1:])])
    zs = _wmean(C, idx - frm, idx - to)
    return float(np.quantile(np.einsum("ij,jk,ik->i", zs, M, zs), q))


def run(ds, P, nm, Zn, Cn, cfg):
    recon.SLOW_FROM, recon.SLOW_TO, recon.FAST, recon.K_FAST, recon.Q_MAX = \
        cfg["SLOW_FROM"], cfg["SLOW_TO"], cfg["FAST"], cfg["K_FAST"], cfg["Q_MAX"]
    rm = ReconModel(); rm.nm, rm.kind = nm, "T2"
    rm.M = np.linalg.inv(Cn + cfg["LAM"] * np.eye(Cn.shape[0]))
    rm.thr_slow = threshold(Zn, rm.M, ds.normal_bounds, cfg["SLOW_FROM"], cfg["SLOW_TO"], cfg["Q"])
    rows = []
    for w in P["windows"]:
        if not w["localisable"]:
            continue
        X = ds.streams[w["stream"]].X
        S, _, _ = rm.window(X, w["upd"][:cfg["N"]], True, True)
        s = (S / (S.sum(1, keepdims=True) + EPS)).mean(0)
        rows.append(rank_eval(s, ds.tags, ds.stage, w["points"], w["stages"])["rr"])
    return float(np.mean(rows)), len(rows)


if __name__ == "__main__":
    saved = {k: getattr(recon, k) for k in ["SLOW_FROM", "SLOW_TO", "FAST", "K_FAST", "Q_MAX"]}
    out = []
    for name in ["swat", "hai"]:
        ds = Dm.load(name)
        P = pickle.load(open(f"cache/{name}.pkl", "rb"))
        nm = pickle.load(open(f"cache/{name}_models.pkl", "rb"))["nm"]
        Zn = (ds.Xn - nm.mu) / nm.sd
        Cn = Zn.T @ Zn / len(Zn)
        base, n = run(ds, P, nm, Zn, Cn, BASE)
        out.append(dict(ds=name, param="baseline", value="frozen", mrr=base, n=n))
        log(name, "baseline", round(base, 3))
        for p, vals in GRID.items():
            for v in vals:
                cfg = dict(BASE); cfg[p] = v
                m, n = run(ds, P, nm, Zn, Cn, cfg)
                out.append(dict(ds=name, param=p, value=v, mrr=m, n=n))
                log(name, p, v, round(m, 3))
        del Zn
    for k, v in saved.items():
        setattr(recon, k, v)
    df = pd.DataFrame(out)
    df.to_csv("results/sweep_lensr.csv", index=False)
    piv = df.pivot_table(index=["param", "value"], columns="ds", values="mrr").round(3)
    print(piv.to_string())
