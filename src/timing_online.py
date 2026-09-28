"""Online per-update cost: running means maintained incrementally (O(N) per update); timed part = the method's own computation."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import time, pickle, numpy as np, pandas as pd, shap
import data as Dm
from recon import ReconModel, greedy, _cum, _wmean, Q_MAX, K_FAST

def per_call(fn, reps):
    fn(); t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / reps * 1000

rows = []
for name in ["swat", "hai"]:
    ds = Dm.load(name); M = pickle.load(open(f"cache/{name}_models.pkl", "rb")); L = pickle.load(open(f"cache/{name}_lens2_models.pkl", "rb"))
    nm = M["nm"]; X = list(ds.streams.values())[-1].X
    rm = ReconModel().fit(ds.Xn, nm, "T2", ds.normal_bounds)
    Z = (X - nm.mu) / nm.sd; C = _cum(Z)
    us = np.random.default_rng(0).integers(4000, len(X) - 10, 50)
    zf = _wmean(C, us + 1 - 10, us + 1); zs = _wmean(C, us - 1800, us - 300)
    k = [0]
    def lens_full():
        i = k[0] % len(us); k[0] += 1
        z = zf[i].copy(); _, Q, fQ, _ = greedy(zs[i], rm.M, Q_MAX, thr=rm.thr_slow)
        if len(Q): z[Q] -= fQ
        greedy(z, rm.M, K_FAST)
    def lens_fast():
        i = k[0] % len(us); k[0] += 1; greedy(zf[i], rm.M, K_FAST)
    F = _wmean(C, us - 10, us)
    def gdn():
        i = k[0] % len(us); k[0] += 1; np.abs(Z[us[i]] - F[i] @ L["fc"].Wf - L["fc"].med) / L["fc"].iqr
    rf = M["rf"]; rf.n_jobs = 1; ex = shap.TreeExplainer(rf)
    def tshap():
        i = k[0] % len(us); k[0] += 1; ex.shap_values(X[us[i]:us[i] + 1], check_additivity=False)
    def knn():
        i = k[0] % len(us); k[0] += 1; L["knn"].score(X[us[i]:us[i] + 1])
    for m, f, r in [("LENS-R, full update", lens_full, 50), ("LENS-R, ranking stage only", lens_fast, 50),
                    ("GDN-style residual", gdn, 200), ("Conditional reference (kNN)", knn, 20), ("TreeSHAP on RF", tshap, 30)]:
        rows.append(dict(ds=name, tags=len(ds.tags), method=m, ms=per_call(f, r)))
    print(name, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv("results/timing_online.csv", index=False)
print(df.pivot(index="method", columns="ds", values="ms").round(2).to_string())
import platform; print(platform.processor() or platform.machine(), os.cpu_count(), "cores; timed on 1 thread")
