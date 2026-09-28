"""Development evaluation of LENS-R on SWaT and HAI (WADI untouched). Usage: python3 run_recon.py"""
import pickle, time, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import data as Dm
from lens import EPS
from recon import ReconModel
from drift_run import build
from run2 import lens2_windows, scores as base_scores
from evaluate import rank_eval, random_expect, summarise, paired, boot_ci

N = 10
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
VARIANTS = {"full": (True, True), "no drift correction": (False, True),
            "no sparse repair": (True, False), "classic single-tag RBC": (False, False)}


def norm_mean(A):
    return (A / (A.sum(1, keepdims=True) + EPS)).mean(0)


if __name__ == "__main__":
    per_all, loc_rows, drift_log, examples = [], [], {}, []
    for name in ["swat", "hai"]:
        P, _ = build(name)
        M = pickle.load(open(f"cache/{name}_models.pkl", "rb"))
        mods = pickle.load(open(f"cache/{name}_lens2_models.pkl", "rb"))
        ds = Dm.load(name)
        P = lens2_windows(ds, P, mods)
        rms = {k: ReconModel().fit(ds.Xn, M["nm"], k, ds.normal_bounds) for k in ["T2", "SPE"]}
        log(name, "models; SPE components:", rms["SPE"].k, "slow thresholds:", {k: round(r.thr_slow, 1) for k, r in rms.items()})
        ws = [w for w in P["windows"] if w["localisable"]]
        qcount = {}
        for w in ws:
            X = ds.streams[w["stream"]].X
            sc = {k: v for k, v in base_scores(P, w).items()
                  if k in ("Linear forecast residual (GDN-style)", "LENS-SW (state-switched, exploratory)",
                           "Simple fix: sliding-baseline deviation", "LENS-U (original, label-free)",
                           "Conditional reference (CondAttr-style kNN)", "Raw deviation |z|", "TreeSHAP on RF",
                           "LENS v2 without CUSUM")}
            for kind, rm in rms.items():
                for vname, (dc, sp) in VARIANTS.items():
                    S, Qs, ev = rm.window(X, w["upd"][:N], dc, sp)
                    sc[f"LENS-R {kind} ({vname})"] = norm_mean(S)
                    if vname == "full":
                        for Q in Qs:
                            for q in Q:
                                qcount[(kind, ds.tags[q])] = qcount.get((kind, ds.tags[q]), 0) + 1
                        if kind == "T2" and len(examples) < 12:
                            top = int(np.argmax(norm_mean(S)))
                            e = ev[-1].get(top)
                            examples.append(dict(ds=name, attack=w["label"], truth=";".join(w["points"]), top=ds.tags[top],
                                                 reading=float(X[w["upd"][N - 1], top]), expected=e))
            li = max(rms["T2"].localisability[ds.tags.index(p)] for p in w["points"])
            for m, s in sc.items():
                r = rank_eval(s, ds.tags, ds.stage, w["points"], w["stages"])
                per_all.append(dict(ds=name, split=w["split"], attack=f"{name}_{w['label']}", method=m, localisability=li, **r))
        drift_log[name] = sorted([(k[0], k[1], v) for k, v in qcount.items()], key=lambda x: -x[2])[:12]
        log(name, "done")

    per = pd.DataFrame(per_all)
    per.to_csv("results/recon_per_attack.csv", index=False)
    keys = [m for m in per.method.unique()]
    for label, sel in [("SWaT all (33)", per.ds == "swat"), ("HAI all (58)", per.ds == "hai"),
                       ("POOLED ALL (91)", per.ds.notna()), ("POOLED TEST (35)", per.split == "test")]:
        d = per[sel]; g = d.groupby("method")
        t = pd.DataFrame({"n": g.size(), "MRR": g.rr.mean(), "Top@1": g.top1.mean(), "Top@3": g.top3.mean(),
                          "Top@5": g.top5.mean(), "Stage": g.stage_hit.mean()}).sort_values("MRR", ascending=False)
        print(f"\n=== {label} ===\n", t.round(3).to_string())
        t.to_csv(f"results/recon_{label.split(' (')[0].replace(' ', '_').lower()}.csv")
    pa = per
    best = pa.groupby("method").rr.mean().filter(like="LENS-R").idxmax()
    print("\nBest LENS-R variant on pooled all (development):", best)
    comp = ["Linear forecast residual (GDN-style)", "LENS-SW (state-switched, exploratory)", "Simple fix: sliding-baseline deviation",
            "LENS-U (original, label-free)", "Conditional reference (CondAttr-style kNN)", f"LENS-R {best.split()[1]} (classic single-tag RBC)",
            f"LENS-R {best.split()[1]} (no drift correction)", f"LENS-R {best.split()[1]} (no sparse repair)"]
    print(pd.DataFrame([paired(pa, best, b) for b in comp if b != best])[["b", "n", "mean_diff", "wins", "losses", "p"]].to_string())
    pt = pa[pa.split == "test"]
    print("\n(test only)\n", pd.DataFrame([paired(pt, best, b) for b in comp + ["TreeSHAP on RF"] if b != best])[["b", "n", "mean_diff", "wins", "losses", "p"]].to_string())
    print("\nLocalisability index vs reciprocal rank (Spearman, pooled all):")
    rows = []
    for m, d in pa.groupby("method"):
        r = spearmanr(d.localisability, d.rr); rows.append((m, round(r.statistic, 3), round(r.pvalue, 4)))
    print(pd.DataFrame(rows, columns=["method", "rho", "p"]).sort_values("rho", ascending=False).to_string())
    q = pa[pa.method == best].assign(bin=lambda d: pd.qcut(d.localisability, 3, labels=["low", "mid", "high"], duplicates="drop"))
    print("\nBest variant MRR by localisability tercile:", q.groupby("bin", observed=True).rr.agg(["mean", "size"]).round(3).to_dict())
    print("\nMost often quarantined (drifting) tags:", drift_log)
    print("\nExample operator output (top tag, reading, expected):")
    print(pd.DataFrame(examples).to_string())
    json.dump(dict(best=best, drift=drift_log), open("results/recon_meta.json", "w"), indent=1, default=str)
