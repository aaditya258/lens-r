"""Pre-registered drift-injection experiment on HAI (PREREG.md). Usage: python3 inject.py"""
import pickle, time
import numpy as np
import pandas as pd
import data as Dm
from lens import Config, indicators, EPS
from drift import DriftModel
from lens2 import all_variants
from select_variant import lens_u, lens_sw
from evaluate import rank_eval

N = 10
MS = [0, 3, 6, 10]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)


def norm_mean(A):
    A = A[:N]
    return (A / (A.sum(1, keepdims=True) + EPS)).mean(0)


def run_condition(ds, att, nm, dm, mods, cfg, tags_by_stream, m):
    rows = []
    for sname, s in ds.streams.items():
        X = s.X.copy()
        ramp = np.minimum(1.0, np.arange(len(X)) / 7200.0)[:, None]
        cols = tags_by_stream[sname]
        if m > 0 and len(cols):
            X[:, cols] += (m * nm.sd[cols])[None, :].astype(np.float32) * ramp
        ind = indicators(X, nm, None, cfg)
        upd = ind["upd"]
        V = all_variants(X, upd, mods["cm1"], mods["cm0"])
        fc = mods["fc"].score_stream(X, upd)
        for _, a in att[att.stream == sname].iterrows():
            u0 = int(np.searchsorted(upd, a.onset)); u = np.arange(u0, u0 + N)
            w = {k: ind[k][u] for k in ("H", "D", "RU", "BU")}
            w.update(dm.features(X, upd[u]))
            sc = {k: norm_mean(v[u]) for k, v in V.items() if k != "Static context residual (no history, no adaptation, no CUSUM)"}
            sc["Linear forecast residual (GDN-style)"] = fc[u].mean(0)
            sc["Conditional reference (CondAttr-style kNN)"] = mods["knn"].score(X[upd[u]]).mean(0)
            sc["Simple fix: sliding-baseline deviation"] = w["D_sw"].mean(0)
            sc["LENS-U (original, label-free)"] = lens_u(w)
            sc["LENS-SW (state-switched, exploratory)"] = lens_sw(w)
            sc["Raw deviation |z|"] = w["D"].mean(0)
            for meth, v in sc.items():
                rows.append(dict(attack=a.label, method=meth, m=m, **rank_eval(v, ds.tags, ds.stage, a.points, a.stages)))
    return rows


if __name__ == "__main__":
    ds = Dm.load("hai")
    nm = pickle.load(open("cache/hai_models.pkl", "rb"))["nm"]
    mods = pickle.load(open("cache/hai_lens2_models.pkl", "rb"))
    cfg = Config()
    dm = DriftModel().fit(ds.Xn, nm)
    att = ds.attacks
    targets = sorted({p for ps in att.points for p in ps})
    nuniq = np.array([len(np.unique(ds.Xn[::50, j])) for j in range(len(ds.tags))])
    cont = [j for j, t in enumerate(ds.tags) if nuniq[j] > 6 and t not in targets]
    rng = np.random.default_rng(0)
    distract = sorted(rng.choice(cont, 5, replace=False).tolist())
    log("distractor tags:", [ds.tags[j] for j in distract])
    tgt_idx = {s: sorted({ds.tags.index(p) for ps in att[att.stream == s].points for p in ps}) for s in ds.streams}
    rows = []
    for scen, tb in [("D: distractor drift", {s: distract for s in ds.streams}), ("T: target pre-drift", tgt_idx)]:
        for m in MS:
            if m == 0 and scen.startswith("T"):
                base = [dict(r, scenario=scen) for r in rows if r["m"] == 0]
                rows += base
                continue
            r = run_condition(ds, att, nm, dm, mods, cfg, tb, m)
            rows += [dict(x, scenario=scen) for x in r]
            log(scen, "m =", m, "done")
    df = pd.DataFrame(rows)
    df.to_csv("results/inject_per_attack.csv", index=False)
    tab = df.groupby(["scenario", "method", "m"]).rr.mean().unstack("m").round(3)
    tab["drop 0->10"] = (tab[10] - tab[0]).round(3)
    tab.to_csv("results/inject_summary.csv")
    for scen in tab.index.get_level_values(0).unique():
        print(f"\n=== {scen}: MRR vs drift size (sigma) on all 58 HAI attacks ===")
        print(tab.loc[scen].sort_values(10, ascending=False).to_string())
