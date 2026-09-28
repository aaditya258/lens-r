"""Confirmatory evaluation of LENS v2 (PREREG.md). Usage: python3 run2.py"""
import pickle, time, json
import numpy as np
import pandas as pd
import data as Dm
from lens import EPS
from lens2 import ContextModel, KNNReference, ForecastResidual, all_variants
from drift_run import build
from select_variant import lens_u, lens_sw
from evaluate import rank_eval, random_expect, summarise, paired, boot_ci

N = 10
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)


def norm_mean(A):
    A = A[:N]
    return (A / (A.sum(1, keepdims=True) + EPS)).mean(0)


def fit_models(ds, nm):
    return dict(cm1=ContextModel().fit(ds.Xn, nm, ds.normal_bounds, True),
                cm0=ContextModel().fit(ds.Xn, nm, ds.normal_bounds, False),
                knn=KNNReference().fit(ds.Xn, nm),
                fc=ForecastResidual().fit(ds.Xn, nm, ds.normal_bounds))


def lens2_windows(ds, P, mods):
    """Attach LENS v2 variant scores and new baselines to every attack window."""
    by_stream = {}
    for w in P["windows"]:
        by_stream.setdefault(w["stream"], []).append(w)
    for sname, ws in by_stream.items():
        X = ds.streams[sname].X
        upd_all = np.arange(49, len(X), 3)                       # same update grid as LENS
        V = all_variants(X, upd_all, mods["cm1"], mods["cm0"])
        fc = mods["fc"].score_stream(X, upd_all)
        for w in ws:
            pos = np.searchsorted(upd_all, w["upd"])
            assert (upd_all[pos] == w["upd"]).all()
            w["v2"] = {k: v[pos] for k, v in V.items()}
            w["fc"] = fc[pos]
            if w["localisable"]:
                w["knn"] = mods["knn"].score(X[w["upd"][:N]])
    return P


def scores(P, w):
    s = {k: norm_mean(v) for k, v in w["v2"].items()}
    s["Conditional reference (CondAttr-style kNN)"] = w["knn"].mean(0)
    s["Linear forecast residual (GDN-style)"] = w["fc"][:N].mean(0)
    s["Simple fix: sliding-baseline deviation"] = w["D_sw"][:N].mean(0)
    s["LENS-U (original, label-free)"] = lens_u(w)
    s["LENS-SW (state-switched, exploratory)"] = lens_sw(w)
    s["Raw deviation |z|"] = w["D"][:N].mean(0)
    s["RF Gini (static)"] = P["gini"]
    if "shap" in w:
        s["TreeSHAP on RF"] = w["shap"]
    return s


def evaluate(P, ws):
    rows = []
    for w in ws:
        for m, sc in scores(P, w).items():
            rows.append(dict(attack=w["label"], method=m, **rank_eval(sc, P["tags"], P["stage"], w["points"], w["stages"])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    DATA, per_ds = {}, {}
    for name in ["swat", "hai"]:
        P, _ = build(name)                                        # windows + drift features (D_sw etc.)
        with open(f"cache/{name}_models.pkl", "rb") as fh:
            M = pickle.load(fh)
        ds = Dm.load(name)
        mods = fit_models(ds, M["nm"])
        log(name, "models fitted; own-history weight (median |v|):", round(float(np.median(np.abs(mods['cm1'].v))), 3))
        DATA[name] = lens2_windows(ds, P, mods)
        with open(f"cache/{name}_lens2_models.pkl", "wb") as fh:
            pickle.dump(mods, fh)
        log(name, "windows scored")

    FULL = "LENS v2 (full)"
    out = {}
    for name, P in DATA.items():
        for split in ["test", "all"]:
            ws = [w for w in P["windows"] if w["localisable"] and (split == "all" or w["split"] == split)]
            per = evaluate(P, ws)
            rnd = pd.DataFrame([random_expect(len(P["tags"]), len(w["points"])) for w in ws]).mean()
            tab = summarise(per, rnd)
            per.to_csv(f"results/v2_{name}_{split}_per_attack.csv", index=False)
            tab.to_csv(f"results/v2_{name}_{split}_summary.csv")
            per_ds[(name, split)] = per
            print(f"\n=== {P['name']} | {split} ({len(ws)} attacks) ===")
            print(tab.round(3).to_string())

    def pool(split):
        return pd.concat([per_ds[(n, split)].assign(attack=lambda d, n=n: n + "_" + d.attack.astype(str)) for n in DATA])

    for split, label in [("test", "POOLED TEST (35) - primary"), ("all", "POOLED ALL (91) - secondary")]:
        pp = pool(split)
        pp.to_csv(f"results/v2_pooled_{split}_per_attack.csv", index=False)
        g = pp.groupby("method")
        t = pd.DataFrame({"attacks": g.size(), "MRR": g.rr.mean(), "Top@1": g.top1.mean(), "Top@3": g.top3.mean(),
                          "Top@5": g.top5.mean(), "Stage hit": g.stage_hit.mean(),
                          "MRR 95% CI": {m: np.round(boot_ci(d.rr.values), 3) for m, d in g}}).sort_values("MRR", ascending=False)
        t.to_csv(f"results/v2_pooled_{split}_summary.csv")
        print(f"\n=== {label} ===")
        print(t.round(3).to_string())
        others = [m for m in t.index if m != FULL]
        tests = pd.DataFrame([paired(pp, FULL, b) for b in others])
        tests.to_csv(f"results/v2_pooled_{split}_tests.csv", index=False)
        print(tests[["b", "n", "mean_diff", "wins", "losses", "ties", "p"]].to_string())
        out[split] = dict(summary=t.drop(columns=["MRR 95% CI"]).round(4).to_dict(), tests=tests.to_dict("records"))

    tt = pd.read_csv("results/v2_pooled_test_tests.csv").set_index("b")
    crit = {k: tt.loc[k, ["mean_diff", "p"]].to_dict() for k in
            ["Simple fix: sliding-baseline deviation", "Conditional reference (CondAttr-style kNN)", "TreeSHAP on RF"]}
    ok = ((crit["Simple fix: sliding-baseline deviation"]["mean_diff"] > 0 and crit["Conditional reference (CondAttr-style kNN)"]["mean_diff"] > 0)
          and (min(crit["Simple fix: sliding-baseline deviation"]["p"], crit["Conditional reference (CondAttr-style kNN)"]["p"]) < 0.05)
          and not (crit["TreeSHAP on RF"]["mean_diff"] < 0 and crit["TreeSHAP on RF"]["p"] < 0.05))
    print("\nPRE-REGISTERED PRIMARY COMPARISONS:", json.dumps(crit, indent=1), "\nSUCCESS CRITERION MET:", ok)
    out["criterion"] = dict(comparisons=crit, met=bool(ok))
    with open("results/v2_summary.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
