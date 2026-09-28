"""Localisation evaluation: LENS vs baselines, per attack.

Usage: python3 evaluate.py [n_updates]
Tuning uses validation attacks only (supervised LENS). LENS-U uses equal weights (no labels).
"""
import sys, pickle, itertools, json
import numpy as np
import pandas as pd
from math import comb
from scipy.stats import rankdata, wilcoxon
from lens import minmax, modes, fuse

N_UPD = int(sys.argv[1]) if len(sys.argv) > 1 else 10
KS = (1, 3, 5)
GRID = np.array([c for c in itertools.product(range(11), repeat=4) if sum(c) == 10]) / 10.0
EQUAL = np.full((3, 4), 0.25)


def load(name):
    with open(f"cache/{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def win_parts(w, n, R_static=None, sup=True):
    """Normalised indicator stacks (n, N) for one attack window."""
    H, D = minmax(w["H"][:n]), minmax(w["D"][:n])
    if sup:
        R = np.broadcast_to(minmax(R_static[None, :]), H.shape)
        B = minmax(w["BS"][:n])
    else:
        R, B = minmax(w["RU"][:n]), minmax(w["BU"][:n])
    return H, R, B, D


def lens_score(w, n, coef, th, sup, R_static=None, detector="rf", switched=True):
    H, R, B, D = win_parts(w, n, R_static, sup)
    f = w[f"f_{detector}"][:n] if sup else np.zeros(n)
    m = modes(w["r"][:n], f, th) if switched else np.zeros(n, np.int8)
    return fuse(H, R, B, D, m, coef).mean(0), m


def rank_eval(score, tags, stage, points, stages):
    idx = [tags.index(p) for p in points if p in tags]
    ranks = rankdata(-score, method="average")
    best = ranks[idx].min()
    top = int(np.argmax(score))
    r = dict(best_rank=best, rr=1.0 / best, stage_hit=int(stage[top] in stages), top1=tags[top])
    for k in KS:
        r[f"top{k}"] = int(best <= k)
    return r


def random_expect(N, m):
    r = {f"top{k}": 1 - comb(N - m, k) / comb(N, k) for k in KS}
    pgt = [comb(N - m, j) / comb(N, j) for j in range(N + 1)]      # P(best rank > j)
    r["rr"] = sum((pgt[j - 1] - pgt[j]) / j for j in range(1, N + 1))
    return r


def methods(P, w, n, coefs):
    """Score vector per method for one window."""
    th, tags = P["th"], P["tags"]
    s = {}
    s["Raw deviation |z|"] = w["D"][:n].mean(0)
    s["Entropy H only"] = minmax(w["H"][:n]).mean(0)
    s["Belief B only (label-free)"] = minmax(w["BU"][:n]).mean(0)
    s["Relevance R only (label-free)"] = minmax(w["RU"][:n]).mean(0)
    s["Relevance R only (supervised, static)"] = P["R_sup"].copy()
    s["RF Gini (static)"] = P["gini"].copy()
    s["XGBoost gain (static)"] = P["xgb_gain"].copy()
    if "shap" in w:
        s["TreeSHAP on RF"] = w["shap"]
    s["LENS-U fixed (equal)"] = lens_score(w, n, EQUAL, th, False, switched=False)[0]
    s["LENS-U (equal, mode-switched)"] = lens_score(w, n, EQUAL, th, False)[0]
    if coefs is not None:
        s["LENS fixed (tuned)"] = lens_score(w, n, coefs["fixed"], th, True, P["R_sup"], switched=False)[0]
        s["LENS (tuned, mode-switched)"] = lens_score(w, n, coefs["switched"], th, True, P["R_sup"])[0]
        s["LENS (tuned, switched, XGB alarms)"] = lens_score(w, n, coefs["switched"], th, True, P["R_sup"], "xgb")[0]
    return s


def tune(P, n):
    """Coefficients for supervised LENS, chosen on validation attacks only (objective: MRR)."""
    val = [w for w in P["windows"] if w["split"] == "val" and w["localisable"]]
    tags, stage, th = P["tags"], P["stage"], P["th"]
    parts = [win_parts(w, n, P["R_sup"], True) for w in val]
    mm = [modes(w["r"][:n], w["f_rf"][:n], th) for w in val]

    def mrr(coef, switched):
        tot = 0.0
        for w, (H, R, B, D), m in zip(val, parts, mm):
            sc = fuse(H, R, B, D, m if switched else np.zeros(n, np.int8), coef).mean(0)
            tot += rank_eval(sc, tags, stage, w["points"], w["stages"])["rr"]
        return tot / len(val)

    best_f = max(GRID, key=lambda c: mrr(np.tile(c, (3, 1)), False))
    fixed = np.tile(best_f, (3, 1))
    sw = fixed.copy()
    for _ in range(2):                                 # coordinate ascent over modes
        for k in range(3):
            def obj(c, k=k):
                t = sw.copy(); t[k] = c
                return mrr(t, True)
            sw[k] = max(GRID, key=obj)
    occ = np.bincount(np.concatenate(mm), minlength=3) / sum(len(x) for x in mm)
    return dict(fixed=fixed, switched=sw, val_mrr_fixed=mrr(fixed, False), val_mrr_switched=mrr(sw, True),
                val_mode_occupancy=occ.round(3).tolist(), n_val=len(val))


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.quantile(bs, [0.025, 0.975])


def evaluate(P, split, coefs, n, exclude_disputed=False):
    tags, stage = P["tags"], P["stage"]
    ws = [w for w in P["windows"] if w["localisable"] and (split == "all" or w["split"] == split)
          and not (exclude_disputed and w["disputed"])]
    rows = []
    for w in ws:
        for meth, sc in methods(P, w, n, coefs if split != "all" else None).items():
            r = rank_eval(sc, tags, stage, w["points"], w["stages"])
            rows.append(dict(attack=w["label"], method=meth, n_points=len(w["points"]), **r))
    per = pd.DataFrame(rows)
    rnd = pd.DataFrame([random_expect(len(tags), len([p for p in w["points"] if p in tags])) for w in ws]).mean()
    return per, rnd


def summarise(per, rnd):
    g = per.groupby("method")
    t = pd.DataFrame({"attacks": g.size(), "Top@1": g.top1.mean(), "Top@3": g.top3.mean(), "Top@5": g.top5.mean(),
                      "Stage hit": g.stage_hit.mean(), "MRR": g.rr.mean()})
    ci = {m: boot_ci(d.rr.values) for m, d in g}
    t["MRR 95% CI"] = [f"[{ci[m][0]:.2f}, {ci[m][1]:.2f}]" for m in t.index]
    t.loc["Random (expected)"] = [t.attacks.iloc[0], rnd.top1, rnd.top3, rnd.top5, np.nan, rnd.rr, ""]
    return t.sort_values("MRR", ascending=False)


def paired(per, a, b):
    x = per[per.method == a].set_index("attack").rr
    y = per[per.method == b].set_index("attack").rr
    j = x.index.intersection(y.index)
    d = x[j] - y[j]
    p = wilcoxon(x[j], y[j]).pvalue if (d != 0).any() else 1.0
    return dict(a=a, b=b, n=len(j), mean_diff=round(d.mean(), 3), wins=int((d > 0).sum()),
                losses=int((d < 0).sum()), ties=int((d == 0).sum()), p=round(p, 4))


if __name__ == "__main__":
    out = {}
    for name in ["swat", "hai"]:
        P = load(name)
        coefs = tune(P, N_UPD)
        res = {}
        for split in ["test", "all"]:
            per, rnd = evaluate(P, split, coefs, N_UPD)
            per.to_csv(f"results/{name}_{split}_per_attack_n{N_UPD}.csv", index=False)
            tab = summarise(per, rnd)
            tab.to_csv(f"results/{name}_{split}_summary_n{N_UPD}.csv")
            res[split] = (per, tab)
            print(f"\n=== {P['name']} | {split} attacks | first {N_UPD} updates ({N_UPD*3}s) ===")
            print(tab.round(3).to_string())
        per = res["test"][0]
        tests = [paired(per, "LENS (tuned, mode-switched)", "Raw deviation |z|"),
                 paired(per, "LENS (tuned, mode-switched)", "TreeSHAP on RF"),
                 paired(per, "LENS-U (equal, mode-switched)", "Raw deviation |z|"),
                 paired(per, "LENS-U (equal, mode-switched)", "TreeSHAP on RF"),
                 paired(per, "LENS (tuned, mode-switched)", "LENS fixed (tuned)")]
        pa = res["all"][0]
        tests += [paired(pa, "LENS-U (equal, mode-switched)", "Raw deviation |z|"),
                  paired(pa, "LENS-U (equal, mode-switched)", "LENS-U fixed (equal)")]
        print(pd.DataFrame(tests).to_string())
        print("tuned coefficients (H,R,B,D) fixed:", coefs["fixed"][0], "\nswitched:\n", coefs["switched"],
              "\nval MRR fixed/switched:", round(coefs["val_mrr_fixed"], 3), round(coefs["val_mrr_switched"], 3),
              "| val attacks:", coefs["n_val"], "| val mode occupancy N/S/A:", coefs["val_mode_occupancy"])
        print("detection (test partition, positive = Attack):", {k: {m: round(float(v), 3) for m, v in d.items()} for k, d in P["detection"].items()})
        out[name] = dict(coefs={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in coefs.items()},
                         tests=tests, detection={k: {m: float(v) for m, v in d.items()} for k, d in P["detection"].items()})
    with open(f"results/summary_n{N_UPD}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
