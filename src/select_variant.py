"""Choose the LENS variant on validation attacks only (pooled SWaT + HAI), then report test.

LENS-SW (state-switched): each sensor's state decides its reference.
  stable sensor   (drift <= Z_D): level = |z| vs normal model,  relation = residual vs normal model (RU)
  drifting sensor (drift >  Z_D): level = change vs recent past (C), relation = change of residual (RC)
  score = equal-weight mean of min-max normalised (H, level, relation).
"""
import numpy as np
import pandas as pd
from lens import minmax
from drift import lens_da, Z_D
from drift_run import build
from evaluate import rank_eval, random_expect, summarise, paired, boot_ci
from lens import EPS

N = 10


def lens_u(w):
    H, R, B, D = (minmax(w[k][:N]) for k in ("H", "RU", "BU", "D"))
    s = (H + R + B + D) / 4
    return (s / (s.sum(1, keepdims=True) + EPS)).mean(0)


def lens_sw(w):
    drifting = w["drift"][:N] > Z_D
    level = np.where(drifting, w["C"][:N], w["D"][:N])
    rel = np.where(drifting, w["RC"][:N], w["RU"][:N])
    s = (minmax(w["H"][:N]) + minmax(level) + minmax(rel)) / 3
    return (s / (s.sum(1, keepdims=True) + EPS)).mean(0)


VARIANTS = {
    "LENS-U (original, label-free)": lens_u,
    "Simple fix: sliding-baseline deviation": lambda w: w["D_sw"][:N].mean(0),
    "LENS-DA (drift-aware)": lambda w: lens_da(w["H"][:N], w["D_g"][:N], w["C"][:N], w["RC"][:N]).mean(0),
    "LENS-SW (state-switched)": lens_sw,
}
OTHERS = {
    "TreeSHAP on RF": lambda w: w.get("shap"),
    "Raw deviation |z|": lambda w: w["D"][:N].mean(0),
    "RF Gini (static)": None,
}


def evaluate(P, ws):
    rows = []
    for w in ws:
        sc = {k: f(w) for k, f in VARIANTS.items()}
        if "shap" in w:
            sc["TreeSHAP on RF"] = w["shap"]
        sc["Raw deviation |z|"] = w["D"][:N].mean(0)
        sc["RF Gini (static)"] = P["gini"]
        for m, s in sc.items():
            rows.append(dict(attack=w["label"], method=m, **rank_eval(s, P["tags"], P["stage"], w["points"], w["stages"])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    DATA = {n: build(n)[0] for n in ["swat", "hai"]}
    val = pd.concat([evaluate(P, [w for w in P["windows"] if w["split"] == "val" and w["localisable"]]).assign(ds=n)
                     for n, P in DATA.items()])
    vm = val[val.method.isin(VARIANTS)].groupby("method").rr.mean().sort_values(ascending=False)
    print("VALIDATION MRR (pooled SWaT 7 + HAI 10 attacks):\n", vm.round(3).to_string())
    print("per dataset:\n", val[val.method.isin(VARIANTS)].groupby(["ds", "method"]).rr.mean().unstack(0).round(3).to_string())
    chosen = vm.index[0]
    print("\nSELECTED ON VALIDATION:", chosen)
    for n, P in DATA.items():
        for split in ["test", "all"]:
            ws = [w for w in P["windows"] if w["localisable"] and (split == "all" or w["split"] == split)]
            per = evaluate(P, ws)
            rnd = pd.DataFrame([random_expect(len(P["tags"]), len(w["points"])) for w in ws]).mean()
            tab = summarise(per, rnd)
            per.to_csv(f"results/select_{n}_{split}_per_attack.csv", index=False)
            tab.to_csv(f"results/select_{n}_{split}_summary.csv")
            print(f"\n=== {P['name']} | {split} ===")
            print(tab.round(3).to_string())
            tests = [paired(per, chosen, b) for b in VARIANTS if b != chosen] + \
                    [paired(per, chosen, b) for b in ["TreeSHAP on RF", "Raw deviation |z|", "RF Gini (static)"]]
            print(pd.DataFrame(tests)[["b", "n", "mean_diff", "wins", "losses", "ties", "p"]].to_string())
    # pooled test across both datasets (35 attacks)
    pooled = pd.concat([evaluate(P, [w for w in P["windows"] if w["localisable"] and w["split"] == "test"]).assign(attack=lambda d, n=n: n + "_" + d.attack.astype(str))
                        for n, P in DATA.items()])
    g = pooled.groupby("method").rr
    print("\n=== POOLED TEST (SWaT 11 + HAI 24) ===")
    print(pd.DataFrame({"MRR": g.mean(), "Top@3": pooled.groupby("method").top3.mean(),
                        "CI": {m: np.round(boot_ci(d.values), 2) for m, d in g}}).sort_values("MRR", ascending=False).round(3).to_string())
    print(pd.DataFrame([paired(pooled, chosen, b) for b in VARIANTS if b != chosen] +
                       [paired(pooled, chosen, "TreeSHAP on RF"), paired(pooled, chosen, "Raw deviation |z|")])[["b", "n", "mean_diff", "wins", "losses", "p"]].to_string())
    pooled.to_csv("results/select_pooled_test_per_attack.csv", index=False)
