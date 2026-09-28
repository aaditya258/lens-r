"""Compute drift features for every attack window and evaluate LENS-DA vs the simple fix,
ablations and earlier baselines. Usage: python3 drift_run.py [n_updates]"""
import sys, pickle, json
import numpy as np
import pandas as pd
import data as Dm
from lens import minmax
from drift import DriftModel, lens_da, sensor_state
from evaluate import rank_eval, random_expect, summarise, paired, methods, tune

N_UPD = int(sys.argv[1]) if len(sys.argv) > 1 else 10

ABL = {
    "LENS-DA (full)": ("H", "D_g", "C", "RC"),
    "LENS-DA without drift gate (D_abs)": None,       # handled below
    "LENS-DA without relational change": ("H", "D_g", "C"),
    "LENS-DA without level change": ("H", "D_g", "RC"),
    "LENS-DA without entropy": ("D_g", "C", "RC"),
    "LENS-DA without abs deviation": ("H", "C", "RC"),
}


def build(name):
    with open(f"cache/{name}.pkl", "rb") as fh:
        P = pickle.load(fh)
    with open(f"cache/{name}_models.pkl", "rb") as fh:
        M = pickle.load(fh)
    ds = Dm.load(name)
    dm = DriftModel().fit(ds.Xn, M["nm"])
    for w in P["windows"]:
        f = dm.features(ds.streams[w["stream"]].X, w["upd"])
        w.update({k: v for k, v in f.items()})
    return P, dm


def score_all(P, w, n, coefs):
    s = methods(P, w, n, coefs)
    H, Dg, C, RC = w["H"][:n], w["D_g"][:n], w["C"][:n], w["RC"][:n]
    s["Simple fix: sliding-baseline deviation"] = w["D_sw"][:n].mean(0)
    s["Level change C only"] = minmax(C).mean(0)
    s["Relational change RC only"] = minmax(RC).mean(0)
    for k, use in ABL.items():
        if use is None:
            s[k] = lens_da(H, w["D"][:n], C, RC).mean(0)
        else:
            s[k] = lens_da(H, Dg, C, RC, use).mean(0)
    return s


def run(name, split, coefs, n):
    P = DATA[name]
    ws = [w for w in P["windows"] if w["localisable"] and (split == "all" or w["split"] == split)]
    rows = []
    for w in ws:
        for meth, sc in score_all(P, w, n, coefs if split != "all" else None).items():
            rows.append(dict(attack=w["label"], method=meth, **rank_eval(sc, P["tags"], P["stage"], w["points"], w["stages"])))
    per = pd.DataFrame(rows)
    rnd = pd.DataFrame([random_expect(len(P["tags"]), len(w["points"])) for w in ws]).mean()
    return per, summarise(per, rnd)


if __name__ == "__main__":
    DATA, out = {}, {}
    keep = ["LENS-DA (full)", "LENS-DA without drift gate (D_abs)", "LENS-DA without relational change",
            "LENS-DA without level change", "LENS-DA without entropy", "LENS-DA without abs deviation",
            "Simple fix: sliding-baseline deviation", "Level change C only", "Relational change RC only",
            "LENS-U fixed (equal)", "TreeSHAP on RF", "Raw deviation |z|", "RF Gini (static)",
            "LENS (tuned, mode-switched)", "Random (expected)"]
    for name in ["swat", "hai"]:
        DATA[name], dm = build(name)
        P = DATA[name]
        coefs = tune(P, N_UPD)
        res = {}
        for split in ["test", "all"]:
            per, tab = run(name, split, coefs, N_UPD)
            per.to_csv(f"results/drift_{name}_{split}_per_attack_n{N_UPD}.csv", index=False)
            tab.to_csv(f"results/drift_{name}_{split}_summary_n{N_UPD}.csv")
            res[split] = per
            print(f"\n=== {P['name']} | {split} | first {N_UPD} updates ===")
            print(tab.reindex([k for k in keep if k in tab.index]).round(3).to_string())
        full, simple = "LENS-DA (full)", "Simple fix: sliding-baseline deviation"
        tests = []
        for sp in ["test", "all"]:
            per = res[sp]
            for b in [simple, "LENS-U fixed (equal)", "Raw deviation |z|", "TreeSHAP on RF", "Relational change RC only"]:
                t = paired(per, full, b); t["split"] = sp; tests.append(t)
        tdf = pd.DataFrame(tests); print(tdf.to_string()); tdf.to_csv(f"results/drift_{name}_tests_n{N_UPD}.csv", index=False)
        # drift-state diagnostics on the attack windows
        allw = [w for w in P["windows"] if w["localisable"]]
        st = np.concatenate([sensor_state(w["drift"][:N_UPD], w["C"][:N_UPD], w["RC"][:N_UPD]).ravel() for w in allw])
        tags = P["tags"]
        drifting = pd.Series(np.concatenate([w["drift"][:1].ravel() > 3 for w in allw]).reshape(len(allw), -1).mean(0), index=tags)
        print("sensor-state occupancy in attack windows (stable/drifting/abrupt):", np.bincount(st, minlength=3) / len(st))
        print("sensors most often in 'drifting' state at onset:", drifting.sort_values(ascending=False).head(6).round(2).to_dict())
