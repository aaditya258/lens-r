"""Drift-injection check for LENS-R (same distractor protocol as inject.py). Usage: python3 inject_recon.py"""
import pickle
import numpy as np
import pandas as pd
import data as Dm
from lens import EPS
from recon import ReconModel
from evaluate import rank_eval

N = 10
DISTRACT = ["P1_FT02", "P1_FT03Z", "P2_24Vdc", "P2_VIBTR03", "P4_HT_PO"]   # same seed-0 choice as inject.py


def nm_(A):
    return (A / (A.sum(1, keepdims=True) + EPS)).mean(0)


if __name__ == "__main__":
    ds = Dm.load("hai")
    nm = pickle.load(open("cache/hai_models.pkl", "rb"))["nm"]
    mods = pickle.load(open("cache/hai_lens2_models.pkl", "rb"))
    rm = ReconModel().fit(ds.Xn, nm, "T2", ds.normal_bounds)
    cols = [ds.tags.index(t) for t in DISTRACT]
    rows = []
    for m in [0, 3, 6, 10]:
        for sname, s in ds.streams.items():
            X = s.X.copy()
            if m:
                X[:, cols] += (m * nm.sd[cols])[None, :].astype(np.float32) * np.minimum(1, np.arange(len(X)) / 7200)[:, None]
            upd_all = np.arange(49, len(X), 3)
            fc = mods["fc"].score_stream(X, upd_all)
            for _, a in ds.attacks[ds.attacks.stream == sname].iterrows():
                u0 = int(np.searchsorted(upd_all, a.onset)); upd = upd_all[u0:u0 + N]
                sc = {"LENS-R T2 (full)": nm_(rm.window(X, upd, True, True)[0]),
                      "LENS-R T2 (no drift correction)": nm_(rm.window(X, upd, False, True)[0]),
                      "Linear forecast residual (GDN-style)": fc[u0:u0 + N].mean(0)}
                for k, v in sc.items():
                    rows.append(dict(m=m, method=k, attack=a.label, **rank_eval(v, ds.tags, ds.stage, a.points, a.stages)))
        print("m =", m, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/inject_recon_per_attack.csv", index=False)
    prev = pd.read_csv("results/inject_summary.csv")
    prev = prev[prev.scenario.str.startswith("D") & prev.method.isin(
        ["Simple fix: sliding-baseline deviation", "LENS-SW (state-switched, exploratory)", "LENS v2 without CUSUM", "LENS-U (original, label-free)"])]
    tab = df.groupby(["method", "m"]).rr.mean().unstack("m")
    tab.columns = [str(c) for c in tab.columns]
    prev = prev.set_index("method")[["0", "3", "6", "10"]]
    out = pd.concat([tab, prev]).round(3)
    out["drop 0->10"] = (out["10"] - out["0"]).round(3)
    out.to_csv("results/inject_recon_summary.csv")
    print(out.sort_values("10", ascending=False).to_string())
