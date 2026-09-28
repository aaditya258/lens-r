"""Diagnostics reported in the paper text and in Table V.
(a) Share of normal-labelled SWaT attack-week samples in which each analyser exceeds 3 sigma (Introduction).
(b) Top-3 tags for HAI test4 attacks A402-A406 with and without 3-sigma drift on five non-target tags (Table V).
(c) LENS-R misses (rank worse than 5) by attack type and number of stages (Results, failure cases).
Usage: python diagnostics.py  (after run_recon.py)"""
import pickle, re
import numpy as np
import pandas as pd
import data as Dm
from lens import EPS
from lens2 import all_variants

out = []

# (a) analyser drift in SWaT
ds = Dm.load("swat"); nm = pickle.load(open("cache/swat_models.pkl", "rb"))["nm"]
s = ds.streams["attack"]
frac = (nm.deviation(s.X[s.y == 0][::30]) > 3).mean(0)
top = pd.Series(frac, index=ds.tags).sort_values(ascending=False).head(6).round(2)
print("(a) Share of normal attack-week samples with |z| > 3 (SWaT):\n", top.to_string())
out.append(top.rename("share").reset_index().assign(item="swat_drift_share"))

# (b) smearing under drift, HAI test4
dh = Dm.load("hai"); nmh = pickle.load(open("cache/hai_models.pkl", "rb"))["nm"]
mods = pickle.load(open("cache/hai_lens2_models.pkl", "rb"))
dis = [dh.tags.index(t) for t in ["P1_FT02", "P1_FT03Z", "P2_24Vdc", "P2_VIBTR03", "P4_HT_PO"]]
st = dh.streams["test4"]; att = dh.attacks[(dh.attacks.stream == "test4") & dh.attacks.label.isin(["A402", "A403", "A404", "A405", "A406"])]
rows = []
for m in [0, 3]:
    X = st.X.copy()
    X[:, dis] += (m * nmh.sd[dis])[None, :].astype(np.float32) * np.minimum(1, np.arange(len(X)) / 7200)[:, None]
    upd = np.arange(49, len(X), 3)
    V = all_variants(X, upd, mods["cm1"], mods["cm0"])["LENS v2 without CUSUM"]
    for _, a in att.iterrows():
        u0 = np.searchsorted(upd, a.onset); A = V[u0:u0 + 10]
        sc = (A / (A.sum(1, keepdims=True) + EPS)).mean(0)
        rows.append(dict(attack=a.label, drift_sigma=m, targets=";".join(a.points), top3=", ".join(dh.tags[i] for i in np.argsort(-sc)[:3])))
tb = pd.DataFrame(rows).pivot(index=["attack", "targets"], columns="drift_sigma", values="top3")
print("\n(b) Top-3 tags, adaptive cross-sensor design, without and with 3-sigma drift (Table V):\n", tb.to_string())
tb.to_csv("results/diag_table5_smearing.csv")

# (c) misses by attack type
d = pd.read_csv("results/recon_per_attack.csv"); d = d[d.method == "LENS-R T2 (full)"]
pts = {}
for n in ["swat", "hai"]:
    for _, r in Dm.load(n).attacks.iterrows():
        pts[f"{n}_{r.label}"] = (r.points, len(r.stages))


def actuator_or_setpoint(p):
    return bool(re.match(r"^(MV|P)\d", p) or re.search(r"(CV\d+D|LCP\d+D|_SCO|_B\d{4}|AutoSD|ManualSD|_RTR|VTR\d|PP04$|PP04SP)", p))


d = d.assign(only_act=d.attack.map(lambda a: all(actuator_or_setpoint(p) for p in pts[a][0])),
             multi_stage=d.attack.map(lambda a: pts[a][1] > 1), miss=d.best_rank > 5)
print("\n(c) Misses (rank > 5):", int(d.miss.sum()), "of", len(d),
      "| of which actuator/set-point only:", int((d.miss & d.only_act).sum()),
      "| miss rate actuator/set-point only:", round(d[d.only_act].miss.mean(), 2),
      "| involving a sensor:", round(d[~d.only_act].miss.mean(), 2),
      "| multi-stage miss rate:", round(d[d.multi_stage].miss.mean(), 2), "single-stage:", round(d[~d.multi_stage].miss.mean(), 2))
