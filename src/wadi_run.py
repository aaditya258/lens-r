"""Confirmatory WADI evaluation of LENS-R (PREREG_WADI.md). Run once. Usage: python3 wadi_run.py"""
import os, time, json, hashlib
import numpy as np
import pandas as pd
from lens import Config, NormalModel, EPS
from recon import ReconModel
from lens2 import KNNReference, ForecastResidual
from drift import two_timescale
from evaluate import rank_eval, random_expect, paired, boot_ci

assert hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon.py"), "rb").read()).hexdigest() == \
    "6a8fb2546c6e97d4d90ffe71d17937ced408606f96bdc73331cc2a21b2b12c19", "recon.py changed since freezing"
N = 10
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
from paths import DATA, GT
W = f"{DATA}/wadi"


def load(keep_constant=False):
    n = pd.read_csv(f"{W}/WADI_14days_new.csv")
    n.columns = [c.strip() for c in n.columns]
    n = n.dropna(subset=["Row"]).reset_index(drop=True)
    row = n["Row"].astype(np.int64).to_numpy()
    ts = pd.Timestamp("2017-09-25 18:00:00") + pd.to_timedelta(row - 1, unit="s")
    mmss = n["Time"].astype(str).str.strip().str.extract(r"(\d+):(\d+)").astype(float)
    chk = float(((ts.minute == mmss[0]) & (ts.second == mmss[1])).mean())
    a = pd.read_parquet(f"{W}/attack.parquet")
    lab = [c for c in a.columns if "LABLE" in c][0]
    cols = [c for c in n.columns if c not in ("Row", "Date", "Time") and c in a.columns]
    Xn_df = n[cols].apply(pd.to_numeric, errors="coerce")
    miss = Xn_df.isna().mean()
    keep = [c for c in cols if miss[c] <= 0.5 and (keep_constant or Xn_df[c].nunique(dropna=True) > 1)]
    dropped = [c for c in cols if c not in keep]
    Xn = Xn_df[keep].ffill().bfill().to_numpy(np.float32)
    Xa = a[keep].apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy(np.float32)
    gaps = np.flatnonzero(np.diff(row) != 1) + 1
    bounds = np.r_[0, gaps, len(row)]
    return dict(Xn=Xn, Xa=Xa, ts_a=pd.to_datetime(a["ts"]).to_numpy(), y=(a[lab] == -1).to_numpy().astype(int),
                tags=keep, dropped=dropped, bounds=bounds, ts_check=chk, n_rows=len(row), ts_range=(str(ts.min()), str(ts.max())))


def attacks(D):
    gt = pd.read_csv(f"{GT}/wadi_a2_attacks.csv", parse_dates=["start", "end"])
    gt = gt[gt.localisable].copy()
    y, ts = D["y"], D["ts_a"]
    d = np.diff(np.r_[0, y, 0]); s = np.flatnonzero(d == 1)
    on = []
    for _, r in gt.iterrows():
        if str(r.attack) == "13":                                    # label error: table time + 3 s (addendum)
            on.append(int(np.searchsorted(ts, np.datetime64(r.start + pd.Timedelta(seconds=3)))))
        else:
            k = np.argmin(np.abs((ts[s] - np.datetime64(r.start)).astype("timedelta64[s]").astype(int)))
            on.append(int(s[k]))
    gt["onset"] = on
    gt["points"] = gt.points_eval.str.split(";")
    gt["stages"] = gt.points.apply(lambda ps: sorted({p.split("_")[0] for p in ps}))
    return gt.reset_index(drop=True)


def main(keep_constant=False, injection=True):
    tag = "secondary_keepconst" if keep_constant else "primary"
    print(f"\n########## {tag.upper()} ##########")
    D = load(keep_constant)
    log("normal rows", D["n_rows"], D["ts_range"], "segments", len(D["bounds"]) - 1, "| mm:ss check", round(D["ts_check"], 5),
        "| tags kept", len(D["tags"]), "dropped", len(D["dropped"]))
    tags = D["tags"]; stage = np.array([t.split("_")[0] for t in tags])
    att = attacks(D)
    miss = {p for ps in att.points for p in ps} - set(tags)
    att = att[att.points.apply(lambda ps: any(p in tags for p in ps))].reset_index(drop=True)
    log("attacks scored", len(att), list(att.attack), "| target tags dropped by data rule:", miss)
    cfg = Config()
    nm = NormalModel().fit(D["Xn"], cfg); log("normal model fitted")
    rm = ReconModel().fit(D["Xn"], nm, "T2", D["bounds"]); log("LENS-R fitted; slow threshold", round(rm.thr_slow, 1))
    fc = ForecastResidual().fit(D["Xn"], nm, D["bounds"]); log("GDN-style fitted")
    knn = KNNReference().fit(D["Xn"], nm); log("kNN fitted")
    Xa = D["Xa"]
    upd_all = np.arange(49, len(Xa), 3)

    def score_all(X, u):
        s = {}
        s["LENS-R T2 (full)"] = norm_mean(rm.window(X, u, True, True)[0])
        s["LENS-R without drift correction"] = norm_mean(rm.window(X, u, False, True)[0])
        s["GDN-style forecast residual"] = fc.score_stream(X, u).mean(0)
        sh, ref = two_timescale(X.astype(np.float64), u)
        s["Simple drift fix"] = np.minimum(np.abs(X[u] - ref) / nm.sd, 50).mean(0)
        s["CondAttr-style kNN"] = knn.score(X[u]).mean(0)
        s["Raw deviation |z|"] = nm.deviation(X[u]).mean(0)
        return s

    rows, qlog = [], {}
    for _, a in att.iterrows():
        u0 = int(np.searchsorted(upd_all, a.onset)); u = upd_all[u0:u0 + N]
        _, Qs, ev = rm.window(Xa, u, True, True)
        for Q in Qs:
            for q in Q:
                qlog[tags[q]] = qlog.get(tags[q], 0) + 1
        for m, sc in score_all(Xa, u).items():
            rows.append(dict(attack=str(a.attack), method=m, **rank_eval(sc, tags, stage, [p for p in a.points if p in tags], a.stages)))
    per = pd.DataFrame(rows)
    per.to_csv(f"results/wadi_{tag}_per_attack.csv", index=False)
    rnd = pd.DataFrame([random_expect(len(tags), len([p for p in ps if p in tags])) for ps in att.points]).mean()

    def table(p, label):
        g = p.groupby("method")
        t = pd.DataFrame({"n": g.size(), "MRR": g.rr.mean(), "Top@1": g.top1.mean(), "Top@3": g.top3.mean(),
                          "Top@5": g.top5.mean(), "Stage": g.stage_hit.mean(),
                          "MRR 95% CI": {k: np.round(boot_ci(v.rr.values), 3) for k, v in g}}).sort_values("MRR", ascending=False)
        print(f"\n=== WADI {label} ===\n", t.round(3).to_string(), f"\nRandom expected MRR {rnd.rr:.3f}, Top@3 {rnd.top3:.3f}")
        return t

    tab = table(per, f"{tag}: {len(att)} attacks")
    tab.to_csv(f"results/wadi_{tag}_summary.csv")
    FULL = "LENS-R T2 (full)"
    comps = ["GDN-style forecast residual", "LENS-R without drift correction", "Simple drift fix", "CondAttr-style kNN", "Raw deviation |z|"]
    tests = pd.DataFrame([paired(per, FULL, b) for b in comps]); tests.to_csv(f"results/wadi_{tag}_tests.csv", index=False)
    print(tests[["b", "n", "mean_diff", "wins", "losses", "ties", "p"]].to_string())
    for excl in ["13", "8"]:
        if excl in set(per.attack):
            table(per[per.attack != excl], f"{tag} sensitivity: excluding attack {excl}")
    claimA = all(not (r.mean_diff < 0 and r.p < 0.05) for r in tests.itertuples() if r.b in comps[:4]) and \
        tests.set_index("b").loc["Simple drift fix", "mean_diff"] > 0
    log("drifting tags quarantined in attack windows:", dict(sorted(qlog.items(), key=lambda x: -x[1])[:10]))
    if not injection:
        json.dump(dict(claimA_secondary=bool(claimA), n_attacks=len(att)), open(f"results/wadi_{tag}_claims.json", "w"), indent=1)
        print("\n(secondary) claim-A criterion on this set:", bool(claimA))
        return

    # ---- Claim B: drift injection (5 non-target continuous tags, seed 0, 2 h ramp, m in {0, 6, 10}) ----
    targets = {p for ps in att.points for p in ps}
    nuniq = np.array([len(np.unique(D["Xn"][::50, j])) for j in range(len(tags))])
    cont = [j for j, t in enumerate(tags) if nuniq[j] > 6 and t not in targets]
    distract = sorted(np.random.default_rng(0).choice(cont, 5, replace=False).tolist())
    log("distractor tags:", [tags[j] for j in distract])
    inj = []
    for m in [0, 6, 10]:
        X = Xa.copy()
        if m:
            X[:, distract] += (m * nm.sd[distract])[None, :].astype(np.float32) * np.minimum(1, np.arange(len(X)) / 7200)[:, None]
        for _, a in att.iterrows():
            u0 = int(np.searchsorted(upd_all, a.onset)); u = upd_all[u0:u0 + N]
            for meth, sc in score_all(X, u).items():
                inj.append(dict(m=m, attack=str(a.attack), method=meth,
                                **rank_eval(sc, tags, stage, [p for p in a.points if p in tags], a.stages)))
        log("injection m =", m, "done")
    inj = pd.DataFrame(inj); inj.to_csv("results/wadi_inject_per_attack.csv", index=False)
    it = inj.groupby(["method", "m"]).rr.mean().unstack("m").round(3); it["drop 0->10"] = (it[10] - it[0]).round(3)
    it.to_csv("results/wadi_inject_summary.csv")
    print("\n=== WADI drift injection: MRR vs drift size (sigma) ===\n", it.sort_values(10, ascending=False).to_string())
    claimB = (it.loc[FULL, "drop 0->10"] > -0.05) and (it.loc["GDN-style forecast residual", "drop 0->10"] < -0.10) and \
        (it.loc["LENS-R without drift correction", "drop 0->10"] < -0.10)
    res = dict(claimA=bool(claimA), claimB=bool(claimB), n_attacks=len(att), tags=len(tags), dropped=D["dropped"],
               distractors=[tags[j] for j in distract], quarantined=qlog)
    print("\nPRE-REGISTERED CLAIM A (accuracy) MET:", res["claimA"], "\nPRE-REGISTERED CLAIM B (drift robustness) MET:", res["claimB"])
    json.dump(res, open("results/wadi_claims.json", "w"), indent=1, default=str)


def norm_mean(A):
    return (A / (A.sum(1, keepdims=True) + EPS)).mean(0)


if __name__ == "__main__":
    main(False, True)
    main(True, False)
