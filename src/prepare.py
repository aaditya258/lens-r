"""Fit models on training data and cache everything the evaluation needs.

Usage: python3 prepare.py swat|hai
Output: cache/<name>.pkl with per-attack windows of all indicators, detector outputs,
TreeSHAP attributions, static importances and timing.
"""
import sys, time, pickle, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier
import shap
import data as D
from lens import Config, NormalModel, SupervisedModel, indicators

NMAX = 20          # updates kept after onset (n = 10 is the pre-registered horizon)
SEED = 0


def training_set(ds, stride_n, stride_a):
    Xs, ys = [ds.Xn[::stride_n]], [np.zeros(len(ds.Xn[::stride_n]), np.int8)]
    for s in ds.streams.values():
        m = s.split == "train"
        Xs.append(s.X[m][::stride_a]); ys.append(s.y[m][::stride_a])
    return np.concatenate(Xs), np.concatenate(ys)


def main(name):
    t0 = time.time()
    cfg = Config()
    ds = D.load(name)
    log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
    log(ds.name, "loaded")

    nm = NormalModel().fit(ds.Xn, cfg)
    stride_n = 5 if ds.name == "SWaT" else 10
    Xtr, ytr = training_set(ds, stride_n, 2)
    log("training rows", Xtr.shape, "attack share", round(ytr.mean(), 3))
    sm = SupervisedModel().fit(nm, Xtr, ytr, cfg, seed=SEED)

    rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                n_jobs=2, random_state=SEED).fit(Xtr, ytr)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=SEED).fit(Xtr, ytr)
    log("detectors trained")

    # r(t) on normal data -> mode thresholds (normal data only)
    Dn = nm.deviation(ds.Xn[:: cfg.DT][::5])
    rn = (Dn > cfg.Z_TH).mean(1)
    th = dict(f1=cfg.F1, f2=cfg.F2, r1=float(np.quantile(rn, cfg.R_Q1)), r2=float(np.quantile(rn, cfg.R_Q2)))
    log("mode thresholds", th)

    expl = shap.TreeExplainer(rf)
    windows, det = [], {"rf": [], "xgb": []}
    for sname, s in ds.streams.items():
        ind = indicators(s.X, nm, sm, cfg)
        upd = ind["upd"]
        Xu = s.X[upd]
        prob = {"rf": rf.predict_proba(Xu)[:, 1], "xgb": xgb.predict_proba(Xu)[:, 1]}
        f = {}
        for k, p in prob.items():
            yhat = (p >= 0.5).astype(float)
            c = np.cumsum(np.r_[0, yhat])
            lo = np.maximum(np.arange(len(yhat)) - cfg.WM + 1, 0)
            f[k] = (c[np.arange(1, len(yhat) + 1)] - c[lo]) / (np.arange(len(yhat)) - lo + 1)
            det[k].append(dict(stream=sname, split=s.split[upd], y=s.y[upd], p=p))
        log(sname, "indicators", {k: v.shape for k, v in ind.items() if k != "upd"})
        for _, a in ds.attacks[ds.attacks.stream == sname].iterrows():
            u0 = int(np.searchsorted(upd, a.onset))
            u = np.arange(u0, min(u0 + NMAX, len(upd)))
            w = dict(label=a.label, split=a.split, points=a.points, stages=a.stages,
                     localisable=bool(a.localisable), disputed=bool(a.get("disputed", False)),
                     duration_s=int(a.duration_s), stream=sname, upd=upd[u],
                     H=ind["H"][u], D=ind["D"][u], RU=ind["RU"][u], BU=ind["BU"][u], BS=ind["BS"][u],
                     r=ind["r"][u], f_rf=f["rf"][u], f_xgb=f["xgb"][u], X=Xu[u])
            if a.split in ("val", "test") and a.localisable:
                sv = expl.shap_values(Xu[u[:10]], check_additivity=False)
                sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
                w["shap"] = np.abs(sv).mean(0)
            windows.append(w)
        # r(t) over normal-labelled test updates, and mode occupancy (for reporting)
        del ind
    log("windows + TreeSHAP done")

    # detection metrics on the test partition (positive class = Attack), at update times
    detm = {}
    for k in det:
        y = np.concatenate([d["y"][d["split"] == "test"] for d in det[k]])
        p = np.concatenate([d["p"][d["split"] == "test"] for d in det[k]])
        pr, rc, f1, _ = precision_recall_fscore_support(y, p >= 0.5, average="binary", zero_division=0)
        detm[k] = dict(precision=pr, recall=rc, f1=f1, auc=roc_auc_score(y, p), n=len(y), attack_share=y.mean())
    log("detection", detm)

    out = dict(name=ds.name, tags=ds.tags, stage=ds.stage, cfg={k: v for k, v in vars(Config).items() if not k.startswith("_") and not callable(v)}, th=th,
               R_sup=sm.R, gini=rf.feature_importances_, xgb_gain=xgb.feature_importances_,
               windows=windows, detection=detm, attacks=ds.attacks.drop(columns=[], errors="ignore"))
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/{name}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    with open(f"cache/{name}_models.pkl", "wb") as fh:
        pickle.dump(dict(nm=nm, sm=sm, rf=rf, xgb=xgb, th=th, cfg=cfg), fh)
    log("saved")


if __name__ == "__main__":
    main(sys.argv[1])
