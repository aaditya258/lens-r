"""Publication figures (black and white, vector PDF, IEEE column widths). Usage: python3 figures.py"""
import os, pickle, shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from paths import FIGURES as OUT, RESULTS, CACHE
os.makedirs(OUT, exist_ok=True)
USETEX = shutil.which("latex") is not None      # LaTeX is optional: without it a standard serif font is used
if USETEX:
    mpl.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{mathptmx}\usepackage[T1]{fontenc}"})
else:
    mpl.rcParams.update({"font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix"})


def it(s):
    return rf"\textit{{{s}}}" if USETEX else s


PCT = r"\%" if USETEX else "%"
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "hatch.linewidth": 0.5,
})
COL, DCOL = 3.5, 7.16

# one fixed style per method, used in every figure
STYLE = {
    "LENS-R":                         dict(color="black", ls="-",  marker="o", mfc="black"),
    "LENS-R w/o drift correction":    dict(color="black", ls="--", marker="o", mfc="white"),
    "GDN-style forecast residual":    dict(color="0.35",  ls="-.", marker="^", mfc="0.35"),
    "Sliding-baseline deviation":     dict(color="0.55",  ls=":",  marker="s", mfc="white"),
    "Conditional reference (kNN)":    dict(color="0.55",  ls="-",  marker="x", mfc="0.55"),
    "Raw deviation":                  dict(color="0.75",  ls="-",  marker="D", mfc="0.75"),
}
MAP = {  # file method names -> figure names
    "LENS-R T2 (full)": "LENS-R", "LENS-R T2 (no drift correction)": "LENS-R w/o drift correction",
    "LENS-R without drift correction": "LENS-R w/o drift correction",
    "Linear forecast residual (GDN-style)": "GDN-style forecast residual", "GDN-style forecast residual": "GDN-style forecast residual",
    "Simple fix: sliding-baseline deviation": "Sliding-baseline deviation", "Simple drift fix": "Sliding-baseline deviation",
    "Conditional reference (CondAttr-style kNN)": "Conditional reference (kNN)", "CondAttr-style kNN": "Conditional reference (kNN)",
    "Raw deviation |z|": "Raw deviation",
}
R = RESULTS


def fig_drift():
    h = pd.read_csv(f"{R}/inject_recon_per_attack.csv")
    h0 = pd.read_csv(f"{R}/inject_per_attack.csv")
    h0 = h0[h0.scenario.str.startswith("D")]
    h = pd.concat([h, h0[h0.method.isin(["Simple fix: sliding-baseline deviation", "Conditional reference (CondAttr-style kNN)", "Raw deviation |z|"])]])
    w = pd.read_csv(f"{R}/wadi_inject_per_attack.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True)
    for ax, df, title, xs in [(axes[0], h, "HAI (58 attacks)", [0, 3, 6, 10]), (axes[1], w, "WADI (8 attacks)", [0, 6, 10])]:
        df = df.assign(name=df.method.map(MAP)).dropna(subset=["name"])
        g = df.groupby(["name", "m"]).rr.mean()
        for name, st in STYLE.items():
            if name in g.index.get_level_values(0):
                s = g.loc[name].reindex(xs)
                ax.plot(xs, s.values, label=name, color=st["color"], ls=st["ls"], marker=st["marker"],
                        mfc=st["mfc"], mec=st["color"], mew=0.7, lw=1.2 if name == "LENS-R" else 0.9, zorder=3 if name == "LENS-R" else 2)
        ax.set_title(title, pad=3)
        ax.set_xticks(xs); ax.set_xlabel(r"Injected drift ($\sigma$)")
        ax.set_ylim(0, 0.45); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.grid(axis="y", lw=0.3, color="0.85"); ax.set_axisbelow(True)
    axes[0].set_ylabel("MRR")
    hdl, lab = axes[0].get_legend_handles_labels()
    fig.legend(hdl, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.53, -0.40), handlelength=2.6, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(f"{OUT}/fig_drift.pdf"); plt.close(fig)


def boot(x, B=4000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.quantile(b, [0.025, 0.975])


def fig_accuracy():
    d = pd.read_csv(f"{R}/recon_per_attack.csv").assign(name=lambda x: x.method.map(MAP)).dropna(subset=["name"])
    w = pd.read_csv(f"{R}/wadi_primary_per_attack.csv").assign(name=lambda x: x.method.map(MAP), ds="wadi").dropna(subset=["name"])
    rnd = {"swat": 0.114, "hai": 0.091, "wadi": 0.089}
    plants = [("swat", "SWaT (33)"), ("hai", "HAI (58)"), ("wadi", "WADI (8)")]
    names = list(STYLE)
    hatches = ["", "////", "", "\\\\\\\\", "....", ""]
    fills = ["black", "white", "0.45", "white", "white", "0.8"]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    bw = 0.13
    for pi, (ds, lab) in enumerate(plants):
        src = w if ds == "wadi" else d[d.ds == ds]
        for mi, name in enumerate(names):
            v = src[src.name == name].rr.values
            if not len(v):
                continue
            x = pi + (mi - (len(names) - 1) / 2) * bw
            lo, hi = boot(v)
            ax.bar(x, v.mean(), bw * 0.9, color=fills[mi], edgecolor="black", lw=0.5, hatch=hatches[mi],
                   label=name if pi == 0 else None)
            ax.plot([x, x], [lo, hi], color="black", lw=0.5)
        ax.plot([pi - 0.45, pi + 0.45], [rnd[ds]] * 2, color="black", lw=0.6, ls=(0, (1, 1.5)),
                label="Random ranking" if pi == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([p[1] for p in plants])
    ax.set_ylabel(f"MRR (95{PCT} CI)"); ax.set_ylim(0, 0.75)
    ax.grid(axis="y", lw=0.3, color="0.85"); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.13), handlelength=1.6, columnspacing=1.0)
    fig.savefig(f"{OUT}/fig_accuracy.pdf"); plt.close(fig)


def fig_case():
    import data as Dm
    from recon import ReconModel, greedy, _cum, _wmean
    from scipy.stats import rankdata
    ds = Dm.load("swat")
    nm = pickle.load(open(f"{CACHE}/swat_models.pkl", "rb"))["nm"]
    rm = ReconModel().fit(ds.Xn, nm, "T2")
    a = ds.attacks[ds.attacks.label == "8"].iloc[0]
    X = ds.streams["attack"].X
    j = ds.tags.index("DPIT301")
    upd = np.arange(a.onset - 300, a.stop + 300, 3)
    S1, Q, _ = rm.window(X, upd, True, True)
    S0, _, _ = rm.window(X, upd, False, True)
    Z = (X - nm.mu) / nm.sd
    C = _cum(Z); zf = _wmean(C, upd + 1 - 10, upd + 1)
    exp = []
    for t in range(len(upd)):                     # reconstruction of FIT401 after drift correction
        z = zf[t].copy()
        if len(Q[t]):
            zs = _wmean(C, np.array([upd[t] - 1800]), np.array([upd[t] - 300]))[0]
            _, Qt, fQ, _ = greedy(zs, rm.M, 10, rm.thr_slow); z[Qt] -= fQ
        b = rm.M @ z
        exp.append(nm.mu[j] + nm.sd[j] * (z[j] - b[j] / rm.M[j, j]))
    rank = lambda S: np.array([rankdata(-s, method="average")[j] for s in S])
    Dz = np.abs(Z[upd])
    tmin = (upd - a.onset) / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12))
    for ax in (ax1, ax2):
        ax.axvspan(0, (a.stop - a.onset) / 60.0, color="0.9", lw=0)
    ax1.plot(tmin, X[upd, j], color="black", lw=1.0, label="Reported DPIT301")
    ax1.plot(tmin, exp, color="black", lw=0.9, ls="--", label="Expected (reconstructed)")
    ax1.set_ylabel("Pressure (kPa)"); ymax = float(np.nanmax(X[upd, j])) * 1.12
    ax1.set_ylim(0, ymax)
    ax1.set_ylim(0, ymax * 1.15)
    ax1.legend(loc="upper left", ncol=2, handlelength=2.0, columnspacing=1.0, bbox_to_anchor=(0, 1.02))
    ax1.text(15.6, ymax * 0.62, it("attack"), fontsize=7, ha="right", style="normal" if USETEX else "italic")
    ax2.plot(tmin, rank(S1), color="black", lw=1.0, label="LENS-R")
    ax2.plot(tmin, rank(S0), color="black", lw=0.9, ls="--", label="w/o drift correction")
    ax2.plot(tmin, rank(Dz), color="0.6", lw=0.9, label="Raw deviation")
    ax2.set_yscale("log"); ax2.invert_yaxis(); ax2.set_ylim(55, 0.8)
    ax2.set_yticks([1, 3, 10, 51]); ax2.set_yticklabels(["1", "3", "10", "51"])
    ax2.set_ylabel("Rank of DPIT301"); ax2.set_xlabel("Minutes from attack onset")
    ax2.legend(loc="lower left", ncol=3, handlelength=1.8, columnspacing=0.8, bbox_to_anchor=(0, -0.62))
    ax2.set_xlim(tmin[0], tmin[-1])
    fig.savefig(f"{OUT}/fig_case.pdf"); plt.close(fig)
    return dict(first_top3=float(tmin[(tmin >= 0) & (rank(S1) <= 3)].min()) if ((tmin >= 0) & (rank(S1) <= 3)).any() else None,
                drifting=sorted({ds.tags[q] for x in Q for q in x}))


def fig_method():
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(DCOL, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, txt, fill="white", ls="-", bold=False):
        ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec="black", lw=0.6, ls=ls))
        t = (rf"\textbf{{{txt}}}" if USETEX else txt) if bold else txt
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7, linespacing=1.25)
        return (x, y, w, h)

    def arrow(p, q, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.6, color="black", ls=ls,
                                     shrinkA=0, shrinkB=0))

    H = 5.2
    # left: data
    box(0.5, 12.4, 11.5, H, "Plant historian\n$x(t)$, 1 Hz")
    box(15.5, 12.4, 12.0, H, "Standardise\n$z=(x-\\mu)/\\sigma$")
    arrow((12.0, 15.0), (15.5, 15.0))
    # slow lane (top)
    ys, yf = 22.0, 2.8
    box(32.0, ys, 13.5, H, "Mean over\n$[t{-}30, t{-}5)$ min")
    box(50.0, ys, 15.0, H, "Greedy repair\nwhile $J > \\theta$")
    box(69.5, ys, 14.0, H, "Drifting tags $Q$,\noffsets $f_Q$")
    # fast lane (bottom)
    box(32.0, yf, 13.5, H, "Mean over\nlast 10 s")
    box(50.0, yf, 15.0, H, "Subtract offsets\non $Q$")
    box(69.5, yf, 14.0, H, "Greedy\nreconstruction")
    # outputs
    box(87.5, ys, 12.0, H, "Drift list\n(maintenance)", fill="0.9")
    box(87.5, yf, 12.0, H, "Ranked tags +\nexpected values", fill="0.9")
    # lane labels
    ax.text(32.0, ys + H + 1.0, it("Slow stage: drift estimation"), fontsize=7, ha="left", va="bottom", style="normal" if USETEX else "italic")
    ax.text(32.0, yf - 1.0, it("Fast stage: attack localisation"), fontsize=7, ha="left", va="top", style="normal" if USETEX else "italic")
    # arrows
    arrow((27.5, 16.6), (32.0, ys + H / 2))
    arrow((27.5, 13.4), (32.0, yf + H / 2))
    arrow((45.5, ys + H / 2), (50.0, ys + H / 2))
    arrow((65.0, ys + H / 2), (69.5, ys + H / 2))
    arrow((83.5, ys + H / 2), (87.5, ys + H / 2))
    arrow((45.5, yf + H / 2), (50.0, yf + H / 2))
    arrow((65.0, yf + H / 2), (69.5, yf + H / 2))
    arrow((83.5, yf + H / 2), (87.5, yf + H / 2))
    arrow((76.5, ys), (57.5, yf + H))          # offsets feed the correction
    ax.text(68.2, 14.6, "$f_Q$", fontsize=7, ha="left", va="center")
    # normal-data model (dashed)
    box(0.5, 1.8, 24.0, 7.4, "Learned from normal data only:\n$\\mu,\\ \\sigma,\\ M=(\\Sigma+\\lambda I)^{-1}$,\nthreshold $\\theta$", ls="--")
    arrow((12.5, 9.2), (21.5, 12.4), ls="--")
    # detector (unchanged, parallel)
    box(0.5, 22.0, 13.5, H, "Existing detector\n(unchanged)", ls=":")
    arrow((6.25, 17.6), (6.25, 22.0), ls=":")
    ax.text(17.2, 24.6, it("alarm"), fontsize=7, ha="left", va="center", style="normal" if USETEX else "italic")
    arrow((14.0, 24.6), (16.8, 24.6), ls=":")
    fig.savefig(f"{OUT}/fig_method.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_method(); print("method ok")
    fig_drift(); print("drift ok")
    fig_accuracy(); print("accuracy ok")
    print("case", fig_case())
