"""Worked example: one LENS-R update, step by step, for SWaT attack 23 (15 s after onset).

Runs without the datasets. Inputs (derived from normal data and one time instant, no raw data files):
  swat_attack23_inputs.csv  per tag: normal mean mu, standard deviation sigma, reading at t,
                            fast-window mean z_fast (t-10 s, t], slow-window mean z_slow [t-1800 s, t-300 s)
  swat_precision_M.csv      M = (Sigma + 0.01 I)^-1 from standardised normal data
  swat_attack23_meta.json   threshold theta and settings

Usage: python worked_example.py
Every number printed here appears in the worked example of the paper.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
inp = pd.read_csv(os.path.join(HERE, "swat_attack23_inputs.csv"))
M = pd.read_csv(os.path.join(HERE, "swat_precision_M.csv"), index_col=0).to_numpy()
meta = json.load(open(os.path.join(HERE, "swat_attack23_meta.json")))
tags = list(inp.tag)
theta = meta["theta"]


def J_after(z, S):
    """Inconsistency left after reconstructing the tags in S, and their fault magnitudes (Eq. 3)."""
    b = M @ z
    J0 = float(z @ b)
    if not S:
        return J0, np.zeros(0)
    f = np.linalg.solve(M[np.ix_(S, S)] + 1e-6 * np.eye(len(S)), b[S])
    return J0 - float(b[S] @ f), f


def greedy(z, steps, thr=None, verbose=True):
    R, J, gain = [], J_after(z, [])[0], np.zeros(len(z))
    if verbose:
        print(f"    start: J = {J:.1f}")
    for _ in range(steps):
        if thr is not None and J <= thr:
            break
        cand = [(J_after(z, R + [j])[0], j) for j in range(len(z)) if j not in R]
        Jn, j = min(cand)
        gain[j] = J - Jn
        if verbose:
            print(f"    repair {tags[j]:8s} gain = {J - Jn:9.1f}   J -> {Jn:.1f}")
        J = Jn; R.append(j)
    for j in range(len(z)):
        if j not in R:
            gain[j] = J - J_after(z, R + [j])[0]
    return gain, R, J_after(z, R)[1], J


print(f"SWaT attack {meta['attack']} (targets {', '.join(meta['targets'])}), {meta['time']}, "
      f"{meta['seconds_after_onset']} s after onset\n")

print("Step 1. Slow stage (drift): z_slow = mean over [t-1800 s, t-300 s)")
print(f"    threshold theta = {theta:.2f}")
_, Q, fQ, _ = greedy(inp.z_slow.to_numpy(), meta["max_drift_tags"], thr=theta)
print("    drifting tags and offsets f_Q:", ", ".join(f"{tags[q]} {f:+.2f}" for q, f in zip(Q, fQ)) or "none")

print("\nStep 2. Drift correction: z_c = z_fast - offsets on Q")
z = inp.z_fast.to_numpy().copy()
for q, f in zip(Q, fQ):
    print(f"    {tags[q]:8s} z_fast = {z[q]:+.2f}  ->  z_c = {z[q] - f:+.2f}")
    z[q] -= f

print("\nStep 3. Fast stage (ranking): greedy reconstruction of z_c, 5 steps")
g, R, fR, _ = greedy(z, meta["fast_steps"])
s = g / g.sum()
print("\n    Rank  Tag       z_c      score")
for r, i in enumerate(np.argsort(-s)[:5], 1):
    mark = "  <- target" if tags[i] in meta["targets"] else ""
    print(f"    {r:4d}  {tags[i]:8s} {z[i]:+7.2f}  {s[i]:.3f}{mark}")

print("\nSame ranking WITHOUT drift correction (z_fast used directly):")
g0, _, _, _ = greedy(inp.z_fast.to_numpy(), meta["fast_steps"], verbose=False)
s0 = g0 / g0.sum()
for r, i in enumerate(np.argsort(-s0)[:5], 1):
    mark = "  <- target" if tags[i] in meta["targets"] else ""
    print(f"    {r:4d}  {tags[i]:8s} {inp.z_fast[i]:+7.2f}  {s0[i]:.3f}{mark}")

print("\nStep 4. Expected value of the top tag (Eq. 6): x_hat = mu + sigma * (z_c - f)")
top = int(np.argmax(s)); k = R.index(top)
mu, sg, xr = inp.mu[top], inp.sigma[top], inp.reading_t[top]
print(f"    {tags[top]}: mu = {mu:.3f}, sigma = {sg:.3f}, z_c = {z[top]:.3f}, f = {fR[k]:.3f}")
print(f"    x_hat = {mu:.3f} + {sg:.3f} x ({z[top]:.3f} - {fR[k]:.3f}) = {mu + sg * (z[top] - fR[k]):.3f}"
      f"   (reported reading {xr:.3f})")

print("\nFor comparison, raw deviation |z_fast| would rank:",
      ", ".join(f"{tags[i]} ({abs(inp.z_fast[i]):.1f})" for i in np.argsort(-np.abs(inp.z_fast.to_numpy()))[:5]))
