"""Data loading, splits and ground truth for SWaT A1 and HAI 22.04.

Splits are chronological and attack-wise (never by sample):
  SWaT: 35 labelled episodes -> train 0-16, validation 17-23, test 24-34.
  HAI : test1+test2 (24 attacks) train, test3 (10) validation, test4 (24) test.
Normal-only statistics always come from the normal files only.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from paths import DATA as ROOT, GT
SWAT_DROP_STARTUP = 21600  # first 6 h of SWaT normal data (tank filling / unstable start-up)


@dataclass
class Stream:
    name: str
    ts: np.ndarray          # datetime64, 1 Hz
    X: np.ndarray           # float32 (T, N)
    y: np.ndarray           # int8 (T,)  1 = attack
    split: np.ndarray       # object (T,) 'train' | 'val' | 'test' per sample


@dataclass
class Dataset:
    name: str
    tags: list
    stage: np.ndarray       # stage label per tag
    Xn: np.ndarray          # normal-only data (float32)
    streams: dict           # name -> Stream
    attacks: pd.DataFrame   # one row per labelled episode
    notes: list = field(default_factory=list)
    normal_bounds: np.ndarray = None   # start indices of separate normal files (None = one contiguous file)


def _episodes(y):
    d = np.diff(np.r_[0, y.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # start idx, end idx (exclusive)


def load_swat():
    n = pd.read_parquet(f"{ROOT}/swat/normal.parquet")
    a = pd.read_parquet(f"{ROOT}/swat/attack_ts.parquet")
    tags = [c for c in n.columns if c not in ("Timestamp", "Normal/Attack")]
    Xn = n[tags].to_numpy(np.float32)[SWAT_DROP_STARTUP:]
    X = a[tags].to_numpy(np.float32)
    y = a["y"].to_numpy(np.int8)
    ts = a["ts"].to_numpy()
    s, e = _episodes(y)
    gt = pd.read_csv(f"{GT}/swat_a1_attacks.csv", parse_dates=["start", "end"]).sort_values("start").reset_index(drop=True)
    assert len(gt) == len(s) == 35
    gt["stream"] = "attack"
    gt["onset"] = s
    gt["stop"] = e
    assert (ts[s] == gt["start"].to_numpy()).all(), "episode starts do not match ground truth"
    gt["split"] = ["train"] * 17 + ["val"] * 7 + ["test"] * 11
    gt["points"] = gt["points_eval"].fillna("").apply(lambda v: [p for p in v.split(";") if p])
    gt["stages"] = gt["stages"].fillna("").apply(lambda v: [p for p in v.split(";") if p])
    gt["label"] = gt["attacks"].astype(str)
    # per-sample split: cut half-way between the last episode of one split and the first of the next
    cut1 = (e[16] + s[17]) // 2
    cut2 = (e[23] + s[24]) // 2
    split = np.empty(len(y), dtype=object)
    split[:cut1], split[cut1:cut2], split[cut2:] = "train", "val", "test"
    stage = np.array(["S" + "".join(ch for ch in t if ch.isdigit())[0] for t in tags])
    return Dataset("SWaT", tags, stage, Xn, {"attack": Stream("attack", ts, X, y, split)}, gt,
                   notes=[f"normal rows used: {len(Xn)} (first {SWAT_DROP_STARTUP} dropped)"])


def load_hai():
    d = f"{ROOT}/hai2204"
    tr = [pd.read_csv(f"{d}/train{i}.csv") for i in range(1, 7)]
    tags = [c for c in tr[0].columns if c not in ("timestamp", "Attack")]
    Xn = np.concatenate([t[tags].to_numpy(np.float32) for t in tr])
    bounds = np.cumsum([0] + [len(t) for t in tr])
    del tr
    gt = pd.read_csv(f"{GT}/hai2204_attacks.csv")
    split_of = {"test1": "train", "test2": "train", "test3": "val", "test4": "test"}
    streams, rows = {}, []
    for f in ["test1", "test2", "test3", "test4"]:
        df = pd.read_csv(f"{d}/{f}.csv")
        ts = pd.to_datetime(df["timestamp"]).to_numpy()
        y = df["Attack"].to_numpy(np.int8)
        streams[f] = Stream(f, ts, df[tags].to_numpy(np.float32), y, np.full(len(y), split_of[f], dtype=object))
        s, e = _episodes(y)
        g = gt[gt.file == f].reset_index(drop=True)
        assert len(g) == len(s)
        for (_, r), si, ei in zip(g.iterrows(), s, e):
            rows.append(dict(label=r.id, stream=f, onset=si, stop=ei, start=ts[si], duration_s=ei - si,
                             points=r.points_eval.split(";"), stages=r.stages.split(";"),
                             localisable=True, disputed=bool(r.disputed), split=split_of[f]))
    stage = np.array([t[:2] for t in tags])
    return Dataset("HAI", tags, stage, Xn, streams, pd.DataFrame(rows), notes=[f"normal file bounds {bounds.tolist()}"],
                   normal_bounds=bounds)


def load(name):
    return load_swat() if name.lower() == "swat" else load_hai()


if __name__ == "__main__":
    for nm in ["swat", "hai"]:
        ds = load(nm)
        a = ds.attacks
        print(ds.name, "tags", len(ds.tags), "normal", ds.Xn.shape, ds.notes)
        print(a.groupby("split").agg(n=("label", "size"), localisable=("localisable", "sum")))
        missing = {p for ps in a.points for p in ps} - set(ds.tags)
        print("points not in tags:", missing)
