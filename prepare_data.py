"""Convert the raw dataset files into the formats used by the experiments.

Usage:
    python prepare_data.py --swat RAW_SWAT_DIR --wadi RAW_WADI_DIR [--hai RAW_HAI_DIR]

RAW_SWAT_DIR : folder with SWaT_Dataset_Normal_v1.xlsx and SWaT_Dataset_Attack_v0.xlsx (SWaT A1, Dec 2015, from iTrust)
RAW_WADI_DIR : folder with WADI_14days_new.csv (or WADI_14days_new.zip) and WADI_attackdataLABLE.csv (WADI A2, 2019, from iTrust)
RAW_HAI_DIR  : folder with HAI 22.04 train1..6.csv and test1..4.csv. If omitted, the files are downloaded from
               https://github.com/icsdataset/hai (about 730 MB).

Output (default ./data, override with the environment variable LENSR_DATA):
    data/swat/normal.parquet, data/swat/attack_ts.parquet
    data/hai2204/train1..6.csv, test1..4.csv
    data/wadi/WADI_14days_new.csv, data/wadi/attack.parquet
"""
import argparse, os, shutil, sys, urllib.request, zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from paths import DATA  # noqa: E402

HAI_FILES = [f"train{i}" for i in range(1, 7)] + [f"test{i}" for i in range(1, 5)]
HAI_URL = "https://media.githubusercontent.com/media/icsdataset/hai/master/hai-22.04/{}.csv"


def read_excel(path):
    try:
        return pd.read_excel(path, engine="calamine", header=1)
    except Exception:
        return pd.read_excel(path, engine="openpyxl", header=1)


def swat(raw):
    out = os.path.join(DATA, "swat"); os.makedirs(out, exist_ok=True)
    for name, fn in [("normal", "SWaT_Dataset_Normal_v1.xlsx"), ("attack", "SWaT_Dataset_Attack_v0.xlsx")]:
        print(f"SWaT: reading {fn} (about one minute)", flush=True)
        df = read_excel(os.path.join(raw, fn))
        df.columns = [str(c).strip() for c in df.columns]
        df["Timestamp"] = df["Timestamp"].astype(str)
        df["Normal/Attack"] = df["Normal/Attack"].astype(str)
        if name == "normal":
            df.to_parquet(os.path.join(out, "normal.parquet"))
            assert len(df) == 495000, f"unexpected SWaT normal length {len(df)}"
        else:
            df["ts"] = pd.to_datetime(df["Timestamp"].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
            lab = df["Normal/Attack"].str.strip().str.replace(" ", "").str.lower()
            df["y"] = (lab == "attack").astype(int)
            assert len(df) == 449919, f"unexpected SWaT attack length {len(df)}"
            df.to_parquet(os.path.join(out, "attack_ts.parquet"))
    print("SWaT: done")


def hai(raw):
    out = os.path.join(DATA, "hai2204"); os.makedirs(out, exist_ok=True)
    for f in HAI_FILES:
        dst = os.path.join(out, f + ".csv")
        if raw:
            shutil.copyfile(os.path.join(raw, f + ".csv"), dst)
        elif not os.path.exists(dst):
            print(f"HAI: downloading {f}.csv", flush=True)
            urllib.request.urlretrieve(HAI_URL.format(f), dst)
        with open(dst) as fh:
            head = fh.readline()
        assert head.startswith("timestamp"), f"{dst} is not a data file (Git LFS pointer?)"
    print("HAI: done")


def wadi(raw):
    out = os.path.join(DATA, "wadi"); os.makedirs(out, exist_ok=True)
    csv = os.path.join(raw, "WADI_14days_new.csv")
    if os.path.exists(csv):
        shutil.copyfile(csv, os.path.join(out, "WADI_14days_new.csv"))
    else:
        with zipfile.ZipFile(os.path.join(raw, "WADI_14days_new.zip")) as z:
            member = [m for m in z.namelist() if m.endswith("WADI_14days_new.csv")][0]
            with z.open(member) as src, open(os.path.join(out, "WADI_14days_new.csv"), "wb") as dst:
                shutil.copyfileobj(src, dst)
    print("WADI: rebuilding attack-file timestamps", flush=True)
    a = pd.read_csv(os.path.join(raw, "WADI_attackdataLABLE.csv"), header=1)
    a.columns = [c.strip() for c in a.columns]
    a = a.dropna(subset=["Row"]).reset_index(drop=True)
    # The Time column keeps only mm:ss. Rows are consecutive 1 Hz samples starting 2017-10-09 18:00:00.
    ts = pd.Timestamp("2017-10-09 18:00:00") + pd.to_timedelta(np.arange(len(a)), unit="s")
    mmss = a["Time"].astype(str).str.strip().str.extract(r"(\d+):(\d+)").astype(int)
    ok = ((ts.minute == mmss[0].to_numpy()) & (ts.second == mmss[1].to_numpy())).mean()
    assert ok == 1.0 and len(a) == 172801, f"WADI timestamp check failed ({ok:.4f}, {len(a)} rows)"
    a = a.drop(columns=["Row", "Date", "Time"])
    a = a.assign(ts=ts)
    a.to_parquet(os.path.join(out, "attack.parquet"))
    print("WADI: done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--swat", help="folder with the SWaT Excel files")
    p.add_argument("--wadi", help="folder with the WADI files")
    p.add_argument("--hai", help="folder with HAI 22.04 CSV files (omit to download)")
    p.add_argument("--skip-hai", action="store_true", help="do not prepare HAI")
    a = p.parse_args()
    if a.swat:
        swat(a.swat)
    if not a.skip_hai:
        hai(a.hai)
    if a.wadi:
        wadi(a.wadi)
    print(f"Prepared data is in {DATA}")
