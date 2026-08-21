#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, urllib.request, zipfile
from pathlib import Path
import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m"
COLS = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def months(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    stop = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= stop:
        yield cur.year, cur.month
        cur = cur + pd.offsets.MonthBegin(1)

def normalize_epoch(v):
    v=int(v)
    return v // 1000 if v > 10**14 else v

def download(url: str, dest: Path):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2020-02-14T09:00:00Z")
    ap.add_argument("--end", default="2026-05-28T00:00:00Z")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw=out/"monthly_zips"; raw.mkdir(exist_ok=True)
    start=pd.Timestamp(args.start); end=pd.Timestamp(args.end)
    frames=[]; archive_manifest=[]
    for y,m in months(start,end):
        name=f"BTCUSDT-5m-{y:04d}-{m:02d}.zip"
        path=raw/name
        url=f"{BASE}/{name}"
        if not path.exists(): download(url,path)
        with zipfile.ZipFile(path) as z:
            members=[n for n in z.namelist() if not n.endswith("/")]
            if len(members)!=1: raise RuntimeError(f"ZIP_MEMBER_FAIL {name}: {members}")
            with z.open(members[0]) as f:
                df=pd.read_csv(f, header=None, names=COLS)
        df["open_time"]=df["open_time"].map(normalize_epoch)
        ts=pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.insert(0,"timestamp",ts)
        frames.append(df[["timestamp","open","high","low","close","volume"]])
        archive_manifest.append({"file":name,"sha256":sha256(path),"rows":len(df),"url":url})
    all_df=pd.concat(frames, ignore_index=True)
    for c in ["open","high","low","close","volume"]:
        all_df[c]=pd.to_numeric(all_df[c], errors="raise")
    all_df=all_df[(all_df.timestamp>=start)&(all_df.timestamp<=end)].copy()
    all_df=all_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    exp=pd.date_range(start,end,freq="5min",tz="UTC")
    missing=exp.difference(pd.DatetimeIndex(all_df.timestamp))
    extra=pd.DatetimeIndex(all_df.timestamp).difference(exp)
    if len(missing) or len(extra) or len(all_df)!=len(exp):
        raise RuntimeError(f"CONTINUITY_FAIL rows={len(all_df)} expected={len(exp)} missing={len(missing)} extra={len(extra)}")
    bad=(all_df.high<all_df[["open","close","low"]].max(axis=1)) | (all_df.low>all_df[["open","close","high"]].min(axis=1)) | (all_df.volume<0)
    if bad.any(): raise RuntimeError(f"OHLC_FAIL {int(bad.sum())}")
    out_csv=out/"BTCUSDT_5m_CANONICAL.csv"
    all_df.to_csv(out_csv,index=False)
    manifest={"source":"Binance public spot monthly klines","start":str(start),"end":str(end),"rows":len(all_df),"expected_rows":len(exp),"sha256":sha256(out_csv),"archives":archive_manifest}
    (out/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","rows":len(all_df),"csv":str(out_csv),"sha256":manifest["sha256"]},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
