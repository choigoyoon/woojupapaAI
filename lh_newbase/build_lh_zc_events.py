#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
FAST,SLOW,SIGNAL=144,312,108

def ema(x, span):
    a=2.0/(span+1.0); out=np.empty(len(x),dtype=float); out[0]=x[0]
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out

def hist(close):
    line=ema(close,FAST)-ema(close,SLOW)
    return line-ema(line,SIGNAL)

def load(path):
    df=pd.read_csv(path)
    need=["timestamp","open","high","low","close","volume"]
    if any(c not in df for c in need): raise RuntimeError("COLUMN_FAIL")
    df=df[need].copy(); df["timestamp"]=pd.to_datetime(df.timestamp,utc=True)
    if len(df)!=660853: raise RuntimeError(f"ROW_COUNT_FAIL {len(df)}")
    if df.timestamp.duplicated().any(): raise RuntimeError("DUPLICATE_FAIL")
    if not df.timestamp.diff().dropna().eq(pd.Timedelta(minutes=5)).all(): raise RuntimeError("GAP_FAIL")
    return df

def build(df):
    h=hist(df.close.to_numpy(float)); s=np.sign(h).astype(np.int8)
    nz=np.flatnonzero(s)
    if not len(nz): raise RuntimeError("MACD_SIGN_FAIL")
    s[:nz[0]]=s[nz[0]]
    for i in range(nz[0]+1,len(s)):
        if s[i]==0:s[i]=s[i-1]
    starts=[0]+[i for i in range(1,len(s)) if s[i]!=s[i-1]]
    hi=df.high.to_numpy(float); lo=df.low.to_numpy(float); events=[]
    for n,(st,zc) in enumerate(zip(starts[:-1],starts[1:]),1):
        en=zc-1; sg=int(s[st])
        if sg>0: p=st+int(np.argmax(hi[st:en+1])); side="H"; price=hi[p]
        else: p=st+int(np.argmin(lo[st:en+1])); side="L"; price=lo[p]
        events.append({"event_no":n,"side":side,"pivot_idx":p,"pivot_time":df.at[p,"timestamp"].isoformat(),"pivot_price":float(price),"segment_start_idx":st,"segment_end_idx":en,"next_zc_idx":zc,"next_zc_time":df.at[zc,"timestamp"].isoformat(),"memory_ready_idx":zc,"memory_ready_time":df.at[zc,"timestamp"].isoformat()})
    return events

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ohlcv",required=True); ap.add_argument("--out",required=True); ap.add_argument("--strict-4136",action="store_true")
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df=load(a.ohlcv); ev=build(df); L=sum(x["side"]=="L" for x in ev); H=sum(x["side"]=="H" for x in ev)
    p=out/"LH_EVENTS.jsonl"
    with p.open("w",encoding="utf-8") as f:
        for x in ev:f.write(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n")
    manifest={"rows":len(df),"events":len(ev),"L":L,"H":H,"ohlcv_sha256":sha256(a.ohlcv),"events_sha256":sha256(p),"strict_target":{"events":4136,"L":2068,"H":2068},"strict_match":(len(ev),L,H)==(4136,2068,2068)}
    (out/"EVENT_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    if a.strict_4136 and not manifest["strict_match"]: raise RuntimeError(f"EVENT_COUNT_FAIL total={len(ev)} L={L} H={H}")
if __name__=="__main__":main()
