from __future__ import annotations
import hashlib, json, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

START = pd.Timestamp('2020-02-14T09:00:00Z')
END = pd.Timestamp('2026-05-28T00:00:00Z')
EXPECTED_ROWS = 660_853
FAST,SLOW,SIGNAL = 144,312,108
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
BASE5='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m'
BASE1='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m'

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def months(start,end):
    cur=pd.Timestamp(start.year,start.month,1,tz='UTC'); stop=pd.Timestamp(end.year,end.month,1,tz='UTC')
    while cur<=stop:
        yield cur.year,cur.month; cur += pd.offsets.MonthBegin(1)

def norm_epoch(v):
    v=int(v); return v//1000 if v>10**14 else v

def dl(url,p):
    if p.exists(): return
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=120) as r,p.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b: break
            f.write(b)

def read_zip(path):
    with zipfile.ZipFile(path) as z:
        member=[n for n in z.namelist() if not n.endswith('/')]
        if len(member)!=1: raise RuntimeError(f'ZIP_MEMBER_FAIL {path.name}')
        with z.open(member[0]) as f: df=pd.read_csv(f,header=None,names=COLS)
    df['open_time']=df['open_time'].map(norm_epoch)
    df.insert(0,'timestamp',pd.to_datetime(df.open_time,unit='ms',utc=True))
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='raise')
    return df[['timestamp','open','high','low','close','volume']]

def ema(x,span):
    a=2/(span+1); out=np.empty(len(x)); out[0]=x[0]
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out

def event_count(df):
    c=df.close.to_numpy(float); line=ema(c,FAST)-ema(c,SLOW); h=line-ema(line,SIGNAL)
    s=np.sign(h).astype(np.int8); nz=np.flatnonzero(s)
    if not len(nz): raise RuntimeError('NO_SIGN')
    s[:nz[0]]=s[nz[0]]
    for i in range(nz[0]+1,len(s)):
        if s[i]==0:s[i]=s[i-1]
    starts=[0]+[i for i in range(1,len(s)) if s[i]!=s[i-1]]
    events=[]; hi=df.high.to_numpy(float); lo=df.low.to_numpy(float)
    for st,zc in zip(starts[:-1],starts[1:]):
        en=zc-1; sg=int(s[st])
        if sg>0: p=st+int(np.argmax(hi[st:en+1])); side='H'; price=float(hi[p])
        else: p=st+int(np.argmin(lo[st:en+1])); side='L'; price=float(lo[p])
        events.append({'side':side,'pivot_idx':p,'pivot_time':df.at[p,'timestamp'].isoformat(),'pivot_price':price,'zc_idx':zc,'zc_time':df.at[zc,'timestamp'].isoformat()})
    H=sum(x['side']=='H' for x in events); L=sum(x['side']=='L' for x in events)
    return events,H,L

def main():
    out=Path('lh_run_repaired'); raw5=out/'raw5'; raw1=out/'raw1'; raw5.mkdir(parents=True,exist_ok=True); raw1.mkdir(exist_ok=True)
    frames=[]
    for y,m in months(START,END):
        name=f'BTCUSDT-5m-{y:04d}-{m:02d}.zip'; p=raw5/name; dl(f'{BASE5}/{name}',p); frames.append(read_zip(p))
    df=pd.concat(frames,ignore_index=True)
    df=df[(df.timestamp>=START)&(df.timestamp<=END)].sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)
    exp=pd.date_range(START,END,freq='5min',tz='UTC'); missing=exp.difference(pd.DatetimeIndex(df.timestamp))
    repairs=[]; unresolved=[]
    by_month={}
    for ts in missing: by_month.setdefault((ts.year,ts.month),[]).append(ts)
    for (y,m),targets in sorted(by_month.items()):
        name=f'BTCUSDT-1m-{y:04d}-{m:02d}.zip'; p=raw1/name; dl(f'{BASE1}/{name}',p); one=read_zip(p).set_index('timestamp')
        for ts in targets:
            wanted=pd.date_range(ts,periods=5,freq='1min',tz='UTC')
            got=one.reindex(wanted)
            if got[['open','high','low','close','volume']].isna().any().any():
                unresolved.append(ts.isoformat()); continue
            repairs.append({'timestamp':ts,'open':float(got.iloc[0].open),'high':float(got.high.max()),'low':float(got.low.min()),'close':float(got.iloc[-1].close),'volume':float(got.volume.sum()),'recovered_from_1m':1})
    rep=pd.DataFrame(repairs)
    if unresolved:
        pd.DataFrame({'timestamp':unresolved}).to_csv(out/'UNRESOLVED_GAPS.csv',index=False)
        raise RuntimeError(f'UNRESOLVED_1M_GAPS {len(unresolved)}')
    df['recovered_from_1m']=0
    df=pd.concat([df,rep],ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)
    idx=pd.DatetimeIndex(df.timestamp); missing2=exp.difference(idx); extra=idx.difference(exp)
    if len(df)!=EXPECTED_ROWS or len(missing2) or len(extra): raise RuntimeError(f'REPAIR_CONTINUITY_FAIL rows={len(df)} missing={len(missing2)} extra={len(extra)}')
    events,H,L=event_count(df)
    df.to_csv(out/'BTCUSDT_5m_REPAIRED_FROM_1M.csv',index=False)
    with (out/'LH_EVENTS.jsonl').open('w',encoding='utf-8') as f:
        for i,x in enumerate(events,1): f.write(json.dumps({'event_no':i,**x},ensure_ascii=False,separators=(',',':'))+'\n')
    result={'status':'PASS_SOURCE_REPAIRED','rows':len(df),'original_missing_5m':len(missing),'recovered_from_actual_1m':len(repairs),'unresolved':0,'segments_events':len(events),'H':H,'L':L,'matches_4136':(len(events),H,L)==(4136,2068,2068),'source_sha256':sha256(out/'BTCUSDT_5m_REPAIRED_FROM_1M.csv')}
    (out/'REPAIR_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
