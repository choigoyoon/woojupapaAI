from __future__ import annotations
import hashlib, json, urllib.parse, urllib.request, zipfile
from pathlib import Path
import pandas as pd

START=pd.Timestamp('2020-02-14T09:00:00Z')
END=pd.Timestamp('2026-05-28T00:00:00Z')
EXPECTED=660_853
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
BINANCE='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m'
BITSTAMP='https://www.bitstamp.net/api/v2/ohlc/btcusd/'

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def months():
    cur=pd.Timestamp(START.year,START.month,1,tz='UTC'); stop=pd.Timestamp(END.year,END.month,1,tz='UTC')
    while cur<=stop:
        yield cur.year,cur.month; cur+=pd.offsets.MonthBegin(1)

def norm_epoch(v):
    v=int(v); return v//1000 if v>10**14 else v

def download(url,p):
    if p.exists(): return
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=120) as r,p.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b: break
            f.write(b)

def read_binance(path):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if not n.endswith('/')]
        with z.open(names[0]) as f: d=pd.read_csv(f,header=None,names=COLS)
    d['open_time']=d.open_time.map(norm_epoch); d['timestamp']=pd.to_datetime(d.open_time,unit='ms',utc=True)
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='raise')
    return d[['timestamp','open','high','low','close','volume']]

def contiguous_runs(times):
    if not len(times): return []
    times=sorted(times); runs=[]; st=pr=times[0]
    for t in times[1:]:
        if t-pr==pd.Timedelta(minutes=5): pr=t; continue
        runs.append((st,pr)); st=pr=t
    runs.append((st,pr)); return runs

def bitstamp_run(st,en):
    # Fetch a small padded window, then retain exact requested 5m timestamps only.
    qs=urllib.parse.urlencode({'step':300,'limit':1000,'start':int((st-pd.Timedelta(minutes=5)).timestamp()),'end':int((en+pd.Timedelta(minutes=5)).timestamp())})
    req=urllib.request.Request(BITSTAMP+'?'+qs,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=60) as r: obj=json.load(r)
    rows=obj['data']['ohlc']; d=pd.DataFrame(rows)
    d['timestamp']=pd.to_datetime(pd.to_numeric(d.timestamp),unit='s',utc=True)
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='raise')
    return d[['timestamp','open','high','low','close','volume']]

def main():
    out=Path('lh_5m_bitstamp_repair'); raw=out/'raw5'; raw.mkdir(parents=True,exist_ok=True)
    frames=[]
    for y,m in months():
        name=f'BTCUSDT-5m-{y:04d}-{m:02d}.zip'; p=raw/name; download(f'{BINANCE}/{name}',p); frames.append(read_binance(p))
    b=pd.concat(frames,ignore_index=True); b=b[(b.timestamp>=START)&(b.timestamp<=END)].sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)
    expected=pd.date_range(START,END,freq='5min',tz='UTC'); missing=expected.difference(pd.DatetimeIndex(b.timestamp)); repairs=[]; unresolved=[]
    for st,en in contiguous_runs(list(missing)):
        d=bitstamp_run(st,en).set_index('timestamp')
        wanted=pd.date_range(st,en,freq='5min',tz='UTC')
        for ts in wanted:
            if ts not in d.index: unresolved.append(ts); continue
            r=d.loc[ts]
            if isinstance(r,pd.DataFrame): r=r.iloc[-1]
            repairs.append({'timestamp':ts,'open':float(r.open),'high':float(r.high),'low':float(r.low),'close':float(r.close),'volume':float(r.volume),'fill_source':'bitstamp_btcusd_5m'})
    if unresolved:
        pd.DataFrame({'timestamp':[x.isoformat() for x in unresolved]}).to_csv(out/'UNRESOLVED.csv',index=False)
        raise RuntimeError(f'BITSTAMP_UNRESOLVED {len(unresolved)}')
    b['fill_source']='binance_btcusdt_5m'; rep=pd.DataFrame(repairs)
    full=pd.concat([b,rep],ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)
    idx=pd.DatetimeIndex(full.timestamp); miss2=expected.difference(idx); extra=idx.difference(expected)
    if len(full)!=EXPECTED or len(miss2) or len(extra): raise RuntimeError(f'CONTINUITY_FAIL rows={len(full)} missing={len(miss2)} extra={len(extra)}')
    bad=(full.high<full[['open','close','low']].max(axis=1)) | (full.low>full[['open','close','high']].min(axis=1)) | (full.volume<0)
    if bad.any(): raise RuntimeError(f'OHLC_FAIL {int(bad.sum())}')
    csv=out/'BTCUSDT_5m_FILLED_SAME_LINEAGE.csv'; full.to_csv(csv,index=False)
    # aggregate repaired hours for local comparison with the existing 1h filled lineage
    ext=full[full.fill_source.eq('bitstamp_btcusd_5m')].copy(); ext['hour']=ext.timestamp.dt.floor('h')
    hourly=ext.groupby('hour').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),filled_5m_count=('timestamp','size')).reset_index()
    hourly.to_csv(out/'BITSTAMP_REPAIRED_HOURS.csv',index=False)
    result={'status':'PASS','rows':len(full),'original_binance_rows':len(b),'original_missing':len(missing),'bitstamp_repaired':len(rep),'unresolved':0,'repaired_hours':int(hourly.hour.nunique()),'sha256':sha256(csv),'first':full.timestamp.iloc[0].isoformat(),'last':full.timestamp.iloc[-1].isoformat()}
    (out/'REPAIR_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
