from __future__ import annotations
import json, hashlib, urllib.request, urllib.parse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

START=pd.Timestamp('2020-02-14T09:00:00Z'); END=pd.Timestamp('2026-05-28T00:00:00Z')
EXPECTED_ROWS=660_853
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
MONTHLY='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m'
DAILY='https://data.binance.vision/data/spot/daily/klines/BTCUSDT/5m'

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def get_json(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read().decode('utf-8'))

def download(url,p):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=120) as r, open(p,'wb') as f:
  while True:
   b=r.read(1<<20)
   if not b: break
   f.write(b)

def read_zip(p, source='binance'):
 with zipfile.ZipFile(p) as z:
  names=[n for n in z.namelist() if not n.endswith('/')]
  if len(names)!=1: raise RuntimeError(f'ZIP_MEMBER_FAIL {p}')
  with z.open(names[0]) as f: x=pd.read_csv(f,header=None,names=COLS)
 x['open_time']=x['open_time'].map(lambda v:int(v)//1000 if int(v)>10**14 else int(v))
 x.insert(0,'timestamp',pd.to_datetime(x.open_time,unit='ms',utc=True))
 x=x[['timestamp','open','high','low','close','volume']].copy(); x['fill_source']=source
 return x

def months(a,b):
 cur=pd.Timestamp(a.year,a.month,1,tz='UTC'); stop=pd.Timestamp(b.year,b.month,1,tz='UTC')
 while cur<=stop:
  yield cur.year,cur.month; cur+=pd.offsets.MonthBegin(1)

def normalize(df):
 for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='raise')
 return df[(df.timestamp>=START)&(df.timestamp<=END)].sort_values('timestamp').drop_duplicates('timestamp',keep='first').reset_index(drop=True)

def fetch_bitstamp(day):
 st=int(pd.Timestamp(day,tz='UTC').timestamp()); en=st+86400-1
 q=urllib.parse.urlencode({'step':300,'limit':1000,'start':st,'end':en})
 obj=get_json('https://www.bitstamp.net/api/v2/ohlc/btcusd/?'+q)
 rows=[]
 for r in obj.get('data',{}).get('ohlc',[]):
  rows.append({'timestamp':pd.to_datetime(int(r['timestamp']),unit='s',utc=True),'open':r['open'],'high':r['high'],'low':r['low'],'close':r['close'],'volume':r['volume'],'fill_source':'bitstamp_btcusd_5m'})
 return pd.DataFrame(rows)

def fetch_bybit(day):
 st=int(pd.Timestamp(day,tz='UTC').timestamp()*1000); en=st+86400_000-1
 url='https://api.bybit.com/v5/market/kline?'+urllib.parse.urlencode({'category':'linear','symbol':'BTCUSDT','interval':'5','start':st,'end':en,'limit':1000})
 obj=get_json(url); rows=[]
 for r in obj.get('result',{}).get('list',[]):
  rows.append({'timestamp':pd.to_datetime(int(r[0]),unit='ms',utc=True),'open':r[1],'high':r[2],'low':r[3],'close':r[4],'volume':r[5],'fill_source':'bybit_linear_btcusdt_5m'})
 return pd.DataFrame(rows)

def main():
 out=Path('lh_run'); mr=out/'monthly'; dr=out/'daily_fill'; mr.mkdir(parents=True,exist_ok=True); dr.mkdir(exist_ok=True)
 frames=[]; archives=[]
 for y,m in months(START,END):
  n=f'BTCUSDT-5m-{y:04d}-{m:02d}.zip'; p=mr/n
  if not p.exists(): download(f'{MONTHLY}/{n}',p)
  x=read_zip(p,'binance_btcusdt_5m'); frames.append(x); archives.append({'file':n,'rows':len(x),'sha256':sha256(p)})
 df=normalize(pd.concat(frames,ignore_index=True)); exp=pd.date_range(START,END,freq='5min',tz='UTC')
 initial_missing=exp.difference(pd.DatetimeIndex(df.timestamp))
 # Binance daily confirms whether monthly omissions are real.
 daily=[]
 for day in sorted(set(ts.strftime('%Y-%m-%d') for ts in initial_missing)):
  n=f'BTCUSDT-5m-{day}.zip'; p=dr/n
  if not p.exists(): download(f'{DAILY}/{n}',p)
  daily.append(read_zip(p,'binance_btcusdt_5m_daily'))
 if daily: df=normalize(pd.concat([df,*daily],ignore_index=True))
 missing=exp.difference(pd.DatetimeIndex(df.timestamp))
 ext_rows=[]; external_audit=[]
 for day in sorted(set(ts.strftime('%Y-%m-%d') for ts in missing)):
  need=set(exp[(exp.strftime('%Y-%m-%d')==day)].intersection(missing))
  bs=fetch_bitstamp(day)
  bs_map=set(pd.DatetimeIndex(bs.timestamp)) if len(bs) else set()
  picked=bs[bs.timestamp.isin(need)].copy() if len(bs) else pd.DataFrame()
  left=need-set(pd.DatetimeIndex(picked.timestamp)) if len(picked) else need
  by=pd.DataFrame()
  if left:
   try: by=fetch_bybit(day)
   except Exception: by=pd.DataFrame()
   if len(by): picked=pd.concat([picked,by[by.timestamp.isin(left)]],ignore_index=True)
  if len(picked): ext_rows.append(picked)
  external_audit.append({'day':day,'needed':len(need),'bitstamp_available':sum(t in bs_map for t in need),'filled_total':len(set(pd.DatetimeIndex(picked.timestamp))) if len(picked) else 0})
 if ext_rows: df=normalize(pd.concat([df,*ext_rows],ignore_index=True))
 remaining=exp.difference(pd.DatetimeIndex(df.timestamp)); extra=pd.DatetimeIndex(df.timestamp).difference(exp)
 df['was_missing_on_binance']=df.fill_source.ne('binance_btcusdt_5m') & df.fill_source.ne('binance_btcusdt_5m_daily')
 df['external_fill']=df.fill_source.str.startswith(('bitstamp','bybit'))
 source_status='PASS' if len(df)==EXPECTED_ROWS and not len(remaining) and not len(extra) else 'FAIL'
 df.to_csv(out/'BTCUSDT_5m_FILLED_CANONICAL.csv',index=False)
 result={'source_status':source_status,'rows':len(df),'expected_rows':EXPECTED_ROWS,'initial_missing':len(initial_missing),'remaining_missing':len(remaining),'external_fill_rows':int(df.external_fill.sum()),'fill_source_counts':df.fill_source.value_counts().to_dict(),'external_audit':external_audit,'monthly_archives':archives}
 (out/'SOURCE_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
 if source_status!='PASS': raise RuntimeError('SOURCE_CONTRACT_FAIL')
if __name__=='__main__': main()
