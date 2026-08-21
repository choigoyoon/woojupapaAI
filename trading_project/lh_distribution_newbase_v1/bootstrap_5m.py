from __future__ import annotations
import json, hashlib, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

START=pd.Timestamp('2020-02-14T09:00:00Z'); END=pd.Timestamp('2026-05-28T00:00:00Z')
EXPECTED_ROWS=660_853; FAST,SLOW,SIGNAL=144,312,108
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
MONTHLY='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m'
DAILY='https://data.binance.vision/data/spot/daily/klines/BTCUSDT/5m'

def sha256(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def download(url,p):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=120) as r, open(p,'wb') as f:
  while True:
   b=r.read(1<<20)
   if not b: break
   f.write(b)

def read_zip(p):
 with zipfile.ZipFile(p) as z:
  names=[n for n in z.namelist() if not n.endswith('/')]
  if len(names)!=1: raise RuntimeError(f'ZIP_MEMBER_FAIL {p}')
  with z.open(names[0]) as f: x=pd.read_csv(f,header=None,names=COLS)
 x['open_time']=x['open_time'].map(lambda v:int(v)//1000 if int(v)>10**14 else int(v))
 x.insert(0,'timestamp',pd.to_datetime(x.open_time,unit='ms',utc=True))
 return x[['timestamp','open','high','low','close','volume']]

def months(a,b):
 cur=pd.Timestamp(a.year,a.month,1,tz='UTC'); stop=pd.Timestamp(b.year,b.month,1,tz='UTC')
 while cur<=stop:
  yield cur.year,cur.month; cur+=pd.offsets.MonthBegin(1)

def ema(x,span):
 a=2/(span+1); out=np.empty(len(x)); out[0]=x[0]
 for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
 return out

def normalize(df):
 for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='raise')
 return df[(df.timestamp>=START)&(df.timestamp<=END)].sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)

def main():
 out=Path('lh_run'); mr=out/'monthly'; dr=out/'daily_fill'; mr.mkdir(parents=True,exist_ok=True); dr.mkdir(exist_ok=True)
 frames=[]; archives=[]
 for y,m in months(START,END):
  n=f'BTCUSDT-5m-{y:04d}-{m:02d}.zip'; p=mr/n
  if not p.exists(): download(f'{MONTHLY}/{n}',p)
  x=read_zip(p); frames.append(x); archives.append({'file':n,'rows':len(x),'sha256':sha256(p)})
 df=normalize(pd.concat(frames,ignore_index=True)); exp=pd.date_range(START,END,freq='5min',tz='UTC')
 initial_missing=exp.difference(pd.DatetimeIndex(df.timestamp))
 fill_frames=[]; fill_files=[]
 for day in sorted(set(ts.strftime('%Y-%m-%d') for ts in initial_missing)):
  n=f'BTCUSDT-5m-{day}.zip'; p=dr/n
  if not p.exists(): download(f'{DAILY}/{n}',p)
  x=read_zip(p); fill_frames.append(x); fill_files.append({'file':n,'rows':len(x),'sha256':sha256(p)})
 if fill_frames: df=normalize(pd.concat([df,*fill_frames],ignore_index=True))
 missing=exp.difference(pd.DatetimeIndex(df.timestamp)); extra=pd.DatetimeIndex(df.timestamp).difference(exp)
 source_status='PASS' if len(df)==EXPECTED_ROWS and not len(missing) and not len(extra) else 'FAIL'
 df.to_csv(out/'BTCUSDT_5m_CANONICAL.csv',index=False)
 c=df.close.to_numpy(float); line=ema(c,FAST)-ema(c,SLOW); hist=line-ema(line,SIGNAL); sign=np.sign(hist).astype(np.int8)
 nz=np.flatnonzero(sign); sign[:nz[0]]=sign[nz[0]]
 for i in range(nz[0]+1,len(sign)):
  if sign[i]==0: sign[i]=sign[i-1]
 starts=[0]+[i for i in range(1,len(sign)) if sign[i]!=sign[i-1]]; hi=df.high.to_numpy(float); lo=df.low.to_numpy(float); seg=[]
 for j in range(len(starts)-1):
  st,zc=starts[j],starts[j+1]; en=zc-1
  if sign[st]>0: p=st+int(np.argmax(hi[st:en+1])); side='H'
  else: p=st+int(np.argmin(lo[st:en+1])); side='L'
  seg.append({'side':side,'pivot_idx':p,'pivot_time':df.at[p,'timestamp'].isoformat(),'zc_idx':zc,'zc_time':df.at[zc,'timestamp'].isoformat()})
 cases=[]
 for i,(a,b) in enumerate(zip(seg[:-1],seg[1:]),1): cases.append({'case_id':i,'side':a['side'],'start_idx':a['pivot_idx'],'start_time':a['pivot_time'],'zc_idx':a['zc_idx'],'zc_time':a['zc_time'],'target_idx':b['pivot_idx'],'target_time':b['pivot_time']})
 with (out/'lh_cases.jsonl').open('w',encoding='utf-8') as f:
  for r in cases: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 H=sum(r['side']=='H' for r in cases); L=sum(r['side']=='L' for r in cases)
 result={'source_status':source_status,'rows':len(df),'expected_rows':EXPECTED_ROWS,'initial_missing':len(initial_missing),'daily_fill_days':len(fill_files),'remaining_missing':len(missing),'extra':len(extra),'segments':len(seg),'cases':len(cases),'H':H,'L':L,'first_case':cases[0] if cases else None,'last_case':cases[-1] if cases else None,'monthly_archives':archives,'daily_fill_files':fill_files}
 (out/'BOOTSTRAP_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
 if source_status!='PASS': raise RuntimeError('SOURCE_CONTRACT_FAIL')
if __name__=='__main__': main()
