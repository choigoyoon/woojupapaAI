from __future__ import annotations
import json, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

START = pd.Timestamp('2026-05-27T04:00:00Z')
END = pd.Timestamp('2026-05-27T05:00:00Z')
OUT = Path('lh_case4135_bybit')
OUT.mkdir(parents=True, exist_ok=True)

params = {
    'category':'linear','symbol':'BTCUSDT','interval':'5',
    'start':str(int(START.timestamp()*1000)),
    'end':str(int((END-pd.Timedelta(milliseconds=1)).timestamp()*1000)),
    'limit':'200',
}
url='https://api.bybit.com/v5/market/kline?'+urllib.parse.urlencode(params)
req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=60) as r:
    data=json.loads(r.read().decode('utf-8'))
if data.get('retCode') != 0:
    raise RuntimeError(data)
rows=data['result']['list']
rec=[]
for x in rows:
    ts=pd.to_datetime(int(x[0]), unit='ms', utc=True)
    rec.append({'timestamp':ts,'open':float(x[1]),'high':float(x[2]),'low':float(x[3]),'close':float(x[4]),'volume':float(x[5]),'fill_source':'bybit_linear_btcusdt_5m'})
df=pd.DataFrame(rec).sort_values('timestamp').reset_index(drop=True)
exp=pd.date_range(START, END-pd.Timedelta(minutes=5), freq='5min', tz='UTC')
if len(df)!=12 or not pd.DatetimeIndex(df.timestamp).equals(exp):
    raise RuntimeError(f'BYBIT_5M_CONTRACT_FAIL rows={len(df)} first={df.timestamp.min() if len(df) else None} last={df.timestamp.max() if len(df) else None}')
df.to_csv(OUT/'CASE4135_BYBIT_5M.csv', index=False)
result={
    'status':'PASS','rows':len(df),'hour_low':float(df.low.min()),'hour_high':float(df.high.max()),
    'low_time':df.loc[df.low.idxmin(),'timestamp'].isoformat(),
    'matches_expected_l_75235_70':abs(float(df.low.min())-75235.70)<1e-9,
}
(OUT/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
