"""§113 诊断:定位复现差距的来源(前复权市值 vs 真实市值 / 上市天数口径)。"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,'/home/user/open-xquant/examples/ashare-bull-stock-study/rps_pool_study')
from codex_r10_replication import DATA, TOP_N, WEIGHT, WINDOWS, metrics, pct, run_window

z=np.load('/home/user/oxq-panel/codex_r10_matrices.npz',allow_pickle=True)
idx = pd.DatetimeIndex(z['idx'])
codes = list(z['codes'])
OP,CL,SUSP,LU,LD,OK=z['OP'],z['CL'],z['SUSP'],z['LU'],z['LD'],z['OK']
LOGCAP,TMEAN=z['LOGCAP'],z['TMEAN']
nt, ns = len(idx), len(codes)

# 前复权市值 = float_mv × (qfq_close/raw_close)  -> log 加 log(ratio)
ADJ=np.full((nt,ns),np.nan,np.float32)
LD365 = np.zeros((nt, ns), bool)
ST = np.zeros((nt, ns), bool)
t0=time.time()
for j,c in enumerate(codes):
    x=pd.read_parquet(f"{DATA}/{c}.parquet",columns=["close","raw_close","listed_days","is_st"])
    if getattr(x.index, 'tz', None) is not None:
        x.index = x.index.tz_localize(None)
    x=x.reindex(idx)
    r=pd.to_numeric(x['close'],errors='coerce')/pd.to_numeric(x['raw_close'],errors='coerce')
    ADJ[:,j]=np.log(r.where(r>0)).to_numpy(np.float32)
    ldv=pd.to_numeric(x['listed_days'],errors='coerce').to_numpy(float)
    LD365[:,j]=ldv>=250          # 放宽到 250 日历日
    ST[:,j]=x['is_st'].fillna(True).to_numpy(bool)
    if (j + 1) % 2000 == 0:
        print(f"  {j+1}/{ns} ({time.time()-t0:.0f}s)", flush=True)
LOGCAP_QFQ=LOGCAP+ADJ
print(f"辅助矩阵完成 ({time.time()-t0:.0f}s)",flush=True)

b=pd.read_parquet(f"{DATA}/510300.parquet",columns=["close"])
b.index=pd.to_datetime(b.index).tz_localize(None)
cal = pd.DatetimeIndex(b.index.unique()).sort_values()
cal = cal[(cal >= '2014-01-01') & (cal <= '2026-08-20')]
cal_pos = pd.Index(idx).get_indexer(cal)
reb = cal_pos[::20]
ipos=pd.Index(idx)

def build(capm, okm):
    sel={}
    for t in reb:
        m=okm[t]&np.isfinite(capm[t])&np.isfinite(TMEAN[t])
        e=np.flatnonzero(m)
        if len(e) < TOP_N * 3:
            continue
        s=(pct(-capm[t,e].astype(float))+pct(-TMEAN[t,e].astype(float)))/2
        sel[int(t)]=(e[np.argsort(-s,kind='stable')[:TOP_N]],np.full(TOP_N,WEIGHT))
    return sel

def go(name, capm, okm, wins=('train','full')):
    sel = build(capm, okm)
    out = []
    for w in wins:
        d0, d1 = WINDOWS[w]
        w0=int(ipos.get_indexer([pd.Timestamp(d0)],method='bfill')[0])
        w1=int(ipos.get_indexer([pd.Timestamp(d1)],method='ffill')[0])
        eq, days, tr, fz = run_window(OP, CL, SUSP, LU, LD, sel, cal_pos, w0, w1)
        m = metrics(eq, days, idx)
        out.append((w, m['total'], m['cagr']))
    print(f"{name:34s} "+"  ".join(f"{w}:{t:+9.2%}(年化{c:+6.2%})" for w,t,c in out),flush=True)
    return out

OKL=OK.copy()  # 基线
print("\nCodex 目标:              train:+401.92%(年化+30.89%)  full:+3135.06%(年化+33.62%)\n")
go("① 基线(真实市值,365日)", LOGCAP, OK)
go("② 前复权市值(前视?)",    LOGCAP_QFQ, OK)
ok250 = OK | (LD365 & ~ST & ~SUSP & ~LU)   # 只放宽上市天数
ok250 = ok250 & ~ST & ~SUSP & ~LU
go("③ 真实市值,上市>=250日历日", LOGCAP, ok250)
go("④ 前复权市值+250日历日",   LOGCAP_QFQ, ok250)
