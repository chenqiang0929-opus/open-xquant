"""验证四只股票在带上限口径下的信号时点。"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0,"/home/user/open-xquant/examples/ashare-bull-stock-study/rps_pool_study")
from codex_r10_replication import DATA
T={"300308":"中际旭创","300476":"胜宏科技","300100":"双林股份","603119":"浙江荣泰",
   "600066":"宇通客车"}
t0=time.time()
codes=[os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
       if os.path.basename(f)[:-8]!="510300"]
cols=["close","turnover","volume","is_st","is_suspended","listed_days"]
d={c:{} for c in cols}
for c in codes:
    x=pd.read_parquet(f"{DATA}/{c}.parquet",columns=cols)
    if getattr(x.index,"tz",None) is not None: x.index=x.index.tz_localize(None)
    for k in cols: d[k][c]=x[k]
cldf=pd.DataFrame(d["close"]).sort_index(); idx=cldf.index; nt,ns=cldf.shape
def al(k,f=np.nan): return pd.DataFrame(d[k]).sort_index().reindex(index=idx,columns=cldf.columns).fillna(f)
trn=al("turnover")
ok=(~al("is_st",True).astype(bool).to_numpy()&~al("is_suspended",True).astype(bool).to_numpy()
    &(al("listed_days",0).to_numpy()>=250)&(al("volume",0).to_numpy()>0))
cl=cldf.where(cldf>0).ffill().to_numpy(np.float64); ok&=np.isfinite(cl)
dfc=pd.DataFrame(cl)
lo250=dfc.rolling(250,min_periods=250).min().to_numpy()
t20=trn.rolling(20,min_periods=10).mean().to_numpy(); t60=trn.rolling(60,min_periods=30).mean().to_numpy()
with np.errstate(all="ignore"):
    rec=cl/np.where(lo250>0,lo250,np.nan)-1.0
    tacc=t20/np.where(t60>0,t60,np.nan)-1.0
fmax=pd.DataFrame(cl[::-1]).rolling(60,min_periods=1).max().to_numpy()[::-1]
fwd=np.full_like(cl,np.nan); fwd[:-1]=fmax[1:]
with np.errstate(all="ignore"): up=fwd/np.where(cl>0,cl,np.nan)-1.0
me=pd.Series(np.arange(nt),index=idx).groupby([idx.year,idx.month]).last().to_numpy()
cp={c:j for j,c in enumerate(cldf.columns)}
print(f"面板就绪 ({time.time()-t0:.0f}s)  上限口径:距低点前30% & 换手加速前30% & 距低点≤100%,降序取前100\n")
for code,nm in T.items():
    j=cp[code]
    print("="*104); print(f"{nm} {code}"); print("="*104)
    print(f"{'观察日':<12}{'收盘':>8}{'距低点':>8}{'分位':>7}{'换手加速':>9}{'分位':>7}"
          f"{'入选':>7}{'排名':>7}{'未来60日':>9}{'达50%':>7}")
    hits=0
    for t in me:
        t=int(t)
        if idx[t]<pd.Timestamp("2023-01-01"): continue
        fut = t<=nt-61
        m=ok[t]&np.isfinite(rec[t])&np.isfinite(tacc[t])
        if fut: m&=np.isfinite(up[t])
        e=np.flatnonzero(m)
        if len(e)<100 or not m[j]: 
            print(f"{str(idx[t].date()):<12}  不合格"); continue
        qr=pd.Series(rec[t,e]).rank(pct=True).to_numpy()
        qt=pd.Series(tacc[t,e]).rank(pct=True).to_numpy()
        k=int(np.flatnonzero(e==j)[0])
        pool=e[(qr>=.70)&(qt>=.70)]
        pool=pool[rec[t,pool]<=1.00]
        sel=pool[np.argsort(-rec[t,pool],kind="stable")[:100]] if len(pool) else np.array([],int)
        inx = j in set(sel.tolist())
        rk = int(np.flatnonzero(sel==j)[0])+1 if inx else 0
        u = up[t,j] if fut and np.isfinite(up[t,j]) else np.nan
        if inx: hits+=1
        print(f"{str(idx[t].date()):<12}{cl[t,j]:>8.2f}{rec[t,j]:>8.0%}{qr[k]:>7.2f}"
              f"{tacc[t,j]:>9.0%}{qt[k]:>7.2f}{'**是**' if inx else '—':>7}"
              f"{(str(rk) if inx else '—'):>7}"
              f"{(f'{u:+.0%}' if np.isfinite(u) else '—'):>9}"
              f"{('★' if np.isfinite(u) and u>=0.5 else ''):>7}")
    print(f"  → 2023-01 以来共入选 {hits} 次\n")
