"""趋势持续性按年份:金叉/死叉后维持同向排列多少周(描述性)。"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0,"/home/user/open-xquant/examples/ashare-bull-stock-study/rps_pool_study")
from codex_r10_replication import DATA
t0=time.time()
codes=[os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
       if os.path.basename(f)[:-8]!="510300"]
cl_,ok_={},{}
for c in codes:
    x=pd.read_parquet(f"{DATA}/{c}.parquet",columns=["close","volume","is_st","is_suspended","listed_days"])
    if getattr(x.index,"tz",None) is not None: x.index=x.index.tz_localize(None)
    cl_[c]=pd.to_numeric(x["close"],errors="coerce")
    ok_[c]=((~x["is_st"].fillna(True).astype(bool))&(~x["is_suspended"].fillna(True).astype(bool))
            &(pd.to_numeric(x["listed_days"],errors="coerce").fillna(0)>=250)
            &(pd.to_numeric(x["volume"],errors="coerce").fillna(0)>0))
cldf=pd.DataFrame(cl_).sort_index(); idx=cldf.index; nt,ns=cldf.shape
okdf=pd.DataFrame(ok_).sort_index().reindex(index=idx,columns=cldf.columns).fillna(False)
cl=cldf.where(cldf>0).ffill().to_numpy(np.float64)
wk=pd.Series(np.arange(nt),index=idx).groupby([idx.isocalendar().year,idx.isocalendar().week]).last()
wpos=np.sort(wk.to_numpy()); wyear=idx[wpos].year.to_numpy()
wdf=pd.DataFrame(cl[wpos])
m20=wdf.rolling(20,min_periods=20).mean().to_numpy(); m60=wdf.rolling(60,min_periods=60).mean().to_numpy()
bw=m20>m60; fw=np.isfinite(m20)&np.isfinite(m60)
wok=okdf.to_numpy()[wpos]
nw=len(wpos)
print(f"周线 {nw} 周 × {ns} 股 ({time.time()-t0:.0f}s)\n")
rows=[]
for j in range(ns):
    f=fw[:,j]
    if f.sum()<60: continue
    b=bw[:,j]; o=wok[:,j]
    i=1
    while i<nw:
        if not (f[i] and f[i-1]):
            i+=1; continue
        if b[i]!=b[i-1] and o[i]:
            k=i
            while k+1<nw and f[k+1] and b[k+1]==b[i]: k+=1
            rows.append((int(wyear[i]), "金叉" if b[i] else "死叉", k-i+1))
            i=k+1
        else: i+=1
r=pd.DataFrame(rows,columns=["year","type","weeks"])
r=r[(r.year>=2014)&(r.year<=2025)]
print("="*76); print("每次金叉/死叉后,维持同向排列的周数(按发生年份)"); print("="*76)
print(f"{'年份':<7}{'金叉次数':>9}{'金叉中位周':>11}{'金叉>52周占比':>14}"
      f"{'死叉次数':>9}{'死叉中位周':>11}{'死叉>52周占比':>14}")
for y,g in r.groupby("year"):
    a=g[g.type=="金叉"]; b2=g[g.type=="死叉"]
    print(f"{y:<7}{len(a):>9,}{a.weeks.median():>11.0f}{(a.weeks>52).mean():>14.1%}"
          f"{len(b2):>9,}{b2.weeks.median():>11.0f}{(b2.weeks>52).mean():>14.1%}")
print("\n分段汇总")
for lo,hi,nm in [(2014,2019,"2014-2019"),(2020,2025,"2020-2025")]:
    g=r[(r.year>=lo)&(r.year<=hi)]
    a=g[g.type=="金叉"]; b2=g[g.type=="死叉"]
    print(f"  {nm}:金叉 n={len(a):,} 中位 {a.weeks.median():.0f} 周,>52周 {(a.weeks>52).mean():.1%}"
          f" | 死叉 n={len(b2):,} 中位 {b2.weeks.median():.0f} 周,>52周 {(b2.weeks>52).mean():.1%}")
print("\n注:2024-2025 的样本被面板末日(2026-08-03)截断,持续期偏低,不可直接比较。")
