"""X01-v1 2026 Top10 的前瞻观察(n=10,不足以下判据,仅描述)。"""
import glob, os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0,"/home/user/open-xquant/examples/ashare-bull-stock-study/rps_pool_study")
from codex_r10_replication import DATA
from industry_neutral import build_industry
TOP=[("301232","飞沃科技"),("000547","航天发展"),("688577","浙海德曼"),
     ("301005","超捷股份"),("000592","平潭发展"),("688788","科思科技"),
     ("301529","福赛科技"),("301117","佳缘科技"),("688228","开普云"),
     ("688353","华盛锂电")]
t0=time.time()
codes=[os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
       if os.path.basename(f)[:-8]!="510300"]
cl,mv={},{}
for c in codes:
    x=pd.read_parquet(f"{DATA}/{c}.parquet",columns=["close","float_mv"])
    if getattr(x.index,"tz",None) is not None: x.index=x.index.tz_localize(None)
    cl[c]=pd.to_numeric(x["close"],errors="coerce").where(lambda s:s>0)
    mv[c]=pd.to_numeric(x["float_mv"],errors="coerce").where(lambda s:s>0)
q=pd.DataFrame(cl).sort_index(); m=pd.DataFrame(mv).sort_index().reindex(index=q.index,columns=q.columns)
idx=q.index; assert q.shape==(3297,5232), q.shape
ind,_,_=build_industry(list(q.columns), idx)
print(f"面板 {q.shape} ({time.time()-t0:.0f}s)",flush=True)
ip=pd.Index(idx)
t25=int(ip.get_indexer([pd.Timestamp("2025-12-31")],method="ffill")[0])
t24=int(ip.get_indexer([pd.Timestamp("2024-12-31")],method="ffill")[0])
tN=len(idx)-1
a=q.to_numpy(); mvv=m.to_numpy()/1e8
cp={c:j for j,c in enumerate(q.columns)}
print(f"观察日 {idx[t25].date()}  末日 {idx[tN].date()}\n")
print(f"{'代码':<8}{'名称':<9}{'2025涨幅':>10}{'2026至今':>10}{'观察日市值(亿)':>14}  同市值同行业对照 2026 中位/胜出分位")
rows=[]
for c,nm in TOP:
    j=cp[c]
    r25=a[t25,j]/a[t24,j]-1; r26=a[tN,j]/a[t25,j]-1
    i0=ind[t25,j]
    e=np.flatnonzero(np.isfinite(mvv[t25]) & (ind[t25]>=0) & np.isfinite(a[tN]) & np.isfinite(a[t25]))
    o=e[np.argsort(mvv[t25,e],kind="stable")]
    rk={int(x):i for i,x in enumerate(o)}
    p0=rk.get(j)
    if p0 is None or i0<0:
        print(f"{c:<8}{nm:<9}{r25:>10.1%}{r26:>10.1%}{mvv[t25,j]:>14.1f}  "
              f"观察日不在对照候选池(市值或行业缺失),跳过对照")
        rows.append({"代码":c,"名称":nm,"r2025":r25,"r2026":r26,"mv":mvv[t25,j],
                     "对照中位":np.nan,"对照数":0,"胜出分位":np.nan})
        continue
    lo,hi=max(0,p0-25),min(len(o)-1,p0+25)
    cand=np.array([x for x in o[lo:hi+1] if ind[t25,x]==i0 and x!=j])
    if len(cand)<5: cand=np.array([x for x in o if ind[t25,x]==i0 and x!=j])
    if len(cand)<2:
        rows.append({"代码":c,"名称":nm,"r2025":r25,"r2026":r26,"mv":mvv[t25,j],
                     "对照中位":np.nan,"对照数":0,"胜出分位":np.nan}); continue
    cr=a[tN,cand]/a[t25,cand]-1
    pct=float((cr<r26).mean())
    rows.append({"代码":c,"名称":nm,"r2025":r25,"r2026":r26,"mv":mvv[t25,j],
                 "对照中位":float(np.nanmedian(cr)),"对照数":len(cand),"胜出分位":pct})
    print(f"{c:<8}{nm:<9}{r25:>10.1%}{r26:>10.1%}{mvv[t25,j]:>14.1f}  "
          f"{np.nanmedian(cr):>7.1%} / {pct:>5.0%}")
d=pd.DataFrame(rows)
print(f"\nTop10 2026 至今  中位 {d.r2026.median():+.1%}  均值 {d.r2026.mean():+.1%}")
print(f"同市值同行业对照 中位的中位 {d['对照中位'].median():+.1%}")
print(f"胜出分位 中位 {d['胜出分位'].median():.0%}(50% = 与对照无差别)")
print(f"\n2025 年涨幅 中位 {d.r2025.median():+.1%}  最大 {d.r2025.max():+.1%}")
print(f"2025 年已翻倍(>100%)的只数:{int((d.r2025>1.0).sum())}/10")
d.to_csv("/home/user/oxq-panel/x01_top10_probe.csv",index=False,encoding="utf-8-sig")
print("落库 /home/user/oxq-panel/x01_top10_probe.csv")
