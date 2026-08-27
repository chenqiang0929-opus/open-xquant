"""19 只名单:三档信号 + 选择偏差量化。"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0,"/home/user/open-xquant/examples/ashare-bull-stock-study/rps_pool_study")
from codex_r10_replication import DATA
from startup_threshold_scan import load_labels
t0=time.time()
XLS="/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/f48a5b4d-___20260827.xls"
px=pd.read_excel(XLS,dtype=str); px=px.rename(columns={px.columns[1]:"名称"})
px["代码"]=px["代码"].str.zfill(6)
pool=dict(zip(px.代码,px.名称)); indm=dict(zip(px.代码,px["一二级行业"]))
L19=["001309","688322","688041","301338","301171","688428","688392","603163",
     "688372","688525","603061","301345","688361","603119","688347","301498",
     "603193","301413","001280"]
codes=[os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
       if os.path.basename(f)[:-8]!="510300"]
cols=["close","volume","is_st","is_suspended","listed_days"]
d={c:{} for c in cols}
for c in codes:
    x=pd.read_parquet(f"{DATA}/{c}.parquet",columns=cols)
    if getattr(x.index,"tz",None) is not None: x.index=x.index.tz_localize(None)
    for k in cols: d[k][c]=x[k]
cldf=pd.DataFrame(d["close"]).sort_index(); idx=cldf.index; nt,ns=cldf.shape
def al(k,f=np.nan): return pd.DataFrame(d[k]).sort_index().reindex(index=idx,columns=cldf.columns).fillna(f)
ok=(~al("is_st",True).astype(bool).to_numpy()&~al("is_suspended",True).astype(bool).to_numpy()
    &(al("listed_days",0).to_numpy()>=250)&(al("volume",0).to_numpy()>0))
cl=cldf.where(cldf>0).to_numpy(np.float64); ok&=np.isfinite(cl)
dfc=pd.DataFrame(cl)
lo250=dfc.rolling(250,min_periods=250).min().to_numpy(); ma20=dfc.rolling(20,min_periods=20).mean().to_numpy()
with np.errstate(all="ignore"):
    rec=cl/np.where(lo250>0,lo250,np.nan)-1.0
    r120=cl/np.roll(cl,120,axis=0)-1.0; r120[:120]=np.nan
    r60=cl/np.roll(cl,60,axis=0)-1.0; r60[:60]=np.nan
ab=pd.DataFrame(cl>ma20).rolling(120,min_periods=120).mean().to_numpy()
rps60=pd.DataFrame(np.where(ok,r60,np.nan)).rank(axis=1,pct=True).to_numpy()*100
l1s,_=load_labels(); cp={c:j for j,c in enumerate(cldf.columns)}; ip=pd.Index(idx)
print(f"面板就绪 ({time.time()-t0:.0f}s)\n")
def tier(t,j):
    b=rec[t,j]>=0.40 and r120[t,j]>=0.10
    if not b: return "—"
    if ab[t,j]>=0.55 and rps60[t,j]>=90: return "强确认"
    if ab[t,j]>=0.55 and rps60[t,j]>=80: return "标准"
    return "观察"
# 选择偏差量化
print("="*92); print("一、选择偏差:这 19 只 vs 你给的 663 只池子里其余的"); print("="*92)
for ty,ds,de in [(2024,"2023-12-29","2024-12-31"),(2025,"2024-12-31","2025-12-31")]:
    t=int(ip.get_indexer([pd.Timestamp(ds)],method="ffill")[0])
    te=int(ip.get_indexer([pd.Timestamp(de)],method="ffill")[0])
    grp={"这19只":[],"池内其余":[]}
    for c in pool:
        j=cp.get(c)
        if j is None or not ok[t,j] or not np.isfinite(cl[t,j]) or not np.isfinite(cl[te,j]): continue
        r=cl[te,j]/cl[t,j]-1
        grp["这19只" if c in L19 else "池内其余"].append((r,(ty,c) in l1s))
    print(f"\n{ty} 年:")
    for k,v in grp.items():
        if not v: continue
        rr=np.array([a for a,_ in v]); hh=np.array([b for _,b in v])
        print(f"  {k:<8} n={len(v):4d}  中位涨幅 {np.median(rr):+7.1%}  "
              f"平均 {rr.mean():+7.1%}  翻倍率 {hh.mean():6.2%}  正收益 {(rr>0).mean():5.1%}")
print("\n"+"="*92); print("二、19 只逐只:三个时点的档位与实际结果"); print("="*92)
print(f"{'名称':<10}{'代码':<8}{'2024-01档':<10}{'2024实际':>9}{'2025-01档':<10}{'2025实际':>9}"
      f"{'当前档':<9}{'当前RPS60':>9}{'距低点':>8}{'120日':>8}")
rows=[]
for c in L19:
    j=cp.get(c); nm=pool.get(c,c); line=f"{nm:<10}{c:<8}"
    rec_={}
    for tag,ds,de,ty in [("2024","2023-12-29","2024-12-31",2024),("2025","2024-12-31","2025-12-31",2025)]:
        t=int(ip.get_indexer([pd.Timestamp(ds)],method="ffill")[0])
        te=int(ip.get_indexer([pd.Timestamp(de)],method="ffill")[0])
        if j is None or not ok[t,j] or not np.isfinite(rec[t,j]) or not np.isfinite(ab[t,j]):
            line+=f"{'不合格':<10}{'—':>9}"; rec_[tag]=("不合格",np.nan); continue
        tg=tier(t,j); r=cl[te,j]/cl[t,j]-1
        star="*" if (ty,c) in l1s else ""
        line+=f"{tg:<10}{r:>8.1%}{star}"; rec_[tag]=(tg,r)
    tN=int(ip.get_indexer([pd.Timestamp("2026-08-03")],method="ffill")[0])
    if j is not None and ok[tN,j] and np.isfinite(rec[tN,j]) and np.isfinite(ab[tN,j]):
        line+=f"{tier(tN,j):<9}{rps60[tN,j]:>9.0f}{rec[tN,j]:>8.0%}{r120[tN,j]:>8.0%}"
        cur=(tier(tN,j),rps60[tN,j],rec[tN,j],r120[tN,j],ab[tN,j])
    else:
        line+=f"{'不合格':<9}{'—':>9}{'—':>8}{'—':>8}"; cur=("不合格",)*5
    print(line)
    rows.append({"名称":nm,"代码":c,"行业":indm.get(c),
                 "2024档":rec_["2024"][0],"2024实际":rec_["2024"][1],
                 "2025档":rec_["2025"][0],"2025实际":rec_["2025"][1],
                 "当前档":cur[0],"当前RPS60":cur[1],"当前距低点":cur[2],
                 "当前120日":cur[3],"当前MA20持续":cur[4]})
print("\n注:实际涨幅后的 * 表示该年翻倍(普查口径)")
pd.DataFrame(rows).to_csv("/home/user/oxq-panel/list19_signal.csv",index=False,encoding="utf-8-sig")
print(f"\n落库 /home/user/oxq-panel/list19_signal.csv ({time.time()-t0:.0f}s)")
