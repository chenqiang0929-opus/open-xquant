"""检验用户的反例:贝泰妮/海尔生物 + B>104周 档的分布与年份集中度。"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0,"/home/user/open-xquant/examples/ashare-bull-stock-study/rps_pool_study")
from codex_r10_replication import DATA
from industry_neutral import build_industry
t0=time.time()
codes=[os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
       if os.path.basename(f)[:-8]!="510300"]
cols=["close","float_mv","volume","is_st","is_suspended","listed_days"]
d={c:{} for c in cols}
for c in codes:
    x=pd.read_parquet(f"{DATA}/{c}.parquet",columns=cols)
    if getattr(x.index,"tz",None) is not None: x.index=x.index.tz_localize(None)
    for k in cols: d[k][c]=x[k]
cldf=pd.DataFrame(d["close"]).sort_index(); idx=cldf.index; nt,ns=cldf.shape
def al(k,f=np.nan): return pd.DataFrame(d[k]).sort_index().reindex(index=idx,columns=cldf.columns).fillna(f)
ok=(~al("is_st",True).astype(bool).to_numpy() & ~al("is_suspended",True).astype(bool).to_numpy()
    & (al("listed_days",0).to_numpy()>=250) & (al("volume",0).to_numpy()>0))
cl=cldf.where(cldf>0).ffill().to_numpy(np.float64); ok&=np.isfinite(cl)
wk=pd.Series(np.arange(nt),index=idx).groupby([idx.isocalendar().year,idx.isocalendar().week]).last()
wpos=np.sort(wk.to_numpy()); wdf=pd.DataFrame(cl[wpos])
m20=wdf.rolling(20,min_periods=20).mean().to_numpy(); m60=wdf.rolling(60,min_periods=60).mean().to_numpy()
bw=m20>m60; fw=np.isfinite(m20)&np.isfinite(m60); nw=len(wpos)
dur=np.zeros((nw,ns),np.int32)
for i in range(1,nw):
    s=fw[i]&fw[i-1]&(bw[i]==bw[i-1]); dur[i]=np.where(s,dur[i-1]+1,1)
dur=np.where(fw,dur,0)
src=np.searchsorted(wpos,np.arange(nt),side="right")-1; vs=src>=0
sa=np.zeros((nt,ns),bool); sd=np.zeros((nt,ns),np.int32); sf=np.zeros((nt,ns),bool)
sa[vs]=bw[src[vs]]; sd[vs]=dur[src[vs]]; sf[vs]=fw[src[vs]]
me=pd.Series(np.arange(nt),index=idx).groupby([idx.year,idx.month]).last().to_numpy()
HOR=250
cp={c:j for j,c in enumerate(cldf.columns)}
print(f"面板就绪 ({time.time()-t0:.0f}s)\n")
print("="*84); print("一、用户的两只反例,逐月末实测(周线口径)"); print("="*84)
for code,nm in [("300957","贝泰妮"),("688139","海尔生物")]:
    j=cp[code]; rows=[]
    for t in me:
        t=int(t)
        if t<60 or t>nt-HOR-1: continue
        if not (ok[t,j] and sf[t,j] and sd[t,j]>0): continue
        fr=cl[t+HOR,j]/cl[t,j]-1
        rows.append((idx[t].date(),"A" if sa[t,j] else "B",int(sd[t,j]),fr))
    r=pd.DataFrame(rows,columns=["日期","排列","持续周","未来250日"])
    b=r[r.排列=="B"]
    print(f"\n{nm} {code}:可测月末 {len(r)},其中空头排列 {len(b)}")
    if len(b):
        print(f"  空头排列买入持有250日:胜率 {(b['未来250日']>0).mean():.1%} "
              f"中位 {b['未来250日'].median():+.1%} 最差 {b['未来250日'].min():+.1%}")
        for lo,hi,tag in [(1,52,"<52周"),(52,104,"52-104周"),(104,10**6,">104周")]:
            g=b[(b.持续周>=lo)&(b.持续周<hi)]
            if len(g): print(f"    {tag:<10} n={len(g):3d} 胜率 {(g['未来250日']>0).mean():6.1%} "
                             f"中位 {g['未来250日'].median():+7.1%}")
    print(f"  最后 5 个月末:{r.tail(5)[['日期','排列','持续周','未来250日']].to_string(index=False)}")
# B>104周 全样本分布与年份
rows=[]
for t in me:
    t=int(t)
    if t<60 or t>nt-HOR-1: continue
    e=np.flatnonzero(ok[t]&sf[t]&(sd[t]>0))
    for j in e:
        rows.append((t,idx[t].year,bool(sa[t,j]),int(sd[t,j]),cl[t+HOR,j]/cl[t,j]-1))
p=pd.DataFrame(rows,columns=["t","year","A","dur","fr"])
b=p[(~p.A)&(p.dur>104)&np.isfinite(p.fr)]
print("\n"+"="*84); print(f"二、B >104周 档:n={len(b):,} 的收益分位数"); print("="*84)
for q in (0.05,0.10,0.25,0.50,0.75,0.90,0.95):
    print(f"  {q:.0%} 分位 {b.fr.quantile(q):+8.1%}")
print(f"  亏损占比 {(b.fr<=0).mean():.1%}   跌超 30% 占比 {(b.fr<-0.30).mean():.1%}")
print("\n"+"="*84); print("三、B >104周 档的年份分布(这是关键)"); print("="*84)
g=b.groupby("year").agg(n=("fr","size"),胜率=("fr",lambda s:(s>0).mean()),中位=("fr","median"))
g["占比"]=g.n/len(b)
print(g.assign(胜率=lambda x:(x.胜率*100).round(1),中位=lambda x:(x.中位*100).round(1),
               占比=lambda x:(x.占比*100).round(1)).to_string())
