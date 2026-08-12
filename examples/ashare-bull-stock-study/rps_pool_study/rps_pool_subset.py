"""补对照:去掉「取市值最小20只」这层,直接看过滤器对**整个子集**的影响

═══ 为什么必须补 ═══
`rps_pool_factors.py` 的组合构建是「池内筛选 → 取市值最小的 20 只」。
但**「取市值最小的 20 只」本身就是本 session 反复证明过的强因子**
(§48:再平衡收益完全集中在市值最小的那一半)。
于是八个二级因子的差异有一部分只是「它们各自剩下的股票里,最小的 20 只是谁」——
这层混淆必须拆掉。

本脚本改用第四十三节的原始口径:**整个子集等权,不做仓位截断**,
直接回答「这个过滤器有没有把池子的平均收益抬高」。
两种口径都报,差异本身就是信息。
"""
import glob, os, time
import numpy as np, pandas as pd
SP = os.path.dirname(os.path.abspath(__file__)); DATA = f"{SP}/oxq_stock_market_fixed"
COST, SEED, N_RAND = 0.003, 20260812, 300
t0 = time.time()
d = {c: {} for c in ["close","float_mv","is_st","listed_days","turnover","bp_correct",
                     "operating_cash_flow","net_income"]}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300": continue
    try: x = pd.read_parquet(f, columns=list(d))
    except Exception: continue
    if x.empty: continue
    for c in d: d[c][k] = pd.to_numeric(x[c], errors="coerce")
CL = pd.DataFrame(d["close"]).sort_index(); CL.index = CL.index.tz_localize(None)
def al(k):
    f = pd.DataFrame(d[k]).sort_index(); f.index = f.index.tz_localize(None)
    return f.reindex(index=CL.index, columns=CL.columns)
MV,ST,LD,TURN,BP = al("float_mv"),al("is_st"),al("listed_days"),al("turnover"),al("bp_correct")
OCF,NI = al("operating_cash_flow"),al("net_income")
CL = CL.where(CL>0); idx = CL.index; NT,NC = CL.shape; A = CL.to_numpy(float)
MVa,STa,LDa = MV.to_numpy(float),ST.to_numpy(float),LD.to_numpy(float)
del d
last_valid = np.full(NC,-1)
for j in range(NC):
    fv = np.flatnonzero(np.isfinite(A[:,j]))
    if fv.size: last_valid[j] = fv[-1]
RPS250 = (CL.pct_change(250).rank(axis=1,pct=True)*100).to_numpy(float)
BP_PCT = BP.rank(axis=1,pct=True).to_numpy(float)
MV_PCT = MV.rank(axis=1,pct=True).to_numpy(float)
TURN_PCT = TURN.rolling(20,min_periods=10).mean().rank(axis=1,pct=True).to_numpy(float)
RMDD_PCT = pd.DataFrame((CL/CL.rolling(20,min_periods=10).max()-1).to_numpy(float)).rank(axis=1,pct=True).to_numpy(float)
OCF_NI = (OCF/NI.where(NI>0)).to_numpy(float)
NI_TTM = pd.read_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet").reindex(index=idx,columns=CL.columns).to_numpy(float)
RV_TTM = pd.read_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet").reindex(index=idx,columns=CL.columns).to_numpy(float)
CQ = pd.read_parquet(f"{SP}/clean_growth_c_qyoy.parquet").reindex(index=idx,columns=CL.columns).to_numpy(float)
ACCEL = CQ - np.roll(CQ,63,axis=0); ACCEL[:63] = np.nan
s0 = idx.searchsorted(pd.Timestamp("2014-06-30")); eN = NT-1; CUT = 575
YRS = (idx[eN]-idx[s0]).days/365.25
_ds=[x for x in CL.resample("ME").last().index if idx[s0]<=x<=idx[eN]]
RB = np.array(sorted({idx.searchsorted(x,side="right")-1 for x in _ds}|{s0,eN}))
def seg_ret(a,b,cols):
    p0,p1 = A[a,cols],A[b,cols].copy()
    for k in np.flatnonzero(~np.isfinite(p1)):
        lv = last_valid[cols[k]]; p1[k] = A[lv,cols[k]] if lv>=a else np.nan
    r = p1/p0-1; return np.where(np.isfinite(r),r,0.0)
eq=1.0
for a,b in zip(RB[:-1],RB[1:]):
    ci=np.flatnonzero(np.isfinite(A[a])); ci=ci[np.isfinite(A[b,ci])]
    if ci.size>=5: eq*=1+float(np.nanmean(A[b,ci]/A[a,ci]-1))
EW=eq**(1/YRS)-1
print(f"锚点:全市场等权月度【§48口径】 {EW:+.2%}(应 +12.25%)"); assert abs(EW-0.1225)<0.003
print("锚点通过\n")
def pool_at(t):
    ok=(np.isfinite(A[t])&(LDa[t]>=250)&(STa[t]!=1)&np.isfinite(RPS250[t])&(RPS250[t]>90))
    return np.flatnonzero(ok)
FACTORS={
 "【池基线】不筛": lambda t,c: np.ones(c.size,bool),
 "1 双增长": lambda t,c: (NI_TTM[t,c]>0)&(RV_TTM[t,c]>0),
 "2 盈利加速 且 当季>25%": lambda t,c: (ACCEL[t,c]>0)&(CQ[t,c]>0.25),
 "3 高估值 BP最低30%": lambda t,c: BP_PCT[t,c]<=0.30,
 "4 破净 BP最高30%": lambda t,c: BP_PCT[t,c]>=0.70,
 "5 小市值 最小50%": lambda t,c: MV_PCT[t,c]<=0.50,
 "6 抗跌 rmdd20最浅50%": lambda t,c: RMDD_PCT[t,c]>=0.50,
 "7 换手不冷 分位>30%": lambda t,c: TURN_PCT[t,c]>0.30,
 "8 现金流不差 OCF/NI≥0": lambda t,c: OCF_NI[t,c]>=0,
}
def run_subset(fn, rng=None, n_pick=None, seg=None):
    """整个子集等权,**不做仓位截断**(第四十三节原始口径)。"""
    lo,hi = (s0,eN) if seg is None else seg
    rb = RB[(RB>=lo)&(RB<=hi)]; eqv=1.0; cnts=[]
    for a,b in zip(rb[:-1],rb[1:]):
        c = pool_at(a)
        if c.size<5: continue
        m = fn(a,c); m = np.where(np.isfinite(m.astype(float)),m,False).astype(bool)
        sel = c[m]
        if rng is not None:
            k = min(max(n_pick if n_pick else sel.size,1), c.size)
            sel = rng.choice(c,k,replace=False)
        if sel.size==0: continue
        cnts.append(sel.size)
        # 子集换手按 100% 计一半成本(与 §43 同,过滤器变动没那么大)
        eqv *= 1 + float(np.mean(seg_ret(a,b,sel))) - COST
    yrs=(idx[min(hi,eN)]-idx[lo]).days/365.25
    return (eqv**(1/yrs)-1 if eqv>0 and yrs>0 else -1.0, float(np.median(cnts)) if cnts else 0)
print("="*112); print("整个子集等权(无仓位截断)—— 与第四十三节同口径"); print("="*112)
print(f"{'因子':<26}{'子集只数中位':>14}{'全期年化':>12}{'2015-05前':>12}{'2015-05后':>12}{'p(全期)':>10}{'p(后段)':>10}")
res={}
for nm,fn in FACTORS.items():
    a_all,cnt = run_subset(fn); a_pre,_ = run_subset(fn,seg=(s0,CUT)); a_post,_ = run_subset(fn,seg=(CUT,eN))
    p1=p2=np.nan
    if not nm.startswith("【池基线】"):
        k=int(round(cnt)); rng=np.random.default_rng(SEED)
        full=np.array([run_subset(fn,rng=rng,n_pick=k)[0] for _ in range(N_RAND)])
        rng2=np.random.default_rng(SEED+1)
        post=np.array([run_subset(fn,rng=rng2,n_pick=k,seg=(CUT,eN))[0] for _ in range(N_RAND)])
        p1=float((full>=a_all).mean()); p2=float((post>=a_post).mean())
    res[nm]={"子集只数":cnt,"全期":a_all,"前段":a_pre,"后段":a_post,"p_全期":p1,"p_后段":p2}
    print(f"{nm:<26}{cnt:>14.0f}{a_all:>+12.2%}{a_pre:>+12.2%}{a_post:>+12.2%}{p1:>10.4f}{p2:>10.4f}   ({time.time()-t0:.0f}s)")
b=res["【池基线】不筛"]
print(f"\n  池基线:全期 {b['全期']:+.2%}、前段 {b['前段']:+.2%}、后段 {b['后段']:+.2%}")
print(f"  全市场等权基准 {EW:+.2%}")
print(f"\n  ── 相对池基线的增量(全期 / 后段)──")
for nm,v in res.items():
    if nm.startswith("【池基线】"): continue
    print(f"    {nm:<26}{(v['全期']-b['全期'])*100:>+8.2f}pp{(v['后段']-b['后段'])*100:>+10.2f}pp")
pd.DataFrame(res).T.to_csv(f"{SP}/rps_pool_subset.csv")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: rps_pool_subset.csv")
