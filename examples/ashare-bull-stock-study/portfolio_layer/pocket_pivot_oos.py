"""赢家变体的分期检验:+10.52% 是不是只是 2015 的产物?

§55 已证明:七种买点在 2015-05 之后**全部**塌到 +0.4~+1.7%/笔。
所以「口袋支点 ∩ 距60日高点5%内」通过三条判据之后,**必须再问这一句**,
否则等于拿一个只在 2013-2015 有效的东西当发现。

分两段各自跑组合级,并各自做 300 次随机对照(随机子集也限定在同一段内,
否则对照组含有另一段的收益,比较不公平)。
"""
import glob, os, time
import numpy as np, pandas as pd
SP = os.path.dirname(os.path.abspath(__file__)); DATA = f"{SP}/oxq_stock_market_fixed"
COST_PF, SLOTS, SEED, N_RAND = 0.003, 10, 20260810, 300
t0 = time.time()
o,h,l,c,mv,vo = {},{},{},{},{},{}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300": continue
    x = pd.read_parquet(f, columns=["open","high","low","close","float_mv","volume"])
    if x.empty: continue
    o[k]=pd.to_numeric(x["open"],errors="coerce"); h[k]=pd.to_numeric(x["high"],errors="coerce")
    l[k]=pd.to_numeric(x["low"],errors="coerce");  c[k]=pd.to_numeric(x["close"],errors="coerce")
    mv[k]=pd.to_numeric(x["float_mv"],errors="coerce"); vo[k]=pd.to_numeric(x["volume"],errors="coerce")
OP=pd.DataFrame(o).sort_index(); OP.index=OP.index.tz_localize(None)
HI=pd.DataFrame(h).set_axis(OP.index); LO=pd.DataFrame(l).set_axis(OP.index)
CL=pd.DataFrame(c).set_axis(OP.index); MV=pd.DataFrame(mv).set_axis(OP.index)
VO=pd.DataFrame(vo).set_axis(OP.index)
OP=OP.where(OP>0);HI=HI.where(HI>0);LO=LO.where(LO>0);CL=CL.where(CL>0)
MA50=CL.rolling(50,min_periods=50).mean(); idx=OP.index; NT=len(idx)
OPa,LOa,CLa,MVa=OP.to_numpy(float),LO.to_numpy(float),CL.to_numpy(float),MV.to_numpy(float)
col_of={cd:i for i,cd in enumerate(OP.columns)}
_m=pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet",columns=["close"])["close"],errors="coerce")
_m.index=_m.index.tz_localize(None); mkt=_m.reindex(idx).ffill()
mkt_ok=(mkt>mkt.rolling(200,min_periods=200).mean()).to_numpy()
_rmax60=CL.rolling(60,min_periods=60).max(); _rmin60=CL.rolling(60,min_periods=60).min()
BASE_OK=(((_rmax60-_rmin60)/_rmin60.replace(0,np.nan)).shift(1)<0.50).to_numpy()
BRK60=(CLa>_rmax60.shift(1).to_numpy())&BASE_OK
NEAR60=(CLa>=_rmax60.shift(1).to_numpy()*0.95)&BASE_OK
pc=CL.shift(1); dnv=VO.where(CL<pc,0.0)
PP=((CL>pc)&(VO>dnv.rolling(10,min_periods=5).max().shift(1))&(CL>(HI+LO)/2)
    &(CL>MA50)&(MA50>MA50.shift(10))&((CL/MA50-1)<=0.10)).to_numpy()
LAST_OK=NT-1-252
def to_events(hit,gap=60):
    cs,ds=[],[]
    for j,cd in enumerate(OP.columns):
        last=-10**9
        for q in np.flatnonzero(hit[:,j]):
            if q-last<gap or q==0 or q>LAST_OK: continue
            last=q; cs.append(cd); ds.append(int(q))
    return pd.DataFrame({"code":cs,"dp":ds})
def run_pf(evs,t_lo,t_hi):
    by_day={d:g["code"].tolist() for d,g in evs.groupby("dp")}
    cash,holds=1.0,{}; equity=np.zeros(NT)
    for t in range(t_lo,t_hi+1):
        for code in list(holds):
            hd=holds[code]; ci=hd["ci"]
            op_t,lo_t,cl_t=OPa[t,ci],LOa[t,ci],CLa[t,ci]; ex=None
            if not np.isfinite(cl_t): ex=hd["last"]
            else:
                hd["last"]=cl_t
                if np.isfinite(lo_t) and lo_t<=hd["stop_px"]:
                    ex=op_t if (np.isfinite(op_t) and op_t<hd["stop_px"]) else hd["stop_px"]
                elif t-hd["t_in"]>=252: ex=cl_t
            if ex is not None and np.isfinite(ex) and ex>0:
                cash+=hd["shares"]*ex*(1-COST_PF); del holds[code]
        cands=by_day.get(t-1,[]); free=SLOTS-len(holds)
        if cands and free>0 and mkt_ok[t]:
            cands=[cd for cd in cands if cd not in holds and np.isfinite(OPa[t,col_of[cd]]) and OPa[t,col_of[cd]]>0]
            cands.sort(key=lambda cd: MVa[t,col_of[cd]] if np.isfinite(MVa[t,col_of[cd]]) else np.inf)
            for cd in cands[:free]:
                alloc=cash/(SLOTS-len(holds)) if SLOTS>len(holds) else 0
                if alloc<=0: break
                px=OPa[t,col_of[cd]]
                holds[cd]={"entry":px,"t_in":t,"last":px,"ci":col_of[cd],
                           "stop_px":px*(1-0.10),"shares":alloc*(1-COST_PF)/px}
                cash-=alloc
        equity[t]=cash+sum(hd["shares"]*(CLa[t,hd["ci"]] if np.isfinite(CLa[t,hd["ci"]]) else hd["last"]) for hd in holds.values())
    eq=pd.Series(equity[t_lo:t_hi+1],index=idx[t_lo:t_hi+1])
    eq=eq[eq>0]
    if len(eq)<100: return np.nan
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    return (eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1
BASE=to_events(BRK60); WIN=to_events(PP&NEAR60); CUT=575
print(f"锚点 全期:60日新高 {len(BASE):,} 笔 → {run_pf(BASE,200,NT-1):+.2%}(应 +6.34%)")
print(f"赢家变体 口袋支点∩距60日高点5%内 {len(WIN):,} 笔\n")
SEGS={"2015-05前":(200,CUT),"2015-05后":(CUT,NT-1)}
rng=np.random.default_rng(SEED)
for sn,(a,b) in SEGS.items():
    we=WIN[(WIN.dp>=a)&(WIN.dp<=b)]; be=BASE[(BASE.dp>=a)&(BASE.dp<=b)]
    real=run_pf(we,a,b); base_a=run_pf(be,a,b)
    anns=np.array([run_pf(be.iloc[rng.choice(len(be),min(len(we),len(be)),replace=False)],a,b)
                   for _ in range(N_RAND)])
    anns=anns[np.isfinite(anns)]
    p=float((anns>=real).mean())
    print(f"【{sn}】 赢家变体 {len(we):,} 笔 → **{real:+.2%}**   同期 60日新高全集 {base_a:+.2%}")
    print(f"    {len(anns)} 次随机对照:中位 {np.median(anns):+.2%}  "
          f"95%区间 [{np.quantile(anns,.025):+.2%}, {np.quantile(anns,.975):+.2%}]  **p={p:.4f}**")
    print(f"    → {'**该段成立**' if p<0.0125 else '**该段不成立**'}  ({time.time()-t0:.0f}s)\n")
