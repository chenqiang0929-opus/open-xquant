"""步骤5:新口径(当年横截面分位)搬到两个大池 —— 第六十节那次失败的重做

═══ 为什么重做 ═══
第六十节用**绝对阈值**把三个特征搬到大池,失败(净期望转负、缩量方向反转)。
诊断说那可能是「固定阈值随时间变严」造成的。现在改成**当年横截面分位**再试一次。
定义与第六十节完全一致(调整期起点 = 事件日前 250 日最高价那一根),
**只把阈值口径从绝对值换成当年分位**,其余一行不动。

═══ 判据(计划里写死) ═══
两个大池**至少一个**的「三条全中」净期望 **不为负**。
另报各单条,与第六十节逐格对照。
**锚点**:60日新高突破池 70,310 笔 / 组合 +6.34%。
"""
import glob, os, time
import numpy as np, pandas as pd
SP=os.path.dirname(os.path.abspath(__file__)); DATA=f"{SP}/oxq_stock_market_fixed"
COST_TRADE,COST_PF,SLOTS,SEED,N_RAND=0.003,0.003,10,20260812,300
Q_KEEP=0.40; PEAK_WIN=250; SPLIT="2020-01-01"
t0=time.time()
o,h,l,c,mv,vo={},{},{},{},{},{}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k=os.path.basename(f)[:-8]
    if k=="510300": continue
    x=pd.read_parquet(f,columns=["open","high","low","close","float_mv","volume"])
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
OPa,HIa,LOa,CLa=OP.to_numpy(float),HI.to_numpy(float),LO.to_numpy(float),CL.to_numpy(float)
MVa,VOa=MV.to_numpy(float),VO.to_numpy(float)
TRa=np.maximum(HIa-LOa,np.maximum(np.abs(HIa-np.roll(CLa,1,0)),np.abs(LOa-np.roll(CLa,1,0))))
codes=list(OP.columns); col_of={cd:i for i,cd in enumerate(codes)}
_m=pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet",columns=["close"])["close"],errors="coerce")
_m.index=_m.index.tz_localize(None); mkt=_m.reindex(idx).ffill()
mkt_ok=(mkt>mkt.rolling(200,min_periods=200).mean()).to_numpy()
LAST_OK=NT-1-252
_rmax60=CL.rolling(60,min_periods=60).max(); _rmin60=CL.rolling(60,min_periods=60).min()
BASE_OK=(((_rmax60-_rmin60)/_rmin60.replace(0,np.nan)).shift(1)<0.50).to_numpy()
BRK60=(CLa>_rmax60.shift(1).to_numpy())&BASE_OK
pc=CL.shift(1); dnv=VO.where(CL<pc,0.0)
PP=((CL>pc)&(VO>dnv.rolling(10,min_periods=5).max().shift(1))&(CL>(HI+LO)/2)
    &(CL>MA50)&(MA50>MA50.shift(10))&((CL/MA50-1)<=0.10)).to_numpy()
def to_events(hit,gap=60):
    cs,ds=[],[]
    for j,cd in enumerate(codes):
        last=-10**9
        for q in np.flatnonzero(hit[:,j]):
            if q-last<gap or q==0 or q>LAST_OK: continue
            last=q; cs.append(cd); ds.append(int(q))
    return pd.DataFrame({"code":cs,"dp":ds})
def add_metrics(ev):
    dep,shr,conv=[],[],[]
    for cd,dp in zip(ev.code.to_numpy(),ev.dp.to_numpy()):
        j=col_of[cd]; t=int(dp); lo_i=max(t-PEAK_WIN,0)
        seg_h=HIa[lo_i:t,j]
        if seg_h.size<40 or np.all(~np.isfinite(seg_h)):
            dep.append(np.nan);shr.append(np.nan);conv.append(np.nan); continue
        pk=lo_i+int(np.nanargmax(seg_h)); hi_pk=HIa[pk,j]
        lows=LOa[pk:t,j]; lows=lows[np.isfinite(lows)]
        dep.append(1-lows.min()/hi_pk if lows.size and np.isfinite(hi_pk) and hi_pk>0 else np.nan)
        va=VOa[pk:t,j]; va=va[np.isfinite(va)]; vp=VOa[max(pk-60,0):pk,j]; vp=vp[np.isfinite(vp)]
        shr.append(va.mean()/vp.mean() if va.size and vp.size and vp.mean()>0 else np.nan)
        tn=TRa[max(t-20,0):t,j]; tn=tn[np.isfinite(tn)]; tp=TRa[max(pk-60,0):pk,j]; tp=tp[np.isfinite(tp)]
        conv.append(tn.mean()/tp.mean() if tn.size and tp.size and tp.mean()>0 else np.nan)
    ev=ev.copy(); ev["深度"],ev["缩量比"],ev["收敛比"]=dep,shr,conv
    ev["date"]=idx[ev.dp.to_numpy()]; ev["year"]=ev["date"].dt.year
    for col in ("深度","缩量比","收敛比"):
        ev[col+"✓"]=ev[col]<=ev.groupby("year")[col].transform(lambda s:s.quantile(Q_KEEP))
    ev["满足条数"]=ev[["深度✓","缩量比✓","收敛比✓"]].sum(axis=1)
    return ev
def run_pf(evs,lo,hi,):
    by_day={}
    for cd,dp in zip(evs.code.to_numpy(),evs.dp.to_numpy()): by_day.setdefault(int(dp),[]).append(cd)
    cash,holds=1.0,{}; equity=np.zeros(NT)
    for t in range(lo,hi+1):
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
                           "stop_px":px*0.90,"shares":alloc*(1-COST_PF)/px}
                cash-=alloc
        equity[t]=cash+sum(hd["shares"]*(CLa[t,hd["ci"]] if np.isfinite(CLa[t,hd["ci"]]) else hd["last"]) for hd in holds.values())
    eq=pd.Series(equity[lo:hi+1],index=idx[lo:hi+1]); eq=eq[eq>0]
    if len(eq)<100: return np.nan,np.nan
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    return (eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1, float((eq/eq.cummax()-1).min())
def trade_ret(evs):
    out=[]
    for code,grp in evs.groupby("code",sort=False):
        ci=col_of[code]; op,ll,cc=OPa[:,ci],LOa[:,ci],CLa[:,ci]
        for dp in grp["dp"].to_numpy():
            e=int(dp)+1; entry=op[e] if e<NT else np.nan
            if not np.isfinite(entry) or entry<=0: continue
            stop,last,ex=entry*0.90,entry,None
            end=min(e+252,NT-1)
            for t in range(e,end+1):
                if not np.isfinite(cc[t]): continue
                last=cc[t]
                if np.isfinite(ll[t]) and ll[t]<=stop:
                    ex=op[t] if (np.isfinite(op[t]) and op[t]<stop) else stop; break
            if ex is None: ex=cc[end] if np.isfinite(cc[end]) else last
            if np.isfinite(ex) and ex>0: out.append(ex/entry-1)
    return np.array(out)
POOLS={"60日新高突破":to_events(BRK60),"口袋支点":to_events(PP)}
_a,_=run_pf(POOLS["60日新高突破"],200,NT-1)
print(f"锚点:{len(POOLS['60日新高突破']):,} 笔(应 70,310)、组合全期 {_a:+.2%}(应 +6.34%)")
assert abs(len(POOLS["60日新高突破"])-70310)<=50 and abs(_a-0.0634)<0.004
print("锚点通过")
S0=idx.searchsorted(pd.Timestamp(SPLIT)); ALPHA=0.05/4; rows=[]
for pname,ev in POOLS.items():
    ev=add_metrics(ev); OUT=ev[ev.date>=SPLIT].reset_index(drop=True)
    print(f"\n{'='*110}\n池:{pname}   OOS {len(OUT):,} 笔   阈值=当年横截面最优 {Q_KEEP:.0%}\n{'='*110}")
    F={"【基线】不筛":np.ones(len(OUT),bool),
       "深度 最浅40%":OUT["深度✓"].to_numpy(),
       "缩量比 最缩40%":OUT["缩量比✓"].to_numpy(),
       "收敛比 最收敛40%":OUT["收敛比✓"].to_numpy(),
       "**三条全中**":(OUT.满足条数==3).to_numpy()}
    print(f"{'配置':<20}{'事件数':>9}{'选中率':>8}{'胜率':>9}{'净期望':>10}{'年化':>10}{'最大回撤':>10}")
    res={}
    for nm,m in F.items():
        s=OUT[m]
        if len(s)<30: continue
        r=trade_ret(s); a,dd=run_pf(s,S0,NT-1)
        res[nm]={"池":pname,"配置":nm,"事件数":len(s),"选中率":m.mean(),"胜率":(r>0).mean(),
                 "净期望":r.mean()-COST_TRADE,"年化":a,"回撤":dd}
        v=res[nm]
        print(f"{nm:<20}{len(s):>9,}{m.mean():>8.1%}{v['胜率']:>9.2%}{v['净期望']:>+10.2%}{a:>+10.2%}{dd:>10.1%}   ({time.time()-t0:.0f}s)")
    for nm,m in F.items():
        if nm.startswith("【基线】") or nm not in res: continue
        k=res[nm]["事件数"]; rg=np.random.default_rng(SEED)
        anns=np.array([run_pf(OUT.iloc[rg.choice(len(OUT),k,replace=False)],S0,NT-1)[0] for _ in range(N_RAND)])
        anns=anns[np.isfinite(anns)]; p=float((anns>=res[nm]["年化"]).mean()); res[nm]["p"]=p
        q=np.quantile(anns,[0.025,0.975])
        print(f"  随机对照 {nm:<20} 实际 {res[nm]['年化']:+.2%}  中位 {np.median(anns):+.2%}"
              f"  [{q[0]:+.2%}, {q[1]:+.2%}]  **p={p:.4f}**   ({time.time()-t0:.0f}s)")
    rows.extend(res.values())
R=pd.DataFrame(rows); R.to_csv(f"{SP}/adaptive_transfer.csv",index=False)
print(f"\n{'='*110}\n判定:两个大池至少一个的「三条全中」净期望不为负\n{'='*110}")
for pname in POOLS:
    t=R[(R.池==pname)&(R.配置=="**三条全中**")]
    if len(t): print(f"  {pname}: 净期望 **{t.净期望.iloc[0]:+.2%}**  年化 {t.年化.iloc[0]:+.2%}"
                     f"  p={t.p.iloc[0]:.4f}   {'✓ 不为负' if t.净期望.iloc[0]>=0 else '**✗ 为负**'}")
print(f"\n  对照第六十节(绝对阈值):60日新高 -0.02%、口袋支点 -0.43%")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: adaptive_transfer.csv")
