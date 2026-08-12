"""OOS 验证:2014-2019 选出的三个调整期特征,拿到 2020-2026 上验

═══ 选择集给出的结论(不许再改) ═══
三条纪律(自身零分布 p<0.05、超公平天花板 1.18、两段方向一致)全过的只有 3 个,
**全部来自第二段调整期**:
  ② 波动收缩 <0.8×      lift 1.37   P(交易赚钱|特征) 24.77%
  ② 调整期缩量 <0.8×    lift 1.33   24.06%
  ② 调整深度 浅50%      lift 1.21   21.95%
(基准 18.08%)

第一段的特征方向**反了**:强势期涨幅高 0.84、换手高 0.84、涨停≥3次 0.77、放量>1.5× 0.85
第三段的:买点日量比≥1.5 → 0.85、市值小50% → 0.84

═══ 验证(事前写死,不调参) ═══
把这三个特征原样搬到 **2020-01-01 之后的 9,248 笔事件**上:
  ① 组合级年化 ≥ **+7.22%**
  ② **300 次**同选中率随机对照(从同期事件里随机抽),p < 0.05/4 = **0.0125**
**阈值 0.8× / 中位数 全部用选择集定的值,验证集上一个都不动。**
"""
import glob, os, time
import numpy as np, pandas as pd
SP = os.path.dirname(os.path.abspath(__file__)); DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF, SLOTS, SEED, N_RAND = 0.003, 0.003, 10, 20260812, 300
t0 = time.time()
o,h,l,c,mv = {},{},{},{},{}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300": continue
    x = pd.read_parquet(f, columns=["open","high","low","close","float_mv"])
    if x.empty: continue
    o[k]=pd.to_numeric(x["open"],errors="coerce"); h[k]=pd.to_numeric(x["high"],errors="coerce")
    l[k]=pd.to_numeric(x["low"],errors="coerce");  c[k]=pd.to_numeric(x["close"],errors="coerce")
    mv[k]=pd.to_numeric(x["float_mv"],errors="coerce")
OP=pd.DataFrame(o).sort_index(); OP.index=OP.index.tz_localize(None)
HI=pd.DataFrame(h).set_axis(OP.index); LO=pd.DataFrame(l).set_axis(OP.index)
CL=pd.DataFrame(c).set_axis(OP.index); MV=pd.DataFrame(mv).set_axis(OP.index)
OP=OP.where(OP>0);HI=HI.where(HI>0);LO=LO.where(LO>0);CL=CL.where(CL>0)
idx=OP.index; NT=len(idx)
OPa,LOa,CLa,MVa=OP.to_numpy(float),LO.to_numpy(float),CL.to_numpy(float),MV.to_numpy(float)
col_of={cd:i for i,cd in enumerate(OP.columns)}
_m=pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet",columns=["close"])["close"],errors="coerce")
_m.index=_m.index.tz_localize(None); mkt=_m.reindex(idx).ffill()
mkt_ok=(mkt>mkt.rolling(200,min_periods=200).mean()).to_numpy()
P=pd.read_parquet(f"{SP}/seq_feature_panel.parquet")
IN=P[P.date<"2020-01-01"]; OUT=P[P.date>="2020-01-01"].reset_index(drop=True)
# **阈值全部取自选择集**,验证集不重算
THR_DEPTH=float(IN.D_深度.median())
print(f"选择集定的阈值:调整深度中位 {THR_DEPTH:.3f}(验证集上不重算)")
FILTERS={
 "【基线】全部OOS事件": np.ones(len(OUT),bool),
 "② 波动收缩 <0.8×": (OUT.D_波动收缩<0.8).fillna(False).to_numpy(),
 "② 调整期缩量 <0.8×": (OUT.D_缩量比<0.8).fillna(False).to_numpy(),
 "② 调整深度 ≤选择集中位": (OUT.D_深度<=THR_DEPTH).fillna(False).to_numpy(),
 "**三条全中**": ((OUT.D_波动收缩<0.8)&(OUT.D_缩量比<0.8)&(OUT.D_深度<=THR_DEPTH)).fillna(False).to_numpy(),
}
def run_pf(evs, lo=None, hi=None):
    lo = idx.searchsorted(pd.Timestamp("2020-01-01")) if lo is None else lo
    hi = NT-1 if hi is None else hi
    by_day={}
    for _,r in evs.iterrows(): by_day.setdefault(int(r.dp),[]).append(r.code)
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
print(f"\n{'='*112}\nOOS 验证 2020-01 ~ 2026-08(选择时完全没看过)\n{'='*112}")
print(f"{'配置':<26}{'事件数':>9}{'选中率':>8}{'交易胜率':>10}{'净期望':>10}{'年化':>10}{'最大回撤':>10}")
res={}
for nm,m in FILTERS.items():
    sub=OUT[m]
    if len(sub)<30:
        print(f"{nm:<26}{len(sub):>9,}   样本不足"); continue
    a,dd=run_pf(sub)
    res[nm]={"事件数":len(sub),"选中率":m.mean(),"胜率":(sub.trade>0).mean(),
             "净期望":sub.trade.mean()-COST_TRADE,"年化":a,"回撤":dd}
    v=res[nm]
    print(f"{nm:<26}{len(sub):>9,}{m.mean():>8.1%}{v['胜率']:>10.2%}{v['净期望']:>+10.2%}{a:>+10.2%}{dd:>10.1%}   ({time.time()-t0:.0f}s)")
print(f"\n{'='*112}\n随机对照 × {N_RAND}(从同期 OOS 事件里随机抽同样多)\n{'='*112}")
ALPHA=0.05/4
print(f"  Bonferroni:4 个过滤器 → 需 **p < {ALPHA:.4f}**\n")
for nm,m in FILTERS.items():
    if nm.startswith("【基线】") or nm not in res: continue
    k=res[nm]["事件数"]; rng=np.random.default_rng(SEED)
    anns=np.array([run_pf(OUT.iloc[rng.choice(len(OUT),k,replace=False)])[0] for _ in range(N_RAND)])
    anns=anns[np.isfinite(anns)]
    p=float((anns>=res[nm]["年化"]).mean()); res[nm]["p"]=p
    q=np.quantile(anns,[0.025,0.975])
    print(f"  {nm}(抽 {k:,} 笔)  实际 **{res[nm]['年化']:+.2%}**  随机中位 {np.median(anns):+.2%}"
          f"  [{q[0]:+.2%}, {q[1]:+.2%}]  **p={p:.4f}**   ({time.time()-t0:.0f}s)")
print(f"\n{'='*112}\n判定(事前判据,验证集上未调任何参数)\n{'='*112}")
for nm,v in res.items():
    if nm.startswith("【基线】"): continue
    c1=v["年化"]>=0.0722; c2=v.get("p",1)<ALPHA
    print(f"  {nm}: ① 年化 {v['年化']:+.2%} {'✓' if c1 else '✗'}   ② p={v.get('p',np.nan):.4f} {'✓' if c2 else '✗'}"
          f"   **{'算发现' if (c1 and c2) else '不算发现'}**")
b=res["【基线】全部OOS事件"]
print(f"\n  对照:OOS 基线(不筛)胜率 {b['胜率']:.2%}、净期望 {b['净期望']:+.2%}、年化 {b['年化']:+.2%}")
print("\n  ── 胜率有没有被提上去(选择集 vs 验证集)──")
for nm,m in FILTERS.items():
    if nm not in res: continue
    mi = (IN.D_波动收缩<0.8) if "波动" in nm else (IN.D_缩量比<0.8) if "缩量" in nm else \
         (IN.D_深度<=THR_DEPTH) if "深度" in nm else \
         ((IN.D_波动收缩<0.8)&(IN.D_缩量比<0.8)&(IN.D_深度<=THR_DEPTH)) if "三条" in nm else np.ones(len(IN),bool)
    mi=pd.Series(mi).fillna(False).to_numpy()
    print(f"    {nm:<26} 选择集 {(IN[mi].trade>0).mean():.2%}  →  验证集 {res[nm]['胜率']:.2%}")
pd.DataFrame(res).T.to_csv(f"{SP}/seq_feature_oos.csv")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: seq_feature_oos.csv")
