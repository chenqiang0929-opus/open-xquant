"""离场规则组合级结果的**随机种子离散度** —— 判断 E随机选 +10.12% 是不是单次抽样运气

═══ 为什么必须做这一步 ═══
主表出现结论相反的一对:
  E 条件+10周线 / 小市值优先  年化 +1.02%
  E 条件+10周线 / 随机选      年化 **+10.12%**(全表最高)
"随机选"只跑了 1 个种子。10个空位、每年 20-50 笔,单笔大赢家能决定整条净值曲线,
**单种子结果不可采信**。本脚本对每条规则跑 20 个种子,报告年化的分布。

判据:若基线A 与某规则的种子分布**重叠严重**(如 A 的 75% 分位高于该规则的中位数),
则该规则的"改进"落在噪音内,不能算数。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
N_SEED = 20
SLOTS, COST = 10, 0.003
INF = float("inf")

# 停牌/退市持仓的平仓折价。诊断脚本 diag_ruleA_vs_stage1.py 实测:
# 交易级 607 笔"到期日收盘为NaN"的持仓,按最后有效成交价计隐含平均收益 **+103.0%**,
# 使净期望从 +3.75% 抬到 +4.61%(+0.86pp)。**按最后成交价平仓是乐观处理**
# ——停牌期间根本卖不掉,长停复牌常大幅低开。用 HAIRCUT 检验结论对该假设的敏感度。
HAIRCUT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
TAG = "" if HAIRCUT == 1.0 else f"_haircut{int(HAIRCUT*100)}"

RULES = {
    "A": dict(name="基线:-10%固定,无止盈,252日", stop=0.10, ma_mode="none",
              trail=None, arm=None, max_hold=252),
    "B": dict(name="纯10周线", stop=None, ma_mode="pure",
              trail=None, arm=None, max_hold=252),
    "C": dict(name="固定+10周线接管", stop=0.10, ma_mode="takeover",
              trail=None, arm=None, max_hold=252),
    "D": dict(name="条件移动止盈(涨100%后)", stop=0.10, ma_mode="none",
              trail=0.20, arm=1.00, max_hold=252),
    "E": dict(name="条件+10周线(涨100%后)", stop=0.10, ma_mode="arm100",
              trail=None, arm=1.00, max_hold=252),
    "F": dict(name="基线+504日", stop=0.10, ma_mode="none",
              trail=None, arm=None, max_hold=504),
}

t0 = time.time()
o, h, l, c, mv = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close", "float_mv"])
    if x.empty:
        continue
    o[k] = pd.to_numeric(x["open"], errors="coerce"); h[k] = pd.to_numeric(x["high"], errors="coerce")
    l[k] = pd.to_numeric(x["low"], errors="coerce"); c[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
OP = pd.DataFrame(o).sort_index(); OP.index = OP.index.tz_localize(None)
HI = pd.DataFrame(h).set_axis(OP.index); LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index); MV = pd.DataFrame(mv).set_axis(OP.index)
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
MA50 = CL.rolling(50, min_periods=50).mean()
idx = OP.index; NT = len(idx); pos = {d: i for i, d in enumerate(idx)}
OPa, HIa, LOa = OP.to_numpy(), HI.to_numpy(), LO.to_numpy()
CLa, MVa, MAa = CL.to_numpy(), MV.to_numpy(), MA50.to_numpy()
col_of = {cd: i for i, cd in enumerate(OP.columns)}
del o, h, l, c, mv
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv", usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev = ev[ev.code.isin(OP.columns)].copy()
ev["dp"] = ev["D"].map(pos); ev = ev.dropna(subset=["dp"]); ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < NT - 5]
by_day = {d: g["code"].tolist() for d, g in ev.groupby("dp")}

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()


def step(rc, hd, op_t, hi_t, lo_t, cl_t, ma_t):
    stop_f, ma_mode, trail, arm = rc["stop"], rc["ma_mode"], rc["trail"], rc["arm"]
    taken_over = ma_mode in ("takeover", "arm100") and hd["armed_ma"]
    if stop_f is not None and not taken_over and np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
        return (op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"])
    if np.isfinite(hi_t) and hi_t > hd["peak"]:
        hd["peak"] = hi_t
    if arm is not None and hd["peak"] >= hd["entry"] * (1 + arm):
        if trail is not None:
            hd["armed"] = True
        if ma_mode == "arm100":
            hd["armed_ma"] = True
    if trail is not None and hd["armed"]:
        tp = hd["peak"] * (1 - trail)
        if np.isfinite(lo_t) and lo_t <= tp:
            return op_t if (np.isfinite(op_t) and op_t < tp) else tp
    if ma_mode != "none" and np.isfinite(ma_t) and np.isfinite(cl_t):
        if ma_mode == "takeover" and not hd["armed_ma"] and cl_t > ma_t:
            hd["armed_ma"] = True
        elif hd["armed_ma"] and cl_t < ma_t:
            hd["pending"] = True
    return None


def run(rc, pick, seed, keep=1.0):
    """keep<1:每次随机保留该比例的突破事件 —— 给'小市值优先'这条确定性路径加误差棒。"""
    rng = np.random.default_rng(seed)
    day_pool = by_day
    if keep < 1.0:
        r2 = np.random.default_rng(seed + 777)
        day_pool = {d: [cd for cd in v if r2.random() < keep] for d, v in by_day.items()}
    cash, holds, equity = 1.0, {}, np.zeros(NT)
    mh, start = rc["max_hold"], 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]; ci = hd["ci"]
            op_t, hi_t, lo_t, cl_t, ma_t = OPa[t, ci], HIa[t, ci], LOa[t, ci], CLa[t, ci], MAa[t, ci]
            if hd["pending"]:
                ex = op_t if np.isfinite(op_t) else (cl_t if np.isfinite(cl_t) else hd["last"])
            elif not np.isfinite(cl_t):
                ex = hd["last"] * HAIRCUT      # 停牌/退市:最后有效价 × 折价系数
            else:
                hd["last"] = cl_t
                ex = step(rc, hd, op_t, hi_t, lo_t, cl_t, ma_t)
                if ex is None and t - hd["t_in"] >= mh:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST)
                del holds[code]
        cands = day_pool.get(t - 1, [])
        free = SLOTS - len(holds)
        if cands and free > 0 and mkt_ok[t]:
            cands = [cd for cd in cands if cd not in holds
                     and np.isfinite(OPa[t, col_of[cd]]) and OPa[t, col_of[cd]] > 0]
            if cands:
                if pick == "small":
                    # 同一 seed 下对"小市值优先"只影响并列打散,故仍先按市值排序
                    cands.sort(key=lambda cd: MVa[t, col_of[cd]]
                               if np.isfinite(MVa[t, col_of[cd]]) else np.inf)
                else:
                    rng.shuffle(cands)
                for cd in cands[:free]:
                    alloc = cash / (SLOTS - len(holds)) if SLOTS > len(holds) else 0
                    if alloc <= 0:
                        break
                    px = OPa[t, col_of[cd]]
                    holds[cd] = {"entry": px, "peak": px, "t_in": t, "last": px,
                                 "stop_px": px * (1 - rc["stop"]) if rc["stop"] else -INF,
                                 "armed": False, "armed_ma": rc["ma_mode"] == "pure",
                                 "pending": False, "ci": col_of[cd],
                                 "shares": alloc * (1 - COST) / px}
                    cash -= alloc
        equity[t] = cash + sum(hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]])
                                               else hd["last"]) for hd in holds.values())
    eq = pd.Series(equity[start:], index=idx[start:])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return ann, (r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan), \
        (eq / eq.cummax() - 1).min()


rows = []
for mode, pick, keep in (("随机选", "random", 1.0), ("小市值优先(90%事件重抽样)", "small", 0.9)):
    print(f"\n{'='*118}")
    print(f"{mode}:每条规则 {N_SEED} 次的分布(10只/510300择时/0.3%成本)")
    print(f"{'='*118}")
    print(f"{'':<3}{'规则':<28}{'中位年化':>10}{'最差':>9}{'最好':>9}{'25%':>9}{'75%':>9}"
          f"{'>0比例':>8}{'中位Sharpe':>11}{'中位回撤':>10}")
    for key, rc in RULES.items():
        res = [run(rc, pick, 20260810 + s, keep) for s in range(N_SEED)]
        a = np.array([x[0] for x in res])
        sh = np.array([x[1] for x in res])
        dd = np.array([x[2] for x in res])
        rows.append({"选股": mode, "规则": key, "说明": rc["name"],
                     "中位年化": np.median(a), "均值": a.mean(),
                     "最差": a.min(), "最好": a.max(),
                     "q25": np.quantile(a, .25), "q75": np.quantile(a, .75),
                     "正收益比例": (a > 0).mean(), "中位Sharpe": np.median(sh),
                     "中位回撤": np.median(dd), "年化标准差": a.std()})
        r = rows[-1]
        print(f"{key:<3}{rc['name']:<28}{r['中位年化']:>+10.2%}{r['最差']:>+9.2%}"
              f"{r['最好']:>+9.2%}{r['q25']:>+9.2%}{r['q75']:>+9.2%}"
              f"{r['正收益比例']:>8.0%}{r['中位Sharpe']:>+11.3f}{r['中位回撤']:>10.2%}"
              f"   ({time.time()-t0:.0f}s)")

S = pd.DataFrame(rows)
S.to_csv(f"{SP}/breakout_exit_rules_seeds{TAG}.csv", index=False)

print(f"\n{'='*118}\n判断:各规则中位数是否超出基线A 的路径噪音\n{'='*118}")
for mode in S["选股"].unique():
    sub = S[S["选股"] == mode]
    a = sub[sub.规则 == "A"].iloc[0]
    print(f"\n[{mode}]  基线A 中位 {a['中位年化']:+.2%}、25~75%分位 "
          f"{a['q25']:+.2%}~{a['q75']:+.2%}、路径标准差 {a['年化标准差']:.2%}")
    for _, r in sub.iterrows():
        if r.规则 == "A":
            continue
        if r["中位年化"] > a["q75"]:
            v = "**中位数超过A的75%分位 → 超出路径噪音**"
        elif r["中位年化"] < a["q25"]:
            v = "**中位数低于A的25%分位 → 显著更差**"
        else:
            v = "落在A的路径噪音范围内"
        print(f"  {r.规则} 中位 {r['中位年化']:+.2%} / Sharpe {r['中位Sharpe']:+.3f} / "
              f"回撤 {r['中位回撤']:.2%}  → {v}")

print(f"\n对照:全市场等权基准 OOS 年化 +7.22% / Sharpe 0.423 / 回撤 -32.77%")
print("主表里 E随机选 +10.12% 是单个种子的结果,对照上面的分布判断其代表性。")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: breakout_exit_rules_seeds.csv")
