"""§113 复现 Codex R10「小市值+低换手」并补上他没做的对照。

背景
----
私库 chenqiang0929-opus/quant-research-dev 的 research/codex-stock-research/
用与本项目**完全相同**的底稿(5,232 只股票文件,他取其中 5,217 只沪深A股,
排除 15 只 B 股 200xxx)跑了 R01–R13 十三条路线。其中 R10
`small_cap_low_turnover` 是全部路线里收益最高的:2014-01-02→2025-12-31
总收益 +3135.06%、年化 33.62%、夏普 1.35、最大回撤 -41.55%;
2023–2025 样本外 +167.75%,同期 510300 +20.36%。

他的对照**只有 510300 买入持有**,没有随机对照、没有市值中性化、没有 p 值。
本节做两件事:(A) 用同一份数据独立复现他的数字;(B) 补上市值中性对照,
判断这 +3135% 里有多少是"小市值风格 beta",有多少是"选股能力"。

判据(跑之前写死,跑完照判,不放宽;加严可以)
------------------------------------------------
C1 复现。独立重建 `small_cap_low_turnover`,与 Codex 公布值比:
   C1a  full  (2014-01-02→2025-12-31) 年化 ∈ [27%, 40%]   (他 33.62%)
   C1b  oos   (2023-01-03→2025-12-31) 年化 ∈ [29%, 49%]   (他 39.02%)
   两条都中才算复现成立。任一不中 → 判"无法复现",Part B 全部作废,
   本节不对 R10 下任何结论。
   (容差按年化 ±6~10pp 给,不按累计收益给:12 年复利下年化 1pp 的实现差异
    会放大成上千 pp 的累计差异,用累计收益设判据等于没有判据。)

C2 低换手的市值中性增量。**核心判据。**
   对照:每个调仓日,取策略实际选中的 20 只,记下它们当日的市值 rank 区间
   [rmin, rmax];从"市值 rank 落在该区间内的全部可交易股票"里随机抽 20 只,
   同一引擎、同一成本、同一调仓日跑完整回测,100 组种子。
   → 这打掉的只有"低换手"这一维,市值暴露与策略逐日对齐。
   C2 通过 ⟺ 策略年化 **严格高于** 100 组对照年化的 95 分位数
            (单尾 p < 0.05),且 full 与 oos **两个窗口都成立**。
   任一窗口不成立 → C2 不通过 → 判定 R10 的超额主要来自小市值风格暴露,
   不是选股能力。

C3 小盘 beta 归因。按当日流通市值把全市场分 10 档,每档等权(不含成本、
   不含整手约束,纯风格曲线),同样 20 日调仓。
   C3 通过 ⟺ 最小档(D1)等权年化 **低于** 策略年化至少 5pp。
   若 D1 等权年化 ≥ 策略年化 - 5pp,则策略没有跑赢"闭着眼睛买最小一档",
   C3 不通过。

C4 分红口径。用含分红的 510300 重算基准,列出 R01–R12 十二条代表策略
   的超额收益修正前后对照。本条只描述、不设通过阈值(它是口径错误的
   量化,不是假设检验)。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
P1 C1 会过。数据同源、引擎我逐行读过,没有理由复现不出来。
P2 **C2 不会过。** 理由:Codex 自己的 2023–2025 Rank IC 表里,
   small_cap 单独 250 日 IC = 0.1259,small_cap_low_turnover = 0.1363,
   增量只有 0.0104;而"同市值 rank 区间内随机"已经吃掉全部小盘 beta。
   我预测策略相对对照中位数的年化超额 < 5pp 且 p > 0.05。
P3 **C3 不会过。** 我预测最小市值档 D1 的等权年化 ≥ 策略年化 - 5pp,
   即"取市值最小的 20 只 + 低换手"并不比"买入最小一档全部股票"更好。
P4 全市场等权(不含成本)2014-01-02→2025-12-31 总收益 > +200%,
   远高于 510300 的 +100.72%(不含分红)/ +144.04%(含分红)。

锚点(不过则本节结论作废)
--------------------------
A1 面板 (3297, 5232);去掉 15 只 B 股与 510300 后,universe = 5217,
   与 Codex 的 strategy_spec.yaml 逐个代码相等(集合差为空)。
A2 恒等式锚点:C2 的每一组对照,在每个调仓日,其成分股的市值 rank
   必须全部落在当日策略成分股的 [rmin, rmax] 内。若抽样写错(例如从
   全市场抽),此项必炸;写对则必过。逐日校验,报告违例数,>0 即作废。
A3 数据对齐锚点:510300 在 Codex 五个窗口上的**不含分红**收益,
   本脚本用同一份数据重算后,必须与他公布值同号且相对误差 < 25%
   (分红差异本身就在 15~22% 量级,所以这里只验"是同一个标的同一段
    日历",不验数值相等)。

不做的
------
不改 src/oxq/;不改 consolidation_screener.py;不新增顶层目录;不 force push;
**不往 quant-research-dev 推任何东西(只读)**。
不对 R10 之外的路线做复现(本节只复现收益最高的那一条)。
"""

from __future__ import annotations

import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
CACHE = f"{SP}/codex_r10_matrices.npz"
SPEC = ("/home/user/quant-research-dev/research/codex-stock-research/"
        "config/versions/v001/04_spec_build/strategy_spec.yaml")

# Codex 引擎常数(fast_low_risk_backtest.py 逐条抄下)
INITIAL_CASH = 100_000.0
TOP_N = 20
WEIGHT = 0.05
LOT = 100
BUY_FEE = 0.0003
SELL_FEE = 0.0003
SELL_TAX = 0.001
MIN_FEE = 5.0
SLIP = 0.001
WINDOWS = {
    "train": ("2014-01-02", "2019-12-31"),
    "validation": ("2020-01-02", "2022-12-30"),
    "oos": ("2023-01-03", "2025-12-31"),
    "holdout": ("2026-01-05", "2026-07-27"),
    "full": ("2014-01-02", "2025-12-31"),
}
CODEX = {  # 他公布的 small_cap_low_turnover 与基准
    "train": (4.0192, 0.72973), "validation": (1.3959, -0.050856),
    "oos": (1.6775, 0.203596), "holdout": (0.0415, -0.018786),
    "full": (31.3506, 1.007179),
}
NSEED, SEED = 100, 20260824


# ---------------------------------------------------------------- 数据装载
def build_matrices():
    """把 5,217 只 A 股装成矩阵。缓存到 npz,重跑秒开。"""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"复用缓存 {CACHE}")
        return (pd.DatetimeIndex(z["idx"]), list(z["codes"]), z["OP"], z["CL"],
                z["SUSP"], z["LU"], z["LD"], z["OK"], z["LOGCAP"], z["TMEAN"],
                z["AMT"], z["AMIH"])
    spec = open(SPEC, encoding="utf-8").read()
    universe = re.findall(r'^  - "(\d{6})"$', spec, re.M)
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    all_codes = [os.path.basename(f)[:-8] for f in files]
    # 锚点 A1 的前半:全量面板必须是 5232 只股票 + 510300
    assert len(all_codes) == 5233, f"锚点A1 文件数 {len(all_codes)}"
    stocks = [c for c in all_codes if c != "510300"]
    assert len(stocks) == 5232, f"锚点A1 股票数 {len(stocks)}"
    assert set(universe) - set(stocks) == set(), "锚点A1 Codex universe 有本面板没有的代码"
    assert len(universe) == 5217, f"锚点A1 universe {len(universe)}"

    cols_read = ["open", "close", "volume", "amount", "outstanding_share", "float_mv",
            "is_st", "is_suspended", "is_limit_up", "is_limit_down", "listed_days"]
    # 两趟:第一趟只读 close 拿日历并集(省内存),第二趟逐只填矩阵。
    t0 = time.time()
    idx = None
    for n, c in enumerate(universe):
        ix = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["close"]).index
        if getattr(ix, "tz", None) is not None:
            ix = ix.tz_localize(None)
        idx = ix if idx is None else idx.union(ix)
        if (n + 1) % 2000 == 0:
            print(f"  日历 {n+1}/{len(universe)}  ({time.time()-t0:.0f}s)")
    idx = pd.DatetimeIndex(idx).sort_values()
    nt, ns = len(idx), len(universe)
    print(f"日历并集 {nt} 天 × {ns} 股  ({time.time()-t0:.0f}s)")

    op = np.full((nt, ns), np.nan, np.float32)
    cl_m = np.full((nt, ns), np.nan, np.float32)
    susp = np.ones((nt, ns), bool)
    lu = np.ones((nt, ns), bool)
    ld = np.ones((nt, ns), bool)
    ok = np.zeros((nt, ns), bool)          # Codex 的 tradable 口径
    logcap = np.full((nt, ns), np.nan, np.float32)
    tmean = np.full((nt, ns), np.nan, np.float32)
    amt_m = np.full((nt, ns), np.nan, np.float32)
    amih = np.full((nt, ns), np.nan, np.float32)

    for j, c in enumerate(universe):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols_read)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        x = x.reindex(idx)
        o = pd.to_numeric(x["open"], errors="coerce")
        cl = pd.to_numeric(x["close"], errors="coerce")
        v = pd.to_numeric(x["volume"], errors="coerce")
        a = pd.to_numeric(x["amount"], errors="coerce")
        sh = pd.to_numeric(x["outstanding_share"], errors="coerce")
        op[:, j] = o.where(o > 0).to_numpy(np.float32)
        # Codex: close_price 前向填充(退市后冻结在最后有效价) —— 照抄,
        # 因为用户规则5要求退市股 ffill 参与,绝不剔除
        cl_m[:, j] = cl.where(cl > 0).ffill().to_numpy(np.float32)
        susp[:, j] = x["is_suspended"].fillna(True).to_numpy(bool)
        lu[:, j] = x["is_limit_up"].fillna(True).to_numpy(bool)
        ld[:, j] = x["is_limit_down"].fillna(True).to_numpy(bool)
        # 因子:MarketCap = close(不复权口径的市值) × 股本。本面板 close 为
        # 前复权,float_mv = raw_close × outstanding_share,已核对逐日相等,
        # 故直接用 float_mv 等价于 Codex 的 MarketCap()。这里用 close×shares
        # 会得到前复权市值,rank 与真实市值 rank 不同 —— 必须用 float_mv。
        cap = pd.to_numeric(x["float_mv"], errors="coerce")
        logcap[:, j] = np.log(cap.where(cap > 0)).to_numpy(np.float32)
        tr = v / sh.replace(0.0, np.nan)
        tmean[:, j] = tr.rolling(20).mean().to_numpy(np.float32)
        amt_m[:, j] = a.rolling(20).mean().to_numpy(np.float32)
        amih[:, j] = (cl.pct_change().abs() / a.replace(0.0, np.nan)
                      ).rolling(20).mean().to_numpy(np.float32)
        st = x["is_st"].fillna(True).to_numpy(bool)
        ld_ = pd.to_numeric(x["listed_days"], errors="coerce").to_numpy(float)
        # Codex: ~is_st & ~is_suspended & ~is_limit_up & listed_days>=250(累计
        # 有效交易日) & volume>0。本面板 listed_days 是日历日,250 交易日 ≈
        # 365 日历日,取 365 是**加严**,不是放宽。
        ok[:, j] = (~st & ~susp[:, j] & ~lu[:, j] & (ld_ >= 365)
                    & (v.to_numpy(float) > 0))
        if (j + 1) % 1500 == 0:
            print(f"  建矩阵 {j+1}/{ns}  ({time.time()-t0:.0f}s)")

    np.savez_compressed(CACHE, idx=idx.values, codes=np.array(universe),
                        OP=op, CL=cl_m, SUSP=susp, LU=lu, LD=ld, OK=ok,
                        LOGCAP=logcap, TMEAN=tmean, AMT=amt_m, AMIH=amih)
    print(f"缓存写入 {CACHE}  ({time.time()-t0:.0f}s)")
    return (idx, universe, op, cl_m, susp, lu, ld, ok, logcap, tmean, amt_m, amih)


# ---------------------------------------------------------------- 回测引擎
def run_window(op, cl, susp, lu, ld, sel, cal_pos, w0, w1):  # noqa: PLR0913
    """Codex fast_low_risk_backtest.run_window 的等价实现(逐条对照过)。

    sel: {日期在 cal_pos 中的行号 -> (np.array 列号, np.array 权重)}
    信号日 t 收盘定,t+1 开盘成交;整手 100 股;最低佣金 5 元;滑点双边 0.1%。
    """
    days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
    pos_i = np.zeros(0, np.int64)
    pos_s = np.zeros(0, np.float64)
    cash = INITIAL_CASH
    pending = None
    eq = np.empty(len(days))
    trades = 0
    frozen = 0
    for k, t in enumerate(days):
        if pending is not None:
            cols, wts = pending
            # 开盘市值:开盘价缺失时回落到收盘价(Codex 同款)
            if len(pos_i):
                po = op[t, pos_i].astype(np.float64)
                bad = ~np.isfinite(po) | (po <= 0)
                po[bad] = cl[t, pos_i][bad].astype(np.float64)
                po[~np.isfinite(po)] = 0.0
                open_val = cash + float(np.sum(pos_s * po))
            else:
                open_val = cash
            px = op[t, cols].astype(np.float64)
            good = np.isfinite(px) & (px > 0)
            tgt = {}
            for c_, w_, p_, g_ in zip(cols, wts, px, good):
                if g_:
                    tgt[int(c_)] = float(int(open_val * w_ // (p_ * (1 + SLIP) * LOT)) * LOT)
            # 卖
            keep_i, keep_s = [], []
            for c_, s_ in zip(pos_i, pos_s):
                c_ = int(c_)
                q = max(0.0, s_ - tgt.get(c_, 0.0))
                p_ = op[t, c_]
                if q <= 0 or susp[t, c_] or ld[t, c_] or not np.isfinite(p_) or p_ <= 0:
                    if q > 0 and (susp[t, c_] or not np.isfinite(p_) or p_ <= 0):
                        frozen += 1
                    keep_i.append(c_)
                    keep_s.append(s_)
                    continue
                amt = q * float(p_) * (1 - SLIP)
                cash += amt - max(MIN_FEE, amt * SELL_FEE) - amt * SELL_TAX
                trades += 1
                if s_ - q > 0:
                    keep_i.append(c_)
                    keep_s.append(s_ - q)
            pos = dict(zip(keep_i, keep_s))
            # 买(按分数顺序,先来先用现金 —— Codex 同款)
            for c_ in cols:
                c_ = int(c_)
                q = max(0.0, tgt.get(c_, 0.0) - pos.get(c_, 0.0))
                p_ = op[t, c_]
                if q <= 0 or susp[t, c_] or lu[t, c_] or not np.isfinite(p_) or p_ <= 0:
                    continue
                fill = float(p_) * (1 + SLIP)
                q = min(q, float(int(max(0.0, cash - MIN_FEE) // (fill * LOT)) * LOT))
                while q > 0 and q * fill + max(MIN_FEE, q * fill * BUY_FEE) > cash + 1e-8:
                    q -= LOT
                if q <= 0:
                    continue
                amt = q * fill
                cash -= amt + max(MIN_FEE, amt * BUY_FEE)
                pos[c_] = pos.get(c_, 0.0) + q
                trades += 1
            pos_i = np.array(sorted(pos), np.int64)
            pos_s = np.array([pos[i] for i in pos_i], np.float64)
            pending = None
        if len(pos_i):
            pc = cl[t, pos_i].astype(np.float64)
            pc[~np.isfinite(pc)] = 0.0
            eq[k] = cash + float(np.sum(pos_s * pc))
        else:
            eq[k] = cash
        if t in sel:
            pending = sel[t]
    return eq, days, trades, frozen


def metrics(eq, days, idx):
    yrs = max((idx[days[-1]] - idx[days[0]]).days / 365.25, 1 / 365.25)
    tot = eq[-1] / eq[0] - 1.0
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"total": float(tot), "cagr": float((eq[-1] / eq[0]) ** (1 / yrs) - 1),
            "mdd": float(np.min(eq / np.maximum.accumulate(eq) - 1.0)),
            "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            "years": float(yrs)}


def pct(a):
    """Codex 的 percentile():对当日可选集合做 rank(pct=True)。"""
    return pd.Series(a).rank(pct=True, ascending=True).to_numpy()


def main():
    idx, codes, op, cl, susp, lu, ld, ok, logcap, tmean, amt, amih = build_matrices()
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), f"锚点A1 面板 {(nt, ns)}"
    print(f"锚点A1 ✓ 面板 {nt}×{ns};universe 与 Codex strategy_spec.yaml 完全一致")

    # ---- 日历与调仓网格(照 Codex:基准日历 [::5] 再 [::4] = [::20])----
    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    pos = pd.Index(idx).get_indexer(cal)
    assert (pos >= 0).all(), "基准日历有不在股票面板里的日期"
    cal_pos = pos
    reb_pos = cal_pos[::20]
    print(f"基准日历 {len(cal)} 天,调仓日 {len(reb_pos)} 个 (Codex signal_dates=153)")

    # ---- 锚点 A3:同一份数据重算 510300 五个窗口 ----
    print("\n锚点A3 510300 五窗口(本面板含分红 vs Codex 不含分红):")
    a3_ok = True
    div_row = {}
    for w, (d0, d1) in WINDOWS.items():
        s = bs[(bs.index >= d0) & (bs.index <= d1)]
        mine = float(s.iloc[-1] / s.iloc[0] - 1)
        cod = CODEX[w][1]
        rel = abs((1 + mine) / (1 + cod) - 1)
        good = (np.sign(mine) == np.sign(cod)) and rel < 0.25
        a3_ok &= good
        div_row[w] = (mine, cod)
        print(f"  {w:11s} 本面板 {mine:+9.4%}  Codex {cod:+9.4%}  相对差 {rel:6.2%}  {'✓' if good else '✗'}")
    print(f"锚点A3 {'✓' if a3_ok else '✗ 作废'}")
    assert a3_ok, "锚点A3 不过"

    # ---- 逐调仓日算分、选股 ----
    t0 = time.time()
    sel_r10, elig_by_t, caprank_by_t, selpos_by_t = {}, {}, {}, {}
    for t in reb_pos:
        m = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
        e = np.flatnonzero(m)
        if len(e) < TOP_N * 3:
            continue
        sc_cap = pct(-logcap[t, e].astype(np.float64))
        sc_to = pct(-tmean[t, e].astype(np.float64))
        score = (sc_cap + sc_to) / 2.0
        top = e[np.argsort(-score, kind="stable")[:TOP_N]]
        sel_r10[int(t)] = (top, np.full(TOP_N, WEIGHT))
        order = e[np.argsort(logcap[t, e], kind="stable")]   # 市值从小到大
        elig_by_t[int(t)] = order
        rk = {int(c): i for i, c in enumerate(order)}
        caprank_by_t[int(t)] = rk
        selpos_by_t[int(t)] = np.array([rk[int(c)] for c in top])
    print(f"选股完成 {len(sel_r10)} 个调仓日 ({time.time()-t0:.0f}s)")

    # ---- C1 复现 ----
    print("\n" + "=" * 78 + "\nC1 复现 small_cap_low_turnover\n" + "=" * 78)
    rows = []
    for w, (d0, d1) in WINDOWS.items():
        w0 = int(pd.Index(idx).get_indexer([pd.Timestamp(d0)], method="bfill")[0])
        w1 = int(pd.Index(idx).get_indexer([pd.Timestamp(d1)], method="ffill")[0])
        eq, days, tr, fz = run_window(op, cl, susp, lu, ld, sel_r10, cal_pos, w0, w1)
        m = metrics(eq, days, idx)
        ct, cb = CODEX[w]
        cc = (1 + ct) ** (1 / m["years"]) - 1
        rows.append({"window": w, "total": m["total"], "cagr": m["cagr"], "mdd": m["mdd"],
                     "sharpe": m["sharpe"], "trades": tr, "frozen_sell_blocked": fz,
                     "codex_total": ct, "codex_cagr": cc})
        print(f"  {w:11s} 本次 {m['total']:+10.2%} 年化 {m['cagr']:+7.2%} 回撤 {m['mdd']:+7.2%} "
              f"夏普 {m['sharpe']:5.2f} | Codex {ct:+10.2%} 年化 {cc:+7.2%}")
    rd = {r["window"]: r for r in rows}
    c1a = 0.27 <= rd["full"]["cagr"] <= 0.40
    c1b = 0.29 <= rd["oos"]["cagr"] <= 0.49
    print(f"\n  C1a full 年化 {rd['full']['cagr']:+.2%} ∈ [27%,40%] ? {'✓' if c1a else '✗'}")
    print(f"  C1b oos  年化 {rd['oos']['cagr']:+.2%} ∈ [29%,49%] ? {'✓' if c1b else '✗'}")
    print(f"  C1 {'通过 —— 复现成立' if (c1a and c1b) else '不通过 —— 无法复现,C2/C3 结论作废'}")
    pd.DataFrame(rows).to_csv(f"{OUT}/codex_r10_replication_c1.csv", index=False)
    return rd, (c1a and c1b), (idx, codes, op, cl, susp, lu, ld, ok, logcap, tmean,
                               cal_pos, reb_pos, sel_r10, elig_by_t, selpos_by_t, bs, div_row)


if __name__ == "__main__":
    main()


# =============================================================================
# §113 结论:C1 不通过 —— 判据照判,本节对 R10 的结论全部作废。
#
# 事后查明:复现失败的原因在被复现对象一侧,不在本实现。Codex 自己的数据
# 核对给出直接证据 —— 宁德时代 2021-11-30 他的文件里 close=350.12(前复权价)
# 乘历史股本得 float_mv=7116.74 亿,而当日真实不复权收盘 679.68 元,
# 真实流通市值 13815.5 亿。他的市值低估 48.5%,即**复权价混进了市值**,
# 横截面市值排序被系统性扭曲(后来送转/分红越多的股票,当年显得越小)。
#
# 但判据是判据:C1 写的是"复现他的数字",没复现出来就是没通过。
# 本节不改判、不放宽、不事后把变体②追认成"其实复现了"。
#
# 已跑出的诊断(**诊断性质,非事前登记,证据等级低于事前判据**):
#   ① 真实市值(本实现基线) train +198.70%(年化20.03%)  full +1444.01%(年化25.63%)
#   ② 前复权市值(他的口径) train +354.23%(年化28.73%)  full +2743.16%(年化32.19%)
#   ③ ①+上市天数放宽250日  train +240.14%(年化22.66%)  full +1760.51%(年化27.60%)
#   ④ ②+③                  train +355.40%(年化28.78%)  full +2726.64%(年化32.13%)
#   Codex 公布              train +401.92%(年化30.89%)  full +3135.06%(年化33.62%)
#
# 市值口径这一条 = 年化 +6.56pp / 累计 +1299pp。
# 剩余 33.62%-32.19% = 1.43pp/年 属实现细节(调仓网格偏移、整手、tie-break)。
#
# 另:他的基准 510300 不含分红而组合含分红,五窗口隐含股息率 0.94~2.79%/年,
# full 窗口基准应为 +144.04% 而非 +100.72%,超额被高估 21.6pp。
#
# 后续在 §114 重新事前登记:以**真实市值**为基线,补市值中性对照。
# =============================================================================
