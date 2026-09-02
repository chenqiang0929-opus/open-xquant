"""第一七九节 事前登记:补齐「确认失败组」的分段中性化超额(结果未跑)。

起因
----
Codex 2026-09-02 复核信指出:在他的事件内部比较里,留出段 2022-2026 的
「确认成立 − 确认失败」在 5/20/60 日**转负**(−0.22% / −0.07% / −1.14%),
与我第一七五节「留出段超额仍为正 +1.60pp」看似冲突。他问要不要完全对齐重跑。

**两边其实不冲突,是基准不同。** 现有三格已经咬得上:

    确认成立(留出段 60 日)  我 +0.87%   他 +1.00%   —— 几乎重合
    同日同市值同行业对照      我 +0.49%               —— 我的 200 组种子中位
    确认失败(留出段 60 日)              他 +2.15%
    确认率                   我 56.4%    他 55.54%

**缺的是「确认失败组 vs 同日对照」的中性化超额。**
第一七四节的 I3(b) 只报了全样本原始收益 +3.16%,没有按段拆,也没有中性化。
本节只补这一格。

做法
----
不改第一七四/一七五节已提交的 `longplat_pullback.py`,另写本脚本,
把 `okc`(确认成立)与 `~okc`(确认失败)**两组走完全相同的对照流程**:
- 事件源:Codex v3 `long_platform_breakouts.csv`(36,297 行),**不改他的事件**;
- 组合固定 **N=10、k=3%**(第一七五节留出段唯一两段都为正的那个);
- **两组的入场日都是确认日 t+10**(失败组也一样)—— 这样才可比,
  也与 Codex「前向收益起点=确认日收盘价」的口径一致;
- 对照:同一确认日、同市值名次 ±25、同申万一级随机抽 1 只,200 组种子;
- **成立组与失败组各自独立调用一次 `controls()`,各自用同一个种子起头** ——
  这样成立组的抽样序列与第一五七/一七五节逐字一致,锚点才有意义;
- 时间切分沿用:训练段 2013→2021-12 / 留出段 2022-01-01 起,按**突破日**切。

判据
----
**本节不设通过/不通过判据。** 第一七四/一七五节已对回踩确认下过判定
(全样本 0/4、留出段 0/4 均不通过),**本节不重判、不翻案,只补描述** ——
与第七六/八六/一七七节同规格。

锚点(不过则本节作废)
----------------------
O1 (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) 事件映射率 ≥ 90%;
   (c) **确认率 = 56.4%**(与第一七四节逐字一致);
   (d) **成立组 留出段 60 日:事件收益 +0.87%、对照中位 +0.49%**
       (与第一七五节逐字一致,四舍五入到 0.01pp);
   (e) 两次对照抽样的市值名次偏离 > 25 的违例均 = 0。

必报
----
训练段 / 留出段 × 成立组 / 失败组 × 5 个持有期,
逐格给出事件收益、对照中位、超额pp、年化超额pp、单尾 p。

**本文件不构成任何投资建议。**
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_neutral import NBR, SEED  # noqa: E402
from industry_neutral import build_industry  # noqa: E402
from panel_cache import cached  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR",
                      "/home/user/oxq-panel-0828/oxq_stock_market_fixed")
OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
EV = os.environ.get("OXQ_EV", "/root/.claude/uploads/"
                    "e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
                    "443008ba-long_platform_breakouts1.csv")
HOLD = (5, 20, 60, 120, 250)
NSEED, NN, KK, MAXN = 200, 10, 0.03, 10


def ann(r, h):
    return (1.0 + r) ** (250.0 / h) - 1.0 if r > -1 else np.nan


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    p = cached("panel", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:panel 缓存必须已存在")))
    idx = pd.DatetimeIndex(p["idx"])
    cl, okm = p["cl"], p["okm"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点O1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点O1a 末日 {idx[-1].date()}"
    print(f"锚点O1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)
    mvm = cached("mv", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:mv 缓存必须已存在")))["mv"]
    ind, _, _ = build_industry(codes, idx)

    e = pd.read_csv(EV, encoding="utf-8-sig", dtype={"code": str})
    e["code"] = e["code"].str.zfill(6)
    e["breakout_date"] = pd.to_datetime(e["breakout_date"])
    pos = {c: j for j, c in enumerate(codes)}
    e["j"] = e["code"].map(pos).fillna(-1).astype(int)
    e["t"] = pd.Index(idx).get_indexer(e["breakout_date"])
    mapped = (e["j"] >= 0) & (e["t"] >= 0)
    rate = float(mapped.mean())
    print(f"锚点O1b 事件映射率 {rate:.1%} {'✓' if rate >= .90 else '✗ 作废'}",
          flush=True)
    if rate < 0.90:
        return
    e = e[mapped & (e["t"] < nt - max(HOLD) - MAXN)].copy()   # 与 §175 同一 trim
    tt, jj = e["t"].to_numpy(), e["j"].to_numpy()
    ratio = (e["upper_before_breakout"].to_numpy(float)
             / e["breakout_close"].to_numpy(float))
    upper = cl[tt, jj] * ratio

    lo_n = np.full(len(e), np.nan)
    for i in range(len(e)):
        seg = cl[tt[i] + 1:tt[i] + 1 + NN, jj[i]]
        lo_n[i] = np.nanmin(seg) if np.isfinite(seg).any() else np.nan
    okc = np.isfinite(lo_n) & (lo_n >= upper * (1.0 - KK))
    cr = float(okc.mean())
    print(f"锚点O1c 确认率 {cr:.1%} {'✓' if abs(cr - 0.564) < 0.0005 else '✗ 作废'}"
          f"(§174 = 56.4%)", flush=True)
    if abs(cr - 0.564) >= 0.0005:
        return
    ct = tt + NN

    def controls(cts):
        rng = np.random.default_rng(SEED)
        out = np.full((NSEED, len(cts)), -1, np.int32)
        viol, cache = 0, {}
        for k, (t, j) in enumerate(zip(cts, jj, strict=True)):
            if t < 0 or t >= nt:
                continue
            if t not in cache:
                el = np.flatnonzero(okm[t] & np.isfinite(mvm[t]) & (ind[t] >= 0))
                if not len(el):
                    cache[t] = None
                else:
                    o = el[np.argsort(mvm[t, el], kind="stable")]
                    rk = np.full(ns, -1, np.int32)
                    rk[o] = np.arange(len(o), dtype=np.int32)
                    cache[t] = (o, rk)
            if cache[t] is None:
                continue
            o, rk = cache[t]
            p_, i0 = rk[j], ind[t, j]
            if p_ < 0:
                continue
            lo, hi = max(0, p_ - NBR), min(len(o), p_ + NBR + 1)
            cand = o[lo:hi]
            cand = cand[(ind[t, cand] == i0) & (cand != j)]
            if not len(cand):
                continue
            out[:, k] = rng.choice(cand, NSEED, replace=True)
            viol += int(np.any(np.abs(rk[out[:, k]] - p_) > NBR))
        return out, viol

    runnable = ct < nt - max(HOLD)
    # 两组各自独立一次 —— 成立组这一路与 §175 逐字同源
    cs_ok, v1 = controls(np.where(okc & runnable, ct, -1))
    cs_no, v2 = controls(np.where((~okc) & runnable, ct, -1))
    print(f"锚点O1e 抽样违例 成立 {v1} / 失败 {v2} "
          f"{'✓' if v1 == 0 and v2 == 0 else '✗ 作废'} ({time.time()-t0:.0f}s)",
          flush=True)
    if v1 or v2:
        return

    split = int(np.searchsorted(idx.values, np.datetime64("2022-01-01")))
    segs = (("训练段13-21", tt < split), ("留出段22-26", tt >= split))
    groups = (("确认成立", okc, cs_ok), ("确认失败", ~okc, cs_no))

    rows, w = [], 104
    print(f"\n{'='*w}\nN={NN} k={KK:.0%};两组入场日都是确认日 t+{NN}\n{'='*w}")
    print(f"{'段':<12}{'组':<10}{'事件':>8}{'持有':>5}{'事件收益':>10}"
          f"{'对照中位':>10}{'超额pp':>9}{'年化超额pp':>12}{'p':>8}")
    for sn, sm in segs:
        for gn, gm, cs in groups:
            for h in HOLD:
                m = gm & runnable & sm & (cs[0] >= 0)
                p0 = cl[np.clip(ct, 0, nt - 1), jj]
                p1 = cl[np.clip(ct + h, 0, nt - 1), jj]
                with np.errstate(all="ignore"):
                    r = p1 / np.where(p0 > 0, p0, np.nan) - 1.0
                m = m & np.isfinite(r)
                if m.sum() < 30:
                    continue
                a = float(np.nanmean(r[m]))
                cm = np.empty(NSEED)
                for s in range(NSEED):
                    ci = cs[s][m]
                    cp0 = cl[ct[m], ci]
                    cp1 = cl[np.clip(ct[m] + h, 0, nt - 1), ci]
                    with np.errstate(all="ignore"):
                        cm[s] = np.nanmean(cp1 / np.where(cp0 > 0, cp0, np.nan) - 1)
                med = float(np.nanmedian(cm))
                rec = {"段": sn, "组": gn, "n": int(m.sum()), "持有": h,
                       "事件收益": a, "对照中位": med, "超额pp": (a - med) * 100,
                       "年化超额pp": (ann(a, h) - ann(med, h)) * 100,
                       "p": float((np.sum(cm >= a) + 1) / (NSEED + 1))}
                rows.append(rec)
                print(f"{sn:<12}{gn:<10}{rec['n']:>8,}{h:>5}{a:>+10.2%}"
                      f"{med:>+10.2%}{rec['超额pp']:>+9.2f}"
                      f"{rec['年化超额pp']:>+12.2f}{rec['p']:>8.4f}")

    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/longplat_fail_group.csv", index=False, encoding="utf-8-sig")
    z = d[(d["段"] == "留出段22-26") & (d["组"] == "确认成立") & (d["持有"] == 60)]
    ok = len(z) and abs(z["事件收益"].iloc[0] - 0.0087) < 5e-5 \
        and abs(z["对照中位"].iloc[0] - 0.0049) < 5e-5
    print(f"\n锚点O1d 成立组留出段 60 日 事件 {z['事件收益'].iloc[0]:+.2%} "
          f"对照 {z['对照中位'].iloc[0]:+.2%} {'✓' if ok else '✗ 作废'}"
          f"(§175 = +0.87% / +0.49%)")
    if not ok:
        print("锚点O1d 不过 —— 本节按登记作废,不出结论。")
        return

    print(f"\n{'='*w}\n给 Codex 第 2、3 问的那一格(留出段、60 日)\n{'='*w}")
    for gn in ("确认成立", "确认失败"):
        y = d[(d["段"] == "留出段22-26") & (d["组"] == gn) & (d["持有"] == 60)].iloc[0]
        print(f"  {gn}:事件 {y['事件收益']:+.2%}、对照 {y['对照中位']:+.2%}、"
              f"超额 {y['超额pp']:+.2f}pp、年化 {y['年化超额pp']:+.2f}pp、"
              f"p {y['p']:.4f}(n={y['n']:,})")
    print(f"\n落库 {OUT}/longplat_fail_group.csv ({time.time()-t0:.0f}s)")
    print("本表是状态记录,不是买点,不构成任何投资建议。")


if __name__ == "__main__":
    main()
