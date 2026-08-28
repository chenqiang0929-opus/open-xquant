"""§129 Part D:把两篇研报的「启动信号」放进已有的牛股特征表(描述项)。

沿用 `bull_features/bull_feature_scan.py` 的口径,否则不可比:
  牛股 = 该自然年收盘涨幅 > 100%
  t*   = 该年内「从 t 到年末的最大涨幅」最大的那一天(即该年最大涨幅的起点)
  所有特征**在 t* 当日测量,只用 <= t* 的信息**
报 P(特征|牛股)、P(特征|非牛)、P(牛股|特征)、lift = P(牛股|特征) ÷ 基准率。

纪律 A(噪音上界)照搬那一节:年内打乱牛股标签 200 次,每次记录**所有特征里的
最高 lift**,得到「纯噪音下 best-of-N 能到多少」。**lift 必须超过这条线才算发现。**
本部分不新增判据 —— 这条线是 scan 那一节定下的,此处只是把研报特征放进同一张表。

Part D2 自查(本轮加的,描述项)
------------------------------
t* 是**用当年的后见之明**挑出来的(该年最大涨幅的起点)。因此
「在 t* 上测出的 lift」不是可交易概率 —— 事前你不知道 t* 在哪一天。
为把「特征本身的信息」与「t* 这个选点方式带来的机械效应」分开,
本轮追加:**在同一年内随机取一天 t_rand(同样要求 t>=310)重测同一批特征**。
若某特征在 t* 上 lift 高、在 t_rand 上塌回 ~1,则它是 t* 的产物,不是启动信号。

与 bull_feature_scan 的差异(如实记录)
--------------------------------------
本部分跑在 **Codex universe 5217** 上(复用 §113 缓存),原表跑在全面板 5232 上,
差 15 只 B 股(200xxx)。基准率会有小数点级差异,**基准与 lift 在同一样本内自洽**。
原表的噪音上界是 best-of-20(中位 1.47 / 95分位 2.45);本部分特征只有 9 个,
best-of-9 的上界必然更低,**故重算一遍,两条线都印出来**。

不做的
------
不改 bull_feature_scan.py;不新增判据;不因某个门槛好看就把它当规则。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, OUT  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import build_fund  # noqa: E402

N_PERM, SEED = 200, 20260826
Y0, Y1 = 2013, 2025


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    cl, logcap = z["CL"], z["LOGCAP"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点C1a"
    m = np.load(f"{OUT}/davis_mats.npz")
    raw = m["raw"]
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点C1a TTM"

    # 未 ffill 的收盘价 —— 用来界定「还活着」的窗口(与原表一致,退市后不再计年)
    t0 = time.time()
    live = np.zeros((nt, ns), bool)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        s = pd.to_numeric(x["close"], errors="coerce").reindex(idx)
        live[:, j] = (s > 0).fillna(False).to_numpy()
    print(f"存续窗口 {time.time()-t0:.0f}s;锚点C1a ✓ {nt}×{ns};TTM ✓", flush=True)

    hi250 = pd.DataFrame(cl).rolling(250, min_periods=250).max().to_numpy()
    with np.errstate(all="ignore"):
        dd = cl / np.where(hi250 > 0, hi250, np.nan) - 1.0
        eps = fm["eps_ttm"]
        pe = raw / np.where(eps > 0, eps, np.nan)
        ni = fm["ni_ttm"]
        nip = np.roll(ni, 250, axis=0)
        yoy = ni / np.where(nip != 0, np.abs(nip), np.nan) - 1.0
        yoy[:250] = np.nan
        mv = np.exp(logcap.astype(np.float64)) / 1e8
        r20 = cl / np.roll(cl, 20, axis=0) - 1.0
        r20[:20] = np.nan

    year = idx.year.to_numpy()
    rng0 = np.random.default_rng(SEED)
    rows = []
    for j in range(ns):
        fin = live[:, j]
        if fin.sum() < 300:
            continue
        a = np.where(fin, cl[:, j], np.nan).astype(np.float64)
        for y in range(Y0, Y1 + 1):
            cur = np.flatnonzero((year == y) & fin)
            if cur.size < 100:
                continue
            prev = np.flatnonzero((year == y - 1) & fin)
            if prev.size == 0:
                continue
            yr_ret = a[cur[-1]] / a[prev[-1]] - 1
            fwd_max = np.maximum.accumulate(a[cur][::-1])[::-1]
            t = int(cur[int(np.argmax(fwd_max / a[cur] - 1))])
            if t < 310:
                continue
            elig = cur[cur >= 310]
            tr = int(rng0.choice(elig)) if elig.size else t
            rows.append({"code": codes[j], "year": y, "bull": bool(yr_ret > 1.0),
                         "dd": dd[t, j], "pe": pe[t, j], "yoy": yoy[t, j],
                         "mv": mv[t, j], "r20": r20[t, j],
                         "dd_r": dd[tr, j], "pe_r": pe[tr, j], "yoy_r": yoy[tr, j],
                         "mv_r": mv[tr, j], "r20_r": r20[tr, j]})
    p = pd.DataFrame(rows)
    base = float(p.bull.mean())
    print(f"\n样本 {len(p):,},牛股 {int(p.bull.sum()):,}(基准率 **{base:.2%}**)"
          f"  [原表 5232 只上是 5.37%]  ({time.time()-t0:.0f}s)", flush=True)

    gf3 = (p.dd <= -0.30) & (p.pe > 0) & (p.pe <= 20) & (p.yoy > 0)
    feats = {
        "【广发】距250日高点回撤≤-30%": p.dd <= -0.30,
        "【广发】距250日高点回撤≤-50%": p.dd <= -0.50,
        "【广发】PE_TTM ∈ (0,20]": (p.pe > 0) & (p.pe <= 20),
        "【广发】净利同比(代理)>0": p.yoy > 0,
        "【广发】三条同时(双击起点)": gf3,
        "【安信】流通市值 10~50 亿": (p.mv >= 10) & (p.mv <= 50),
        "【安信】前 20 日涨幅 > -1%": p.r20 > -0.01,
        "【安信】两条同时": (p.mv >= 10) & (p.mv <= 50) & (p.r20 > -0.01),
        "【安信】净利同比(代理)>300%": p.yoy > 3.0,
    }
    b = p.bull.to_numpy()
    mats = {}

    def stat(mask):
        mm = mask.fillna(False).astype(bool).to_numpy()
        n1 = int(mm.sum())
        if n1 < 30:
            return None
        pb = float(b[mm].mean())
        tab = [[int((b & mm).sum()), int((~b & mm).sum())],
               [int((b & ~mm).sum()), int((~b & ~mm).sum())]]
        _, pv = stats.fisher_exact(tab)
        return {"P(特征|牛股)": float(mm[b].mean()), "P(特征|非牛)": float(mm[~b].mean()),
                "P(牛股|特征)": pb, "lift": pb / base, "命中数": n1, "p": float(pv)}

    print(f"\n{'='*104}\n基准牛股率 {base:.2%};lift = P(牛股|特征) ÷ 基准\n{'='*104}")
    print(f"{'特征':<30}{'P(特征|牛股)':>12}{'P(特征|非牛)':>12}"
          f"{'P(牛股|特征)':>13}{'lift':>8}{'命中数':>10}{'Fisher p':>11}")
    res = {}
    for nm, mk in feats.items():
        s = stat(mk)
        if s is None:
            print(f"{nm:<30}{'样本<30':>12}")
            continue
        res[nm] = s
        mats[nm] = mk.fillna(False).astype(bool).to_numpy()
        print(f"{nm:<30}{s['P(特征|牛股)']:>12.1%}{s['P(特征|非牛)']:>12.1%}"
              f"{s['P(牛股|特征)']:>13.2%}{s['lift']:>8.2f}{s['命中数']:>10,}"
              f"{s['p']:>11.2e}")

    print(f"\n{'='*104}\n纪律A 噪音上界:年内打乱牛股标签 {N_PERM} 次,"
          f"每次取所有特征的最高 lift\n{'='*104}")
    rng = np.random.default_rng(SEED)
    yv = p.year.to_numpy()
    yrs = np.unique(yv)
    pos = {y: np.flatnonzero(yv == y) for y in yrs}
    best = []
    for _ in range(N_PERM):
        bb = np.zeros(len(p), bool)
        for y in yrs:
            e = pos[y]
            k = int(b[e].sum())
            if k:
                bb[rng.choice(e, k, replace=False)] = True
        bs = bb.mean()
        best.append(max(bb[mm].mean() / bs for mm in mats.values()))
    best = np.asarray(best)
    hi = float(np.percentile(best, 95))
    top = max(res.items(), key=lambda kv: kv[1]["lift"])
    print(f"  纯噪音 best-of-{len(mats)} lift:中位 **{np.median(best):.2f}**   "
          f"95%分位 **{hi:.2f}**   最大 {best.max():.2f}")
    print("  原表 best-of-20 的同一条线:中位 1.47   95%分位 2.45(仅供对照)")
    print(f"  本表最高 lift:**{top[1]['lift']:.2f}**({top[0]})")
    print(f"  → {'**超出噪音上界**' if top[1]['lift'] > hi else '**未超出噪音上界,不能算发现**'}")

    print(f"\n{'='*104}\nPart D2 自查:同一年内随机取一天 t_rand 重测,"
          f"看 lift 是不是 t* 选点带来的\n{'='*104}")
    feats_r = {
        "【广发】距250日高点回撤≤-30%": p.dd_r <= -0.30,
        "【广发】距250日高点回撤≤-50%": p.dd_r <= -0.50,
        "【广发】PE_TTM ∈ (0,20]": (p.pe_r > 0) & (p.pe_r <= 20),
        "【广发】净利同比(代理)>0": p.yoy_r > 0,
        "【广发】三条同时(双击起点)": (p.dd_r <= -0.30) & (p.pe_r > 0)
                            & (p.pe_r <= 20) & (p.yoy_r > 0),
        "【安信】流通市值 10~50 亿": (p.mv_r >= 10) & (p.mv_r <= 50),
        "【安信】前 20 日涨幅 > -1%": p.r20_r > -0.01,
        "【安信】两条同时": (p.mv_r >= 10) & (p.mv_r <= 50) & (p.r20_r > -0.01),
        "【安信】净利同比(代理)>300%": p.yoy_r > 3.0,
    }
    print(f"{'特征':<30}{'lift @ t*':>11}{'lift @ 随机日':>14}{'命中数(随机)':>14}")
    for nm in feats:
        s2 = stat(feats_r[nm])
        l0 = res.get(nm, {}).get("lift", np.nan)
        if s2 is None:
            print(f"{nm:<30}{l0:>11.2f}{'样本<30':>14}")
            continue
        res.setdefault(nm, {})["lift_随机日"] = s2["lift"]
        res[nm]["命中数_随机日"] = s2["命中数"]
        print(f"{nm:<30}{l0:>11.2f}{s2['lift']:>14.2f}{s2['命中数']:>14,}")

    df = pd.DataFrame(res).T
    df["超噪音上界"] = df["lift"] > hi
    df.to_csv(f"{OUT}/davis_partD.csv", encoding="utf-8-sig")
    print(f"\n落库 {OUT}/davis_partD.csv")


if __name__ == "__main__":
    main()
