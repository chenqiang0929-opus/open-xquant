"""§129 Part C:两个研报口径的充分性检验(事件 + 同市值同行业对照)。

判据、对照设计与数据边界见 `davis_double_click.py` 的事前登记 docstring
(commit 48404fe,早于任何结果)。本文件只是实现,不新增判据、不放宽判据。
复用 `davis_double_click.py` 落下的矩阵缓存 davis_mats.npz,避免重建面板。

实现说明(与登记口径一致,仅为速度改写)
----------------------------------------
把「每个事件、每个种子各算一次前瞻峰值」改成先算好前瞻命中矩阵
hit[t,j] = (max(cl[t+1:t+1+hor,j])/cl[t,j]-1 >= thr),再做纯索引抽样。
数值等价,C1(c) 用逐点断言把「窗口起点严格 > t」钉死。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, NBR, OUT, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import build_fund  # noqa: E402

NSEED, ALPHA, GAP = 500, 0.05 / 2, 250
SEED_BLK = 50


def fwd_hit(cl, hor, thr):
    """hit[t,j]=前瞻窗口 (t, t+hor] 的峰值涨幅 >= thr;val[t,j]=该值可算。"""
    nt, ns = cl.shape
    hit = np.zeros((nt, ns), bool)
    val = np.zeros((nt, ns), bool)
    for s in range(0, ns, 600):
        a = cl[:, s:s + 600].astype(np.float64)
        # rm[t] = max(a[t : t+hor]) —— 反转后做滚动最大再反转回来
        rm = pd.DataFrame(a[::-1]).rolling(hor, min_periods=1).max().to_numpy()[::-1]
        f = np.full_like(a, np.nan)
        f[:-1] = rm[1:]                     # f[t] = max(a[t+1 : t+1+hor])
        with np.errstate(all="ignore"):
            r = f / np.where(a > 0, a, np.nan) - 1.0
        v = np.isfinite(r)
        val[:, s:s + 600] = v
        hit[:, s:s + 600] = v & (r >= thr)
    return hit, val


def main():  # noqa: PLR0912, PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    cl, ok, logcap = z["CL"], z["OK"], z["LOGCAP"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点C1a"
    m = np.load(f"{OUT}/davis_mats.npz")
    raw, ind = m["raw"], m["ind"]
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点C1a TTM"
    print(f"锚点C1a ✓ {nt}×{ns};TTM ✓", flush=True)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    tmin = int(pd.Index(idx).get_indexer(cal).min())

    t0 = time.time()
    hi250 = pd.DataFrame(cl).rolling(250, min_periods=250).max().to_numpy()
    with np.errstate(all="ignore"):
        dd = cl / np.where(hi250 > 0, hi250, np.nan) - 1.0
        eps = fm["eps_ttm"]
        pe = raw / np.where(eps > 0, eps, np.nan)
        ni = fm["ni_ttm"]
        nip = np.roll(ni, 250, axis=0)
        yoy = ni / np.where(nip != 0, np.abs(nip), np.nan) - 1.0
        yoy[:250] = np.nan
        mv = np.exp(logcap.astype(np.float64)) / 1e8            # 亿元
        r20 = cl / np.roll(cl, 20, axis=0) - 1.0
        r20[:20] = np.nan
    print(f"条件矩阵 {time.time()-t0:.0f}s", flush=True)

    # ---- 锚点 C1(c):条件矩阵只用 <=t 的信息(逐点重算断言)----
    rs = np.random.default_rng(7)
    nchk = 0
    for _ in range(4000):
        t = int(rs.integers(300, nt))
        j = int(rs.integers(0, ns))
        if not np.isfinite(dd[t, j]):
            continue
        ref = float(cl[t, j] / np.nanmax(cl[t - 249:t + 1, j]) - 1.0)
        assert abs(ref - dd[t, j]) < 1e-6, f"C1c dd 前视 t={t} j={j}"
        if np.isfinite(yoy[t, j]):
            r2 = float(ni[t, j] / abs(ni[t - 250, j]) - 1.0)
            assert abs(r2 - yoy[t, j]) < 1e-6, f"C1c yoy 前视 t={t} j={j}"
        if np.isfinite(r20[t, j]):
            r3 = float(cl[t, j] / cl[t - 20, j] - 1.0)
            assert abs(r3 - r20[t, j]) < 1e-6, f"C1c r20 前视 t={t} j={j}"
        nchk += 1
    print(f"锚点C1c 条件矩阵因果性 {nchk} 点逐点重算一致 ✓", flush=True)

    fwd = {}
    for hor, thr in ((690, 1.00), (250, 2.00)):
        t1 = time.time()
        fwd[(hor, thr)] = fwd_hit(cl, hor, thr)
        print(f"前瞻矩阵 hor={hor} thr={thr:+.0%} ({time.time()-t1:.0f}s)", flush=True)
    # 锚点 C1(c) 后半:窗口起点严格 > t
    hit0, val0 = fwd[(690, 1.00)]
    nchk = 0
    for _ in range(3000):
        t = int(rs.integers(tmin, nt - 690 - 1))
        j = int(rs.integers(0, ns))
        if not val0[t, j]:
            continue
        pk = float(np.nanmax(cl[t + 1:t + 1 + 690, j]) / cl[t, j] - 1.0)
        assert bool(pk >= 1.00) == bool(hit0[t, j]), f"C1c 前瞻窗口 t={t} j={j}"
        nchk += 1
    print(f"锚点C1c 前瞻窗口起点 > t  {nchk} 点逐点重算一致 ✓", flush=True)

    # ---- 每日「同市值名次 + 同行业」候选池预处理 ----
    order_cache, rank_cache = {}, {}

    def prep(t):
        if t not in order_cache:
            e = np.flatnonzero(ok[t] & np.isfinite(logcap[t]) & (ind[t] >= 0))
            o = e[np.argsort(logcap[t, e], kind="stable")]
            rk = np.full(ns, -1, np.int32)
            rk[o] = np.arange(len(o), dtype=np.int32)
            order_cache[t], rank_cache[t] = o, rk
        return order_cache[t], rank_cache[t]

    def events(mask, hor, thr, tag, sub=None):
        """按 mask 造事件(同股 250 日内不重复),测前瞻峰值 >= thr 的概率。"""
        hit, val = fwd[(hor, thr)]
        mk = mask & val
        if sub is not None:
            mk = mk & sub
        tmax = nt - hor - 1
        ts, js = [], []
        for j in range(ns):
            h = np.flatnonzero(mk[tmin:tmax + 1, j])
            if h.size == 0:
                continue
            h += tmin
            last = -10**9
            for t in h:
                if t - last >= GAP:
                    ts.append(int(t))
                    js.append(j)
                    last = t
        if not ts:
            print(f"{tag:34s} 事件 0 —— 跳过", flush=True)
            return None
        ts = np.asarray(ts)
        js = np.asarray(js)
        p_hit = float(hit[ts, js].mean())

        # 对照候选:同日、同申万一级行业、市值名次 ±NBR
        chunks, nofit = [], 0
        off = np.zeros(len(ts), np.int64)
        lens = np.zeros(len(ts), np.int64)
        keep = np.ones(len(ts), bool)
        pos_f = 0
        for k, (t, j) in enumerate(zip(ts, js, strict=True)):
            o, rk = prep(int(t))
            p0 = rk[j]
            i0 = ind[t, j]
            if p0 < 0 or i0 < 0:
                keep[k] = False
                nofit += 1
                continue
            lo, hi = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
            cand = o[lo:hi + 1]
            cand = cand[ind[t, cand] == i0]
            if len(cand) < 2:
                cand = o[ind[t, o] == i0]
            if len(cand) < 2:
                keep[k] = False
                nofit += 1
                continue
            off[k] = pos_f
            lens[k] = len(cand)
            pos_f += len(cand)
            chunks.append(cand)
        flat = np.concatenate(chunks).astype(np.int64)
        tk, jk, ofk, lnk = ts[keep], js[keep], off[keep], lens[keep]
        rng = np.random.default_rng(SEED)
        cp, viol = [], 0
        for s0 in range(0, NSEED, SEED_BLK):
            nb = min(SEED_BLK, NSEED - s0)
            r = rng.random((nb, len(tk)))
            pick = flat[ofk[None, :] + (r * lnk[None, :]).astype(np.int64)]
            viol += int((ind[tk, pick] != ind[tk, jk][None, :]).sum())
            v = val[tk, pick]
            h = hit[tk, pick] & v
            nv = v.sum(1)
            cp.extend(np.where(nv > 0, h.sum(1) / np.maximum(nv, 1), np.nan))
        cp = np.asarray(cp, float)
        p = (1 + int(np.sum(cp >= p_hit))) / (NSEED + 1)
        print(f"{tag:34s} 事件{len(ts):6d} P(峰值≥{thr:+.0%})={p_hit:6.2%} | "
              f"对照中位{np.nanmedian(cp):6.2%} 95分位{np.nanpercentile(cp,95):6.2%} "
              f"p={p:.4f} {'✓' if p < ALPHA else '✗'} | 行业违例{viol} 无对照{nofit}",
              flush=True)
        return {"tag": tag, "n_ev": len(ts), "p_hit": p_hit,
                "ctrl_med": float(np.nanmedian(cp)),
                "ctrl_p95": float(np.nanpercentile(cp, 95)), "p": p,
                "C2": bool(p < ALPHA), "viol": viol, "nofit": nofit}

    with np.errstate(all="ignore"):
        gf = ok & (dd <= -0.30) & (pe > 0) & (pe <= 20) & (yoy > 0)
        ax = ok & (mv >= 10) & (mv <= 50) & (r20 > -0.01)

    print("\n=== C2 核心判定(α=0.025,Bonferroni 2 个口径)===", flush=True)
    rows = [events(gf, 690, 1.00, "GF 广发口径 690日 峰值≥100%"),
            events(ax, 250, 2.00, "AX 安信口径 250日 峰值≥200%")]

    print("\n=== C3 分层描述(不据此挑最优)===", flush=True)
    with np.errstate(all="ignore"):
        for d0 in (-0.30, -0.40, -0.50):
            for p0 in (15, 20, 25):
                mk = ok & (dd <= d0) & (pe > 0) & (pe <= p0) & (yoy > 0)
                rows.append(events(mk, 690, 1.00, f"GF 回撤≤{d0:.0%} PE≤{p0}"))
        for lo, hi in ((10, 50), (10, 100), (5, 30)):
            mk = ok & (mv >= lo) & (mv <= hi) & (r20 > -0.01)
            rows.append(events(mk, 250, 2.00, f"AX 市值[{lo},{hi}]亿"))

    print("\n=== C4 食品饮料子样本(描述,不参与判定)===", flush=True)
    fb = -99
    try:
        from industry_neutral import build_industry
        _, _, nid = build_industry(codes, idx)
        fb = int(nid.get("食品饮料", -99))
    except Exception as ex:                                   # noqa: BLE001
        print(f"  行业名映射不可用({ex}),跳过 C4", flush=True)
    if fb >= 0:
        rows.append(events(gf, 690, 1.00, "GF·食品饮料内", sub=(ind == fb)))
        rows.append(events(ax, 250, 2.00, "AX·食品饮料内", sub=(ind == fb)))

    df = pd.DataFrame([r for r in rows if r])
    v = int(df["viol"].sum())
    print(f"\n锚点C1b 行业违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}")
    assert v == 0
    core = df[df["tag"].str.startswith(("GF 广发", "AX 安信"))]
    print(f"C2 通过 {int(core['C2'].sum())}/2:"
          f"{', '.join(core.loc[core['C2'], 'tag']) or '无'}")
    df.to_csv(f"{OUT}/davis_partC.csv", index=False)
    print(f"落库 {OUT}/davis_partC.csv")


if __name__ == "__main__":
    main()
