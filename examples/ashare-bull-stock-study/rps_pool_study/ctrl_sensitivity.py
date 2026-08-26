"""§136 事前登记 + 实现:对照口径敏感性 —— 只同行业 vs 同行业+同市值。

起因
----
用户提出:「同行业对照可以,同市值对照我觉得没必要,因为板块效应,
可能同板块是一起上涨的,只是你选的是涨得多的龙头,或者是普涨的。」

**这个意见有实质内容,不该靠论证决定,直接测。**
用户担心的是:同市值对照可能把「在行业内挑龙头」这个能力一起扣掉。
反向风险是:市值是 A 股最强横截面因子之一(第一一五-B 节 small_cap 年化 +33.14%
p=0.0020 通过;large_cap +9.00% 为负对照),不控市值时任何偏小市值的规则
都会自动跑赢,而这跟规则本身无关。

做什么(纯描述,不设通过/不通过判据)
------------------------------------
沿用第一三五节完全相同的样本、特征、阈值候选与标签(不改一个字),
只把对照换成两套并列:
  **对照 I(只同行业)**:同日、同申万一级行业内随机抽一只,**市值不限**
  **对照 IB(同行业+同市值)**:同日、同行业、且流通市值名次 ±25 —— 第一三五节现口径
每个阈值同时报 `lift_ind`(对 I)与 `lift_indmv`(对 IB),各 500 组种子。

**再加一个诊断(回答用户的原话)**:每个阈值下,被选中样本的
**流通市值分位中位数**(同日全市场横截面分位)。
若选中样本系统性偏小市值,则 `lift_ind` 与 `lift_indmv` 的差就是市值维度贡献的部分。

判据
----
**本节不设通过/不通过。** 这是对照口径的敏感性分析,不是假设检验。
锚点仍然要过,不过则作废:
  U1(a) 面板 (3297, 5232);
  U1(b) **两套对照都必须同行业**,行业违例 > 0 即作废;
  U1(c) 对照 IB 的候选必须是对照 I 候选的子集(逐条断言),
        否则「加严」这个说法不成立。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不改第一三五节的脚本与结论;不改阈值候选;不新增顶层目录;不 force push;
**不因为某一套对照更好看就改用它** —— 两套并列呈现,由数据说话。
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
from codex_r10_neutral import NBR, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from industry_neutral import build_industry  # noqa: E402
from startup_threshold_scan import load_labels  # noqa: E402

# 阈值候选与列名:**逐字照抄第一三五节 main() 内的同名字典**,
# 一个值都没改。在此重复定义是为了不改动已落库的第一三五节脚本。
cands_all = {
    "recovery_from_low_250": [("0-20%", (0.0, 0.20)), ("20-40%", (0.20, 0.40)),
                              ("40-60%", (0.40, 0.60)), ("60-100%", (0.60, 1.00)),
                              ("100-200%", (1.00, 2.00)), (">200%", (2.00, 1e9))],
    "close_to_ma_250": [("<0", (-1e9, 0.0)), ("0-10%", (0.0, 0.10)),
                        ("10-20%", (0.10, 0.20)), ("20-40%", (0.20, 0.40)),
                        ("40-60%", (0.40, 0.60)), ("60-100%", (0.60, 1.00)),
                        (">100%", (1.00, 1e9))],
    "rps_60": [(f">={v}", (v, 1e9)) for v in (60, 70, 80, 85, 90, 95)],
    "vol_20_pct": [(f">={v}分位", (v, 1e9)) for v in (50, 60, 70, 80, 90)],
    "above_ma20_share_120": [(f">={int(v*100)}%", (v, 1e9))
                             for v in (.50, .55, .60, .65, .70, .75, .80, .90)],
}
colmap = {"recovery_from_low_250": "rec", "close_to_ma_250": "c2ma",
          "rps_60": "rps60", "vol_20_pct": "vq20",
          "above_ma20_share_120": "ab120"}

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NSEED = 500
Y_ALL, Y_MOD = (2015, 2025), (2019, 2025)


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "float_mv", "volume", "is_st", "is_suspended", "listed_days"]
    d = {c: {} for c in cols}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in cols:
            d[k][c] = x[k]
    cldf = pd.DataFrame(d["close"]).sort_index()
    idx = cldf.index
    nt, ns = cldf.shape
    assert (nt, ns) == (3297, 5232), f"锚点U1a {cldf.shape}"

    def al(k, fill=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(fill)
    mv = al("float_mv").to_numpy() / 1e8
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).to_numpy(np.float64)
    ok &= np.isfinite(cl)
    ind, _, _ = build_industry(list(cldf.columns), idx)
    print(f"锚点U1a ✓ {cldf.shape} ({time.time()-t0:.0f}s)", flush=True)

    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    ma250 = dfc.rolling(250, min_periods=250).mean().to_numpy()
    ma20 = dfc.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        c2ma = cl / np.where(ma250 > 0, ma250, np.nan) - 1.0
        r60 = cl / np.roll(cl, 60, axis=0) - 1.0
        r60[:60] = np.nan
        lr = np.log(cl / np.roll(cl, 1, axis=0))
        lr[0] = np.nan
    v20 = pd.DataFrame(lr).rolling(20, min_periods=20).std().to_numpy()
    ab = pd.DataFrame(cl > ma20).rolling(120, min_periods=120).mean().to_numpy()
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100

    l1s, _ = load_labels()
    yend = pd.Series(np.arange(nt), index=idx).groupby(idx.year).last()
    rows = []
    for y, t in yend.items():
        ty = int(y) + 1
        if not (Y_ALL[0] <= ty <= Y_ALL[1]):
            continue
        e = np.flatnonzero(ok[t] & np.isfinite(rec[t]) & np.isfinite(c2ma[t])
                           & np.isfinite(v20[t]) & np.isfinite(ab[t])
                           & np.isfinite(rps60[t]) & np.isfinite(mv[t])
                           & (ind[t] >= 0))
        vq = pd.Series(v20[t, e]).rank(pct=True).to_numpy() * 100
        mq = pd.Series(mv[t, e]).rank(pct=True).to_numpy() * 100
        for k, j in enumerate(e):
            c = cldf.columns[j]
            rows.append((ty, int(t), int(j), rec[t, j], c2ma[t, j], rps60[t, j],
                         vq[k], ab[t, j], mq[k], (ty, c) in l1s))
    p = pd.DataFrame(rows, columns=["ty", "t", "j", "rec", "c2ma", "rps60",
                                    "vq20", "ab120", "mvq", "L1"])
    print(f"样本 {len(p):,} ({time.time()-t0:.0f}s)", flush=True)

    # 两套候选池
    pre = {}
    for t in p.t.unique():
        e = np.flatnonzero(ok[t] & np.isfinite(mv[t]) & (ind[t] >= 0))
        o = e[np.argsort(mv[t, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        pre[t] = (o, rk)
    tv, jv = p.t.to_numpy(), p.j.to_numpy()

    def build(narrow):
        ch, off, lens = [], np.zeros(len(p), np.int64), np.zeros(len(p), np.int64)
        pos, keep = 0, np.ones(len(p), bool)
        for k in range(len(p)):
            t, j = int(tv[k]), int(jv[k])
            o, rk = pre[t]
            i0 = ind[t, j]
            same = o[ind[t, o] == i0]
            if narrow:
                p0 = rk[j]
                a_, b_ = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
                cand = o[a_:b_ + 1]
                cand = cand[ind[t, cand] == i0]
                if len(cand) < 2:
                    cand = same
            else:
                cand = same
            if len(cand) < 2:
                keep[k] = False
                continue
            off[k], lens[k] = pos, len(cand)
            pos += len(cand)
            ch.append(cand)
        return np.concatenate(ch).astype(np.int64), off, lens, keep

    f_i, off_i, len_i, keep_i = build(False)
    f_b, off_b, len_b, keep_b = build(True)
    # 锚点 U1c:IB 候选 ⊆ I 候选
    bad = 0
    rs = np.random.default_rng(5)
    for k in rs.choice(np.flatnonzero(keep_i & keep_b), 2000, replace=False):
        s_i = set(f_i[off_i[k]:off_i[k] + len_i[k]].tolist())
        s_b = set(f_b[off_b[k]:off_b[k] + len_b[k]].tolist())
        bad += int(not s_b.issubset(s_i))
    print(f"锚点U1c IB候选⊆I候选 抽查2000条 违例 {bad} {'✓' if bad == 0 else '✗ 作废'}",
          flush=True)
    assert bad == 0

    lab1 = np.zeros((nt, ns), bool)
    lab1[tv[p.L1.to_numpy()], jv[p.L1.to_numpy()]] = True
    valid = np.zeros((nt, ns), bool)
    valid[tv, jv] = True

    def draw(flat, off, lens, keep):
        rng = np.random.default_rng(SEED)
        pk = np.full((NSEED, len(p)), -1, np.int64)
        kk = np.flatnonzero(keep)
        for s0 in range(0, NSEED, 50):
            r = rng.random((50, len(kk)))
            pk[s0:s0 + 50, kk] = flat[off[kk][None, :]
                                      + (r * lens[kk][None, :]).astype(np.int64)]
        return pk
    pk_i, pk_b = draw(f_i, off_i, len_i, keep_i), draw(f_b, off_b, len_b, keep_b)
    v = 0
    for pk in (pk_i, pk_b):
        kk = np.flatnonzero((pk >= 0).all(0))
        v += int((ind[tv[kk], pk[:, kk]] != ind[tv[kk], jv[kk]][None, :]).sum())
    print(f"锚点U1b 两套对照的行业违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}", flush=True)
    assert v == 0

    def ctrl_rate(pk, gi):
        q = pk[:, gi]
        g = q >= 0
        cm = np.where(g, lab1[tv[gi][None, :], np.maximum(q, 0)], False)
        cv = np.where(g, valid[tv[gi][None, :], np.maximum(q, 0)], False)
        nv = cv.sum(1)
        return float(np.nanmedian(np.where(nv > 0, cm.sum(1) / np.maximum(nv, 1),
                                           np.nan)))

    def scan(mask, tag):
        sp = p[mask]
        si = np.flatnonzero(mask.to_numpy())
        base = sp.L1.mean()
        print(f"\n{'='*104}\n{tag}:合格 {len(sp):,};全样本基准 {base:.2%}\n{'='*104}")
        print(f"{'变量':<24}{'阈值':<12}{'选中':>7}{'命中率':>8}{'市值分位':>9}"
              f"{'lift_base':>10}{'lift_ind':>10}{'lift_indmv':>11}")
        out = []
        for var, cands in cands_all.items():
            col = sp[colmap[var]].to_numpy()
            for nm, (lo, hi) in cands:
                m = (col >= lo) & (col < hi)
                if int(m.sum()) < 30:
                    continue
                gi = si[m]
                hr = float(sp.L1.to_numpy()[m].mean())
                c_i, c_b = ctrl_rate(pk_i, gi), ctrl_rate(pk_b, gi)
                mq = float(np.median(sp.mvq.to_numpy()[m]))
                r = {"段": tag, "变量": var, "阈值": nm, "选中": int(m.sum()),
                     "命中率": hr, "市值分位中位": mq, "lift_base": hr / base,
                     "lift_ind": hr / c_i if c_i > 0 else np.nan,
                     "lift_indmv": hr / c_b if c_b > 0 else np.nan}
                out.append(r)
                print(f"{var:<24}{nm:<12}{r['选中']:>7,}{hr:>8.2%}{mq:>9.0f}"
                      f"{r['lift_base']:>10.2f}{r['lift_ind']:>10.2f}"
                      f"{r['lift_indmv']:>11.2f}")
        return out

    res = scan(pd.Series(np.ones(len(p), bool)), "全样本 2015–2025")
    res += scan((p.ty >= Y_MOD[0]) & (p.ty <= Y_MOD[1]), "现代段 2019–2025")
    df = pd.DataFrame(res)
    print(f"\n{'='*104}\n市值维度贡献了多少(lift_ind − lift_indmv)\n{'='*104}")
    for tag, g in df.groupby("段", sort=False):
        dd = g["lift_ind"] - g["lift_indmv"]
        print(f"  {tag}:中位 {dd.median():+.2f}  最大 {dd.max():+.2f}  "
              f"lift_ind>1.2 的阈值数 {int((g.lift_ind > 1.2).sum())}/{len(g)}  "
              f"lift_indmv>1.2 的 {int((g.lift_indmv > 1.2).sum())}/{len(g)}")
    df.to_csv(f"{OUT}/ctrl_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/ctrl_sensitivity.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
