"""第九十三节:宇通客车结案 —— 把这个案例身上所有实测值汇到一处

═══ 本节规格:描述性汇总,不设通过/不通过判据 ═══
与 §76 / §86 / §87 同规格。**本节不做新的假设检验、不重判、不翻案。**
§86~§92 已经把三段全部测完,本节只做一件事:
**把宇通身上每一个维度的实测值抽出来,放在同一张表里,回答用户最初那个问题。**

═══ 用户最初的问题(§86 起点,整个项目最好的一个提问)═══
> **「2023 年的宇通客车,我有没有办法在 2024-01 出现买入信号,
>   预测到未来 6 个月涨幅达到 100%?」**

═══ 本节计算什么 ═══
  A  宇通在 legacy 尺子下的全部亮灯月份与**全部突破日**(§89 口径,按日判定)
  B  每个突破日的 6 / 12 个月峰值与期末,以及**它在全样本 11,645 个
     legacy 突破事件里排第几个分位**
  C  2024-01-08 当天,**同市值五分位**的全体同侪的 6 个月峰值分布与宇通排名
  D  第一段体检:锚定强势日 `ts` 是哪天、当天距 250 日高多少、
     `[ts−60, ts]` 窗口内有没有创过新高(接 §90 / §92 的劈分口径)

═══ 锚点(不过则本节数字不可用;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② 四格事件数恒等复现 §89:19,704 / 12,161 / 10,861 / 6,676
     (同一管线,§90 / §92 已两次实证可达)
  ③ 宇通 legacy 突破日含 **2024-01-08**(§89 已预先验证)

**不设判据** —— 本节是结案陈述,不是检验。所有结论性判断都引用 §86~§92 已落库的判据结果。
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from consolidation_screener import (  # noqa: E402
    MIN_ADJ_FLOOR,
    MIN_ADJ_RATIO,
    Q_KEEP,
    THR_ATR,
    THR_DEPTH,
    THR_SHRINK,
    load_panel,
    score_one,
    series_of,
)

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NQ, BRK_CAP, WIN, CODE = 5, 250, 60, "600066"
HOR = [(120, "6个月"), (250, "12个月")]
EXP_EV = {("legacy", "状态"): 19704, ("legacy", "突破"): 12161,
          ("adaptive", "状态"): 10861, ("adaptive", "突破"): 6676}

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    k = list(CL.columns).index("510300")
    STRONG = np.delete(STRONG, k, axis=1)
    CL = CL.drop(columns=["510300"])
    MA100 = MA100.drop(columns=["510300"])
    frames.pop("510300", None)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

codes = list(CL.columns)
JY = codes.index(CODE)
SER = [series_of(frames, idx, c) for c in codes]
MAv = [MA100[c].to_numpy(float) for c in codes]
del frames
Fa = CL.where(CL > 0).ffill().to_numpy(float)
HI250 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(HI250) & (Fa >= HI250 * 0.9999)
GAP250 = (Fa / HI250 - 1.0).astype(np.float32)
del HI250
mvv = {c: pd.to_numeric(pd.read_parquet(f"{DATA}/{c}.parquet",
                                        columns=["float_mv"])["float_mv"],
                        errors="coerce") for c in codes}
MVa = pd.DataFrame(mvv).set_axis(idx).to_numpy(float)
del mvv
QUINT = np.full((NT, NS), -1, dtype=np.int8)
for t in range(NT):
    ok = np.isfinite(MVa[t]) & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if ok.sum() < 50:
        continue
    QUINT[t, ok] = np.searchsorted(np.nanquantile(MVa[t][ok], [.2, .4, .6, .8]),
                                   MVa[t][ok], side="right")
del MVa


def fwd(n, agg):
    if agg == "peak":
        m = pd.DataFrame(Fa[::-1]).rolling(n, min_periods=1).max().to_numpy(float)[::-1]
        out = np.full((NT, NS), np.nan)
        out[:-1] = m[1:]
    else:
        out = np.full((NT, NS), np.nan)
        out[:NT - n] = Fa[n:]
    out = (out / Fa - 1.0).astype(np.float32)
    out[NT - n:] = np.nan
    return out


PK = {n: fwd(n, "peak") for n, _ in HOR}
EN = {n: fwd(n, "end") for n, _ in HOR}
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = sorted(last_td)

ST = {"legacy": [], "adaptive": []}
SEG = {"legacy": {}, "adaptive": {}}
prev = {"legacy": set(), "adaptive": set()}
JY_LIT = []                                   # 宇通 legacy 亮灯月份明细

for mi, p in enumerate(months):
    t = last_td[p]
    sc_l, sc_a = {}, {}
    for j in range(NS):
        h, lo_, c_, v_ = SER[j]
        if not np.isfinite(c_[t]):
            continue
        sd = np.flatnonzero(STRONG[:t + 1, j])
        if sd.size == 0:
            continue
        s_l = score_one(h, lo_, c_, v_, MAv[j], sd, t, legacy=True)
        if s_l is not None:
            sc_l[j] = s_l
        s_a = score_one(h, lo_, c_, v_, MAv[j], sd, t, legacy=False)
        if s_a is not None:
            sc_a[j] = s_a
    if len(sc_a) < 50:
        continue
    adj = np.array([s["调整天数"] for s in sc_a.values()])
    floor = max(MIN_ADJ_FLOOR, int(round(MIN_ADJ_RATIO * np.median(adj))))
    thr = {k: float(np.nanquantile([s[k] for s in sc_a.values()], Q_KEEP))
           for k in ("缩量比", "收敛比", "深度")}
    hits = {
        "legacy": {j: s for j, s in sc_l.items()
                   if s["缩量比"] < THR_SHRINK and s["收敛比"] < THR_ATR
                   and s["深度"] <= THR_DEPTH},
        "adaptive": {j: s for j, s in sc_a.items()
                     if s["调整天数"] >= floor and s["缩量比"] <= thr["缩量比"]
                     and s["收敛比"] <= thr["收敛比"] and s["深度"] <= thr["深度"]},
    }
    jy_was_lit = JY in prev["legacy"]        # 必须在 prev 被覆盖之前取
    for r in ("legacy", "adaptive"):
        for j, s in hits[r].items():
            if j in prev[r]:
                continue
            ST[r].append((t, j))
            if np.isfinite(s["距区间高"]) and s["现价"] > 0:
                pk = s["现价"] / (1 + s["距区间高"])
                if np.isfinite(pk) and pk > 0:
                    SEG[r].setdefault(j, []).append((t, pk, int(s["_ts"])))
        prev[r] = set(hits[r])
    if JY in hits["legacy"]:
        s = hits["legacy"][JY]
        JY_LIT.append(dict(月=str(p), 新事件=not jy_was_lit,
                           调整天数=s["调整天数"], 深度=s["深度"],
                           缩量比=s["缩量比"], 收敛比=s["收敛比"],
                           现价=s["现价"], 区间高=s["现价"] / (1 + s["距区间高"]),
                           ts=str(idx[int(s["_ts"])].date())))
    if (mi + 1) % 40 == 0:
        print(f"  {p}  状态 L{len(ST['legacy']):,}/A{len(ST['adaptive']):,}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

BK = {"legacy": [], "adaptive": []}
for r in ("legacy", "adaptive"):
    for j, segs in SEG[r].items():
        col, cur = Fa[:, j], -1
        for t, pk, ts in sorted(segs):
            if t <= cur:
                continue
            w = np.flatnonzero(col[t + 1:min(t + 1 + BRK_CAP, NT)] > pk)
            if w.size:
                cur = t + 1 + int(w[0])
                BK[r].append((cur, j, t, ts))
CELL = {(r, k): v for r in ("legacy", "adaptive")
        for k, v in (("状态", ST[r]), ("突破", BK[r]))}
print("\n锚点② 四格事件数恒等复现 §89:", flush=True)
a2 = True
for key, want in EXP_EV.items():
    got = len(CELL[key])
    ok = got == want
    a2 &= ok
    print(f"  {'✓' if ok else '✗'} {key[0]}|{key[1]:<4} {got:>7,}  (§89 {want:,})")

W = 116
print(f"\n{'='*W}\nA 宇通 legacy 亮灯月份({len(JY_LIT)} 个月)\n{'='*W}")
print(f"{'月份':<10}{'调整天数':>8}{'深度':>8}{'缩量比':>8}{'收敛比':>8}"
      f"{'现价':>8}{'区间高':>8}   锚定强势日 ts")
for d in JY_LIT:
    print(f"{d['月']:<10}{d['调整天数']:>8}{d['深度']:>8.1%}{d['缩量比']:>8.2f}"
          f"{d['收敛比']:>8.2f}{d['现价']:>8.2f}{d['区间高']:>8.2f}   {d['ts']}")

jbk = sorted([(t, ts0, ts) for t, j, ts0, ts in BK["legacy"] if j == JY])
print(f"\n{'='*W}\nB 宇通 legacy 突破日({len(jbk)} 次)与全样本分位\n{'='*W}")
allbk = [(t, j) for t, j, _, _ in BK["legacy"]]
rows = []
for t, ts0, ts in jbk:
    r = dict(突破日=str(idx[t].date()), 状态月=str(ym[ts0]),
             收盘=float(Fa[t, JY]), 距状态月末交易日=t - ts0)
    for n, hn in HOR:
        pk, en = float(PK[n][t, JY]), float(EN[n][t, JY])
        pool = np.array([PK[n][tt, jj] for tt, jj in allbk])
        pool = pool[np.isfinite(pool)]
        r |= {f"{hn}峰值": pk, f"{hn}期末": en,
              f"{hn}分位": float((pool < pk).mean()) if np.isfinite(pk) else np.nan}
    rows.append(r)
    print(f"  {r['突破日']}  (状态月 {r['状态月']},距月末 {r['距状态月末交易日']:>2} 个交易日)"
          f"  收盘 {r['收盘']:.2f}")
    for _, hn in HOR:
        print(f"      {hn}  峰值 {r[hn+'峰值']:+7.1%}   期末 {r[hn+'期末']:+7.1%}"
              f"   全样本分位 **{r[hn+'分位']:.1%}**")
B = pd.DataFrame(rows)

T24 = int(np.flatnonzero(idx == pd.Timestamp("2024-01-08", tz=idx.tz))[0])
q = int(QUINT[T24, JY])
peers = np.flatnonzero((QUINT[T24] == q) & np.isfinite(PK[120][T24]))
pv = PK[120][T24, peers]
jy6 = float(PK[120][T24, JY])
print(f"\n{'='*W}\nC 2024-01-08 同市值五分位(第 {q} 档)同侪 {len(peers):,} 只 —— 6 个月峰值\n{'='*W}")
for lbl, val in [("中位", np.median(pv)), ("75 分位", np.percentile(pv, 75)),
                 ("90 分位", np.percentile(pv, 90)), ("99 分位", np.percentile(pv, 99))]:
    print(f"  {lbl:<8}{val:+8.1%}")
print(f"  ≥100% 的比例  **{(pv >= 1.0).mean():.2%}**   ({int((pv >= 1.0).sum())} / {len(peers):,} 只)")
print(f"  **宇通 {jy6:+.1%}   排名 {int((pv > jy6).sum())+1} / {len(peers):,}"
      f"   分位 {float((pv < jy6).mean()):.1%}**")

print(f"\n{'='*W}\nD 第一段体检(接 §90 / §92 的劈分口径)\n{'='*W}")
for t, ts0, ts in jbk:
    lo = max(ts - WIN, 0)
    print(f"  突破日 {idx[t].date()}  锚定强势日 ts = {idx[ts].date()}"
          f"   ts 当天距 250 日高 **{float(GAP250[ts, JY]):+.1%}**"
          f"   ts 当天是新高: {'是' if NEWHI[ts, JY] else '否'}"
          f"   [ts−{WIN}, ts] 窗口内创过新高: **{'是' if NEWHI[lo:ts+1, JY].any() else '否'}**")

print(f"\n{'='*W}\n锚点核对\n{'='*W}")
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 四格事件数恒等复现 §89")
a3 = "2024-01-08" in [str(idx[t].date()) for t, _, _ in jbk]
print(f"  {'✓' if a3 else '✗'} 锚点③ 宇通 legacy 突破日含 2024-01-08")
print("\n  本节不设判据(结案陈述,非检验)。")

B.to_csv(f"{OUT}/case_yutong_closure.csv", index=False)
pd.DataFrame(JY_LIT).to_csv(f"{OUT}/case_yutong_lit.csv", index=False)
print(f"\n→ {OUT}/case_yutong_closure.csv + _lit.csv   ({time.time()-t0:.0f}s)")
