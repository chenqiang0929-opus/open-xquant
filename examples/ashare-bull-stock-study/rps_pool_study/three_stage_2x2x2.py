"""第八十八节:三段结构的 2×2×2 —— 尺子 × 第一段 × 突破确认(事前登记)

═══ 起因:用户把假设拆成三段,并指出了第三段 ═══
> ① 第一段上涨,肯定伴随 RPS>90,**且突破 250 日新高**(是不是共性)
> ② 第二段调整形成箱体/平台(是不是也存在共性)
> ③ **第三段必然需要突破箱体** —— **如果一直不突破,那就是横盘或下跌趋势**

**③ 是关键,而且 §87 的数据正好印证它**:§87 全样本 10,861 个「正在整理」事件,
6/12 个月的 ≥100% 比例与同市值随机**无差别**(5.62% vs 5.83%、16.02% vs 16.28%)。
原因就是用户说的:整理有三种结局(向上突破/继续横盘/向下破位),
**筛选器不区分**,抓到的是混合物。筛选器自己也写着「亮灯 42 天 —— 这是状态标记,不是买点」。

═══ 第三个轴:尺子 —— §87 挖出来的矛盾必须一起测 ═══
两把尺子对宇通 2023-2024 给出**完全相反**的判定:

    绝对阈值(legacy)     **42 天**三条全中,首次 2023-10-17
    横截面分位(adaptive)  **0 天** —— 整段没有一天

§61 覆盖表早记着这个形状:宇通 A旧版 291→C新版 23 天(且在 2015 年);
生益 76→**0 天**,判定栏写着「新版丢了」。
**§61 的分位改造修好三处旧毛病,代价是丢掉用户最在意的两个案例。**
**§87 因此测错了对象。本节把尺子当成一个轴,不预设哪把对。**

═══ 2×2×2 = 8 格 ═══
  尺子    legacy(绝对阈值 0.80/0.80/0.352,下限 15 日) / adaptive(当期 40% 分位,自适应下限)
  第一段  A = RPS60>90(现有实现) / B = RPS60>90 **且当日创 250 日新高**(用户版,更严)
  第三段  不要求突破(=§87 口径) / **要求突破**

**突破事件的 PIT 安全定义**:上月末三条全中,**本月末 `距区间高` ≥ 0**
—— 即整理之后确认站上区间高点,入场在确认当月末。**不含前视。**

═══ 一个必须事前声明的偏差(不是事后补的) ═══
「突破」本身就是**已经涨了**。所以突破格对「同市值随机」对照的 lift
**含动量成分**,不能全算作形态的功劳。
→ 本节对突破格**额外报一个对照 B**:同日、同市值档、**且当日也创 250 日新高**的随机股。
  **对照 B 才是隔离出「箱体突破」相对「一般新高」增量的那个。**
→ 但**判据统一用对照 A**(同市值随机),保证 8 格可比;对照 B 只作诊断。

═══ 锚点(不过则全节作废) ═══
  ① 面板 (3297, 5232)
  ② **宇通必须出现在 legacy 尺子、第一段 A、不要求突破 的新事件里,
     月份 ∈ {2023-10, 2023-11, 2023-12}**
     —— §86 用 **同一把 legacy 尺子**实测:42 天,首次 2023-10-17。
     **§87 栽在跨尺子取数,本节的锚点与被测格用的是同一把尺子。**
  ③ 对照零校验:对照 A 的「6个月 ≥100%」须落在全市场同期基础概率 ±3pp 内

═══ 事前判据(跑之前写死,不放宽) ═══
  **前置条件**:某格事件数 **< 300** 则该格不参与判据(沿用 §66 判据①门槛)
  ① 8 格中**至少一格**,其「6 个月 峰值≥100%」对同市值随机的
     **lift ≥ 1.3**(沿用 §77 门槛)**且 Bonferroni 校正后 p < 0.05/8 = 0.00625**

**为什么必须 Bonferroni**:本节同时看 8 格,不校正就是 §53 那个 best-of-N 陷阱;
§85 恰恰栽在「多重比较不校正」上(六个 τ 各设 |z|<2,误杀率 26.5%)。
**这次校正的是判据侧,不是锚点侧 —— 锚点只有 3 项且各自独立可验,不做校正。**

═══ 判据自查(§79 正问 + §83 反问) ═══
**正问:什么会让它通过而不回答我的问题?**
→ 突破 = 已经涨了,lift 可能只是动量 → **堵法:对照 B 诊断 + 正文明写**。
→ 8 格搜索出一个假阳性 → **堵法:Bonferroni**。

**反问:什么会让它不通过而与问题无关?**
→ 突破格样本量被砍太狠 → **堵法:前置条件 n≥300,不足则该格不判,而非判负**。
→ 尺子选错导致测错对象(§87 的教训)→ **堵法:尺子是一个轴,两把都测**。

═══ 事前预测(写下以便被证伪) ═══
**① 不通过。** 但我预测**要求突破的四格会明显好于不要求突破的四格**
(峰值≥100% 至少高 3pp),**只是 lift 达不到 1.3 或过不了 Bonferroni**。
理由:§62「所有提高胜率的过滤器都在削右尾」—— 等突破确认会漏掉直接拉走不回头的那批,
而那批恰恰是右尾。**若①通过,说明「状态→事件」这个改动确实不同于加形状特征,我错了。**
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
NQ, NSEED, SEED = 5, 200, 20260814
H6 = 120
MIN_ADJ_LEGACY = 15
MIN_N = 300
NCELL = 8
ALPHA = 0.05 / NCELL

t0 = time.time()
CL, frames, STRONG_A, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    k = list(CL.columns).index("510300")
    STRONG_A = np.delete(STRONG_A, k, axis=1)
    CL = CL.drop(columns=["510300"])
    MA100 = MA100.drop(columns=["510300"])
    frames.pop("510300", None)
    print("  已剔除 510300(ETF),与全项目 universe 对齐")
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

codes = list(CL.columns)
SER = [series_of(frames, idx, c) for c in codes]
MAv = [MA100[c].to_numpy(float) for c in codes]
del frames
Fa = CL.where(CL > 0).ffill().to_numpy(float)
# 第一段 B:RPS60>90 **且当日创 250 日新高**
HI250 = CL.where(CL > 0).ffill().rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(HI250) & (Fa >= HI250 * 0.9999)
STRONG_B = STRONG_A & NEWHI
print(f"强势日:A {STRONG_A.sum():,}  B(且创250日新高) {STRONG_B.sum():,}"
      f"  ({time.time()-t0:.0f}s)")

mvv = {}
for c in codes:
    mvv[c] = pd.to_numeric(pd.read_parquet(f"{DATA}/{c}.parquet",
                                           columns=["float_mv"])["float_mv"],
                           errors="coerce")
MVa = pd.DataFrame(mvv).set_axis(idx).to_numpy(float)
del mvv
print(f"市值载入完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = sorted(last_td)


def scan(strong, tag):
    """一次扫描同时产出 legacy 与 adaptive 两把尺子的事件。"""
    prev = {"legacy": set(), "adaptive": set()}
    ev = {f"{r}|{b}": [] for r in ("legacy", "adaptive")
          for b in ("nobrk", "brk")}
    for mi, p in enumerate(months):
        t = last_td[p]
        sc = {}
        for j in range(NS):
            h, lo_, c_, v_ = SER[j]
            if not np.isfinite(c_[t]):
                continue
            sd = np.flatnonzero(strong[:t + 1, j])
            if sd.size == 0:
                continue
            s_ = score_one(h, lo_, c_, v_, MAv[j], sd, t)
            if s_ is not None:
                sc[j] = s_
        if len(sc) < 50:
            continue
        adj = np.array([s["调整天数"] for s in sc.values()])
        floor = max(MIN_ADJ_FLOOR, int(round(MIN_ADJ_RATIO * np.median(adj))))
        thr = {k: float(np.nanquantile([s[k] for s in sc.values()], Q_KEEP))
               for k in ("缩量比", "收敛比", "深度")}
        hits = {
            "legacy": {j for j, s in sc.items()
                       if s["调整天数"] >= MIN_ADJ_LEGACY
                       and s["缩量比"] < THR_SHRINK and s["收敛比"] < THR_ATR
                       and s["深度"] <= THR_DEPTH},
            "adaptive": {j for j, s in sc.items()
                         if s["调整天数"] >= floor
                         and s["缩量比"] <= thr["缩量比"]
                         and s["收敛比"] <= thr["收敛比"]
                         and s["深度"] <= thr["深度"]},
        }
        for r in ("legacy", "adaptive"):
            for j in hits[r] - prev[r]:            # 上月未亮、本月亮
                ev[f"{r}|nobrk"].append((str(p), t, j))
            for j in prev[r]:                      # 上月亮、本月确认突破
                if j in sc and np.isfinite(sc[j]["距区间高"]) \
                        and sc[j]["距区间高"] >= 0:
                    ev[f"{r}|brk"].append((str(p), t, j))
            prev[r] = hits[r]
        if (mi + 1) % 40 == 0:
            print(f"  [{tag}] {p}  ({time.time()-t0:.0f}s)", flush=True)
    return ev


EV = {}
for tag, strong in (("A", STRONG_A), ("B", STRONG_B)):
    print(f"\n扫描第一段 {tag} …", flush=True)
    for k, v in scan(strong, tag).items():
        EV[f"{tag}|{k}"] = v
    print(f"  {tag} 完成:" + "  ".join(
        f"{k.split('|', 1)[1]} {len(v):,}" for k, v in EV.items() if k.startswith(tag)))


def pk6(t, j):
    if not np.isfinite(Fa[t, j]) or Fa[t, j] <= 0 or t + H6 >= NT:
        return np.nan
    seg = Fa[t + 1:t + H6 + 1, j]
    return np.nanmax(seg) / Fa[t, j] - 1 if len(seg) else np.nan


rng = np.random.default_rng(SEED)
rows = []
print(f"\n{'='*112}\n2×2×2 结果(6 个月峰值 ≥100%;对照 A = 同日同市值五分位随机)\n{'='*112}")
print(f"{'格':<28}{'事件数':>8}{'≥100%':>9}{'对照A':>9}{'lift':>7}{'p':>9}"
      f"{'对照B(仅突破格)':>16}")
for key in [f"{a}|{r}|{b}" for a in ("A", "B") for r in ("legacy", "adaptive")
            for b in ("nobrk", "brk")]:
    lst = [(t, j) for _, t, j in EV[key] if t + H6 < NT]
    if not lst:
        continue
    v = np.array([pk6(t, j) for t, j in lst])
    v = v[np.isfinite(v)]
    n = len(v)
    obs = float((v >= 1.0).mean()) if n else np.nan
    ca, cb = [], []
    for _ in range(NSEED):
        pa, pb = [], []
        for tt in sorted({t for t, _ in lst}):
            sub = [j for t, j in lst if t == tt]
            base = np.flatnonzero(np.isfinite(Fa[tt]) & (Fa[tt] > 0)
                                  & np.isfinite(MVa[tt]))
            if len(base) < 50:
                continue
            mv = MVa[tt][base]
            q = np.nanquantile(mv, np.linspace(0, 1, NQ + 1)[1:-1])
            nh = base[NEWHI[tt][base]]
            for jj in sub:
                b = int(np.searchsorted(q, MVa[tt, jj]))
                lo2 = -np.inf if b == 0 else q[b - 1]
                hi2 = np.inf if b >= NQ - 1 else q[b]
                band = base[(mv > lo2) & (mv <= hi2)]
                if len(band):
                    pa.append(pk6(tt, int(rng.choice(band))))
                if key.endswith("brk") and len(nh):
                    pb.append(pk6(tt, int(rng.choice(nh))))
        pa = np.array(pa)
        pa = pa[np.isfinite(pa)]
        if len(pa):
            ca.append(float((pa >= 1.0).mean()))
        pb = np.array(pb)
        pb = pb[np.isfinite(pb)]
        if len(pb):
            cb.append(float((pb >= 1.0).mean()))
    ca = np.array(ca)
    ra = float(np.median(ca)) if len(ca) else np.nan
    pv = float((ca >= obs).mean()) if len(ca) else np.nan
    rb = float(np.median(cb)) if len(cb) else np.nan
    lift = obs / ra if ra and ra > 0 else np.nan
    print(f"{key:<28}{n:>8,}{obs:>9.2%}{ra:>9.2%}{lift:>7.2f}{pv:>9.4f}"
          + (f"{rb:>16.2%}" if np.isfinite(rb) else f"{'—':>16}"))
    rows.append(dict(格=key, 事件数=n, ge100=obs, 对照A=ra, lift=lift, p=pv,
                     对照B=rb))
R = pd.DataFrame(rows)

print(f"\n{'='*112}\n锚点核对(不过则全节作废)\n{'='*112}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
yj = codes.index("600066")
yt = [m for m, t, j in EV["A|legacy|nobrk"]
      if j == yj and m in ("2023-10", "2023-11", "2023-12")]
a2 = len(yt) > 0
print(f"  {'✓' if a2 else '✗'} 锚点② 宇通在 legacy|A|nobrk 的 2023-10~12 新事件里"
      + (f"({yt[0]})" if a2 else ""))
if not a2:
    bad.append("锚点②")
base_all = []
for tt in sorted({t for _, t, j in EV["A|adaptive|nobrk"]}):
    if tt + H6 >= NT:
        continue
    b = np.flatnonzero(np.isfinite(Fa[tt]) & (Fa[tt] > 0))
    s = np.array([pk6(tt, int(x)) for x in b])
    s = s[np.isfinite(s)]
    if len(s):
        base_all.append(float((s >= 1.0).mean()))
b100 = float(np.mean(base_all))
ref = R[R["格"] == "A|adaptive|nobrk"]["对照A"].iloc[0]
a3 = abs(ref - b100) <= 0.03
print(f"  {'✓' if a3 else '✗'} 锚点③ 对照零校验:对照A {ref:.2%} vs 全市场基础概率 "
      f"{b100:.2%}(容差 3pp)")
if not a3:
    bad.append("锚点③")

print(f"\n{'='*112}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*112}")
elig = R[R["事件数"] >= MIN_N]
print(f"  前置条件:事件数 ≥{MIN_N} 的格 {len(elig)}/{len(R)}"
      + ("(以下不足 300 的格不参与判据:"
         + ", ".join(R[R['事件数'] < MIN_N]['格']) + ")" if len(elig) < len(R) else ""))
win = elig[(elig["lift"] >= 1.3) & (elig["p"] < ALPHA)]
c1 = len(win) > 0
print(f"  {'✓' if c1 else '✗'} 判据① 至少一格 lift≥1.3 且 p<{ALPHA:.5f}"
      f"(Bonferroni 0.05/8)   {len(win)} 格")
for _, x in win.iterrows():
    print(f"       {x['格']}  lift {x['lift']:.2f}  p={x['p']:.4f}")
brk = R[R["格"].str.endswith("brk")]["ge100"].mean()
nob = R[R["格"].str.endswith("nobrk")]["ge100"].mean()
print(f"  诊断:要求突破四格均值 {brk:.2%} vs 不要求突破四格均值 {nob:.2%}"
      f"(事前预测:突破组至少高 3pp → {'命中' if brk - nob >= 0.03 else '未中'})")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1:
    print("  **结论:有格通过。事前预测被证伪 —— 我错了。**")
else:
    print("  **结论:8 格无一通过。三段完整版(含突破确认)仍不优于同市值随机。**")

R.to_csv(f"{OUT}/three_stage_2x2x2.csv", index=False)
print(f"\n→ {OUT}/three_stage_2x2x2.csv   ({time.time()-t0:.0f}s)")
