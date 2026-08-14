"""信号仪表盘:给人看的观察面板,不做买卖决策(生成 Markdown 报告)

═══ 定位 ═══
用户明确:「不是想做可以执行的系统,而是**带信号的系统**,
可以通过信号来观察,**人工判断是否买入**,你辅助的。」

所以本脚本**不给买卖建议**,只做三件事:
  ① 把闸门、池子、个股三层状态摆出来
  ② **每个信号旁边标注它在 §1~§74 里的证据等级** —— 这是本面板的核心价值
  ③ 把研究里被证伪的信号也一并列出并标红,防止「看着顺眼就用」

═══ 证据等级(全部来自已落库的事前判据检验) ═══
  ✅ 正向  该信号方向在完整控制下通过过事前判据
  ⛔ 负向  该信号作为**选股因子**被实测为负 alpha 或无超额
  ⬜ 未验证 本研究没测过 —— 不代表没用,代表**没有依据**

═══ 三层内容 ═══
  一、闸门层  510300 月线 MACD(§68/§71/§73)——全研究证据最强的一层
  二、池子层  池子规模与整体强度分布(体温计,非择股)
  三、个股层  逐只信号表,含**基地计数状态**(§71/§72 唯一通过检验的个股规则)

═══ 基地计数的口径 ═══
  自**本轮红轴起点**起算(不是自上市起算,也不是自任意低点):
    跟踪区间最高价 H;价格在 H 之下连续 ≥25 交易日后再创新高 = 完成一个基地。
  报出:当前第几个基地、距 H 回撤多少、是否已满 3 基。
  **§71/§72:满 3 基之前不因回撤卖出,是唯一通过检验的个股离场规则。**

═══ 数据口径 ═══
  面板最后交易日即报告基准日,**不是运行当日**。报告里会写明。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUTDIR = ("/home/user/open-xquant/examples/ashare-bull-stock-study/dashboard")
USER_CSV = ("/root/.claude/uploads/95a7873e-a420-5ffc-8d4d-fc8fba4ec34e/"
            "8b6acb64-___20260814.csv")
BASE_MIN, N_BASE = 25, 3
Y_LO, Y_HI = 365, 1095

t0 = time.time()
os.makedirs(OUTDIR, exist_ok=True)
cl, op, vo, ld, mv = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "close", "volume", "listed_days",
                                    "float_mv"])
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
VO = pd.DataFrame(vo).set_axis(CL.index)
LD = pd.DataFrame(ld).set_axis(CL.index)
MV = pd.DataFrame(mv).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
ASOF = idx[-1].date()
print(f"面板 {CL.shape}  基准日 {ASOF}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLf = CL.ffill()
CLa = CLf.to_numpy(float)
last = CLf.iloc[-1]
alive = CL.iloc[-1].notna()
MA100 = CLf.rolling(100, min_periods=100).mean().iloc[-1]
MA300 = CLf.rolling(300, min_periods=300).mean().iloc[-1]
HI250 = CLf.rolling(250, min_periods=100).max().iloc[-1]
V20 = VO.rolling(20, min_periods=10).mean().iloc[-1]
V60 = VO.rolling(60, min_periods=30).mean().iloc[-1]
RPS = {}
for n in (120, 250):
    RPS[n] = ((last / CLf.shift(n).iloc[-1] - 1).where(alive).rank(pct=True) * 100)

ym = idx.to_period("M")

# ── 闸门 ──
mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
mkc = mk["close"].reindex(idx).ffill()
mm = mkc.resample("ME").last()
dif = mm.ewm(span=12, adjust=False).mean() - mm.ewm(span=26, adjust=False).mean()
hist = dif - dif.ewm(span=9, adjust=False).mean()
hist.index = hist.index.to_period("M")
# 面板末月可能是未收月(交易日不足) —— 趋势与状态判断只用已收月
last_p = idx[-1].to_period("M")
n_td_last = int((ym == last_p).sum())
FULL = hist.index[:-1] if n_td_last < 15 else hist.index
hist_full = hist.loc[FULL]
sign = (hist_full > 0).astype(int)
chg = sign[sign.diff().fillna(0) != 0]
cur_start = chg.index[-1]
cur_red = bool(sign.iloc[-1])
nmon = len(sign.loc[cur_start:])
part_note = (f"（{last_p} 仅 {n_td_last} 个交易日，未收月，不参与判断）"
             if n_td_last < 15 else "")
red_start_td = int(np.flatnonzero(ym == cur_start)[0])

# ── 基地计数:自本轮闸门起点起算 ──
seg = CLa[red_start_td:]
run_hi = np.maximum.accumulate(np.where(np.isfinite(seg), seg, -np.inf), axis=0)
nb = np.zeros(NS, int)
below = np.zeros(NS, int)
for t in range(seg.shape[0]):
    c = seg[t]
    newhi = np.isfinite(c) & (c >= run_hi[t]) & (t > 0)
    nb += (newhi & (below >= BASE_MIN)).astype(int)
    below = np.where(newhi, 0, below + 1)
peak = pd.Series(np.where(np.isfinite(run_hi[-1]), run_hi[-1], np.nan), index=CL.columns)
dd_from_peak = (last / peak - 1) * 100
BASE_N = pd.Series(nb, index=CL.columns)
print(f"基地计数完成(自 {cur_start} 起,{seg.shape[0]} 个交易日)  ({time.time()-t0:.0f}s)")


def build(codes, label):
    d = pd.DataFrame(index=pd.Index(codes, name="code"))
    g = lambda s: d.index.map(s)                                     # noqa: E731
    d["收盘"] = g(last)
    d["上市天"] = g(LD.iloc[-1])
    d["流通市值亿"] = g(MV.iloc[-1]) / 1e8
    d["RPS250"] = g(RPS[250])
    d["RPS120"] = g(RPS[120])
    d["MA100"] = g(MA100)
    d["MA300"] = g(MA300)
    d["多头排列"] = d["MA100"] > d["MA300"]
    d["站上20周线"] = d["收盘"] > d["MA100"]
    d["量比20/60"] = g(V20) / g(V60)
    d["距250日新高%"] = (d["收盘"] / g(HI250) - 1) * 100
    d["本轮基地数"] = g(BASE_N)
    d["距区间高点%"] = g(dd_from_peak)
    d["满3基"] = d["本轮基地数"] >= N_BASE
    d["池"] = label
    return d


# 池一:滚动次新(研究口径)
roll = CL.columns[(LD.iloc[-1] >= Y_LO) & (LD.iloc[-1] < Y_HI) & alive]
D1 = build(roll, "滚动次新[1,3)年")
# 池二:用户观察池
up = pd.read_csv(USER_CSV, encoding="gbk", dtype=str)
up.columns = [c.strip() for c in up.columns]
up["code"] = up["代码"].str.zfill(6)
NAME = dict(zip(up["code"], up[up.columns[1]]))
IND = dict(zip(up["code"], up["细分行业"])) if "细分行业" in up else {}
ucodes = [c for c in up["code"] if c in CL.columns]
D2 = build(ucodes, "用户池(2022-08前后上市)")
print(f"滚动池 {len(D1)} 只 / 用户池 {len(D2)} 只(面板覆盖)  ({time.time()-t0:.0f}s)")

ALL = pd.concat([D1, D2])
ALL["名称"] = ALL.index.map(lambda c: NAME.get(c, ""))
ALL["细分行业"] = ALL.index.map(lambda c: IND.get(c, ""))
ALL.to_csv(f"{OUTDIR}/signals_{ASOF:%Y%m%d}.csv", encoding="utf-8-sig")


def health(d):
    n = len(d)
    if not n:
        return "—"
    return (f"{n} 只 · 多头排列 {d['多头排列'].mean():.0%} · "
            f"站上20周线 {d['站上20周线'].mean():.0%} · "
            f"距新高中位 {d['距250日新高%'].median():.0f}% · "
            f"满3基 {d['满3基'].mean():.0%}")


L = []
w = L.append
w(f"# 信号观察面板 · 基准日 {ASOF}")
w("")
w(f"> **这是观察面板，不是买卖建议。** 判断由你做，本表只负责把状态摆清楚，"
  f"并**在每个信号旁标注它在 §1~§74 里的证据等级**。")
w(f">")
w(f"> 数据基准日 **{ASOF}**（面板最后交易日），非运行当日。生成于 "
  f"{time.strftime('%Y-%m-%d %H:%M')}。")
w("")
w("## 证据等级说明")
w("")
w("| 记号 | 含义 |")
w("|---|---|")
w("| ✅ | 该方向在**完整控制**下通过过事前判据（市值中性化 + 同规模随机对照）|")
w("| ⛔ | 该信号作为**选股因子**被实测为负 alpha 或无超额 —— 看可以，据此排序不行 |")
w("| ⬜ | 本研究**没测过** —— 不代表没用，代表**没有依据** |")
w("")
w("---")
w("")
w("## 一、闸门层 ✅ 全研究证据最强的一层")
w("")
w(f"**当前：{'🔴 红轴' if cur_red else '🟢 绿轴'}，自 {cur_start} 起已 {nmon} 个月。**{part_note}")
w("")
w("| 月份 | MACD 柱 | 状态 |")
w("|---|---:|---|")
for p, v in hist.tail(15).items():
    tag = "🔴 红轴" if v > 0 else "🟢 绿轴"
    if p not in FULL:
        tag += f"（未收月，{n_td_last} 个交易日）"
    w(f"| {p} | {v:+.4f} | {tag} |")
w("")
recent = hist_full.tail(12)
slope = "衰减中" if recent.iloc[-1] < recent.iloc[0] else "走强中"
w(f"**柱值趋势：{slope}**（{recent.index[0]} {recent.iloc[0]:+.4f} → "
  f"{recent.index[-1]} {recent.iloc[-1]:+.4f}，仅用已收月）。{part_note}")
w("")
w("**证据**：§68 该规则机械复现四轮牛市，无事后调参。"
  "§73 实测：择时**不创造收益**（−0.70pp/年），但把最大回撤从 −64.8% 降到 −53.6%。")
w("")
w("**⚠️ 柱值高度本身没有被检验过** ⬜ —— §68/§73 只用了**符号**（正/负）。"
  "「柱在衰减所以该减仓」是一个未验证的推断，不是本研究的结论。")
w("")
w("---")
w("")
w("## 二、池子层 ⬜ 体温计，不是择股依据")
w("")
w("| 池 | 状态 |")
w("|---|---|")
w(f"| 滚动次新 [1,3) 年 ✅ | {health(D1)} |")
w(f"| 用户池（2022-08 前后上市）⛔ | {health(D2)} |")
w("")
w(f"**⛔ 用户池已超出验证窗口**：这批股票现已上市约 4 年（中位 "
  f"{D2['上市天']. median():.0f} 天），而 §69 证实的年龄优势窗口是 "
  f"**[365, 1095) 天**。那个「≥500% 密度是同档随机 2.73 倍」的结论"
  f"**不适用于这个池子**。")
w("")
w(f"**滚动池当前 {len(D1)} 只**，是研究口径下真正对应结论的池子。")
w("")
w("---")
w("")
w("## 三、个股层")
w("")
w("### 各信号的证据等级 —— 先看这张表再看数字")
w("")
w("| 信号 | 等级 | 依据 |")
w("|---|---|---|")
w("| 上市年龄落在 [1,3) 年 | ✅ | §69：同市值档内 ≥500% 密度 2.73 vs 1.36，p=0.0000 |")
w("| 本轮基地数 ≥ 3 | ✅ | §71/§72：满 3 基后才允许因回撤离场，超额 +3.0%，"
  "九格敏感性 9/9 显著 |")
w("| RPS250 / RPS120 排序 | ⛔ | §55~§61：作为买点组合级均不显著；"
  "§62：胜率提到最高时组合年化转负 |")
w("| 20/60 周线多头排列 | ⛔ | §57/§62/§63 三次独立确认：趋势作**选股因子**是负 alpha |")
w("| 站上 20 周线 | ⛔ | §70：作离场规则是六条里**最差**（兑现率 13%、实收 +2.4%/笔）|")
w("| 量比 | ⬜ | 本研究未单独检验 |")
w("| 距 250 日新高 | ⬜ | §64 测过「N 新高密度」不显著；本口径未单独测 |")
w("")
w("**怎么用这张表**：✅ 的两项可以当依据；⛔ 的三项**可以看，但不要据此排序选股**；"
  "⬜ 的两项是你的自由裁量区间 —— 用它们没问题，但要知道那是判断，不是证据。")
w("")

for d, nm, note in ((D1, "滚动次新池 [1,3) 年 ✅", "研究口径对应的池子"),
                    (D2, "用户观察池 ⛔", "已超出 [1,3) 年验证窗口，仅供观察")):
    sub = d[d["满3基"]].sort_values("距区间高点%", ascending=False)
    w(f"### {nm} —— 已满 3 基的标的（{len(sub)} / {len(d)} 只）")
    w("")
    w(f"*{note}。按「距区间高点」降序 —— 越靠上离高点越近。*")
    w("")
    if not len(sub):
        w("（本轮闸门起点至今，池内无标的完成 3 个基地）")
        w("")
        continue
    w("| 代码 | 名称 | 细分行业 | 收盘 | 上市天 | 基地数 | 距区间高点% | "
      "距250日新高% | RPS250 | 多头 | 量比 |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|:-:|---:|")
    for c, r in sub.head(40).iterrows():
        w(f"| {c} | {NAME.get(c,'')} | {IND.get(c,'')} | {r['收盘']:.2f} | "
          f"{r['上市天']:.0f} | {r['本轮基地数']:.0f} | {r['距区间高点%']:+.1f} | "
          f"{r['距250日新高%']:+.1f} | {r['RPS250']:.0f} | "
          f"{'✓' if r['多头排列'] else ''} | {r['量比20/60']:.2f} |")
    if len(sub) > 40:
        w(f"\n*（共 {len(sub)} 只，此处列前 40；完整表见 CSV）*")
    w("")

w("---")
w("")
w("## 四、本研究能替你回答与不能替你回答的")
w("")
w("**能回答的：**")
w("")
w("- 闸门当前是红是绿，以及这个规则历史上机械复现过哪几轮牛市（§68）")
w("- 择时值多少钱：**不创造收益，用 0.70pp/年买 11.1pp 最大回撤**（§73）")
w("- 池子该怎么定义：滚动 [1,3) 年，不是固定日期（§69）")
w("- 哪些个股信号是噪声：RPS、均线多头排列、站上 20 周线作**选股**用（§55~§63）")
w("- 唯一通过检验的个股规则：**满 3 基之前不因回撤卖出**（§71/§72）")
w("")
w("**不能回答的：**")
w("")
w("- 具体买哪几只 —— 右尾事前不可分辨（ML AUC 0.57，特征 lift ≈ 1.0）")
w("- 柱值高度该对应多少仓位 —— 未检验")
w("- 行业/主线怎么选 —— 未检验")
w("- 你的自由裁量加上去之后系统会变好还是变坏 —— **本研究无法评价人工判断**")
w("")
w("---")
w("")
w(f"*本面板由 `portfolio_layer/signal_dashboard.py` 生成 · 基准日 {ASOF} · "
  f"数据锚点 3,297 × 5,232 · 完整信号表见 `signals_{ASOF:%Y%m%d}.csv`*")

path = f"{OUTDIR}/dashboard_{ASOF:%Y%m%d}.md"
open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"\n→ {path}")
print(f"→ {OUTDIR}/signals_{ASOF:%Y%m%d}.csv   ({time.time()-t0:.0f}s)")
print(f"\n闸门 {'红轴' if cur_red else '绿轴'} 自 {cur_start} 起 {nmon} 个月")
print(f"滚动池 {len(D1)} 只,满3基 {D1['满3基'].sum()} 只")
print(f"用户池 {len(D2)} 只,满3基 {D2['满3基'].sum()} 只")
