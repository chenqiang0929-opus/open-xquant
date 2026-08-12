"""用真实成交均价(amount/volume)重新标定 raw_close,替代公司行动表反推法。

═══ 为什么换方法 ═══
用户用宁德时代(300750)提供了一组可核对的真值(2021-11-30):
  不复权 680.00 / 后复权 682.66 / 前复权 348.86,三者月涨跌幅均 +6.38%

我用公司行动表反推的 raw_close = 673.42,**偏离真值 -0.97%**。追查发现
`corporate_actions.parquet` **漏了两笔现金分红**:
  2024-04-30 的 10派30.17(3.017元/股)、2025-01-24 的 10派12.30(1.230元/股)
两笔合计 4.247元/股,约占当时不复权价 1.4%,与观测到的 0.98% 缺口吻合。
**公式没错,是公司行动表不全**(它似乎只收录年报分红,漏掉特别分红/中期分红)。

而 `amount/volume` 是真实成交均价,与复权方式无关,**天然免疫公司行动表
的缺漏**。实测 k = (amount/volume)/close 在两次除权之间高度稳定
(300750 各段标准差仅约 1.2%),且在 2023-04-26 的 10转8 处从 1.9403
跌到 1.0712,比值 1.811 ≈ 1.8,与实际方案完全对上。
用该法估计 300750 在 2021-11-30 的不复权价 = **679.84,偏差 -0.024%**。

═══ 方法 ═══
  1. k(t) = (amount/volume) / close   —— 不复权/前复权 的直接观测,含日内噪音
  2. 分段:公司行动表除权日 ∪ **从 k 自身检测到的跳变点**
     (后者能捕捉公司行动表漏掉的事件,如上面那两笔)
  3. 每段取 k 中位数 → 分段常数的 k_step
  4. raw_close = close × k_step
  5. float_mv = raw_close × outstanding_share;bp_correct = BVPS / raw_close

**不做整体归一化**:理论上数据末期 factor=1、k 应等于 1,实测 300750 末段
k=1.0036,那 0.36% 是 VWAP 与收盘价的天然偏离。用末段去归一化反而把这个
偏差摊到全历史(实测会从 -0.024% 恶化到 -0.38%),所以保持原值。

**成交量单位不一致的处理**:已实测有 283 只(5.4%)标的的 volume 以"手"
计而非"股",其 k 会整体放大约 100 倍。按中位数判定并除以 100。

**兜底**:有效 VWAP 天数 < 60,或末段 k 偏离 1 超过 ±25%(说明该股 VWAP
不可信),则保留公司行动表反推的旧 raw_close,并在结果里标记。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
OTH = f"{SP}/mktdata_enriched_others"

JUMP_THRESH = 0.015     # |Δlog k| > 1.5% 视为除权跳变(段内噪音约1.2%)
MIN_SEG = 5             # 段内少于5天则并入相邻段
MIN_VALID = 60          # 有效VWAP天数下限

t0 = time.time()
ca = pd.read_parquet(f"{OTH}/corporate_actions.parquet")
ca["ex_date"] = pd.to_datetime(ca["ex_date"]).dt.tz_localize("UTC")
ex_by_code = {c: sorted(set(g["ex_date"])) for c, g in ca.groupby("code", sort=False)}

files = sorted(glob.glob(f"{DATA}/*.parquet"))
print(f"标的文件: {len(files)}")

stats = []
n_vwap = n_fallback = n_err = 0

for i, p in enumerate(files):
    code = os.path.basename(p)[:-8]
    try:
        df = pd.read_parquet(p)
        close = pd.to_numeric(df["close"], errors="coerce")
        vol = pd.to_numeric(df["volume"], errors="coerce")
        amt = pd.to_numeric(df["amount"], errors="coerce")

        k = (amt / vol.where(vol > 0)) / close.where(close > 0)
        k = k.replace([np.inf, -np.inf], np.nan)
        valid = k.dropna()

        use_vwap = len(valid) >= MIN_VALID
        if use_vwap:
            # 成交量单位:以"手"计的会整体放大约100倍
            if valid.median() > 30:
                k = k / 100.0
                valid = valid / 100.0

            # ---- 分段:公司行动除权日 ∪ k 自身的跳变点 ----
            lk = np.log(k.where(k > 0))
            sm = lk.rolling(5, center=True, min_periods=2).median()
            jumps = set(sm.index[sm.diff().abs() > JUMP_THRESH])
            bounds = sorted({df.index[0]} | jumps |
                            {d for d in ex_by_code.get(code, []) if df.index[0] < d <= df.index[-1]})

            k_step = pd.Series(np.nan, index=df.index)
            edges = bounds + [df.index[-1] + pd.Timedelta(days=1)]
            for a, b in zip(edges[:-1], edges[1:]):
                seg = k.loc[(k.index >= a) & (k.index < b)].dropna()
                if len(seg) >= MIN_SEG:
                    k_step.loc[(k_step.index >= a) & (k_step.index < b)] = seg.median()
            k_step = k_step.ffill().bfill()

            # 末段 k 应≈1(数据末期无后续事件);偏离过大说明该股VWAP不可信
            tail = k_step.dropna()
            if tail.empty or not (0.75 <= tail.iloc[-1] <= 1.25):
                use_vwap = False

        if use_vwap:
            raw_new = close * k_step
            df["raw_close_ca"] = df["raw_close"]         # 保留旧方法结果备查
            df["raw_close"] = raw_new
            df["float_mv"] = raw_new * pd.to_numeric(df["outstanding_share"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df["bp_correct"] = (pd.to_numeric(df["book_value_per_share"], errors="coerce")
                                    / raw_new.replace(0, np.nan))
            df.to_parquet(p)
            n_vwap += 1
            old = pd.to_numeric(df["raw_close_ca"], errors="coerce")
            rel = (raw_new / old - 1).replace([np.inf, -np.inf], np.nan).dropna()
            stats.append({"code": code, "method": "vwap", "n_valid": len(valid),
                          "tail_k": float(tail.iloc[-1]),
                          "median_change": float(rel.median()) if len(rel) else np.nan})
        else:
            n_fallback += 1
            stats.append({"code": code, "method": "ca_fallback", "n_valid": len(valid),
                          "tail_k": np.nan, "median_change": 0.0})
    except Exception as e:
        n_err += 1
        if n_err <= 5:
            print(f"  [ERROR] {code}: {type(e).__name__}: {e}")
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)")

st = pd.DataFrame(stats)
st.to_csv(f"{SP}/refine_raw_close_stats.csv", index=False)
print(f"\n完成: VWAP法 {n_vwap}, 兜底(公司行动表) {n_fallback}, 出错 {n_err}, "
      f"耗时 {time.time()-t0:.0f}s")

v = st[st.method == "vwap"]
print(f"\n{'='*92}\n对旧方法的修正幅度\n{'='*92}")
print(f"  raw_close 变化中位数        : {v.median_change.median():+.3%}")
print(f"  |变化|>2%  的标的比例       : {(v.median_change.abs()>0.02).mean():.1%}")
print(f"  |变化|>10% 的标的比例       : {(v.median_change.abs()>0.10).mean():.1%}")
print(f"  末段 k 中位数(理论应≈1.00): {v.tail_k.median():.4f}")

# ---------------- 用户提供的真值核对 ----------------
print(f"\n{'='*92}\n真值核对: 宁德时代 300750 @ 2021-11-30\n{'='*92}")
d = pd.read_parquet(f"{DATA}/300750.parquet",
                    columns=["close", "raw_close", "raw_close_ca", "float_mv", "outstanding_share"])
d.index = d.index.tz_localize(None)
r = d.loc["2021-11-30"]
print(f"  不复权 raw_close  = {r['raw_close']:8.2f}   雪球真值 680.00   偏差 {r['raw_close']/680-1:+.3%}")
print(f"    (旧公司行动表法 = {r['raw_close_ca']:8.2f}                    偏差 {r['raw_close_ca']/680-1:+.3%})")
print(f"  前复权 close      = {r['close']:8.2f}   雪球真值 348.86   偏差 {r['close']/348.86-1:+.3%}")
print(f"    (雪球含 2026-08-10 的 10派14.11,我的数据止于 2026-07-31,略高属正常)")
print(f"  总市值            = {r['float_mv']/1e12:8.4f} 万亿   雪球约 1.38 万亿")
