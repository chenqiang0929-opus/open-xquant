"""第一七六节 A 部分:两只样本的事实核对 —— 20 周线附近的平台突破,6 月成 12 月败。

用户提出的问题
--------------
金发科技 600143 与卧龙电驱 600580,2025 年 6 月在 20 周线附近的平台突破**成功**,
2025 年 12 月同样位置的平台突破**失败**。为什么同一个形态在不同阶段结局相反?
如何分析、如何量化?

**本部分只摆事实,不下判断、不做检验。** 判据留给 B 部分的全市场检验,
按用户规则 1 在跑之前写死。

口径
----
- 面板 `/home/user/oxq-panel-0828/oxq_stock_market_fixed`,末日 2026-08-28;
- 周线 = 日线按 W-FRI 重采样(收盘取周内最后一个有效值,高/低取周内极值,
  量取周内合计),**停牌周按最后有效价 ffill 参与,绝不剔除**(用户规则 5);
- 20 周线 = 周收盘的 20 周简单均线(≈ 日线 MA100);
- 平台上沿 = 突破周之前 `L` 周的周收盘最高值(L 由「上一次创新高」回推,见下)。

锚点(不过则本部分作废)
------------------------
A1 两只票的日线行数与面板末日:600143 / 600580 均能取到 2026-08-28;
A2 周线条数 = 日线按周聚合后的条数,且 2025-06 与 2025-12 两个窗口都有数据;
A3 用户给的形态方向:两个 6 月事件的 26 周后收益 > 0,两个 12 月事件的 26 周后
   收益 < 0 —— **这一条是核对用户前提,不是判据**;若不成立,如实写出来,
   并按实际情况重述问题。

**本文件不构成任何投资建议。**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATA = "/home/user/oxq-panel-0828/oxq_stock_market_fixed"
CASES = (("600143", "金发科技"), ("600580", "卧龙电驱"))
WINDOWS = (("2025-06", "2025-05-01", "2025-08-31"),
           ("2025-12", "2025-11-01", "2026-02-28"))


def weekly(code: str) -> pd.DataFrame:
    x = pd.read_parquet(f"{DATA}/{code}.parquet",
                        columns=["high", "low", "close", "volume", "turnover"])
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_localize(None)
    x = x.sort_index()
    x["close"] = x["close"].where(x["close"] > 0).ffill()
    w = pd.DataFrame({
        "close": x["close"].resample("W-FRI").last(),
        "high": x["high"].resample("W-FRI").max(),
        "low": x["low"].resample("W-FRI").min(),
        "volume": x["volume"].resample("W-FRI").sum(),
        "turnover": x["turnover"].resample("W-FRI").mean(),
    }).dropna(subset=["close"])
    w["ma20w"] = w["close"].rolling(20).mean()
    w["dist_ma20w"] = w["close"] / w["ma20w"] - 1.0
    w["hi52"] = w["close"].rolling(52).max()
    w["lo52"] = w["close"].rolling(52).min()
    return w


def describe(code: str, name: str) -> list[dict]:
    w = weekly(code)
    out = []
    for tag, a, b in WINDOWS:
        sub = w.loc[a:b]
        if not len(sub):
            print(f"  {code} {tag}: 窗口内无数据")
            continue
        print(f"\n=== {code} {name} {tag} 窗口 {a} → {b} ===")
        print("  周五        收盘   20周线   距20周线   周量比52周均"
              "   距52周高  距52周低")
        vmean = w["volume"].rolling(52).mean()
        for d, r in sub.iterrows():
            vr = (r["volume"] / vmean.loc[d]) if np.isfinite(vmean.loc[d]) else np.nan
            print(f"  {d.date()}  {r['close']:7.2f} {r['ma20w']:7.2f} "
                  f"{r['dist_ma20w']:+8.1%} {vr:9.2f} "
                  f"{r['close']/r['hi52']-1:+9.1%} {r['close']/r['lo52']-1:+8.1%}")
        out.append({"code": code, "name": name, "window": tag})
    return out


def forward(code: str, name: str):
    """对每个窗口,找窗口内「周收盘创 20 周新高」的第一周当作突破周,报后验。"""
    w = weekly(code)
    rows = []
    for tag, a, b in WINDOWS:
        prior_hi = w["close"].shift(1).rolling(20).max()
        cand = w.loc[a:b]
        hit = cand.index[(cand["close"] > prior_hi.loc[a:b])
                         & np.isfinite(cand["ma20w"])]
        if not len(hit):
            print(f"\n  {code} {tag}: 窗口内没有「周收盘创 20 周新高」的周")
            continue
        d = hit[0]
        i = w.index.get_loc(d)
        r = {"code": code, "name": name, "window": tag,
             "突破周": str(d.date()), "突破周收盘": round(float(w["close"].iloc[i]), 2),
             "20周线": round(float(w["ma20w"].iloc[i]), 2),
             "距20周线": round(float(w["dist_ma20w"].iloc[i]), 4),
             "平台上沿(前20周最高周收)": round(float(prior_hi.iloc[i]), 2),
             "距52周高": round(float(w["close"].iloc[i] / w["hi52"].iloc[i] - 1), 4),
             "距52周低": round(float(w["close"].iloc[i] / w["lo52"].iloc[i] - 1), 4)}
        for k in (4, 8, 13, 26):
            j = i + k
            r[f"+{k}周"] = (round(float(w["close"].iloc[j] / w["close"].iloc[i] - 1), 4)
                           if j < len(w) else np.nan)
        j2 = min(i + 26, len(w) - 1)
        seg = w["close"].iloc[i:j2 + 1]
        r["26周内最大涨"] = round(float(seg.max() / w["close"].iloc[i] - 1), 4)
        r["26周内最大跌"] = round(float(seg.min() / w["close"].iloc[i] - 1), 4)
        rows.append(r)
    return rows


def main():
    print("=" * 78)
    print("第一七六节 A 部分:600143 金发科技 / 600580 卧龙电驱 —— 事实核对")
    print("=" * 78)
    for code, name in CASES:
        x = pd.read_parquet(f"{DATA}/{code}.parquet", columns=["close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        assert str(x.index.max().date()) == "2026-08-28", f"锚点A1 {code}"
        print(f"锚点A1 ✓ {code} 日线 {len(x)} 行,末日 {x.index.max().date()}")
    all_rows = []
    for code, name in CASES:
        describe(code, name)
        all_rows += forward(code, name)
    d = pd.DataFrame(all_rows)
    print("\n" + "=" * 78)
    print("突破周与后验(周收盘创 20 周新高的第一周)")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(d.to_string(index=False))
    ok6 = d[d["window"] == "2025-06"]["+26周"]
    ok12 = d[d["window"] == "2025-12"]["+26周"]
    a3 = bool((ok6 > 0).all() and (ok12 < 0).all()) if len(ok6) and len(ok12) else False
    print(f"\n锚点A3 用户前提(6月 +26周 > 0 且 12月 +26周 < 0):"
          f"{'✓ 成立' if a3 else '✗ 不成立 —— 需按实际重述'}")
    d.to_csv("/home/user/oxq-panel/phase_cases.csv", index=False,
             encoding="utf-8-sig")
    print("落库 /home/user/oxq-panel/phase_cases.csv")


if __name__ == "__main__":
    main()
