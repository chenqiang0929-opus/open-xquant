"""第一八三节:独立重算 Codex 交接包的行情窗口日期表(核验,不是新研究)。

Codex 2026-09-02 交接包里,《创业板四段行情_第一轮事实核对》和《阶段验证报告》
第 2 节给出同一张 T0–T4 表,**后面十几节的结论全部建在它上面**。
他的关键数据依赖 `159915.parquet` 已随包给到,哈希与清单逐字节一致
(`cbc8ec66…`),所以这张表是**整个包里我唯一能完全独立复算的东西**。

他的口径(原文照抄)
------------------
- 用 `159915` 创业板 ETF 日线作为创业板指数代理;
- 数据截止日固定 **2026-06-30**,不使用其后数据;
- MA20 / MA60 按**交易日日线收盘价**计算;
- **连续 3 个交易日**保持在均线上方/下方,**以第 3 日作为确认日**。

四个观察窗口(原文):
    2022 反弹  2022-04-01 → 2022-09-30
    2023 短反弹 2022-12-01 → 2023-05-31
    2024 反弹  2024-01-15 → 2024-06-30
    2025 行情  2025-03-15 → 2025-12-31

判据(跑之前写死;本节是核验,判据就是「能不能对上他的表」)
------------------------------------------------------------
R1 锚点:`159915.parquet` sha256 = `cbc8ec66d80b1c8bb1fbb69e560a5676f064229d803
   e8acddd88d543a9aeab8a`(与他 `文件清单_SHA256.csv` 一致);行数 3,291,
   覆盖 2013-01-04 → 2026-07-27,截断后末日 = 2026-06-30。
R2 **主判据**:16 个日期(4 窗口 × 4 个确认日)**逐个复现**。
   全中 = 他的地基可信;有出入 = 逐个列出差异与我算出的日期,**不改他的口径去凑**。
R3 描述:他 §3 提到的三个单日事件(2022-07-12 首次单日跌破 MA20、
   2022-08-24 单日同时跌破 MA20/MA60、2023-02-24 首次单日跌破 MA60)一并核。
R4 描述:他 §11 说全历史机械识别出 **44 个行情段**(短 ≤30 日 22 段 /
   中 31-60 日 13 段 / 长 >60 日 9 段,中位约 30 日)。他没写死合并规则,
   **本节按最直白的一种试算并如实报告是否为 44**,对不上只报数、不指控。

**本文件不构成任何投资建议。**
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

U = "/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd"
SRC = f"{U}/bd9532b1-159915.parquet"
SHA = "cbc8ec66d80b1c8bb1fbb69e560a5676f064229d803e8acddd88d543a9aeab8a"
END = "2026-06-30"
# (窗口名, 人工参考 T0, 观察范围末日)。**搜索起点是 T0,不是观察窗口首日** ——
# 首版我按窗口首日搜,只对上 9/16;那是我读错他的口径,不是他的错。
# 另外「首次三日跌破」他是从**站上 MA60 之后**起搜的(读法B),
# 按「各自站上日起搜」(读法A)则 2025 那格对不上。两种读法都试过,记在 R2 里。
WINS = (("2022反弹", "2022-04-27", "2022-09-30"),
        ("2023短反弹", "2023-01-03", "2023-05-31"),
        ("2024反弹", "2024-02-05", "2024-06-30"),
        ("2025行情", "2025-04-07", "2025-12-31"))
HIS = {"2022反弹": ("2022-05-13", "2022-06-08", "2022-07-21", "2022-08-26"),
       "2023短反弹": ("2023-01-09", "2023-01-09", "2023-02-17", "2023-02-28"),
       "2024反弹": ("2024-02-19", "2024-03-04", "2024-03-27", "2024-04-23"),
       "2025行情": ("2025-05-07", "2025-06-09", "2025-06-23", "2025-11-24")}


def run3(cond):
    """连续 3 日满足 cond,返回「第 3 日」为 True 的布尔序列。"""
    c = cond.astype(bool).to_numpy()
    out = np.zeros(len(c), bool)
    out[2:] = c[2:] & c[1:-1] & c[:-2]
    return pd.Series(out, index=cond.index)


def first(s, lo=None, hi=None):
    z = s
    if lo is not None:
        z = z[z.index >= lo]
    if hi is not None:
        z = z[z.index <= hi]
    z = z[z]
    return str(z.index[0].date()) if len(z) else None


def main():  # noqa: PLR0915
    h = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
    d = pd.read_parquet(SRC).sort_index()
    d.index = pd.DatetimeIndex(d.index)
    ok1 = (h == SHA and len(d) == 3291
           and str(d.index[0].date()) == "2013-01-04"
           and str(d.index[-1].date()) == "2026-07-27")
    print(f"锚点R1 sha256 {'✓' if h == SHA else '✗'};行数 {len(d)};"
          f"{d.index[0].date()} → {d.index[-1].date()} "
          f"{'✓' if ok1 else '✗ 作废'}")
    if not ok1:
        return
    d = d[d.index <= pd.Timestamp(END)]
    print(f"        截断后 {len(d)} 行,末日 {d.index[-1].date()}")

    c = d["close"]
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    up20, up60 = run3(c > ma20), run3(c > ma60)
    dn20, dn60 = run3(c < ma20), run3(c < ma60)

    w = 96
    print(f"\n{'='*w}\nR2 主判据:四窗口 × 四个确认日,逐个与他的表比对\n{'='*w}")
    print(f"{'窗口':<12}{'项':<16}{'他的表':<14}{'我算的':<14}{'':<6}")
    nok = 0
    rows = []
    for name, a, b in WINS:
        m20 = first(up20, a, b)
        m60 = first(up60, m20 or a, b)
        x20 = first(dn20, m60, b) if m60 else None      # 读法B:从站上MA60起搜
        x60 = first(dn60, m60, b) if m60 else None
        xa = first(dn20, m20, b) if m20 else None       # 读法A,仅供对照
        if xa != x20:
            print(f"  〔注〕{name}:读法A(从站上MA20起搜)得 {xa},"
                  f"说明 {m20} 与 {m60} 之间指数曾三日跌破 MA20 后又收复")
        for lab, mine, his in (("三日站上MA20", m20, HIS[name][0]),
                               ("三日站上MA60", m60, HIS[name][1]),
                               ("首次三日跌破MA20", x20, HIS[name][2]),
                               ("首次三日跌破MA60", x60, HIS[name][3])):
            same = (mine == his)
            nok += same
            rows.append({"窗口": name, "项": lab, "他": his, "我": mine,
                         "一致": same})
            print(f"{name:<12}{lab:<16}{his:<14}{str(mine):<14}"
                  f"{'✓' if same else '✗'}")
    print(f"\n**R2:16 个日期复现 {nok}/16 "
          f"{'—— 全中,他的地基可信' if nok == 16 else '—— 有出入,见上表'}**")

    print(f"\n{'='*w}\nR3 他 §3 提到的三个单日事件\n{'='*w}")
    s20, s60 = c < ma20, c < ma60
    e1 = first(s20, "2022-06-15", "2022-09-30")
    e2 = first(s20 & s60, "2022-07-25", "2022-09-30")
    e3 = first(s60, "2023-02-01", "2023-05-31")
    for lab, mine, his in (("2022 首次单日跌破MA20", e1, "2022-07-12"),
                           ("2022 单日同破MA20+MA60", e2, "2022-08-24"),
                           ("2023 首次单日跌破MA60", e3, "2023-02-24")):
        print(f"  {lab}:他 {his} / 我 {mine} {'✓' if mine == his else '✗'}")

    print(f"\n{'='*w}\nR4 全历史机械行情段(他 §11 报 44 段)\n{'='*w}")
    # 最直白的一种:三日站上 MA60 确认开段,三日跌破 MA60 确认收段,不合并
    seg, i, n = [], 0, len(c)
    u60 = up60.to_numpy()
    x60m = dn60.to_numpy()
    while i < n:
        if u60[i]:
            j = i + 1
            while j < n and not x60m[j]:
                j += 1
            seg.append((c.index[i], c.index[min(j, n - 1)], min(j, n - 1) - i))
            i = j + 1
        else:
            i += 1
    ln = np.array([x[2] for x in seg])
    if len(ln):
        print(f"  口径「三日站上MA60 开段 / 三日跌破MA60 收段,不合并」:"
              f"**{len(seg)} 段**(他 44)")
        print(f"    持续交易日 中位 {np.median(ln):.0f}(他约 30);"
              f"短≤30 {int((ln <= 30).sum())}(他 22)、"
              f"中31-60 {int(((ln > 30) & (ln <= 60)).sum())}(他 13)、"
              f"长>60 {int((ln > 60).sum())}(他 9)")
        hit = [k for k, (s_, _, _) in enumerate(seg)
               if str(s_.date()) in [HIS[x][1] for x in HIS]]
        print(f"    四个案例的「站上MA60」日期落在段首的个数:{len(hit)}/4")
    pd.DataFrame(rows).to_csv("/home/user/oxq-panel/codex_index_windows_check.csv",
                              index=False, encoding="utf-8-sig")
    print("\n落库 /home/user/oxq-panel/codex_index_windows_check.csv")
    print("本节是核验,不构成任何投资建议。")


if __name__ == "__main__":
    main()
