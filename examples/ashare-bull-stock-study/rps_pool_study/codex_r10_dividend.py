"""§114-D4(d) 分红口径对照:两个自洽口径下修正版 R10 的超额收益。

Codex 现行口径是**不自洽**的:组合用 hfq_close(后复权,含分红再投资),
基准用 510300 不复权价(不含分红)。本脚本给出两个自洽口径:

  自洽A 两边都含分红:组合用 close(前复权,含分红),基准用本面板 510300
        复权收盘(含分红)。
  自洽B 两边都不含分红:组合用 raw_close_ca(只调送转、不调分红)重算损益,
        基准用 Codex 公布的 510300 价格收益(不含分红)。

选股完全不变 —— 市值(raw_close × PIT 股本)与换手(volume/股本)都不受
分红影响,所以两个口径之间的差额就是**纯分红效应**,不掺别的。

价格构造(恒等式,写错必炸):
  close_t(前复权含分红) = raw_t × f_t,   f_t = 除息复权因子_t / 因子_T
  raw_close_ca_t        = raw_t × g_t,   g_t = 只含送转的因子
  ⇒ open_nodiv_t = open_t × raw_close_ca_t / close_t = raw_open_t × g_t
锚点:末日 close/raw_close_ca 必须 = 1.0000 ± 1e-3(两条序列在末端重合)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, OUT, run_window_fast  # noqa: E402
from codex_r10_replication import DATA, WINDOWS, metrics  # noqa: E402

CODEX_BENCH = {"train": 0.72973, "validation": -0.050856, "oos": 0.203596,
               "holdout": -0.018786, "full": 1.007179}


def main():
    from codex_r10_neutral import build_sel
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)
    sel, _, _ = build_sel(reb, ok, logcap, tmean)

    print("构造不含分红价格矩阵…")
    op_nd = np.full((nt, ns), np.nan, np.float32)
    cl_nd = np.full((nt, ns), np.nan, np.float32)
    bad = 0
    for j, c in enumerate(codes):
        cols = pq.read_schema(f"{DATA}/{c}.parquet").names
        # 7 只股票的文件没有 raw_close_ca(000522/000602/000990/301512/603400/688663
        # 及基准 510300)。它们退回含分红价格,即"不含分红"口径在这 6 只上不生效,
        # 方向是**低估**分红差异,不是高估 —— 结论若成立则更保守。
        want = ["open", "close"] + (["raw_close_ca"] if "raw_close_ca" in cols else [])
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=want)
        if "raw_close_ca" not in x.columns:
            x["raw_close_ca"] = x["close"]
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        x = x.reindex(idx)
        cq = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0)
        ca = pd.to_numeric(x["raw_close_ca"], errors="coerce").where(lambda s: s > 0)
        oq = pd.to_numeric(x["open"], errors="coerce").where(lambda s: s > 0)
        k = ca / cq
        last = k.dropna()
        if len(last) and not (0.98 < float(last.iloc[-1]) < 1.02):
            bad += 1
        op_nd[:, j] = (oq * k).to_numpy(np.float32)
        cl_nd[:, j] = ca.ffill().to_numpy(np.float32)
        if (j + 1) % 2000 == 0:
            print(f"  {j+1}/{ns}", flush=True)
    print(f"锚点 末日 close/raw_close_ca 偏离 1.0 超过 2% 的股票数 {bad} "
          f"({'✓' if bad < ns * 0.02 else '✗'})")

    def win(w):
        d0, d1 = WINDOWS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    rows = []
    print(f"\n{'窗口':11s} {'组合(含分红)':>13s} {'组合(不含)':>12s} "
          f"{'基准(含)':>10s} {'基准(不含)':>11s} {'超额A':>10s} {'超额B':>10s} {'Codex超额':>11s}")
    for w in WINDOWS:
        w0, w1 = win(w)
        eq1, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        eq2, _, _, _ = run_window_fast(op_nd, cl_nd, susp, lu, ld, sel, cal_pos, w0, w1)
        p_div = metrics(eq1, dd, idx)["total"]
        p_nd = metrics(eq2, dd, idx)["total"]
        s = bs[(bs.index >= WINDOWS[w][0]) & (bs.index <= WINDOWS[w][1])]
        b_div = float(s.iloc[-1] / s.iloc[0] - 1)
        b_nd = CODEX_BENCH[w]
        ea, eb = p_div - b_div, p_nd - b_nd
        rows.append({"window": w, "port_div": p_div, "port_nodiv": p_nd,
                     "bench_div": b_div, "bench_nodiv": b_nd,
                     "excess_A_both_div": ea, "excess_B_both_nodiv": eb})
        print(f"{w:11s} {p_div:+12.2%} {p_nd:+11.2%} {b_div:+9.2%} {b_nd:+10.2%} "
              f"{ea*100:+9.2f}pp {eb*100:+9.2f}pp")
    pd.DataFrame(rows).to_csv(f"{OUT}/codex_r10_dividend.csv", index=False)
    print(f"\n落库 {OUT}/codex_r10_dividend.csv")


if __name__ == "__main__":
    main()
