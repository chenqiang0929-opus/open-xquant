"""§148 冻结规则的 2019–2025 逐月选中清单(名单交付,非检验)。

规则(与第一四八节完全相同,一个数没改)
  距一年低点涨幅 ∈ 全市场当日前 30%
  且 换手加速(20日均换手 ÷ 60日均换手 − 1)∈ 全市场当日前 30%

输出每个月末被选中的股票,含名称、申万一级行业、两个条件值、流通市值,
以及**未来 60 日实际最大涨幅**与**是否启动(≥50%)**,供逐期核对。

**与第一四八节的口径差异(必须知道,否则会误读)**
--------------------------------------------------
第一四八节的 60 日去重是在**全部合格观察**上做的(一只股票在窗口内
只保留第一次出现,不论当时是否被选中);
**本清单只对被选中的记录去重**,所以同一只股票可以在多个月重复出现
(间隔 ≥60 日)。
因此本清单的整体启动率 **12.30%** 与第一四八节的 14.07% / 14.17%
**不是同一个口径,不能直接比较**。
本清单适合「每月看名单」这个用途;要看规则的统计效力,以第一四八节为准。

名称来源:quant-research-dev 牛股普查的 code→name(只读)+ 用户提供的次新股 xls;
两者合并后覆盖 97.0% 的行,其余留空只给代码。

**本脚本不设判据,是名单交付,不构成任何买入建议。**
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,"/home/user/open-xquant/examples/ashare-bull-stock-study/rps_pool_study")
from codex_r10_replication import DATA
from industry_neutral import build_industry

t0=time.time()
CEN="/home/user/quant-research-dev/research/bull-stock-census-2010-2025/data"
nm={}
for f in ["intrayear_gt100","multi_year_5x_10x","annual_gt100_main",
          "annual_gt100_listing_year","annual_gt100_delisted"]:
    try:
        x = pd.read_csv(f"{CEN}/{f}.csv", dtype=str)
        x.columns = [c.lstrip("\ufeff") for c in x.columns]
        if "code" in x.columns and "name" in x.columns:
            for c, n in zip(x.code.str.zfill(6), x.name, strict=True):
                if pd.notna(n):
                    nm.setdefault(c, n)
    except Exception as e:                                     # noqa: BLE001
        print(f, "skip", str(e)[:40])
try:
    px=pd.read_excel("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/f48a5b4d-___20260827.xls",dtype=str)
    px = px.rename(columns={px.columns[1]: "名称"})
    px["代码"] = px["代码"].str.zfill(6)
    for c, n in zip(px.代码, px.名称, strict=True):
        nm.setdefault(c, n)
except Exception as e:                                         # noqa: BLE001
    print("xls skip", e)
print(f"名称映射 {len(nm)} 只")
codes=[os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
       if os.path.basename(f)[:-8]!="510300"]
cols=["close","float_mv","turnover","volume","is_st","is_suspended","listed_days"]
d={c:{} for c in cols}
for c in codes:
    x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols)
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_localize(None)
    for k in cols:
        d[k][c] = x[k]
cldf = pd.DataFrame(d["close"]).sort_index()
idx = cldf.index
nt, ns = cldf.shape
assert (nt,ns)==(3297,5232)
def al(k, f=np.nan):
    return pd.DataFrame(d[k]).sort_index().reindex(
        index=idx, columns=cldf.columns).fillna(f)


mv = al("float_mv").to_numpy() / 1e8
trn = al("turnover")
ok=(~al("is_st",True).astype(bool).to_numpy()&~al("is_suspended",True).astype(bool).to_numpy()
    &(al("listed_days",0).to_numpy()>=250)&(al("volume",0).to_numpy()>0))
cl = cldf.where(cldf > 0).to_numpy(np.float64)
ok &= np.isfinite(cl)
ind, inames, nid = build_industry(list(cldf.columns), idx)
id2n = {v: k for k, v in nid.items()} if isinstance(nid, dict) else {}
lo250=pd.DataFrame(cl).rolling(250,min_periods=250).min().to_numpy()
t20 = trn.rolling(20, min_periods=10).mean().to_numpy()
t60 = trn.rolling(60, min_periods=30).mean().to_numpy()
with np.errstate(all="ignore"):
    rec=cl/np.where(lo250>0,lo250,np.nan)-1.0
    tacc=t20/np.where(t60>0,t60,np.nan)-1.0
fmax=pd.DataFrame(cl[::-1]).rolling(60,min_periods=1).max().to_numpy()[::-1]
fwd = np.full_like(cl, np.nan)
fwd[:-1] = fmax[1:]
with np.errstate(all="ignore"):
    up = fwd / np.where(cl > 0, cl, np.nan) - 1.0
me=pd.Series(np.arange(nt),index=idx).groupby([idx.year,idx.month]).last().to_numpy()
colnames = list(cldf.columns)
rows = []
last = {}
for t in me:
    t = int(t)
    if (t > nt - 61 or idx[t] < pd.Timestamp("2019-01-01")
            or idx[t] > pd.Timestamp("2025-12-31")):
        continue
    m = ok[t] & np.isfinite(rec[t]) & np.isfinite(tacc[t]) & np.isfinite(up[t])
    e = np.flatnonzero(m)
    if len(e) < 100:
        continue
    qr = pd.Series(rec[t, e]).rank(pct=True).to_numpy()
    qt = pd.Series(tacc[t, e]).rank(pct=True).to_numpy()
    sel = e[(qr >= 0.70) & (qt >= 0.70)]
    for j in sel:
        if t - last.get(j, -10**9) < 60:
            continue
        last[j] = t
        c = colnames[j]
        rows.append({"观察日":idx[t].date(),"代码":c,"名称":nm.get(c,""),
                     "申万一级":id2n.get(int(ind[t,j]),"") if ind[t,j]>=0 else "",
                     "距一年低点涨幅":round(float(rec[t,j]),4),
                     "换手加速":round(float(tacc[t,j]),4),
                     "流通市值亿":round(float(mv[t,j]),1),
                     "未来60日最大涨幅":round(float(up[t,j]),4),
                     "启动(>=50%)":bool(up[t,j]>=0.50)})
df = pd.DataFrame(rows).sort_values(["观察日", "换手加速"], ascending=[True, False])
p = f"{os.environ.get('OXQ_OUT_DIR', '/home/user/oxq-panel')}/startup_rule_roster_2019_2025.csv"
df.to_csv(p, index=False, encoding="utf-8-sig")
print(f"\n清单 {len(df):,} 行,{df.代码.nunique():,} 只股票,"
      f"{df.观察日.min()} → {df.观察日.max()}")
print(f"有名称的 {(df.名称!='').sum():,} 行 ({(df.名称!='').mean():.1%})")
print(f"启动率 {df['启动(>=50%)'].mean():.2%}")
print("\n按年:")
df["年"] = pd.to_datetime(df.观察日).dt.year
print(df.groupby("年").agg(选中次数=("代码","size"),股票数=("代码","nunique"),
                        启动率=("启动(>=50%)","mean")).assign(
      启动率=lambda x:(x.启动率*100).round(1)).to_string())
print(f"\n落库 {p}  ({time.time()-t0:.0f}s)")
