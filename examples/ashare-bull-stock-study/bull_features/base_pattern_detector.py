"""欧奈尔基底形态检测器:杯柄 / 平底 / 双底

═══ 为什么要重做 ═══
第五十一节测过「基底形态」,结论是 **lift 0.99 —— 没有信息量**
(牛股 39.3% 有「60日振幅<30%」,非牛股 39.8%)。
**但那个结论不能采信,因为测的根本不是基底形态。**
欧奈尔的杯柄有五个约束(前期涨幅、时长、深度、手柄位置、手柄缩量),
「60日振幅<30%」一个都没有 —— 它把所有横盘的都算进去了,
**包括阴跌到没人要的那批**(那些恰恰是最不可能出牛股的)。

═══ 参数取值:全部来自欧奈尔原书,跑之前锁定 ═══
一共 12 个参数。**不做网格搜索** —— 12 个参数的网格是过拟合机器。
敏感性表另跑另报,并明确标注「那不是检验」。

              杯柄              平底           双底
前期涨幅      ≥30%              ≥20%          ≥30%
基底时长      35~325日(7~65周)  ≥25日(5周)   ≥35日(7周)
深度          12~33%            ≤15%          15~35%
附加          手柄深8~15%、长≥5日、          第二低点下破第一低点
              手柄低点在基底上半部、
              手柄均量<基底均量
买点 pivot    手柄高点          基底高点      中间峰高点

═══ 无前视 ═══
`detect_base(...)` 只接收 **[t-W, t-1]** 的切片(**不含 t 当天**),
函数内部无法看到 t 及之后。调用处有断言。

═══ 两个必须说明的实现选择 ═══
1. **杯柄与双底的左沿 = 窗口内最高价那一根**(欧奈尔:「基底始于此前的高点」),
   基底长度随之确定。首版让起点在 [35,325] 上按步长 5 搜索、再要求
   h[s] ≈ 段内最高,结果 **14 个牛股案例一个杯柄都检不出** ——
   格点几乎不可能正好落在峰值那一根上。改后 8 个抽样案例全部形态正确
   (时长 47~160 日、深 24.8~33%、手柄 8.2~11.7%)。
   平底没有「左沿」概念,仍按步长 5 扫时长,取最长的。
2. 三种形态**各自独立判定**,一个窗口可以同时满足多种 ——
   归因时分开报,不做优先级合并。

═══ 一个由定义决定的结构性事实(不是 bug) ═══
在 t\\*(该年最大涨幅的**起点**)上检测,**只可能检出平底**:
杯柄的手柄要求价格已回到左沿 95%、且手柄低点在基底上半部,
而 t\\* 是低点。**杯柄/双底必须在突破日上测**(见 B 部分)。
"""
import numpy as np

# ── 锁定的参数(书里的值,跑之前写死) ──
CUP = dict(prior=0.30, lmin=35, lmax=325, dmin=0.12, dmax=0.33,
           hmin=0.08, hmax=0.15, hlen=5)
FLAT = dict(prior=0.20, lmin=25, lmax=130, dmax=0.15)
DBL = dict(prior=0.30, lmin=35, lmax=325, dmin=0.15, dmax=0.35)
STEP = 5          # 基底起点搜索步长(仅影响速度)


def _argmin(a):
    """nanargmin,遇到全 NaN 返回 -1(停牌股的切片会整段是 NaN)。"""
    return -1 if a.size == 0 or np.all(~np.isfinite(a)) else int(np.nanargmin(a))


def _argmax(a):
    return -1 if a.size == 0 or np.all(~np.isfinite(a)) else int(np.nanargmax(a))


def detect_base(c, h, l, v, prior_min):
    """在窗口 [0..W-1](= 原序列的 t-W..t-1)内找基底。

    返回 dict:三个独立布尔 + 各自的 pivot / 深度 / 起点下标。
    c/h/l/v/prior_min 等长一维数组。
    `prior_min[s]` = 窗口下标 s **之前** 250 日的最低收盘(调用方预先算好,
    用滚动 min 一次算完;放在这里逐候选 concatenate 会慢十几倍)。
    """
    out = {"cup": False, "flat": False, "dbl": False,
           "cup_pivot": np.nan, "flat_pivot": np.nan, "dbl_pivot": np.nan,
           "cup_start": -1, "flat_start": -1, "dbl_start": -1,
           "cup_depth": np.nan, "flat_depth": np.nan, "dbl_depth": np.nan,
           "cup_handle": np.nan}
    W = len(c)
    end = W - 1
    if W < FLAT["lmin"] + 1:
        return out
    ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(l) & (c > 0)
    if ok.sum() < W * 0.6:
        return out

    def prior_of(s):
        lo, cs = prior_min[s], c[s]
        if not (np.isfinite(lo) and lo > 0 and np.isfinite(cs) and cs > 0):
            return np.nan
        return cs / lo - 1

    # ── 基底左沿:欧奈尔的定义就是「基底始于此前的高点」 ──
    # 首版让 s 在 [35,325] 上按步长 5 搜索、再要求 h[s] ≈ 段内最高 ——
    # **14 个案例里一个杯柄都检不出**:格点几乎不可能正好落在峰值那一根上。
    # 改为直接把左沿定义成窗口内最高价所在的那一根,基底长度随之确定,
    # 既符合原书定义,又少一个搜索维度。
    hw = h[:end + 1]
    if np.all(~np.isfinite(hw)):
        return out
    rim = _argmax(hw)
    if rim < 0:
        return out

    # ── 杯柄 ──
    for _once in (0,):
        s = rim
        L = end - s + 1
        if not (CUP["lmin"] <= L <= CUP["lmax"]):
            continue
        hs, ls_, cs = h[s:end + 1], l[s:end + 1], c[s:end + 1]
        if not np.isfinite(h[s]) or np.nanmax(hs) <= 0:
            continue
        left = h[s]
        bot_i = _argmin(ls_)
        if bot_i < 0:
            continue
        bot = ls_[bot_i]
        if not np.isfinite(bot) or bot <= 0:
            continue
        depth = 1 - bot / left
        if not (CUP["dmin"] <= depth <= CUP["dmax"]):
            continue
        if not (prior_of(s) >= CUP["prior"]):
            continue
        # 右沿:杯底之后第一次回到左沿 95% 的位置
        after = hs[bot_i + 1:]
        rec = np.flatnonzero(after >= left * 0.95)
        if rec.size == 0:
            continue
        r = bot_i + 1 + int(rec[0])            # 窗口内下标
        if end - (s + r) + 1 < CUP["hlen"]:    # 手柄至少 5 天
            continue
        hh, hl = hs[r:], ls_[r:]
        htop = np.nanmax(hh)
        hlow = np.nanmin(hl)
        if not (np.isfinite(htop) and np.isfinite(hlow) and htop > 0):
            continue
        hdep = 1 - hlow / htop
        if not (CUP["hmin"] <= hdep <= CUP["hmax"]):
            continue
        if hlow <= bot + (left - bot) * 0.5:   # 手柄低点须在基底上半部
            continue
        vb, vh = v[s:s + r], v[s + r:end + 1]
        vb, vh = vb[np.isfinite(vb)], vh[np.isfinite(vh)]
        if vb.size == 0 or vh.size == 0 or vh.mean() >= vb.mean():
            continue                            # 手柄必须缩量
        out.update(cup=True, cup_pivot=float(htop), cup_start=s,
                   cup_depth=float(depth), cup_handle=float(hdep))
        break

    # ── 平底 ──
    for L in range(FLAT["lmax"], FLAT["lmin"] - 1, -STEP):
        s = end - L + 1
        if s < 0:
            continue
        hs, ls_ = h[s:end + 1], l[s:end + 1]
        top, bot = np.nanmax(hs), np.nanmin(ls_)
        if not (np.isfinite(top) and np.isfinite(bot) and top > 0 and bot > 0):
            continue
        depth = 1 - bot / top
        if depth > FLAT["dmax"]:
            continue
        if not (prior_of(s) >= FLAT["prior"]):
            continue
        out.update(flat=True, flat_pivot=float(top), flat_start=s,
                   flat_depth=float(depth))
        break

    # ── 双底(左沿同样取窗口内最高点) ──
    for _once in (0,):
        s = rim
        L = end - s + 1
        if not (DBL["lmin"] <= L <= DBL["lmax"]):
            continue
        hs, ls_ = h[s:end + 1], l[s:end + 1]
        if not np.isfinite(h[s]):
            continue
        left = h[s]
        n = len(ls_)
        half = max(2, n // 2)
        b1 = _argmin(ls_[:half])
        if b1 < 0 or b1 + 2 >= n:
            continue
        _p = _argmax(hs[b1 + 1:])
        if _p < 0:
            continue
        p = b1 + 1 + _p
        if p + 1 >= n:
            continue
        _b2 = _argmin(ls_[p + 1:])
        if _b2 < 0:
            continue
        b2 = p + 1 + _b2
        if not (ls_[b2] < ls_[b1]):            # 第二个低点必须下破第一个
            continue
        if hs[p] <= ls_[b1] + (left - ls_[b1]) * 0.4:   # 中间峰要够高
            continue
        depth = 1 - min(ls_[b1], ls_[b2]) / left
        if not (DBL["dmin"] <= depth <= DBL["dmax"]):
            continue
        if not (prior_of(s) >= DBL["prior"]):
            continue
        out.update(dbl=True, dbl_pivot=float(hs[p]), dbl_start=s,
                   dbl_depth=float(depth))
        break

    return out


WIN = CUP["lmax"]          # 检测窗口长度(取最长的形态)
PRIOR = 250                # 前期涨幅回看长度
NEED = WIN + PRIOR         # 调用方在 t 之前至少需要这么多历史
