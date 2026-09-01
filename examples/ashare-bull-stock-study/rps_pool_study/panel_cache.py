"""面板派生矩阵的磁盘缓存 —— **纯记忆化,不改任何计算值**。

来由
----
出一次清单要读 5,233 个 parquet **三遍**(主面板 / `load_panel` 的平台面板 /
`build_fund` 的财务面板),实测 308s,其中三段各约 100s。
换个观察日或换个池子就得重来一遍。本模块把这三段的**输出**存成 npz,
第二次起直接读盘。

**设计原则(重要)**
------------------
1. **只缓存输出,不复制逻辑。** 每一块的计算仍然由原函数完成
   (`load_panel` / `vec_screen` / `build_fund`),本模块只负责存和取。
   这样缓存永远不会与实现漂移。
2. **指纹失效。** 缓存键包含:面板目录、目录下 parquet 的**个数 + 总字节数 +
   最大 mtime**、以及本模块的 `SCHEMA` 版本号。面板一变、schema 一改,自动重算。
3. **dtype 逐字保留。** 尤其 `phi/plo` 必须是 **float64** ——
   第一五五节的真 bug 就是 float32 在「打平」处把等于误判成突破,虚增留出段约 4pp。
   本模块存取一律 `np.save` 原 dtype,不做任何降精度。
4. **可校验。** `OXQ_CACHE_VERIFY=1` 时,读缓存后**再算一遍**并逐点比对,
   不一致即报错。用于改动后自查,不是日常路径。

用法
----
    from panel_cache import cached
    cl, okm, ... = cached("panel", DATA, lambda: 原来的那段计算())
"""

from __future__ import annotations

import glob
import hashlib
import os
import time

import numpy as np

SCHEMA = "v1"
CACHE_DIR = os.environ.get("OXQ_CACHE_DIR", "/home/user/oxq-cache")
VERIFY = os.environ.get("OXQ_CACHE_VERIFY", "") == "1"


def fingerprint(panel_dir: str) -> str:
    """目录指纹:parquet 个数 + 总字节 + 最大 mtime。任何一项变了就换 key。"""
    fs = sorted(glob.glob(os.path.join(panel_dir, "*.parquet")))
    n = len(fs)
    size = sum(os.path.getsize(f) for f in fs)
    mt = max((os.path.getmtime(f) for f in fs), default=0.0)
    h = hashlib.sha256(f"{SCHEMA}|{panel_dir}|{n}|{size}|{mt:.0f}".encode()).hexdigest()
    return h[:16]


def _path(tag: str, panel_dir: str, extra: str = "") -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = fingerprint(panel_dir) + (("_" + extra) if extra else "")
    return os.path.join(CACHE_DIR, f"{tag}_{key}.npz")


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) != isinstance(b, np.ndarray):
        return False
    if not isinstance(a, np.ndarray):
        return bool(a == b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def cached(tag: str, panel_dir: str, build, extra: str = ""):
    """build() 返回一个 {名字: ndarray} 的 dict;命中则从盘读回同样的 dict。

    **dtype 与形状逐字保留**,读回的每个数组与 build() 的输出逐点相等
    (含 NaN 位置)—— `OXQ_CACHE_VERIFY=1` 时会真的比对一遍。
    """
    p = _path(tag, panel_dir, extra)
    if os.path.exists(p) and not VERIFY:
        t0 = time.time()
        z = np.load(p, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        print(f"[cache] 命中 {tag}{('/' + extra) if extra else ''} "
              f"({time.time()-t0:.1f}s,{os.path.getsize(p)/1e6:.0f}MB)", flush=True)
        return out
    t0 = time.time()
    out = build()
    assert isinstance(out, dict), f"{tag}: build() 必须返回 dict"
    if os.path.exists(p) and VERIFY:
        z = np.load(p, allow_pickle=True)
        bad = [k for k in out if k not in z.files or not _same(out[k], z[k])]
        assert not bad, f"[cache] 校验不过 {tag}:{bad}"
        print(f"[cache] 校验通过 {tag}:{len(out)} 个数组逐点相等", flush=True)
        return out
    tmp = p + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, p)
    print(f"[cache] 新建 {tag}{('/' + extra) if extra else ''} "
          f"({time.time()-t0:.0f}s,{os.path.getsize(p)/1e6:.0f}MB)", flush=True)
    return out
