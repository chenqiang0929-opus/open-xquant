#!/usr/bin/env bash
# 一条命令恢复面板(容器/机器换了之后)
#
# 起因:面板是**派生物**且位于 git 仓库之外,从未被提交;
# 而研究会话跑在临时容器里,闲置一段时间即被回收。
# 仓库能从 GitHub 拉回,面板不能 —— 已经发生三次(08-13 / 08-14 / 08-14 二次)。
#
# 本脚本把原先散在 README §0.1 的 8 步手工命令 + 验收数字固化下来。
# **每一步跑完立即核对锚点,对不上就退出非零,绝不带着错的面板往下跑。**
#
# 用法:  bash data_prep/recover_panel.sh
# 环境:  可用 SRC_REPO / OXQ_RESEARCH_DIR 覆盖默认路径
#
# ⚠️ 绝不重新下载行情 —— 重下的是另一条价格序列,锚点必然对不上,等于换了个面板。

set -euo pipefail

SRC_REPO="${SRC_REPO:-/workspace/etf-netflow-dev}"
SRC_URL="${SRC_URL:-https://github.com/chenqiang0929-opus/etf-netflow-dev}"
export OXQ_RESEARCH_DIR="${OXQ_RESEARCH_DIR:-/home/user/oxq-panel}"
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PANEL="$OXQ_RESEARCH_DIR/oxq_stock_market_fixed"

step() { echo; echo "═══ $* ═══"; }
fail() { echo "✗ $*" >&2; exit 1; }

# ══════════════════════════════════════════════════════════════════════════
# 0. 预检 —— **一次报出全部缺失**,不要让人一个个试错
#
# 这一段的由来:本脚本首版把「开发容器恰好具备的东西」当成了普遍前提 ——
# git-lfs 已装、open-xquant SDK venv 已缓存、etf-netflow-dev 已挂进 session。
# 换一个干净容器,三样全没有,而 `set -euo pipefail` 会在第 1/6 步直接退出,
# 报错还很难懂。**预检必须先于任何副作用,且一次报全。**
# ══════════════════════════════════════════════════════════════════════════
step "0/6 预检"
PROBLEMS=()

# ── git-lfs ──
if ! command -v git-lfs >/dev/null 2>&1; then
  PROBLEMS+=("git-lfs 未安装(源数据是 LFS 对象,约 791MB)
       修复:  apt-get update && apt-get install -y git-lfs && git lfs install
       或:    conda install -y -c conda-forge git-lfs && git lfs install")
else
  echo "  ✓ git-lfs  $(git lfs version | head -1)"
fi

# ── python:必须能 import pandas/numpy/pyarrow,光有解释器不算 ──
PY=""
for cand in "${OXQ_PYTHON:-}" \
            $(ls -d "$HOME"/.config/open-xquant/sdk-bundles/*/runner/.venv/bin/python 2>/dev/null) \
            "$(command -v python3 || true)"; do
  [ -n "$cand" ] && [ -x "$cand" ] || continue
  if "$cand" -c "import pandas, numpy, pyarrow" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  PROBLEMS+=("找不到带 pandas/numpy/pyarrow 的 python
       (光有 python3 不够 —— 重建脚本要读 parquet)
       修复:  python3 -m pip install pandas numpy pyarrow
       或:    uv pip install pandas numpy pyarrow
       或指定: OXQ_PYTHON=/path/to/python bash \$0")
else
  echo "  ✓ python   $PY"
  echo "             pandas $("$PY" -c 'import pandas;print(pandas.__version__)')"
fi

# ── 源仓库:没有就自动克隆(浅克隆,LFS 稍后单独拉) ──
if [ ! -d "$SRC_REPO/.git" ]; then
  echo "  · 源仓库不在 $SRC_REPO,尝试克隆 $SRC_URL"
  mkdir -p "$(dirname "$SRC_REPO")"
  if GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 "$SRC_URL" "$SRC_REPO" 2>&1 | tail -3; then
    echo "  ✓ 源仓库   已克隆到 $SRC_REPO"
  else
    PROBLEMS+=("无法克隆源仓库 $SRC_URL
       该仓库是私有的,需要本 session 有访问权。
       修复:  让 Agent 用 add_repo 把 chenqiang0929-opus/etf-netflow-dev 挂进 session,
              或手工克隆到 $SRC_REPO 后重跑,
              或用 SRC_REPO=/已有路径 bash \$0 指向已有 clone")
  fi
else
  echo "  ✓ 源仓库   $SRC_REPO"
fi

# ── 研究脚本 ──
for s in rebuild_price_data_fixed.py refine_raw_close_vwap.py fill_fundamentals.py \
         510300_hfq.parquet; do
  [ -f "$STUDY/data_prep/$s" ] || PROBLEMS+=("缺 data_prep/$s —— 仓库不完整?")
done

if [ ${#PROBLEMS[@]} -gt 0 ]; then
  echo
  echo "✗ 预检未通过,共 ${#PROBLEMS[@]} 项:"
  for i in "${!PROBLEMS[@]}"; do
    echo
    echo "  [$((i+1))] ${PROBLEMS[$i]}"
  done
  echo
  echo "  **全部修完再重跑本脚本。以上问题一次性列出,不必逐个试错。**"
  exit 1
fi
echo
echo "python : $PY"
echo "源仓库 : $SRC_REPO"
echo "工作区 : $OXQ_RESEARCH_DIR"
if [ -n "${PREFLIGHT_ONLY:-}" ]; then
  echo
  echo "✓ 预检通过(PREFLIGHT_ONLY 已设,到此为止)。去掉该变量即执行完整恢复。"
  exit 0
fi

# ── 1. 取源数据 ────────────────────────────────────────────────────────────
# 坑 1:本地 checkout 可能是**浅克隆且停在旧 commit**,`ls` 看不到 mktdata_enriched。
#       **不要据此判断数据不存在** —— 先 fetch 看远端。
# 坑 2:`git lfs pull` 只对 HEAD 生效。只 `git checkout FETCH_HEAD -- <路径>` 会得到
#       一堆 133 字节的指针文件,而 `git lfs pull` 会**静默什么都不做**(不报错)。
#       必须先 `git checkout --detach FETCH_HEAD`。
step "1/6 取源数据(fetch → detach → lfs pull)"
cd "$SRC_REPO"
git fetch --depth=1 origin main
git checkout --detach FETCH_HEAD
git lfs pull
[ -d mktdata_enriched ] || fail "mktdata_enriched/ 仍不存在 —— 远端结构变了?"
sz=$(stat -c%s mktdata_enriched/others/financials.parquet 2>/dev/null || echo 0)
[ "$sz" -gt 1000000 ] || fail "financials.parquet 只有 ${sz} 字节 —— 是 LFS 指针,lfs pull 没生效(坑 2)"
echo "✓ 源数据就位(financials.parquet $((sz/1024/1024))MB)"

# ── 2. 摆目录结构(软链,不复制) ─────────────────────────────────────────────
step "2/6 建立软链"
mkdir -p "$OXQ_RESEARCH_DIR/mktdata_enriched_others"
ln -sfn "$SRC_REPO/mktdata_enriched" "$OXQ_RESEARCH_DIR/mktdata_enriched"
ln -sf  "$SRC_REPO/mktdata_enriched/others/corporate_actions.parquet" \
        "$OXQ_RESEARCH_DIR/mktdata_enriched_others/"
echo "✓ 软链完成"

# ── 3. 重建价格面板 ────────────────────────────────────────────────────────
step "3/6 重建价格面板(~150s)"
cd "$OXQ_RESEARCH_DIR"
"$PY" "$STUDY/data_prep/rebuild_price_data_fixed.py"
n=$(ls "$PANEL"/*.parquet 2>/dev/null | wc -l)
[ "$n" -eq 5232 ] || fail "面板文件数 $n ≠ 5232"
echo "✓ 5,232 个 parquet"

# ── 4. VWAP 重标定 raw_close ───────────────────────────────────────────────
# 独立真值核对:宁德时代 300750 @2021-11-30 重建价 679.68 vs 雪球 680.00(−0.047%)。
# 它同时重算 float_mv = raw_close × outstanding_share,而组合层按 float_mv 排序。
step "4/6 VWAP 重标定"
"$PY" "$STUDY/data_prep/refine_raw_close_vwap.py"
echo "✓ 完成"

# ── 5. 补 510300 后复权 ────────────────────────────────────────────────────
# 坑 4:不要用 data/*/kline.parquet 里的 510300 —— 那是**不复权**价,
#       会让 MA200 闸门有 87 天判定不同,组合级锚点必挂。
step "5/6 补 510300 后复权序列"
cp "$STUDY/data_prep/510300_hfq.parquet" "$PANEL/510300.parquet"
echo "✓ 完成"

# ── 6. 财务字段 PIT 回填 ───────────────────────────────────────────────────
# ⚠️ 这一步 08-14 恢复时**漏掉了** —— README 的「已知缺口」写了这件事,
#    但没写进步骤表,导致六列财务字段 0% 覆盖,直到 §75 要用业绩信号时才发现。
#    现在它是恢复链的一等公民。
step "6/6 财务字段 PIT 回填"
"$PY" "$STUDY/data_prep/fill_fundamentals.py"

# ── 全量锚点核对 ───────────────────────────────────────────────────────────
step "锚点核对(任一不过即退出非零)"
"$PY" - <<'PYEOF'
import glob, os, sys
import numpy as np, pandas as pd

SP = os.environ["OXQ_RESEARCH_DIR"]
D = f"{SP}/oxq_stock_market_fixed"
bad = []


def chk(name, got, want, tol=0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    print(f"  {'✓' if ok else '✗'} {name:<34} {got}   (期望 {want})")
    if not ok:
        bad.append(name)


cl, ni = {}, {}
for f in sorted(glob.glob(f"{D}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "net_income"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    ni[k] = x["net_income"]
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
NI = pd.DataFrame(ni).set_axis(CL.index)
CL = CL.where(CL > 0)

chk("面板行数(交易日)", CL.shape[0], 3297)
chk("面板列数(标的)", CL.shape[1], 5232)
chk("起始日", str(CL.index[0].date()), "2013-01-04")
chk("结束日", str(CL.index[-1].date()), "2026-08-03")
chk("有财务标的数", int(NI.notna().any().sum()), 4967)
chk("无财务标的数", int((~NI.notna().any()).sum()), 265)

# 案例锚点:688183 @2024-05-31(§75 用过的那一组数)
F = CL.ffill()
t = F.index.get_indexer([pd.Timestamp("2024-05-31")], method="ffill")[0]
chk("688183 收盘 @2024-05-31", round(float(F.iloc[t]["688183"]), 2), 14.49, 0.01)
chk("688183 MA100", round(float(F.rolling(100, min_periods=100).mean().iloc[t]["688183"]), 2),
    9.69, 0.01)
chk("688183 MA300", round(float(F.rolling(300, min_periods=300).mean().iloc[t]["688183"]), 2),
    10.74, 0.01)
r = (F.iloc[t] / F.shift(50).iloc[t] - 1).where(CL.iloc[t].notna()).rank(pct=True) * 100
chk("688183 RPS50", round(float(r["688183"]), 1), 99.7, 0.2)

if bad:
    print(f"\n✗ {len(bad)} 项锚点对不上:{', '.join(bad)}")
    print("  **不要带着这个面板往下跑。**")
    sys.exit(1)
print("\n✓ 全部锚点通过")
PYEOF

echo
echo "═══ 面板恢复完成 ═══"
echo "  $PANEL"
echo
echo "下一步可跑(会自行 assert 锚点):"
echo "  cd $OXQ_RESEARCH_DIR && $PY $STUDY/bull_features/base_pattern_trade.py"
echo "  → 交易级净期望 +4.61% / 组合年化 +6.34%"
