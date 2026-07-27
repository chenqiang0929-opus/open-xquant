# 自定义数据列与 workspace 组件

> 如何在**不修改 open-xquant 框架**的前提下，在外部研究 workspace 中
> 使用非 OHLCV 数据（资金流、财务、另类因子）开发并验证策略。

本文回答一个具体问题：我的策略依赖 OHLCV 之外的数据列（例如 ETF 资金流净流入），
自定义指标也不在内置的组件清单里——我必须 fork 或修改框架吗？

**不需要。** 框架已经为这两件事各留了一条通道，本文说明它们的位置、
契约和已知缺口。

---

## 1. 数据通道：写进 market parquet，不要用 factor 目录

### 额外列会原样透传

`LocalMarketDataProvider.get_bars()` 读取数据的全部内容是一句
`pd.read_parquet(path)`（`src/oxq/data/market.py:28`）。**没有列白名单，
不拒绝未知列**。读取路径上仅有的两项校验是时区（`market.py:29-35`）
和重复索引（`market.py:81-84`）。

随后 `src/oxq/core/engine.py:132-135` 把整帧 `.copy()` 交给引擎：

```python
self._mktdata[symbol] = market.get_bars(symbol, load_start, end).copy()
```

而 Indicator 拿到的就是这一帧（`src/oxq/core/types.py:281-287`）：

```python
class Indicator(Protocol):
    name: str
    def compute(self, mktdata: pd.DataFrame, **params: object) -> pd.Series: ...
```

所以：**parquet 里有什么列，`compute()` 里就能按名取用什么列。**

### factor 目录不会被 join 进来

`src/oxq/data/factors.py` 提供了 `resolve_factor_dir()` 和 `read_factor()`，
对应 `$OXQ_DATA_DIR/factor/{macro,financial}/`。但 `read_factor()`
**从未被 engine、compiler 或任何 provider 调用**——全仓只有它的定义
（`factors.py:581`）和 `src/oxq/data/__init__.py` 的 re-export。
没有任何 factor → mktdata 的合并逻辑。

佐证：内置的基本面指标 `PE`、`MarketCap`、`TurnoverRate` 依赖
`eps` / `total_shares` 列，而它们的测试直接把这些列内联进 DataFrame
（`tests/indicators/test_pe.py:20`）——因为这些列必须**已经是 market parquet 的列**。

`factor/` 目录目前只服务于 `factor_inspect` / `financial_inspect`
等检查工具（`src/oxq/tools/data.py:568`、`:667`），是一个 staging /
inspection 区域，不是运行时数据源。

**结论：自定义数据列写进 `$OXQ_DATA_DIR/market/{symbol}.parquet`。**

### 没有导入工具，parquet 需自行写入

`src/oxq/cli/main.py` 没有 `data` 命令组。Agent/MCP 侧的
`data_load_symbols`（`src/oxq/tools/data.py:469`）会委托给 downloader，
而 downloader 在 `src/oxq/data/loaders.py:38-39` 和 `:128` 把结果**硬砍成
OHLCV 五列**：

```python
cols = ["open", "high", "low", "close", "volume"]
df = df[cols]
```

因此带自定义列的 parquet 只能自己写。写完可以用
`data_inspect`（`src/oxq/tools/data.py:33-47`）确认列，它会返回
`"columns": list(df.columns)`。

### 索引要求

- **必须 tz-aware。** `market.py:29-35` 对无时区的索引只会打 warning
  然后**静默当作 UTC** —— A 股数据这样会整体错位，比直接报错更危险。
  `engine.py:136-142` 有一道硬性 `raise`，但对已被 localize 的本地 parquet
  不会触发。请在写 parquet 时就带上正确时区（A 股用 `Asia/Shanghai`，
  与 `loaders.py:127` 的 akshare 约定一致）。
- 索引不得有重复值（`market.py:81-84` 会抛 `ValueError`），需排序。

---

## 2. spec 契约：自定义列必须声明

这是唯一的硬约束，也是最容易踩的坑。

`src/oxq/spec/schema.py:79` 的 `required_columns` 是一个**开放 list，不是枚举**：

```python
required_columns: list[str] = field(
    default_factory=lambda: ["open", "high", "low", "close", "volume"]
)
```

校验器 `_validate_compute_params()`（`src/oxq/spec/validator.py:959-984`）
会**只用 `required_columns` 构造一个合成 DataFrame** 做 dry-run：

```python
index = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
frame = pd.DataFrame(
    {column: [1.0, 2.0, 3.0] for column in spec.data.required_columns},
    index=index,
)
```

如果你的指标读 `mktdata["net_inflow"]` 而 spec 没声明该列，
`KeyError` 会被捕获并转成 **fatal `compute_dry_run_failed`**。
声明了就通过：

```yaml
data:
  required_columns: [open, high, low, close, volume, net_inflow, main_net_inflow]
```

Signal 侧同理，`_validate_signal_column_references`（`validator.py:64-97`）
把可用列定义为 `required_columns ∪ indicators.keys()`，
引用未声明的列 → fatal `signal_column_missing`。优化器的 `score_col`
也走同一套（`validator.py:270-273`）。

### 命名冲突

`validator.py:209-216`：**指标输出名不得与原始数据列同名**，
否则 fatal `signal_name_collision`：

```
signal.indicators.{name} must not overwrite raw data column '{name}'
```

所以原始列叫 `net_inflow` 时，指标输出要命名成 `net_inflow_ma5` 之类。

### 数据可用性不是硬校验

声明了但 parquet 里实际缺失的列，不会在编译期报错。
`src/oxq/spec/compiler.py:1788-1794` 的 `_compute_missing_ratio()`
会把它算成 100% 缺失，`src/oxq/audit/research_bias.py:304-305`
在缺失率 > 5% 时只给 **warning**。请自行核对数据是否真的存在。

---

## 3. 组件通道：自定义组件不必进框架仓

框架提供两条 out-of-tree 注册通道，都最终汇入
`src/oxq/core/registry.py` 的四个注册函数（`registry.py:100-117`）。

| 通道 | 实现位置 | 适用场景 | provenance |
|---|---|---|---|
| **component manifest**（推荐） | `src/oxq/core/component_manifest.py` | 组件代码留在研究 workspace 的松散文件里 | 有：`bundle_hash` + run 记录 |
| entry_points | `src/oxq/core/registry.py:377-416` | 组件打包成已安装的发行包 | 无 |

### component manifest

`load_component_manifest()`（`component_manifest.py:71-94`）读取一份 JSON
manifest，校验 bundle hash，把 extension root 前置到 `sys.path`，
然后按 dotted path 导入并注册每个声明的组件
（`_register_manifest_component`，`component_manifest.py:290-312`）。

manifest 声明 `schema_version`、`extension_id`、`extension_root`、
`bundle_hash`，以及 `components[]`（每项含 `kind` / `name` / `module` / `class`）。

护栏包括：`extension_root` 必须在 manifest 自身目录之下；导入的模块文件
必须落在 root 内；声明的 `name` 必须等于类的注册名且不得与已有注册冲突；
拒绝符号链接与 `..` 穿越；`scoped_component_registries()`
（`component_manifest.py:47-68`）保证注册是临时的，不污染全局 registry。

支持 `--component-manifest`（可重复）的命令：

| 命令 | 位置 |
|---|---|
| `oxq spec validate` | `src/oxq/cli/main.py:211` |
| `oxq strategy compile` | `main.py:3725` |
| `oxq registry export` | `main.py:3796` |
| `oxq backtest run` | `main.py:2146` |

另有 `oxq component-manifest hash|validate`（`main.py:3820`）。

约定的 workspace 布局（见 `docs/agent-guide.md` 第 7 节、
`docs/strategy-workflow-artifact-governance.md`）：

```text
<components_dir>/bundles/<bundle_id>/
  component_manifest.json
  component_catalog.json
  custom_components/
```

`components_dir` 从 `.open-xquant/workspace.yaml` 的 `paths.components_dir`
解析，缺省为 `components`。

> 治理化（formal）回测额外要求 manifest 落在
> `workspace_root / components_dir` 之内（`main.py:3506-3519`）；
> 探索性运行（`--allow-unaudited`）不受此限。

### entry_points

任何已安装的发行包只要声明这四个 group 之一，`import oxq` 时会自动注册
（`registry.py:377-416`）：

```
oxq.indicators
oxq.signals
oxq.portfolio_optimizers
oxq.rules
```

加载失败只打 warning，不中断（`registry.py:405-411`）。

### spec 不支持任意 dotted path

`src/oxq/spec/compiler.py:60-91` 的 `_resolve_indicator` 等四个函数是
**纯字典查表**，`strategy_spec.yaml` 里的 `type:` 只能是 registry key，
没有 `module:Class` 语法。组件必须在 compile 之前被注册进 registry。

---

## 4. 有些策略零代码即可跑

写自定义组件之前，先确认是否已被内置件覆盖——以下三个内置件本来就接受任意列名：

| 组件 | 位置 | 用法 |
|---|---|---|
| `Ratio` | `src/oxq/indicators/ratio.py:14-21` | `col_a: main_net_inflow, col_b: volume` |
| `Formula` | `src/oxq/signals/formula.py:13-18` | `mktdata.eval(expr)`，表达式内可引用任意列 |
| `Threshold` | `src/oxq/signals/threshold.py:15-23` | `column: net_inflow, threshold: 0, relationship: gt` |

内置的非 close 指标也都把输入列做成了**带默认值的字符串参数**，
这是全仓的统一约定（`src/oxq/indicators/pe.py:14-22`、
`turnover_rate.py:15-27`、`market_cap.py:14-21`）：

```python
def compute(
    self,
    mktdata: pd.DataFrame,
    volume_col: str = "volume",
    shares_col: str = "total_shares",
) -> pd.Series:
```

自定义指标请沿用这个约定（`flow_col: str = "main_net_inflow"`），
这样列名可以在 spec 里配置而不用改代码。

---

## 5. 边界：自定义 Rule 需要进入框架开发

`docs/agent-guide.md` 第 7 节明确规定：

> workspace-local custom Rule 当前不属于普通 authoring 能力；如果需要 Rule，
> 应阻塞并要求用户明确是否进入 OpenXQuant 框架开发。

Indicator / Signal / PortfolioOptimizer 可以留在 workspace；
需要新的 Rule（bar-by-bar 有状态的风控或约束逻辑）时，
才需要按 `CLAUDE.md` 的组件创建流程改框架仓。

---

## 6. 其他已知缺口

- **手写 parquet 没有 provenance。** `src/oxq/data/manifest.py:63`
  的 `verify_manifest()` 在 `src/oxq/` 内**从未被调用**，
  所以缺少 `.manifest.json` 不会被拦截，但也意味着 `oxq` 无法为这份数据背书。
- **无中文别名。** `src/oxq/core/aliases.py:5-19` 的字段别名表不含
  资金流 / 净流入类字段。`resolve_alias()`（`aliases.py:40-50`）
  对未知名称原样小写返回，不阻塞，只是没有中文便利性。
- **前视偏差需自行处理。** `read_factor()` 提供的 point-in-time /
  `publish_date` 滞后语义（`factors.py:642-646`）在 market-parquet 路径上
  **不可用**。资金流、财务等有发布延迟的列，必须在写 parquet 时就
  预先 `shift`，否则会引入未来信息。

---

## 7. 端到端流程

在你的研究 workspace（不是框架仓）中：

```bash
# 1. 安装框架与 Agent 能力
uv add open-xquant
uv run oxq agent install --all-targets

# 2. 初始化 workspace（生成 .open-xquant/workspace.yaml、versions/、components/ 等）
uv run oxq research init

# 3. 自行写 ingest 脚本，产出带自定义列的 market parquet
#    → $OXQ_DATA_DIR/market/{symbol}.parquet
#    要求：tz-aware、唯一、已排序的 DatetimeIndex；有发布延迟的列预先 shift

# 4. 先用内置件跑通链路（零自定义代码）
uv run oxq spec validate <spec.yaml>
uv run oxq backtest run <spec.yaml> --out ./runs --allow-unaudited --json

# 5. 再引入自定义组件
uv run oxq component-manifest validate <components_dir>/bundles/<id>/component_manifest.json
uv run oxq registry export --component-manifest <manifest> --out component_catalog.json
uv run oxq spec validate <spec.yaml> --component-manifest <manifest>
uv run oxq backtest run <spec.yaml> --component-manifest <manifest> \
  --out ./runs --allow-unaudited --json
```

编写自定义指标时遵循 `agent/skills/create-indicator/SKILL.md` 的流程
（先写手算期望值的测试，再实现），只是产物落在 workspace 的
`custom_components/` 而非 `src/oxq/indicators/`。

---

## 相关文档

- `docs/agent-guide.md` — 安装与 workspace-local component 章节
- `docs/strategy-workflow-artifact-governance.md` — workspace 目录布局与治理契约
- `docs/architecture.md` — spec 结构与整体设计
- `CLAUDE.md` — 内置组件的创建流程（需要改框架时）
