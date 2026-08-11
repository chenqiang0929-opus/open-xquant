# A股牛股研究案例:突破 + 止损交易系统

一次完整的 A股个股量化研究记录,以及其中「突破后跟随」方向的可复现代码。

> 本目录是**研究案例**,不是 oxq 框架的一部分。脚本只依赖
> numpy / pandas / pyarrow,**不 import oxq**,也不修改 `src/oxq/`。

## 一句话结论

| 提议 | 结论 |
|---|---|
| 用 **10周线(MA50)做止损** | **证伪**。六组规则 × 两种选股 × 三种停牌处理下全部明显差于固定止损 |
| **涨幅达 100% 后再启动移动止盈** | 方向对(交易级净期望 +4.61% → +6.12%/笔),但组合级只比基线高 0.81pp,落在路径噪音内 |
| 真正起作用的开关 | **大盘跌破 MA200 不开新仓** —— 同一套规则,年化从 -13.73% 变成 +6.34% |

最优配置仍跑输全市场等权基准(+6.85% / Sharpe 0.419 / 回撤 -58.5%
vs 基准 +7.22% / 0.423 / **-32.8%**)——**收益接近、回撤差一倍**。

细节见 `ETF_research_summary_for_stock_comparison.md` 第四十一、四十二节。
(文件名是历史遗留:研究从 ETF 起步,后转向个股。)

## 目录

```
ETF_research_summary_for_stock_comparison.md   研究记录主文档(53节)
data_prep/         上游数据构建(产出下面两个输入)
breakout_system/   突破+止损系统本体(第41-42节)
rps_pool_study/    RPS动量股池 + 双增长过滤 + 广度择时(第43-47节)
bull_features/     牛股特征:成长字段修正、三个假设、20因子扫描(第50-53节)
results/           所有表格数字的来源 CSV;results/logs/ 是原始运行日志
```

### `rps_pool_study/` —— 用户上传股池引出的一串检验

| 脚本 | 做什么 | 节 |
|---|---|---|
| `rps_pool_prep.py` | 解析用户上传的 RPS>90 每周快照,做前视检查(0/10,007 行有前视) | 43 |
| `rps_consistency_check.py` | 我自己算的 RPS 与用户快照对账(相关 0.990,中位绝对差 0.07) | 44 |
| `rps_growth_test.py` | 主检验:动量池内「净利+收入双增长」过滤 | 43 |
| `rps_growth_decompose.py` | 独立复现 + 拆解(市值/PE/RPS 分档、集中度) | 43 |
| `rps_timing_overlay.py` / `rps_rule_ladder.py` | 叠加择时、规则阶梯 | 44 |
| `breadth_timing_test.py` | 广度择时 OOS(**+10.5pp,但与同开仓率随机开关无法区分,p=0.050**) | 45 |
| `dual_growth_transfer.py` | 双增长能否迁移到全市场/小市值/高BP(**不能**) | 46 |
| `growth_rank_vs_filter.py` | 成长当排序因子 vs 当过滤器 | 47 |

### `bull_features/` —— 牛股到底和非牛股差在哪

| 脚本 | 做什么 | 节 |
|---|---|---|
| `build_clean_growth.py` | **修正被污染的成长字段**(YTD 去累计 + 报告期对齐),已对官方财报核验 | 52 |
| `bull_pattern_hypotheses.py` | 用户三个假设:基底形态 / 均线多头排列 / CANSLIM | 51 |
| `bull_feature_scan.py` | 20 个未用过的因子系统扫描 | 53 |
| `bull_feature_recheck.py` | **复查扫描的噪音上界**(退化特征污染 → 每特征零分布 + 公平天花板) | 53 |
| `oneil_real_thresholds.py` / `oneil_sampling_variance.py` | 欧奈尔真实阈值、抽样方差 | 50 |
| `composite3_replication.py` / `rebalance_bonus_study.py` / `ma50_reconcile.py` | 复合因子复现、再平衡收益、MA50 对账 | 48-49 |

⚠️ `rps_pool_study/` 里的脚本需要用户私有的股池快照(`rps_pool_A/B.parquet`),
**该数据未提交**,这几个脚本在仓库里只作方法留档,不能直接跑通。

## 复现步骤

### 0. 数据依赖(**不在仓库内**)

两个输入体积过大,未提交,需自行生成:

| 产物 | 体积 | 由谁生成 |
|---|---|---|
| `oxq_stock_market_fixed/`(5,232 个 parquet) | >1GB | `data_prep/rebuild_price_data_fixed.py` → `data_prep/refine_raw_close_vwap.py` |
| `oneil_prelaunch_events_fixed.csv`(70,310 个突破事件) | 19.6MB | `data_prep/oneil_prelaunch_attribution.py` |

所有脚本通过环境变量定位工作目录,**默认是脚本自身所在目录**:

```bash
export OXQ_RESEARCH_DIR=/path/to/your/research-workdir
```

把上面两个产物放在 `$OXQ_RESEARCH_DIR` 下,再按顺序运行。

### 1. 运行

```bash
export OXQ_RESEARCH_DIR=/path/to/your/research-workdir

python breakout_system/breakout_trading_system.py    # 阶段1 交易级,18组配置   ~220s
python breakout_system/breakout_portfolio.py         # 阶段2-4 组合级+择时+成本 ~230s
python breakout_system/breakout_exit_rules.py        # 6组离场规则            ~225s
python breakout_system/breakout_exit_rules_seeds.py  # 20次种子/重抽样分布     ~120s
python breakout_system/breakout_exit_rules_halt.py   # 3种停牌/退市处理对比    ~390s
python breakout_system/diag_ruleA_vs_stage1.py       # 基线差异逐项诊断        ~250s
```

`breakout_exit_rules_seeds.py` 接一个可选参数:退市折价系数
(`python ... 0.5` 表示按最后有效价的 50% 平仓)。

`breakout_exit_rules_halt.py` 接 `--trade-only` 跳过组合级。

## 方法上必须保留的几个细节

写在这里是因为**每一条都实测会改变结论**:

1. **入场用突破日次日开盘价**,不用突破日收盘 —— 突破日往往放量大涨,
   用当日收盘等于假设你能在盘中识别突破并成交
2. **止损判断用当日最低价**,止盈用当日最高价 —— 用收盘会系统性低估触发率
3. **跳空穿越止损线 → 按开盘价成交**(实测 12.1% 的止损属此类,影响 -0.25pp)
4. **临时停牌要持有穿越,不能强平**;只有真退市才平仓 ——
   把两者混为一谈会让基线年化从 +6.85% 掉到 +4.69%,**比任何离场规则的影响都大**
5. 择时基准用 **510300**(用"横截面日收益中位数累乘"构造指数是无效的)
6. **两种选股都要跑**(小市值优先 / 随机选)——本案例里两者对条件式止盈
   给出了相反结论,只跑一种会得到假结论
7. **单次回测不可采信**:随机选必须跑多种子,确定性选股用 90% 事件重抽样
   加误差棒。本案例中有个 +10.12% 的单种子结果,在其 20 次分布
   (+0.35% ~ +11.52%)里接近最高值

## 研究过程中自查出的三个错误

都记在主文档第四十二节末尾,这里只列结论:

1. 曾断言「退市/长停那批几乎全亏」—— 实测隐含平均 **+103%**,方向反了
2. 用「任何价格中断一律打5折」做压力测试,得到「所有规则含基线全线转负」,
   差点当成重要发现报出去 —— **是那个检验本身写错了**,它把只停几天的
   临时停牌也按半价强平
3. `breakout_exit_rules_halt.py` 里「到期」判断曾放在「永久终止」之前,
   导致退市折价对目标持仓不生效 —— 已修,修后结论不变

第 2 条最值得记:**一个推翻结论的检验结果,和一个支持结论的结果,
需要同等力度的复核。**

## 后续几轮又自查出的三个(第43-53节)

4. **成长字段整体被污染**:`ni_yoy_252 = net_income / net_income.shift(252) - 1`,
   而 `net_income` 是**年初至今累计**,252 交易日 ≈ 1.04 年必然跨报告期 ——
   茅台 2023-05-04 算出 **-60.4%**(拿单季比全年),修正后 **+20.6%**。
   硬自检:横截面「同比>0比例」的月份极差 **25.1pp → 2.5pp**。
   受影响的五处结论已在第五十二节全部重跑(`build_clean_growth.py`)
5. **基准换手成本算错了两次**:等权全市场再平衡的换手是 **1.9~2.0%/期**,
   不是组合自身的换手率。算错时基准 -1.55%,算对后 +7.48%,超额从 -9.2pp 变 **-18.2pp**
6. **噪音上界被一个退化特征污染**:第五十三节 20 个特征里有一个只有 38 个命中,
   打乱标签时 best-of-20 的最大值几乎每次都被它拿走,天花板被抬到 2.45。
   改成「每个特征各自的零分布 + 只让命中≥500 的参与天花板」后天花板 1.54。
   **结论(0 个发现)没变,但第一次给出的理由是错的** —— 这种「结论对、推理错」
   最容易漏过去

## 贯穿全程的一条纪律

**每个检验在跑之前先写死判据。** 本研究里连续六个检验没达到事前判据
(10周线止损、条件式移动止盈、MA200择时、广度择时、双增长可迁移性、再平衡收益),
**没有一个事后放宽过标准**。主文档里每一节末尾都有「事前判据 vs 实际结果」对照表。
