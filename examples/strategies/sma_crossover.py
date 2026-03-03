"""SMA Crossover Strategy — end-to-end example.

Strategy: Buy when SMA10 crosses above SMA50 (golden cross),
          sell when SMA10 drops below SMA50 (death cross).

Architecture note:
    The strategy definition is provider-agnostic. The Engine doesn't know
    whether it's running a backtest or live trading — it only depends on
    three Protocol interfaces: MarketDataProvider, OrderRouter, FillReceiver.

    "Backtest" = LocalMarketDataProvider + SimBroker
    "Paper"    = RealtimeDataProvider   + SimBroker     (future)
    "Live"     = RealtimeDataProvider   + BrokerAdapter  (future)

Usage:
    # First download data
    python -c "from oxq.data import YFinanceDownloader; YFinanceDownloader().download('AAPL', '2023-01-01', '2024-12-31')"

    # Then run the strategy
    python examples/strategies/sma_crossover.py
"""

from oxq.core import Engine, Strategy
from oxq.data import LocalMarketDataProvider
from oxq.indicators import SMA
from oxq.rules import EntryRule, ExitRule
from oxq.signals import Crossover
from oxq.trade import SimBroker
from oxq.universe import StaticUniverse

# ── 1. Strategy Definition (provider-agnostic) ──────────────────────

strategy = Strategy(
    name="sma_crossover",
    hypothesis="短期均线上穿长期均线的标的在后续持有期内有正超额收益",
    universe=StaticUniverse(("AAPL",)),
    indicators={
        "sma_10": (SMA(), {"period": 10}),
        "sma_50": (SMA(), {"period": 50}),
    },
    signals={
        "sma_10_x_sma_50": (Crossover(), {"fast": "sma_10", "slow": "sma_50"}),
    },
    entry_rules=[EntryRule(signal="sma_10_x_sma_50", shares=100)],
    exit_rules=[ExitRule(fast="sma_10", slow="sma_50")],
)

# ── 2. Choose Providers (this is what makes it "backtest") ──────────

market = LocalMarketDataProvider()
sim_broker = SimBroker()

# ── 3. Run ──────────────────────────────────────────────────────────

engine = Engine()
result = engine.run(
    strategy,
    market=market,
    router=sim_broker,      # SimBroker implements OrderRouter
    receiver=sim_broker,    # SimBroker implements FillReceiver
    start="2023-01-01",
    end="2024-12-31",
)

# ── 4. Results ──────────────────────────────────────────────────────

print("=" * 60)
print(f"Strategy: {strategy.name}")
print(f"Hypothesis: {strategy.hypothesis}")
print("=" * 60)
print(f"Total Return:  {result.total_return():>8.2%}")
print(f"Sharpe Ratio:  {result.sharpe_ratio():>8.2f}")
print(f"Max Drawdown:  {result.max_drawdown():>8.2%}")
print(f"Total Trades:  {len(result.trades):>8d}")
print(f"Final Cash:    {result.portfolio.cash:>12,.2f}")
print()

# Show trades
if result.trades:
    print("Trades:")
    print("-" * 60)
    for fill in result.trades:
        print(
            f"  {fill.filled_at}  {fill.order.side:>4}  "
            f"{fill.order.shares:>4} {fill.order.symbol}  "
            f"@ {fill.filled_price:.2f}"
        )
