"""Tests for SimBroker."""

from decimal import Decimal

import pandas as pd

from oxq.core.types import FillReceiver, Order, OrderRouter
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import SimBroker
from oxq.trade.slippage import PercentageSlippage


def test_sim_broker_satisfies_order_router_protocol() -> None:
    assert isinstance(SimBroker(), OrderRouter)


def test_sim_broker_satisfies_fill_receiver_protocol() -> None:
    assert isinstance(SimBroker(), FillReceiver)


def test_submit_order_returns_id() -> None:
    broker = SimBroker()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    order_id = broker.submit_order(order)
    assert isinstance(order_id, str)
    assert len(order_id) > 0


def test_fill_market_orders() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0, 155.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == Decimal("150")
    assert fills[0].order.symbol == "AAPL"


def test_get_fills_clears_after_read() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    assert len(broker.get_fills()) == 1
    assert len(broker.get_fills()) == 0


def test_multi_symbol_fill() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0]}, index=dates),
        "MSFT": pd.DataFrame({"close": [300.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.submit_order(Order(symbol="MSFT", side="BUY", shares=50))
    broker.fill_pending_orders(mktdata, dates[0])

    fills = broker.get_fills()
    assert len(fills) == 2
    prices = {f.order.symbol: f.filled_price for f in fills}
    assert prices["AAPL"] == Decimal("150")
    assert prices["MSFT"] == Decimal("300")


def test_fee_model() -> None:
    broker = SimBroker(fee_model=PercentageFee(rate=Decimal("0.001"), min_fee=Decimal("5")))
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {"AAPL": pd.DataFrame({"close": [150.0]}, index=dates)}
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fill = broker.get_fills()[0]
    assert fill.fee == Decimal("15.0")  # 150 * 100 * 0.001


def test_slippage_model() -> None:
    broker = SimBroker(slippage_model=PercentageSlippage(rate=Decimal("0.01")))
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {"AAPL": pd.DataFrame({"close": [100.0]}, index=dates)}
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fill = broker.get_fills()[0]
    assert fill.filled_price == Decimal("100") * (1 + Decimal("0.01"))


def test_stop_order_triggers() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 145.0, 138.0]}, index=dates,
        ),
    }
    # Place stop at 140
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))

    # Day 1: close=150 > 140, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: close=145 > 140, no trigger
    broker.process_pending_orders(mktdata, dates[1])
    broker.fill_market_orders(mktdata, dates[1])
    assert len(broker.get_fills()) == 0

    # Day 3: close=138 <= 140, triggers
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].order.order_type == "stop"
    assert fills[0].filled_price == Decimal("140")


def test_limit_order_triggers() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 160.0, 185.0]}, index=dates,
        ),
    }
    # Place limit sell at 180
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="limit", limit_price=Decimal("180"),
    ))

    # Day 1: 150 < 180, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 3: 185 >= 180, triggers at limit price
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == Decimal("180")


def test_trailing_stop_order() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=4)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 110.0, 105.0, 103.0]}, index=dates,
        ),
    }
    # Trail 5% from high
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="trailing_stop", trail_pct=0.05,
    ))

    # Day 1: HWM=100, stop=95, close=100, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: HWM=110, stop=104.5, close=110, no trigger
    broker.process_pending_orders(mktdata, dates[1])
    broker.fill_market_orders(mktdata, dates[1])
    assert len(broker.get_fills()) == 0

    # Day 3: HWM=110, stop=104.5, close=105, no trigger
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    assert len(broker.get_fills()) == 0

    # Day 4: HWM=110, stop=104.5, close=103 <= 104.5, triggers
    broker.process_pending_orders(mktdata, dates[3])
    broker.fill_market_orders(mktdata, dates[3])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].order.order_type == "trailing_stop"


def test_stop_dedup_replaces_old() -> None:
    broker = SimBroker()

    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("145"),
    ))

    open_orders = broker.get_open_orders(symbol="AAPL")
    assert len(open_orders) == 1
    assert open_orders[0].order.stop_price == Decimal("145")
