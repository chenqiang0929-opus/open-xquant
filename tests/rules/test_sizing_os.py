"""Tests for os_* sizing functions."""

from decimal import Decimal

from oxq.rules.sizing import os_equal_weight, os_kelly, os_pct_equity, os_risk_parity


class TestOsEqualWeight:
    def test_basic(self):
        assert os_equal_weight(Decimal("100000"), 5, Decimal("150")) == 133

    def test_single_asset(self):
        assert os_equal_weight(Decimal("100000"), 1, Decimal("50")) == 2000

    def test_zero_assets(self):
        assert os_equal_weight(Decimal("100000"), 0, Decimal("50")) == 0

    def test_zero_price(self):
        assert os_equal_weight(Decimal("100000"), 5, Decimal("0")) == 0


class TestOsPctEquity:
    def test_basic(self):
        assert os_pct_equity(Decimal("100000"), 0.10, Decimal("50")) == 200

    def test_small_pct(self):
        assert os_pct_equity(Decimal("100000"), 0.01, Decimal("150")) == 6

    def test_zero_price(self):
        assert os_pct_equity(Decimal("100000"), 0.10, Decimal("0")) == 0


class TestOsRiskParity:
    def test_basic(self):
        assert os_risk_parity(Decimal("100000"), 0.10, 0.20, Decimal("100")) == 500

    def test_high_volatility_reduces_shares(self):
        assert os_risk_parity(Decimal("100000"), 0.10, 0.40, Decimal("100")) == 250

    def test_zero_volatility(self):
        assert os_risk_parity(Decimal("100000"), 0.10, 0.0, Decimal("100")) == 0

    def test_zero_price(self):
        assert os_risk_parity(Decimal("100000"), 0.10, 0.20, Decimal("0")) == 0


class TestOsKelly:
    def test_basic(self):
        assert os_kelly(Decimal("100000"), 0.6, 2.0, 1.0, Decimal("100")) == 400

    def test_half_kelly(self):
        assert os_kelly(Decimal("100000"), 0.6, 2.0, 1.0, Decimal("100"), fraction=0.5) == 200

    def test_negative_kelly_returns_zero(self):
        assert os_kelly(Decimal("100000"), 0.3, 1.0, 1.0, Decimal("100")) == 0

    def test_zero_avg_loss(self):
        assert os_kelly(Decimal("100000"), 0.6, 2.0, 0.0, Decimal("100")) == 0

    def test_zero_price(self):
        assert os_kelly(Decimal("100000"), 0.6, 2.0, 1.0, Decimal("0")) == 0
