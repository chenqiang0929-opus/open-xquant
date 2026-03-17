"""Tests for ExperimentLog — structured experiment iteration tracking."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult


def _make_result(values: list[float]) -> RunResult:
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    return RunResult(
        portfolio=Portfolio(cash=Decimal(str(values[-1])) if values else Decimal("0")),
        trades=[],
        equity_curve=[(d, v) for d, v in zip(dates, values)],
        mktdata={},
    )


class TestExperiment:
    def test_frozen(self) -> None:
        from oxq.observe.experiment import Experiment
        exp = Experiment(
            name="test", observation="obs", hypothesis="hyp",
            criteria={}, result={}, conclusion="confirmed",
            notes="", timestamp="2024-01-01T00:00:00",
        )
        with pytest.raises(AttributeError):
            exp.name = "changed"


class TestAdd:
    def test_add_single(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog(name="test-batch")
        log.add(
            name="iter1",
            observation="sharpe drops in Q2",
            hypothesis="reduce frequency",
            criteria={"sharpe": "above_baseline"},
            result={"best_freq": 63},
            conclusion="rejected",
        )
        assert len(log.experiments) == 1
        assert log.experiments[0].name == "iter1"
        assert log.experiments[0].conclusion == "rejected"

    def test_add_multiple(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog()
        log.add(name="e1", observation="o1", hypothesis="h1",
                criteria={}, result={}, conclusion="confirmed")
        log.add(name="e2", observation="o2", hypothesis="h2",
                criteria={}, result={}, conclusion="rejected")
        assert len(log.experiments) == 2

    def test_timestamp_auto_generated(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog()
        log.add(name="e1", observation="o", hypothesis="h",
                criteria={}, result={}, conclusion="confirmed")
        assert log.experiments[0].timestamp != ""
        assert "T" in log.experiments[0].timestamp


class TestAddFromStrategy:
    def test_extracts_hypothesis(self) -> None:
        from oxq.core.strategy import Strategy
        from oxq.observe.experiment import ExperimentLog
        from oxq.universe.static import StaticUniverse

        strategy = Strategy(
            name="test-strat",
            universe=StaticUniverse(("A",)),
            indicators={},
            signals={},
            entry_rules=[],
            exit_rules=[],
            hypothesis="vol filter improves drawdown",
            objectives={"max_drawdown": {"max": -0.15}},
        )
        result = _make_result([100, 105, 110, 108, 115])
        log = ExperimentLog()
        log.add_from_strategy(
            strategy=strategy,
            result=result,
            observation="drawdown too deep",
            conclusion="confirmed",
        )
        exp = log.experiments[0]
        assert exp.hypothesis == "vol filter improves drawdown"
        assert exp.name == "test-strat"
        assert "total_return" in exp.result
        assert "sharpe_ratio" in exp.result
        assert "max_drawdown" in exp.result
        assert "annualized_return" in exp.result

    def test_matches_objectives(self) -> None:
        from oxq.core.strategy import Strategy
        from oxq.observe.experiment import ExperimentLog
        from oxq.universe.static import StaticUniverse

        strategy = Strategy(
            name="test",
            universe=StaticUniverse(("A",)),
            indicators={},
            signals={},
            entry_rules=[],
            exit_rules=[],
            objectives={"sortino": {"min": 1.5}, "calmar": {"min": 1.0}},
        )
        result = _make_result([100, 102, 99, 103, 97, 105])
        log = ExperimentLog()
        log.add_from_strategy(
            strategy=strategy, result=result,
            observation="test", conclusion="partial",
        )
        exp = log.experiments[0]
        assert "sortino" in exp.result
        assert "calmar" in exp.result
        assert exp.criteria == {"sortino": {"min": 1.5}, "calmar": {"min": 1.0}}


class TestToDataFrame:
    def test_columns(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog()
        log.add(name="e1", observation="o", hypothesis="h",
                criteria={"k": 1}, result={"r": 2}, conclusion="confirmed")
        df = log.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "name" in df.columns
        assert "conclusion" in df.columns
        assert len(df) == 1

    def test_empty_log(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog()
        df = log.to_dataframe()
        assert len(df) == 0


class TestToMarkdown:
    def test_returns_string(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog()
        log.add(name="e1", observation="o", hypothesis="h",
                criteria={}, result={}, conclusion="confirmed")
        md = log.to_markdown()
        assert isinstance(md, str)
        assert "e1" in md
        assert "confirmed" in md


class TestSerialization:
    def test_roundtrip(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog(name="batch-1")
        log.add(name="e1", observation="obs1", hypothesis="hyp1",
                criteria={"k": 1}, result={"r": 2.5},
                conclusion="confirmed", notes="note1")
        log.add(name="e2", observation="obs2", hypothesis="hyp2",
                criteria={}, result={}, conclusion="rejected")
        d = log.to_dict()
        restored = ExperimentLog.from_dict(d)
        assert restored.name == "batch-1"
        assert len(restored.experiments) == 2
        assert restored.experiments[0].name == "e1"
        assert restored.experiments[0].notes == "note1"
        assert restored.experiments[1].conclusion == "rejected"

    def test_dict_structure(self) -> None:
        from oxq.observe.experiment import ExperimentLog
        log = ExperimentLog(name="test")
        log.add(name="e1", observation="o", hypothesis="h",
                criteria={}, result={}, conclusion="confirmed")
        d = log.to_dict()
        assert "name" in d
        assert "experiments" in d
        assert isinstance(d["experiments"], list)
        assert d["experiments"][0]["name"] == "e1"
