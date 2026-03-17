"""Observe layer — strategy diagnostics and analysis."""

__all__ = [
    "BadPeriod",
    "Experiment",
    "ExperimentLog",
    "MarketStateDetector",
    "StrategyMonitor",
]

_IMPORTS = {
    "BadPeriod": "oxq.observe.monitor",
    "StrategyMonitor": "oxq.observe.monitor",
    "MarketStateDetector": "oxq.observe.detector",
    "Experiment": "oxq.observe.experiment",
    "ExperimentLog": "oxq.observe.experiment",
}


def __getattr__(name: str):
    if name in _IMPORTS:
        import importlib

        module = importlib.import_module(_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
