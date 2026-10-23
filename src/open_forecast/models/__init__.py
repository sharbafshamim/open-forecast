"""Model adapters sharing one fit/predict experiment interface.

PyTorch is imported lazily so tree and naive adapters run without it.
"""

from .tree import TreeForecaster, naive_prediction


def __getattr__(name):
    if name == "DeepForecaster":
        from .deep import DeepForecaster
        return DeepForecaster
    raise AttributeError(name)


__all__ = ["DeepForecaster", "TreeForecaster", "naive_prediction"]
