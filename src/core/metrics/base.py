from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from src.core.experiments.evaluation import (
    EvalConfig,
    EvalResults,
)

EC = TypeVar("EC", bound=EvalConfig)
ER = TypeVar("ER", bound=EvalResults)


class Metric(ABC, Generic[EC, ER]):
    name: str

    @abstractmethod
    def __call__(self, config: EC, results: ER) -> Any: ...
