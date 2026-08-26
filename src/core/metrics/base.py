from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from src.core.experiments.predict import (
    PredictConfig,
    PredictResults,
)

PC = TypeVar("PC", bound=PredictConfig)
PR = TypeVar("PR", bound=PredictResults)


class Metric(ABC, Generic[PC, PR]):
    name: str

    @abstractmethod
    def __call__(self, config: PC, results: PR) -> Any: ...
