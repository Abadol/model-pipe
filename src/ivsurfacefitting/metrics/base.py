from abc import ABC
from typing import Any

from src.core.metrics.base import Metric
from src.ivsurfacefitting.experiments.predict import (
    IVSurfacePredictConfig,
    IVSurfacePredictResults,
)


class IVMetric(Metric[IVSurfacePredictConfig, IVSurfacePredictResults], ABC):
    """
    Abstract data class for a mtric for iv surface fitting.

    Currently forced to take in the predicted and true surfaces, even though some metrics,
    like no aribitrage violation may only require the predicted surface.

    It is only being used for type managing at the moment, if no functionality is rquired, which it probably wont,
    it could simply be defined as a union of types. This is just a quick fix.
    """

    def __init__(self, name) -> None:
        self.name = name

    def __call__(
        self, config: IVSurfacePredictConfig, results: IVSurfacePredictResults
    ) -> Any:
        return f"Metric {self.name} not implemented."
