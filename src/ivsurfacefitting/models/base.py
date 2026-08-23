from abc import ABC, abstractmethod

from src.core.models.base import Model
from src.ivsurfacefitting.experiments.evaluation import (
    IVSurfaceEvalConfig,
    IVSurfaceEvalResults,
)
from src.ivsurfacefitting.experiments.train import (
    IVSurfaceTrainConfig,
    IVSurfaceTrainResults,
)


class IVSurfaceModel(
    Model[
        IVSurfaceEvalConfig,
        IVSurfaceEvalResults,
        IVSurfaceTrainConfig,
        IVSurfaceTrainResults,
    ],
    ABC,
):
    """
    Abstract data class for models that fit iv surfaces.
    """

    def __init__(self, name: str, learnable: bool) -> None:
        self.name = name
        self.learnable = learnable

    @abstractmethod
    def fit(self, eval_config: IVSurfaceEvalConfig) -> IVSurfaceEvalResults: ...
