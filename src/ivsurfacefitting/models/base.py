from abc import ABC, abstractmethod

from src.core.models.base import Model

from src.ivsurfacefitting.experiments.learn import (
    IVSurfaceLearnConfig,
    IVSurfaceLearnResults,
)

from src.ivsurfacefitting.experiments.predict import (
    IVSurfacePredictConfig,
    IVSurfacePredictResults,
)

class IVSurfaceModel(
    Model[
        IVSurfaceLearnConfig,
        IVSurfaceLearnResults,
        IVSurfacePredictConfig,
        IVSurfacePredictResults,
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
    def predict(self, predict_config: IVSurfacePredictConfig) -> IVSurfacePredictResults: ...
