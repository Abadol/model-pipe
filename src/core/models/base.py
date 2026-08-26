from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from src.core.experiments.learn import (
    LearnConfig,
    LearnResults,
)
from src.core.experiments.predict import (
    PredictConfig,
    PredictResults,
)


LC = TypeVar("LC", bound=LearnConfig)
LR = TypeVar("LR", bound=LearnResults)
PC = TypeVar("PC", bound=PredictConfig)
PR = TypeVar("PR", bound=PredictResults)

class Model(ABC, Generic[LC,LR,PC,PR]):
    """
    Abstract data class for models.
    """

    learnable: bool
    name: str

    def __init__(self, name: str, learnable: bool) -> None:
        self.name = name
        self.learnable = learnable

    def learn(self, learn_config: LC) -> LR:
        """
        Learning algorithm.
        """
        if not self.learnable:
            raise ValueError(f"{self.name} cannot learn.")
        else:
            raise ValueError(f"{self.name} learning not implemented.")


    @abstractmethod
    def predict(self, predict_config: PC) -> PR:
        """
        Handles the prediction for the model.

        Note that it is the models predict method responsability to get whatever data structure the model uses for prediction from the config.
        """
        ...

    def load(self, path: Path) -> None:
        """
        Loads the model.
        """
        if not self.learnable:
            raise ValueError(f"{self.name} isnt loadeable.")
        else:
            raise ValueError(f"{self.name} loading not implemented.")

    def save(self, path: Path) -> None:
        """
        Saves the model.
        """
        if not self.learnable:
            raise ValueError(f"{self.name} cannot be saved.")
        else:
            raise ValueError(f"{self.name} saving not implemented.")
