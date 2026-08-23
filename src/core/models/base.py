from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from src.core.experiments.evaluation import (
    EvalConfig,
    EvalResults,
)
from src.core.experiments.train import (
    TrainConfig,
    TrainResults,
)

EC = TypeVar("EC", bound=EvalConfig)
ER = TypeVar("ER", bound=EvalResults)
TC = TypeVar("TC", bound=TrainConfig)
TR = TypeVar("TR", bound=TrainResults)


class Model(ABC, Generic[EC, ER, TC, TR]):
    """
    Abstract data class for models.
    """

    learnable: bool
    name: str

    def __init__(self, name: str, learnable: bool) -> None:
        self.name = name
        self.learnable = learnable

    @abstractmethod
    def fit(self, eval_config: EC) -> ER:
        """
        Handles the fitting for the model.

        Note that it is the models fit method responsability to get whatever data structure the model uses for fitting.
        """
        ...

    def learn(self, train_config: TC) -> TR:
        """
        Learning algorithm.
        """
        if not self.learnable:
            raise ValueError(f"{self.name} cannot learn.")
        else:
            raise ValueError(f"{self.name} learning not implemented.")

    def load(self, path: Path) -> None:
        """
        Loads the model from a file.
        """
        if not self.learnable:
            raise ValueError(f"{self.name} isnt loadeable.")
        else:
            raise ValueError(f"{self.name} loading not implemented.")

    def save(self, path: Path) -> None:
        """
        Saves the model to a file.
        """
        if not self.learnable:
            raise ValueError(f"{self.name} cannot be saved.")
        else:
            raise ValueError(f"{self.name} saving not implemented.")
