from abc import ABC, abstractmethod
from pathlib import Path


class PredictConfig(ABC):
    name: str


class PredictResults(ABC):
    @abstractmethod
    def save(self, path: Path): ...

    @abstractmethod
    def load(self, path: Path): ...
