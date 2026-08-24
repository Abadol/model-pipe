from abc import ABC, abstractmethod
from pathlib import Path


class TrainConfig(ABC):
    name: str


class TrainResults(ABC):
    @abstractmethod
    def save(self, path: Path): ...
