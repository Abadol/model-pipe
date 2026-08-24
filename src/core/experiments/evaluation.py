from abc import ABC, abstractmethod
from pathlib import Path


class EvalConfig(ABC):
    name: str


class EvalResults(ABC):
    @abstractmethod
    def save(self, path: Path): ...

    @abstractmethod
    def load(self, path: Path): ...
