from abc import ABC, abstractmethod
from pathlib import Path


class LearnConfig(ABC):
    name: str


class LearnResults(ABC):
    @abstractmethod
    def save(self, path: Path): ...
