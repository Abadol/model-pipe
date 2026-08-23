from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TrainConfig(ABC):
    name: str

    @abstractmethod
    def getdata(self) -> Any: ...


class TrainResults(ABC):
    @abstractmethod
    def save(self, path: Path): ...
