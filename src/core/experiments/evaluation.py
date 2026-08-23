from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class EvalConfig(ABC):
    name: str

    @abstractmethod
    def getdata(self) -> Any: ...


class EvalResults(ABC):
    @abstractmethod
    def save(self, path: Path): ...

    @abstractmethod
    def load(self, path: Path): ...
