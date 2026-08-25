from pathlib import Path

import pandas as pd

from src.core.experiments.train import TrainConfig, TrainResults


class IVSurfaceTrainConfig(TrainConfig):
    """
    Has all the info required for an IVSurfaceModel to Learn.

    Config can be useful to compare things like optimizer or learning rate used.

    Attributes:
        datapath (Path): Path to the training data.
        name (str)
    """

    def __init__(self, datapath: Path, name: str, config: dict | None = None) -> None:

        self.datapath = datapath
        self.name = name
        if config is None:
            self.config = {}  # Config defaults.
        else:
            self.config = config

    def getdata(self) -> pd.DataFrame:
        """
        Returns the dataframe from its path.
        """
        return pd.read_csv(self.datapath, index_col="id")


class IVSurfaceTrainResults(TrainResults):
    """TODO"""

    # TODO

    def __init__(self) -> None:
        pass

    def save(self, path: Path) -> None:
        pass
