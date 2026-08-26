from pathlib import Path

import pandas as pd

from src.core.experiments.learn import LearnConfig,LearnResults


class IVSurfaceLearnConfig(LearnConfig):
    """
    Has all the info required for an IVSurfaceModel to Learn.

    Config can be useful to compare things like optimizer or learning rate used.

    Attributes:
        datapath (Path): Path to the learning data.
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


class IVSurfaceLearnResults(LearnResults):
    """
    Results from training.

    Currently it only holds the validation loss over epochs.
    """

    def __init__(self,val_losses: pd.DataFrame | None = None) -> None:
        if val_losses is None:
            self.val_losses = pd.DataFrame()
        else:
            self.val_losses = val_losses

    def save(self, path: Path) -> None:
        self.val_losses.to_csv(path / "validation_losses.csv")
