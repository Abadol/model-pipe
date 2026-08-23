from pathlib import Path

import pandas as pd

from src.core.experiments.evaluation import EvalConfig, EvalResults


class IVSurfaceEvalConfig(EvalConfig):
    """
    Has all the info required to run a model Prediction/fit for iv surface fitting.

    Attributes:
        datapath (Path): Path to the data for fitting.
        name (str)
    """

    def __init__(self, datapath: Path, name: str) -> None:

        self.datapath = datapath
        self.name = name

    def getdata(self) -> pd.DataFrame:
        return pd.read_csv(self.datapath)


class IVSurfaceEvalResults(EvalResults):
    """
    Represents the results of an iv surface evaluation.

    Holds the results on the testing set and in the grid for NA metrics.
    """

    def __init__(
        self,
        fit_results: pd.DataFrame | None = None,
        grid_results: pd.DataFrame | None = None,
    ) -> None:

        if fit_results is None:
            self.fit_results = pd.DataFrame()
        else:
            self.fit_results = fit_results  # used for losses metrics
        if grid_results is None:
            self.grid_results = pd.DataFrame()
        else:
            self.grid_results = grid_results  # used for no arbitrage metrics

    def save(self, path: Path):
        self.fit_results.to_csv(path / "fit_results.csv", index=False)
        self.grid_results.to_csv(path / "grid_results.csv", index=False)

    def load(self, path: Path):
        try:
            self.fit_results = pd.read_csv(path / "fit_results.csv")
        except pd.errors.EmptyDataError:
            print(f"No results in path {path}.")
            self.fit_results = pd.DataFrame()
        try:
            self.grid_results = pd.read_csv(path / "grid_results.csv")
        except pd.errors.EmptyDataError:
            self.grid_results = pd.DataFrame()
