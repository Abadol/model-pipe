from pathlib import Path

import numpy as np
import pandas as pd

from src.core.experiments.predict import PredictConfig,PredictResults


class IVSurfacePredictConfig(PredictConfig):
    """
    Has all the info required to run an ivsurfacefitting model prediction algorithm.

    The splitter gets a context data frame for the model to predict on, and the predict data set remains
    the true data, which is used to know where to predict and for metrics later. The grid is required to facilitate the
    no arbitrage metrics later and also easier plotting. If needed the grid's bounds and amount of points
    could be changed.

    Attributes:
        datapath (Path): Path to the data for predicting.
        splitter (IVSplitter): Gets a context subset of data.
        name (str)
    """

    def __init__(self, datapath: Path, name: str, splitter = None) -> None:

        self.datapath = datapath
        self.name = name
        self.splitter = splitter

    def _get_context_data(self) -> pd.DataFrame:
        if self.splitter == None:
            return pd.read_csv(self.datapath, index_col="id")
        else:
            return self.splitter(pd.read_csv(self.datapath, index_col="id"))

    def _get_predict_data(self) -> pd.DataFrame:
        """
        Gets the predict data coordinates and tru values.

        Note that contexxt is a subset of this, and also gets used for metrics, this is by design, since even if context
        may bias the metrics it does so for all models anyways, and in my opinion the interesting thing is the L2 norm 
        difference between the interpolated surface and the real one anyways.
        """
        return pd.read_csv(self.datapath, index_col="id")

    def _get_grid(self) -> pd.DataFrame:
        logmoneyness = np.linspace(-0.4, 0.4, 20)
        maturity = np.linspace(0.01, 3, 20)
        rows = []
        for m in maturity:
            for l in logmoneyness:
                rows.append([l, m])
        return pd.DataFrame(rows, columns=["logmoneyness", "maturity"])

    def getdata(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self._get_predict_data(),
            self._get_context_data(),
            self._get_grid(),
        )


class IVSurfacePredictResults(PredictResults):
    """
    Represents the results of an iv surface prediction.

    This consists of three csv files. The per point results, called predict_results, which gives the
    output of the model in each of the surfaces and points in predict data. The grid_results, which does
    the same but only evaluated in the grid data points. And finally the surface_info, which for each surface (id)
    outputs relevant information, such as for example the encoding of the surface.

    The annoying empty initialization is required for loading.
    """

    def __init__(
        self,
        predict_results: pd.DataFrame | None = None,
        grid_results: pd.DataFrame | None = None,
        surface_info: pd.DataFrame | None = None,
    ) -> None:

        if predict_results is None:
            self.predict_results = pd.DataFrame()
        else:
            self.predict_results = predict_results
        if grid_results is None:
            self.grid_results = pd.DataFrame()
        else:
            self.grid_results = grid_results
        if surface_info is None:
            self.surface_info = pd.DataFrame()
        else:
            self.surface_info = surface_info

    def save(self, path: Path):
        self.predict_results.to_csv(path / "predict_results.csv")
        self.grid_results.to_csv(path / "grid_results.csv")
        self.surface_info.to_csv(path / "surface_info.csv")

    def load(self, path: Path):
        try:
            self.predict_results = pd.read_csv(path / "predict_results.csv", index_col="id")
        except pd.errors.EmptyDataError:
            self.predict_results = pd.DataFrame()
        try:
            self.grid_results = pd.read_csv(path / "grid_results.csv", index_col="id")
        except pd.errors.EmptyDataError:
            self.grid_results = pd.DataFrame()
        try:
            self.surface_info = pd.read_csv(path / "surface_info.csv", index_col="id")
        except pd.errors.EmptyDataError:
            self.surface_info = pd.DataFrame()
