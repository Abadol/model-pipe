import numpy as np
import pandas as pd

from src.ivsurfacefitting.experiments.predict import (
    IVSurfacePredictConfig,
    IVSurfacePredictResults,
)
from src.ivsurfacefitting.metrics.base import IVMetric


class NA(IVMetric):
    def __init__(self) -> None:
        super().__init__("na")

    def __call__(self, config: IVSurfacePredictConfig, results: IVSurfacePredictResults):
        """
        Measures precentage of points in grid that satisfy no arbitrage conditions.
        """

        grids = results.grid_results

        total_na = []

        for _,grid in grids.groupby(level=0):
            grid = grid.sort_values(["maturity", "logmoneyness"])
            matrows = []

            for m,maturity in grid.groupby("maturity"):
                # Finite differences for Durrlemans condition.
                k = maturity["logmoneyness"].to_numpy()
                dk = k[1] - k[0]
                iv = maturity["iv"].to_numpy()
                w = iv*iv * m
                wk = (w[2:] - w[:-2]) / (2*dk)
                wkk = (w[2:] - 2 * w[1:-1] + w[:-2]) / dk**2

                g = (1 - k[1:-1] * wk / (2 * w[1:-1]))**2 - (wk**2 / 4) * (1 / w[1:-1] + 1 / 4) + wkk / 2

                matrows.append((g >= 0).astype(int))

            logrows = []

            for k,logmoneyness in grid.groupby("logmoneyness"):
                # Checks total variance increases.
                mats = logmoneyness["maturity"].to_numpy()
                iv = logmoneyness["iv"].to_numpy()
                w = iv*iv * mats
                g = (w[1:] >= w[:-1]).astype(int)
                logrows.append(g)

            mat = np.asarray(matrows)[:-1] # crop where increments cant be cheked
            log = np.asarray(logrows).T
            log = log[:, 1:-1] # crop where durrleman cant be checked
            combined = mat & log

            proportion = combined.mean().astype(float)
            total_na.append(proportion)

        na_per_id = pd.DataFrame(total_na,index=grids.index.drop_duplicates(),columns=["na"])

        results.surface_info["na"] = na_per_id["na"]

        return na_per_id["na"].mean()
