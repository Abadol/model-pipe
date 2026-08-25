import numpy as np
import pandas as pd

from src.ivsurfacefitting.experiments.evaluation import (
    IVSurfaceEvalConfig,
    IVSurfaceEvalResults,
)
from src.ivsurfacefitting.metrics.base import IVMetric


class RMSE(IVMetric):
    def __init__(self) -> None:
        super().__init__("rmse")

    def __call__(self, config: IVSurfaceEvalConfig, results: IVSurfaceEvalResults):
        """
        Measures mrse of predicted.
        """
        real,_,_ = config.getdata()
        results_fit = results.test_results

        if not real["id"].equals(results_fit["id"]):
            raise ValueError("Index columns must match.")

        error = real["iv"].to_numpy() - results_fit["iv"].to_numpy()

        rmse_per_id = (
            pd.DataFrame(
                {
                    "id": real["id"],
                    "error": error,
                }
            )
            .groupby("id")["error"]
            .apply(lambda x: np.sqrt(np.mean(x**2)))
        )


        results.surface_info["rmse"] = rmse_per_id

        return rmse_per_id.mean()
