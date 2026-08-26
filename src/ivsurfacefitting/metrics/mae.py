import numpy as np
import pandas as pd

from src.ivsurfacefitting.experiments.predict import (
    IVSurfacePredictConfig,
    IVSurfacePredictResults,
)
from src.ivsurfacefitting.metrics.base import IVMetric


class MAE(IVMetric):
    def __init__(self) -> None:
        super().__init__("mae")

    def __call__(self, config: IVSurfacePredictConfig, results: IVSurfacePredictResults):
        """
        Measures mrse of predicted.
        """
        real,_,_ = config.getdata()
        results_fit = results.predict_results
        real = real.sort_index()
        results_fit = results_fit.sort_index()
        
        if not real.index.equals(results_fit.index):
            raise ValueError("Index columns must match.")

        error = real["iv"].to_numpy() - results_fit["iv"].to_numpy()

        mae_per_id = (
            pd.DataFrame(error,index=real.index)
            .groupby(level=0)
            .apply(lambda x: np.mean(np.abs(x)))
        )


        results.surface_info["mae"] = mae_per_id

        return mae_per_id.mean()
