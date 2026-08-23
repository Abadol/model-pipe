import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from src.ivsurfacefitting.experiments.evaluation import (
    IVSurfaceEvalConfig,
    IVSurfaceEvalResults,
)
from src.ivsurfacefitting.models.base import IVSurfaceModel


def eval_ssvi(k, T, sigma0, sigmainf, rho, eta, gamma, lamb):

    sigma = sigmainf + (sigma0 - sigmainf) * np.exp(-lamb * T)

    v = sigma * sigma * T

    phi = eta / (v**gamma)

    w = v / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho * rho))

    return w


class SSVI(IVSurfaceModel):
    """
    Implements the ssvi fitting method naively.

    Remember the SSVI parametrization, wich is given by the formula:

        w(k,T) = TODO LATEX
    """

    def __init__(self, name: str = "SSVI") -> None:
        super().__init__(name, learnable=False)

    def fit(self, eval_config: IVSurfaceEvalConfig) -> IVSurfaceEvalResults:

        data = eval_config.getdata()[["id", "logmoneyness", "maturity", "iv"]]

        final_results = []

        for _, surface in tqdm(data.groupby("id")):

            def func(params, surface=surface):

                pred = eval_ssvi(surface["logmoneyness"], surface["maturity"], *params)

                return np.linalg.norm(surface["iv"] - pred)

            params0 = [0.25, 0.2, -0.5, 1.0, 0.5, 1.5]

            parambounds = [
                [0.001, 2.0],
                [0.001, 2.0],
                [-0.999, 0.999],
                [0.001, 5.0],
                [0.0, 1.0],
                [0.001, 20.0],
            ]

            minimizer = minimize(
                func,
                x0=params0,
                method="L-BFGS-B",
                bounds=parambounds,
            )

            predictions = eval_ssvi(
                surface["logmoneyness"], surface["maturity"], *(minimizer.x)
            )

            results = surface[["id", "logmoneyness", "maturity"]].copy()

            results["iv"] = predictions

            final_results.append(results)

        final_results = pd.concat(final_results, ignore_index=True)

        return IVSurfaceEvalResults(final_results, pd.DataFrame())
