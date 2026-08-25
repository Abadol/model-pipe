from typing import cast

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

    return np.sqrt(w / T)


class SSVI(IVSurfaceModel):
    """
    Implements the ssvi fitting method naively.

    Remember the SSVI parametrization, wich is given by the formula:

        w(k, T) = v(T) / 2 * (
            1 + rho * phi(T) * k
            + sqrt((phi(T) * k + rho)**2 + 1 - rho**2)
        ),

    where w is total variance and:

        sigma(T) = sigma_inf + (sigma_0 - sigma_inf) * exp(-lambda * T),
        v(T)     = sigma(T)**2 * T,
        phi(T)   = eta / v(T)**gamma.

    then

        sigma_SSVI(k, T) = sqrt(w(k, T) / T)
    """

    def __init__(self, name: str = "SSVI") -> None:
        super().__init__(name, learnable=False)

    def fit(self, eval_config: IVSurfaceEvalConfig) -> IVSurfaceEvalResults:

        test, context, grid = eval_config.getdata()

        final_test = []

        final_grid = []

        surface_info = []

        test_groups = dict(tuple(test.groupby(level=0)))

        for id_, context_surface in tqdm(context.groupby(level=0)):

            test_surface = cast(pd.DataFrame, test_groups.get(id_, test.iloc[0:0]))

            k = context_surface["logmoneyness"].to_numpy()
            T = context_surface["maturity"].to_numpy()
            iv = context_surface["iv"].to_numpy()

            def func(params, k=k, T=T, iv=iv):

                pred = eval_ssvi(k, T, *params)
                diff = iv - pred

                return np.sum(diff**2)
            params0 = [0.25, 0.2, -0.5, 1.0, 0.5, 1.5]

            parambounds = [
                [0.001, 3.0],
                [0.001, 2.0],
                [-0.999, 0.999],
                [0.001, 5.0],
                [0.0, 2.0],
                [0.001, 20.0],
            ]

            minimizer = minimize(
                func,
                x0=params0,
                method="L-BFGS-B",
                bounds=parambounds,
            )

            test_predictions = eval_ssvi(
                test_surface["logmoneyness"], test_surface["maturity"], *(minimizer.x)
            )

            test_results = test_surface[[ "logmoneyness", "maturity"]].copy()

            test_results["iv"] = test_predictions

            final_test.append(test_results)
            
            grid_predictions = eval_ssvi(
                grid["logmoneyness"], grid["maturity"], *(minimizer.x)
            )

            grid_results = grid.copy()
            grid_results.index = pd.Index([id_]*len(grid), name = "id")

            grid_results["iv"] = grid_predictions

            final_grid.append(grid_results)

            surface_info.append([id_, *(minimizer.x)])

        final_results = pd.concat(final_test)

        final_grids = pd.concat(final_grid)

        final_info = pd.DataFrame(surface_info, columns=["id","sigma0"," sigmainf"," rho"," eta"," gamma"," lambda"]).set_index("id")

        return IVSurfaceEvalResults(final_results, final_grids, final_info)








