from pathlib import Path

from src.core.pipeline.base import Pipeline
from src.ivsurfacefitting.experiments.predict import (
    IVSurfacePredictConfig,
    IVSurfacePredictResults,
)
from src.ivsurfacefitting.experiments.learn import (
    IVSurfaceLearnConfig,
    IVSurfaceLearnResults,
)
from src.ivsurfacefitting.metrics.base import IVMetric
from src.ivsurfacefitting.models.base import IVSurfaceModel


class IVSurfacePipeline(
    Pipeline[
        IVSurfaceLearnConfig,
        IVSurfaceLearnResults,
        IVSurfacePredictConfig,
        IVSurfacePredictResults,
    ]
):
    """
    Pipeline for implied volatility surface fitting.

    Attributes:
    TODO
    """

    def __init__(
        self,
        train_configs: list[IVSurfaceLearnConfig],
        test_configs: list[IVSurfacePredictConfig],
        models: list[IVSurfaceModel],
        metrics: list[IVMetric],
    ):

        super().__init__(
            train_configs, test_configs, models, metrics, "ivsurfacefitting"
        )

    def _results_exist(self, path: Path) -> bool:
        """
        Checks if results exist in the given directory.

        Args:
            path (Path): Where to look for results.
        """
        pred_path = path / "predict_results.csv"
        grid_path = path / "grid_results.csv"
        surfaces_path = path / "surface_info.csv"
        return pred_path.exists() and grid_path.exists() and surfaces_path.exists()

    def _load_results(self, path: Path) -> IVSurfacePredictResults:
        """
        Loads results from path.
        """
        results = IVSurfacePredictResults()
        results.load(path)
        return results
