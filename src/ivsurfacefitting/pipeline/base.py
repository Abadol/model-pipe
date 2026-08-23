from pathlib import Path

from src.core.pipeline.base import Pipeline
from src.ivsurfacefitting.experiments.evaluation import (
    IVSurfaceEvalConfig,
    IVSurfaceEvalResults,
)
from src.ivsurfacefitting.experiments.train import (
    IVSurfaceTrainConfig,
    IVSurfaceTrainResults,
)
from src.ivsurfacefitting.metrics.base import IVMetric
from src.ivsurfacefitting.models.base import IVSurfaceModel


class IVSurfacePipeline(
    Pipeline[
        IVSurfaceEvalConfig,
        IVSurfaceEvalResults,
        IVSurfaceTrainConfig,
        IVSurfaceTrainResults,
    ]
):
    """
    Pipeline for implied volatility surface fitting.

    Attributes:
    TODO
    """

    def __init__(
        self,
        train_configs: list[IVSurfaceTrainConfig],
        test_configs: list[IVSurfaceEvalConfig],
        models: list[IVSurfaceModel],
        metrics: list[IVMetric],
    ):

        super().__init__(train_configs, test_configs, models, metrics)

    def _results_exist(self, path: Path) -> bool:
        """
        Checks if results exist in the given directory.

        Args:
            path (Path): Where to look for results.
        """
        pred_path = path / "fit_results.csv"
        grid_path = path / "grid_results.csv"
        return pred_path.exists() and grid_path.exists()

    def _load_results(self, path: Path) -> IVSurfaceEvalResults:
        """
        Loads results from path.
        """
        results = IVSurfaceEvalResults()
        results.load(path)
        return results
