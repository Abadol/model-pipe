from pathlib import Path

from src.ivsurfacefitting.experiments.evaluation import IVSurfaceEvalConfig
from src.ivsurfacefitting.experiments.train import IVSurfaceTrainConfig
from src.ivsurfacefitting.metrics.rmse import RMSE
from src.ivsurfacefitting.metrics.mae import MAE
from src.ivsurfacefitting.models.cross_attn_set_encoder_mlp_decoder import (
    CrossAttnEncodeMLPDecoder,
)
from src.ivsurfacefitting.models.ssvi import SSVI
from src.ivsurfacefitting.pipeline.base import IVSurfacePipeline

train_configs = [
    IVSurfaceTrainConfig(
        Path("ivsurfacefitting/datasets/heston/heston_train.csv"), "heston_train"
    ),
    IVSurfaceTrainConfig(
        Path("ivsurfacefitting/datasets/2013/real_train.csv"), "real_train"
    ),
]

splitter = lambda x: x

test_configs = [
    IVSurfaceEvalConfig(
        Path("ivsurfacefitting/datasets/heston/heston_test.csv"),
        "heston_test",
        splitter,
    ),
    IVSurfaceEvalConfig(
        Path("ivsurfacefitting/datasets/2013/real_test.csv"), "real_test", splitter
    ),
]

pipeline = IVSurfacePipeline(
    train_configs,
    test_configs,
    [CrossAttnEncodeMLPDecoder(), SSVI()],
    [RMSE(), MAE()],
)

pipeline.run(forcetrain=False, forcefit=False)
