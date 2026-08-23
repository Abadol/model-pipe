from pathlib import Path

from src.ivsurfacefitting.experiments.evaluation import IVSurfaceEvalConfig
from src.ivsurfacefitting.experiments.train import IVSurfaceTrainConfig
from src.ivsurfacefitting.metrics.mse import RMSE
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

test_configs = [
    IVSurfaceEvalConfig(
        Path("ivsurfacefitting/datasets/heston/heston_test.csv"), "heston_test"
    ),
    IVSurfaceEvalConfig(
        Path("ivsurfacefitting/datasets/2013/real_test.csv"), "real_test"
    ),
]

pipeline = IVSurfacePipeline(
    train_configs,
    test_configs,
    [CrossAttnEncodeMLPDecoder(), SSVI()],
    [RMSE()],
)

pipeline.run(forcetrain=False, forcefit=False)
