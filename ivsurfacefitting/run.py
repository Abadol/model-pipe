import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--forcelearn", action="store_true")
parser.add_argument("--forcepredict", action="store_true")
args = parser.parse_args()

from pathlib import Path

import pandas as pd

from src.ivsurfacefitting.experiments.predict import IVSurfacePredictConfig
from src.ivsurfacefitting.experiments.learn import IVSurfaceLearnConfig
from src.ivsurfacefitting.metrics.rmse import RMSE
from src.ivsurfacefitting.metrics.mae import MAE
from src.ivsurfacefitting.metrics.na import NA
from src.ivsurfacefitting.models.cross_attn_set_encoder_mlp_decoder import (
    CrossAttnEncodeMLPDecoder,
)
from src.ivsurfacefitting.models.ssvi import SSVI
from src.ivsurfacefitting.pipeline.base import IVSurfacePipeline

learn_configs = [
    IVSurfaceLearnConfig(
        Path("ivsurfacefitting/datasets/heston/heston_learn.csv"), "heston_learn"
    ),
    IVSurfaceLearnConfig(
        Path("ivsurfacefitting/datasets/2013/2013_learn.csv"), "2013_learn"
    ),
]

# uses exactly half the data per index for context
def splitter(df: pd.DataFrame):
    end = []
    for _,group in df.groupby(level=0):
        end.append(group.sample(n=len(group)))
    return pd.concat(end)


predict_configs = [
    IVSurfacePredictConfig(
        Path("ivsurfacefitting/datasets/heston/heston_predict.csv"),
        "heston_predict",
        splitter,
    ),
    IVSurfacePredictConfig(
        Path("ivsurfacefitting/datasets/2013/2013_predict.csv"), "2013_predict", splitter
    ),
]

pipeline = IVSurfacePipeline(
    learn_configs,
    predict_configs,
    [CrossAttnEncodeMLPDecoder(), SSVI()],
    [RMSE(), MAE(), NA()],
)

pipeline.run(forcelearn=args.forcelearn, forcepredict=args.forcepredict)
