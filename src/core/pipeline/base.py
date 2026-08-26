from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd

from src.core.experiments.learn import LearnConfig, LearnResults
from src.core.experiments.predict import PredictConfig, PredictResults
from src.core.metrics.base import Metric
from src.core.models.base import Model

LC = TypeVar("LC", bound=LearnConfig)
LR = TypeVar("LR", bound=LearnResults)
PC = TypeVar("PC", bound=PredictConfig)
PR = TypeVar("PR", bound=PredictResults)


class Pipeline(ABC, Generic[LC,LR,PC,PR]):
    """
    General pipeline algorithm.

    Note that everything is very sensible to the names of the datasets and the models.
    Learning results are made by pairs (learn data, model).
    Predicting results are mad eby triplets (learn data, predict data, model).

    Attributes:
    TODO
    """

    def __init__(
        self,
        learn_configs: Sequence[LC],
        predict_configs: Sequence[PC],
        models: Sequence[Model[LC, LR, PC, PR]],
        metrics: Sequence[Metric[PC, PR]],
        problem: str,
    ):
        """
        Initializes all the attributes.

        Also ensures that all necessary directories are created.
        """

        self.learn_configs = learn_configs
        self.predict_configs = predict_configs
        self.models = models
        self.metrics = metrics

        self.path = Path(problem)

        self.results_path = self.path / "results"
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.learn_results_path = self.results_path / "learn"
        self.learn_results_path.mkdir(parents=True, exist_ok=True)

        self.predict_results_path = self.results_path / "predict"
        self.predict_results_path.mkdir(parents=True, exist_ok=True)

        # Create learning and predicting directories

        for predict_config in self.predict_configs:
            for model in self.models:
                if model.learnable:
                    for learn_config in self.learn_configs:
                        p = self._get_learn_path(model, learn_config)
                        p.mkdir(parents=True, exist_ok=True)

                        p = self._get_predict_path(model, learn_config, predict_config)
                        p.mkdir(parents=True, exist_ok=True)

                else:
                    p = self._get_predict_path(model, None, predict_config)
                    p.mkdir(parents=True, exist_ok=True)

    def _get_learn_path(self, model: Model, learn_config: LC) -> Path:
        """
        Gets path for learning directory of the model.
        """
        return self.learn_results_path / model.name / learn_config.name

    def _get_predict_path(
        self, model: Model, learn_config: LC | None, predict_config: PC
    ) -> Path:
        """
        Gets path for learning directory of the model.
        """
        if learn_config is None:
            return (
                self.predict_results_path / model.name / "Nolearn" / predict_config.name
            )  # no learn so results live in same depth
        else:
            return (
                self.predict_results_path
                / model.name
                / learn_config.name
                / predict_config.name
            )

    @abstractmethod
    def _results_exist(self, path: Path) -> bool:
        """
        Checks if results exist in the given directory.

        It depends on the problem to what it means for the results to exist.

        Args:
            path (Path): Where to look for results.
        """
        ...

    @abstractmethod
    def _load_results(self, path: Path) -> PR:
        """
        Loads results from path.
        """
        ...

    def _learn(self,learn_config, model, forcelearn):
        """
        learns and saves a model.
        """
        learn_path = self._get_learn_path(model, learn_config)
        model_path = learn_path / "learned_model.pt"

        if model.learnable and ((not model_path.exists()) or forcelearn):
            print(f"Learning {model.name} for dataset {learn_config.name}.")
            learn_results = model.learn(learn_config)
            print("Learnt.")

            learn_results.save(learn_path)

            print(
                f"Saving {model.name} for dataset {learn_config.name} to {model_path}."
            )
            model.save(model_path)
            print("Saved.")
        elif model.learnable and model_path.exists() and not forcelearn:
            model.load(model_path)


    def _predict(self, learn_config, predict_config, model, forcepredict):
        """
        predicts each model and applies the metrics.
        """
        
        results_path = self._get_predict_path(
            model, learn_config, predict_config
        )

        if not self._results_exist(results_path) or forcepredict:
            if learn_config is None:
                print(f"predictting {model.name} for dataset {predict_config.name}.")
            else:
                print(f"predictting {model.name} with learning in {learn_config.name} for dataset {predict_config.name}.")

            results = model.predict(predict_config)
        else:
            results = self._load_results(results_path)
        
        if learn_config is None:
             row = ["Nolearn", predict_config.name, model.name]
        else:
             row = [learn_config.name, predict_config.name, model.name]

        for metric in self.metrics:
            row.append(metric(predict_config, results))
        results.save(results_path) # metrics may add columns to surface info

        return row


    def run(self, forcelearn: bool = False, forcepredict: bool = False):
        """
        Runs the pipeline.

        Args:
            forcelearn (bool): Relearn models even if learned models already exist.
            forcepredict (bool): Repredict the models even if results already exist.
        """
        final_stats = pd.DataFrame(
            columns=[
                "learn_dataset",
                "predict_dataset",
                "model",
                *[metric.name for metric in self.metrics],
            ]
        )

        rows = []

        for model in self.models:
            if model.learnable:
                for learn_config in self.learn_configs:

                    self._learn(learn_config, model, forcelearn)

                    for predict_config in self.predict_configs:

                        row = self._predict(learn_config, predict_config, model, forcepredict)

                        rows.append(row)

            elif not model.learnable:
                for predict_config in self.predict_configs:

                    row = self._predict(None, predict_config, model, forcepredict)

                    rows.append(row)

        final_stats = pd.DataFrame(
                rows,
                columns=[
                    "learn_dataset",
                    "predict_dataset",
                    "model",
                    *[metric.name for metric in self.metrics],
                ]
        )            
        print(final_stats)

        final_stats.to_csv(self.results_path / "final_statistics.csv", index=False)
