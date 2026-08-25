from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd

from src.core.experiments.evaluation import EvalConfig, EvalResults
from src.core.experiments.train import TrainConfig, TrainResults
from src.core.metrics.base import Metric
from src.core.models.base import Model

EC = TypeVar("EC", bound=EvalConfig)
ER = TypeVar("ER", bound=EvalResults)
TC = TypeVar("TC", bound=TrainConfig)
TR = TypeVar("TR", bound=TrainResults)


class Pipeline(ABC, Generic[EC, ER, TC, TR]):
    """
    General pipeline algorithm.

    Note that everything is very sensible to the names of the datasets and the models.
    Training results are made by pairs (train data, model).
    Testing results are mad eby triplets (train data, test data, model).

    Attributes:
    TODO
    """

    def __init__(
        self,
        train_configs: Sequence[TC],
        test_configs: Sequence[EC],
        models: Sequence[Model[EC, ER, TC, TR]],
        metrics: Sequence[Metric[EC, ER]],
        problem: str,
    ):
        """
        Initializes all the attributes.

        Also ensures that all necessary directories are created.
        """

        self.train_configs = train_configs
        self.test_configs = test_configs
        self.models = models
        self.metrics = metrics

        self.path = Path(problem)

        self.results_path = self.path / "results"
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.train_results_path = self.results_path / "train"
        self.train_results_path.mkdir(parents=True, exist_ok=True)

        self.test_results_path = self.results_path / "test"
        self.test_results_path.mkdir(parents=True, exist_ok=True)

        # Create training and testing directories

        for test_config in self.test_configs:
            for model in self.models:
                if model.learnable:
                    for train_config in self.train_configs:
                        p = self._get_train_path(model, train_config)
                        p.mkdir(parents=True, exist_ok=True)

                        p = self._get_test_path(model, train_config, test_config)
                        p.mkdir(parents=True, exist_ok=True)

                else:
                    p = self._get_test_path(model, None, test_config)
                    p.mkdir(parents=True, exist_ok=True)

    def _get_train_path(self, model: Model, train_config: TC) -> Path:
        """
        Gets path for training directory of the model.
        """
        return self.train_results_path / model.name / train_config.name

    def _get_test_path(
        self, model: Model, train_config: TC | None, test_config: EC
    ) -> Path:
        """
        Gets path for training directory of the model.
        """
        if train_config is None:
            return (
                self.test_results_path / model.name / "NoTrain" / test_config.name
            )  # no train so results live in same depth
        else:
            return (
                self.test_results_path
                / model.name
                / train_config.name
                / test_config.name
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
    def _load_results(self, path: Path) -> ER:
        """
        Loads results from path.
        """
        ...

    def _train(self,train_config, model, forcetrain):
        """
        Trains and saves a model.
        """
        train_path = self._get_train_path(model, train_config)
        model_path = train_path / "trained_model.pt"

        if model.learnable and ((not model_path.exists()) or forcetrain):
            print(f"Learning {model.name} for dataset {train_config.name}.")
            train_results = model.learn(train_config)
            print("Learnt.")

            train_results.save(train_path)

            print(
                f"Saving {model.name} for dataset {train_config.name} to {model_path}."
            )
            model.save(model_path)
            print("Saved.")

    def _fit(self, train_config, test_config, model, forcefit):
        """
        Fits each model and applies the metrics.
        """
        
        results_path = self._get_test_path(
            model, train_config, test_config
        )

        if not self._results_exist(results_path) or forcefit:
            if train_config == None:
                print(f"Fitting {model.name} for dataset {test_config.name}.")
            else:
                print(f"Fitting {model.name} with training in {train_config.name} for dataset {test_config.name}.")

            results = model.fit(test_config)
        else:
            results = self._load_results(results_path)
        
        if train_config == None:
             row = ["NoTrain", test_config.name, model.name]
        else:
             row = [train_config.name, test_config.name, model.name]

        for metric in self.metrics:
            row.append(metric(test_config, results))
        results.save(results_path) # metrics may add columns to surface info

        return row


    def run(self, forcetrain: bool = False, forcefit: bool = False):
        """
        Runs the pipeline.

        Args:
            forcetrain (bool): Retrain models even if trained models already exist.
            forcefit (bool): Refit the models even if results already exist.
        """
        final_stats = pd.DataFrame(
            columns=[
                "train_dataset",
                "test_dataset",
                "model",
                *[metric.name for metric in self.metrics],
            ]
        )

        for model in self.models:
            if model.learnable:
                for train_config in self.train_configs:

                    self._train(train_config, model, forcetrain)

                    for test_config in self.test_configs:

                        row = self._fit(train_config, test_config, model, forcefit)

                        final_stats.loc[len(final_stats)] = row

            elif not model.learnable:
                for test_config in self.test_configs:

                    row = self._fit(None, test_config, model, forcefit)

                    final_stats.loc[len(final_stats)] = row


        print(final_stats)

        final_stats.to_csv(self.results_path / "final_statistics.csv", index=False)
