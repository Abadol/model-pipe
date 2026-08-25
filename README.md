# ModelPipe

A framework for fair comparison of models attempting to solve the same problem.

The idea is to try to standarize the running ad comparison of models that are trying to solve the same task, while keeping the comparison as fair as possible.

The general workflow is an schematic, each problem must implement its own details. It does NOT help if the objective is to run one single model, since the single model implementation still has to be fully done, but it is usefull in the case where one wants to compare multiple models across datasets and metrics.

## Overview

An orchestrator handles the main parts of model fitting and comparing, it makes sure everything is saved in the right directories, handles loading into the metrics and comparing all models. It also keeps all interfaces consistent.

The workflow for an experiment should be:
- Generate or download all datasets in an standarized way that feeds into all models.
- Implement the models to be used, and the metrics one wants to see.
- Choose datasets, models and metrics.
- Press run :D.

The orchestrator will take in all of the collections:
- Train data
- Test data
- Models
- Metrics
It then takes all possible combinations, and stores the trained models, results and metrics for easy comparison.

If you only want to fit or retrain one model, the esier way is to remove the existing results for that specific model, forcetrain or forcefit in pipelin will do so for every model which may be undesireable.

## Templating

Each problem should have their own README explaining a few things, an example can be found in ivsurfacefitting. First what problem they are trying to solve, the specific data types and requirements of everything, both of the datasets and the results. It should also include tables giving brief overvies of how each dataset is generated or downloaded, how each model works (along with reference to the relevant papers) and finally what each metric is measuring.
Optimally, each problem should have scripts for generating/downloading data with the right structure, and then a run script that constructs each of the objects, then simply calling pipeline.run should do everything.

## Requirements

Almost no libraries are used in the genral framework. This is on purpose so that one could in theory replicate this as easily as possible in other languages other than Python. Only heavily used library in the general framework would be Pandas and the inbulit pathlib library in Python.

## TODO

Apart from specific problem TODOs, found in their respective READMEs, fix randomness, add tests, decide on train/learn fit/eval/predict semantics, and reconsider Metrics.
