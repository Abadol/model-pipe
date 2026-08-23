# IVSurface FItting

Considers the problem of fitting an implied volatility surface: given implied volatilities for some asset at some points (Strike, Maturity), can we reproduce the whole surface?

There is extensive literature in this problem, one can find a survey of methodologies in "Implied Volatility Surface: Construction Methodologies and Characteristics" by C. Homescu.

The run is an example, it is up to the user to generate the datasets.

## Data

The data structures for this problem are as follows:
- Train data: a csv with the columns ["id","logmoneyness","maturity","iv"], it may contain others such as call prices or quotes, where iv may be the mid, but these are not strictly required.
- Test data: two csv with the same structure as train data, a context set with the qoutes the model observes for fitting, and a "truth" dataset, which tells the model the pairs (logmoneyness,maturity) where to evaluate the model, and serves as a ground truth for the mterics to measure losses. It also contains an optional grid config, which gives the model a standard grid at which to evaluate, which is needed later by the NA measuring metrics.

## Datasets

|Dataset|Description|Reference|Inputs|
|-|-|-|-|
|Heston|Fourier inversion to comppute call prices at given points.|...|...|
|2013|IV Surfaces for a collection of assets in 2013-01.|...|...|

## Models
|Model|Description|References|
|-|-|-|
|SSVI|Surface Stochastic Volatility Inspired parametrization, parametrization of the surface with easy to check NA conditions.|"Arbitrage-free SVI volatility surfaces" J. Gatheral, A. Jacquier|
|CrossAttnEncMLPDecoder|Encodes observations using set transformer and decodes with fully connected network.|Custom|

## Metrics
|Metric|Description|
|-|-|
|RMSE| Average of the root mean square error per surface.|

## TODO

- Currently heldout data and context data for evaluation are the same, fix this.
- Properly save results, e.g. metric per id, matches well with next todo:
- Save encodings: ssvi parameters, neural encoding, ... .
- Add grid evaluation and na metrics.
- Datasets must have the same number sof rows per id for crossattn model to work, fix this.
- Add more models and datasets, do heston yourself (only thing vibe coded).
