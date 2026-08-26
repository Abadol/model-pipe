# IVSurface FItting

Considers the problem of fitting an implied volatility surface: given implied volatilities for some asset at some points (Strike, Maturity), can we reproduce the whole surface?

There is extensive literature in this problem, one can find a survey of methodologies in "Implied Volatility Surface: Construction Methodologies and Characteristics" by C. Homescu.

The run is an example, it is up to the user to generate the datasets.

## Data

The data structures for this problem are as follows:
- Learn data: a csv with the columns ["id","logmoneyness","maturity","iv"], it may contain others such as call prices or quotes, where iv may be the mid, but these are not strictly required.
- Predict data: a csv with the same structure as learn data. It also gets a splitter that tells us how one should split the test data into context and evaluation parts.

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

- Add more models and datasets, for example heston.
- Add visualization.
- Consider if final results should include the average of metric across surfaces or the statistics of the metric.
