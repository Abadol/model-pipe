"""
Uses the data in a directory to create a learn and a test file.

It asusmes that the directory has pairs of csv files of the form *options.csv and *stocks.csv,
and constructs one single csv with alll the data.

It is intended for the sample data downloadeable at https://historicaldata.net/options.html, just download unzip and put
them into the directory ivsurfacefitting/datasets/2013-raw, then create an empty directory ivsurfacefitting/datasets/2013.

Data cleaning done is that we only take surfaces if they have more than a certain amount of data points with traded volume.
"""

import os
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

PATH = Path("ivsurfacefitting/datasets/2013-raw")
DATA_POINTS = 20

days = os.listdir(PATH)

days = sorted(list(set([str(s[:10]) for s in days])))

pd.DataFrame(columns=["id","logmoneyness","maturity","iv"]).set_index("id").to_csv("ivsurfacefitting/datasets/2013/2013_learn.csv")
pd.DataFrame(columns=["id","logmoneyness","maturity","iv"]).set_index("id").to_csv("ivsurfacefitting/datasets/2013/2013_predict.csv")

i = 0 #enumerate fucks up the tqdm.
for day in tqdm(days):

    options = pd.read_csv(PATH / (day + "options.csv"))
    options = options[options["type"] == "call"]
    options = options[options["volume"] > 0]
    stocks = pd.read_csv(PATH / (day + "stocks.csv"))

    for ticker,option_data in options.groupby("underlying"):

        if len(option_data) >= DATA_POINTS:

            stock_data = stocks[stocks["symbol"] == ticker]

            id_ = str(ticker) + "@" + day
            logmoneyness = np.log(option_data["strike"] / stock_data.iloc[0]["close"])

            date = pd.to_datetime(day)
            option_data["expiration"] = pd.to_datetime(option_data["expiration"])
            maturity = (option_data["expiration"] - date).dt.days / 365

            iv = option_data["implied_volatility"]

            df = pd.DataFrame({
                "id": [id_] * len(logmoneyness),
                "logmoneyness": logmoneyness.to_numpy(),
                "maturity": maturity.to_numpy(),
                "iv": iv,
            }).set_index("id").sample(n = DATA_POINTS)

            if i<len(days)*0.9:
                df.to_csv(
                        "ivsurfacefitting/datasets/2013/2013_learn.csv",
                        mode = 'a',
                        header=False,
                        index=True,
                        )
            else:
                df.to_csv(
                        "ivsurfacefitting/datasets/2013/2013_predict.csv",
                        mode = 'a',
                        header=False,
                        index=True,
                        )

    i += 1

        


