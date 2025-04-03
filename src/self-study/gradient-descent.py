import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# mean squared error
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

data = pd.read_csv("data/clean_weather.csv", index_col=0)
data = data.ffill()
# data.corr()
#
prediction = lambda x, w1=.82, b = 11.99: x * w1 + b

tmax_bins = pd.cut(data["tmax"], 25)

ratios = (data["tmax_tomorrow"] - 11.99) / data["tmax"]
binned_ratio = ratios.groupby(tmax_bins).mean()
binned_tmax = data["tmax"].groupby(tmax_bins).mean()

# print(data.corr())
# print(tmax_bins)
# print(binned_ratio)
# data.plot.scatter("tmax", "tmax_tomorrow")
#
# print(mse(data["tmax_tomorrow"], prediction(data["tmax"])))
# print(mse(data["tmax_tomorrow"], prediction(data["tmax"], .82, 13)))

#plt.plot([30, 120], [prediction(30), prediction(120)], 'green')
plt.scatter(binned_tmax, binned_ratio)

plt.show()