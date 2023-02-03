import datetime as dt
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics as stat
import scipy as sc
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from pandas_datareader import data as pdr

# Yahoo finance and pandas_datareader have some bugs.
# Using the function below from yfinance fixes it
yf.pdr_override()



def getData(stocks, start, end):
    df = pdr.get_data_yahoo(stocks, start, end).dropna(how = "all")
    stockPrices = df["Close"]

    # Returns = Price / Price (lagged by 1)
    returns = stockPrices.pct_change()

    meanReturns = returns.mean()

    # Covariance matrix
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix):
    # Weighted Returns
    weightedReturns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    std2 = np.std(np.dot(weights.T, np.dot(covMatrix, weights)), "omitnan") * np.std(252)
    return weightedReturns, std

# Calculating the time period
end = dt.datetime.now()
start = end - dt.timedelta(days=10000)
'''
# Creating stocklist
stockList = list()
# Reading all ASXcompanies
asxCompanies = pd.read_csv('20200601-etfs.csv')
# Retrieving all the codes
for code in asxCompanies["Code"]:
    stockList.append(code)
# Adding the suffix of .AX
asxStocks = [i + ".AX" for i in stockList]
'''
stocks = ["NIO", "VGS.AX", "VOO"]
weights = np.array([.3, .4, .3])
meanReturns, covMatrix = getData(stocks, start, end)

returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

print(round(returns*100, 2), round(std*100,2))