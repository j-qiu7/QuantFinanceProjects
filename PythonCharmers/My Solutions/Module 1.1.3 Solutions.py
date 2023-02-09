from pandas_datareader import data as pdr
import datetime
import altair as alt
import yfinance as yf
import altair_viewer
from scipy import stats

yf.pdr_override()

today = datetime.datetime.now()
begin = today - datetime.timedelta(days = 1000)

"""
Exercises

1. Compute the increase in price for each day (Close - Open)
2. Plot a histogram of these increases
3. Investigate the `stats.skew` and `stats.kurtosis` functions to compute the third and fourth moment of the dataset.

# download dataframe
aapl = pdr.get_data_yahoo("AAPL", start=begin, end=today)

# Exercise 1
aapl["Gain"] = aapl["Close"] - aapl["Open"]

# Exercise 2

aaplGain = alt.Chart(aapl).mark_bar().encode(
    alt.X("Gain", bin=alt.Bin(maxbins=100)),
    y = "count()"
)

altair_viewer.show(aaplGain)
# Exercise 3

skew = stats.skew(aapl['Gain'])
print("The skew is {}".format(skew))
kurtosis = stats.kurtosis(aapl['Gain'])
print("The kurtosis is {}".format(kurtosis))
"""
"""
Extended Exercise
Quandl has a python module for extracting datasets. The documentation is available at https://www.quandl.com/tools/python

Install this module, and review the documentation to obtain stock prices for the following four tech giants:
* IBM
* Google
* Apple (more up-to-date than our dataset)
* Amazon

Compute the skew and kurtosis of each stock, and compare the results. Looking at the histograms of the stock prices, 
the skew and the kurtosis, what does this tell you about the usefulness of these moments?
"""

stockList = ["IBM", "GOOGL", "AAPL", "AMZN"]

stockDataframes = list()

for stock in stockList:
    df = pdr.get_data_yahoo(stock, start=begin, end=today)
    stockDataframes.append(df)
for i, stockDf in enumerate(stockDataframes):
    stockDf["Gain"] = stockDf["Close"] - stockDf["Open"]
    skew = stats.skew(stockDf["Gain"])
    kurtosis = stats.kurtosis(stockDf['Gain'])
    #print(f"The stock is {stockList[i]}.\nSkew is {skew}.\nKurtosis is {kurtosis}\n")
    chart = alt.Chart(stockDf).mark_bar().encode(
        alt.X("Gain", bin = alt.Bin(maxbins=100)),
        y = "count()"
    ).properties(title = stockList[i])
    altair_viewer.show(chart)
