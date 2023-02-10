import pandas as pd
import altair as alt
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime 
import altair_viewer

yf.pdr_override()

today = datetime.datetime.now()
begin = today - datetime.timedelta(days = 1000)

aapl = pdr.get_data_yahoo("AAPL", start=begin, end=today)

aapl["Gain"] = aapl["Adj Close"].diff()
aapl.dropna(inplace=True)

print(f"skew is {stats.skew(aapl['Gain'])}")

print(f"kurtosis is {stats.kurtosis(aapl['Gain'])}")

aaplGains = alt.Chart(aapl.reset_index(), title = "Apple Gains").mark_bar().encode(
    alt.X("Gain", bin=alt.Bin(maxbins=100)),
    y='count()',
).configure_bar(
    color = "black",
)

altair_viewer.show(aaplGains)

statistic, p = stats.shapiro(aapl['Gain'])

print(p)