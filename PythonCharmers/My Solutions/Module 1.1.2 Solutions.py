'''
1. Plot a histogram of the standard deviation with the following properties:

    1. Mean ($\mu$) of 1, standard deviation ($\sigma$) of 7
    2. $\mu=10, \sigma=1$
    3. $\mu=-10, \sigma=5$
2. Create a python function that accepts two inputs (`mean` and `standard_deviation`) and plots the histogram as per question 1.

3. Investigate the documentation of Altair and overlay these plots on top of each other, with different colours, giving a result that looks like this:
<img src="img/snapshot.png">
'''


# 1

import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import altair as alt
import numpy as np
import vega
import random
import altair_viewer 

'''
meanStdList = [(1,7), (10, 1), (-10, 5)]

for meanStd in meanStdList:
    n = stats.norm(meanStd[0], meanStd[1])
    normal_values = pd.DataFrame({"value": n.rvs(5000)})
    normChart = alt.Chart(normal_values).mark_bar().encode(
        alt.X("value", bin=alt.Bin(maxbins=100)),
        y='count()',
    )
    altair_viewer.show(normChart)
'''
# 2
'''
def generateNormChart(mean,std):
    n = stats.norm(mean, std)
    normal_values = pd.DataFrame({"value": n.rvs(5000)})
    normChart = alt.Chart(normal_values).mark_bar().encode(
        alt.X(shorthand="value", bin=alt.Bin(maxbins=100)),
        y='count()',
    )
    altair_viewer.show(normChart)

generateNormChart(10, 300)
'''
# 3
'''
normChart = alt.LayerChart()
numColors=len(meanStdList)
color=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
       for j in range(numColors)]
for i, meanStd in enumerate(meanStdList):
    n = stats.norm(meanStd[0], meanStd[1])
    normal_values = pd.DataFrame({"value": n.rvs(5000)})
    normChart += alt.Chart(normal_values).mark_bar(color=color[i]).encode(
        alt.X("value", bin=alt.Bin(maxbins=100)),
        y='count()',
    )
altair_viewer.show(normChart)
'''
# Exercises
'''
Using three different distribution classes in `scipy.stats`, run the following experiment:

1. Compute 10,000 random values. Compute and save the mean.
2. Repeat step 1, 1,000 times
3. Plot the histogram of the mean values and observe the shape.

Use the distributions `norm`, `cosine` and `uniform`.

This finding is why the normal distribution is so important, and so commonly found in nature.

**Hint:** See this example for how to do independent axes with Altair: https://altair-viz.github.io/gallery/layered_plot_with_dual_axis.html?highlight=resolve_scale
'''

'''
def get_rv_sample_mean(distribution, numSample = 10000):
    sample = distribution.rvs(numSample)
    return sample.mean()

def get_list_sample_means(distribution, numSample = 10000, numTimes = 1000):
    sampleMeanList = list()
    for i in range(numTimes):
        sampleMeanList.append(get_rv_sample_mean(distribution, numSample))
    return sampleMeanList

distributions = dict([("normal", stats.norm(0,1)),
                 ("cosine", stats.cosine(1,1)),
                 ("uniform", stats.uniform(0, 5))
                ])


means = pd.DataFrame({distribution_name: get_list_sample_means(distribution)
                  for distribution_name, distribution in distributions.items()})

for dist in distributions:
    nplot = alt.Chart(means[[dist]]).mark_bar().encode(alt.X(dist, bin=alt.Bin(maxbins=100)), y='count()',)
    altair_viewer.show(nplot)
'''
'''
#### Extended exercise

Load a random dataset, and perform the same exercise with the real world data - that is, take the mean of each of 1,000 samples of size 10,000 and plot the histogram. What shaped distribution does it look like?
'''
'''
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr

yf.pdr_override()

end = dt.datetime.now()
start = end - dt.timedelta(days=252)


df = pdr.get_data_yahoo("AAPL", start, end)

aaplPrice = df["Close"]

print(aaplPrice.mean())

print(aaplPrice)

applSampleMeanList = list() 
for x in range(1000):
    applSampleMeanList.append(aaplPrice.sample(150).mean())

meanAapl = pd.DataFrame({"Apple Price": applSampleMeanList})
nplot = alt.Chart(meanAapl).mark_bar().encode(alt.X("Apple Price", bin=alt.Bin(maxbins=100)), y='count()',)
altair_viewer.show(nplot)
'''
# Exercises
'''
1. Plot the relationship between the mean of a normal distribution as $x$ (with a fixed standard deviation of 1), and the area under the curve between 0 and 1 as the $y$ axis.
2. Remake the plot, but have the standard deviation vary as $x$, and the mean fixed as 0. How can you characterise the relationship?
'''

x_values_1 = np.linspace(-5, 5, 1000)
normalDist = [stats.norm(x, 1) for x in x_values_1]
y_1 = np.array([dist.cdf(1) - dist.cdf(0)
               for dist in normalDist])
plt.plot(x_values_1, y_1, "r")
plt.show()

df = pd.DataFrame({"x": x_values_1, "y": y_1})
normChart = alt.Chart(df).mark_line().encode(
    x = "x",
    y = "y"
)
altair_viewer.show(normChart)