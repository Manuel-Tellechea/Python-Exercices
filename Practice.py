import pandas as pd
import yfinance as yf
import quandl
import intrinio_sdk
import numpy as np
import cufflinks as cf
import seaborn as sns
import scipy.stats as scs
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

df = yf.download('AAPL', start='2000-01-01',
                 end='2010-12-31', progress=False)
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close / df.adj_close.shift(1))

fig, ax = plt.subplots(3, 1, figsize=(24, 20), sharex=True)
df.adj_close.plot(ax=ax[0])
ax[0].set(title='AAPL time series', ylabel='Stock price ($)')
df.simple_rtn.plot(ax=ax[1])
ax[1].set(ylabel='Simple returns (%)')
df.log_rtn.plot(ax=ax[2])
ax[2].set(xlabel='Date', ylabel='Log returns (%)')
plt.show()
