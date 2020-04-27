#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd


# In[12]:


import yfinance as yf

#define the ticker symbol
tickerSymbol = 'MSFT'

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2020-04-14')

#see your data
tickerDf


# In[17]:


#define the ticker symbol
tickerSymbol = 'MSFT'

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get event data for ticker
tickerData.calendar


# In[9]:


#define the ticker symbol
tickerSymbol = 'MSFT'

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get recommendation data for ticker
tickerData.recommendations


# In[10]:



#define the ticker symbol
tickerSymbol = 'MSFT'

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#info on the company
tickerData.info


# In[ ]:




