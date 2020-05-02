from datetime import datetime
import backtrader as bt
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download('AAPL', start='2000-01-01', end='2010-12-31', progress=False)
#data = pdr.get_data_yahoo('AAPL', start=datetime(2017, 8, 13), end=datetime(2018, 8, 14))

#data = pdr.get_data_yahoo('AAPL', start=datetime(2017, 8, 13), end=datetime(2018, 8, 14))

data.columns=['high', 'low', 'open', 'close', 'volume', 'adj_close']


data['pct']=data.close.pct_change(1)
data['pct2']=data.close.pct_change(5)
data['pct3']=data.close.pct_change(10)

class TestStrategy(bt.Strategy):
    def next(self):
        #print("hello")
        print(','.join(str(x) for x in [self.datetime.datetime(0), self.data.pct2[0]]))

cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)

class PandasData(bt.feeds.PandasData):
    lines = ('adj_close','pct','pct2','pct3')
    params = (
        ('datetime', None),
        ('open','open'),
        ('high','high'),
        ('low','low'),
        ('close','close'),
        ('volume','volume'),
        ('openinterest',None),
        ('adj_close','adj_close'),
        ('pct','pct'),
        ('pct2','pct2'),
        ('pct3','pct3'),
    )


#df = bt.feeds.PandasData(dataname=data)
df=PandasData(dataname=data)
cerebro.adddata(df)
cerebro.run()