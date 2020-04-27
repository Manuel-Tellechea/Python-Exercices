#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import backtrader as bt

class PrintClose(bt.Strategy):
     
    def __init__(self):
         #Keep a reference to the "close" line in the data[0] dataseries
         self.dataclose = self.datas[0].close
    
    def log(self, txt, dt=None):     
        dt = dt or self.datas[0].datetime.date(0)     
        print('%s, %s' % (dt.isoformat(), txt)) #Print date and close

    def next(self):     
        #Log closing price to 2 decimals     
        self.log('Close: %.2f' % self.dataclose[0])

