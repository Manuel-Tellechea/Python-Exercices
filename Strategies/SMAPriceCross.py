from datetime import datetime
import backtrader as bt

class SmaSignal(bt.Signal):
    params = (('period', 20), )

    def __init__(self):
        self.lines.signal = self.data - bt.ind.SMA(period=self.p.period)

data = bt.feeds.YahooFinanceCSVData(dataname='AAPL.csv',
                                        fromdate=datetime(2017, 1, 1),
                                        todate=datetime(2020, 4, 30))

cerebro = bt.Cerebro(stdstats = False)
cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)

print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.plot(style='candlestick',iplot=True, volume=False)
