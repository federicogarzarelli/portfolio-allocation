# -*- coding: utf-8; py-indent-offset:4 -*-
import os
import sys
import pandas as pd
import backtrader as bt
from report import Cerebro
from strategies import uniform


class CrossOver(bt.Strategy):
    """A simple moving average crossover strategy,
    at SMA 50/200 a.k.a. the "Golden Cross Strategy"
    """
    params = (('fast', 50),
              ('slow', 200),
              ('order_pct', 0.95),
              ('market', 'BTC/USD')
              )

    def __init__(self):
        sma = bt.indicators.SimpleMovingAverage
        cross = bt.indicators.CrossOver
        self.fastma = sma(self.data.close,
                          period=self.p.fast,
                          plotname='FastMA')
        sma = bt.indicators.SimpleMovingAverage
        self.slowma = sma(self.data.close,
                          period=self.p.slow,
                          plotname='SlowMA')
        self.crossover = cross(self.fastma, self.slowma)

    def start(self):
        self.size = None

    def log(self, txt, dt=None):
        """ Logging function for this strategy
        """
        dt = dt or self.datas[0].datetime.date(0)
        time = self.datas[0].datetime.time()
        print('%s - %s, %s' % (dt.isoformat(), time, txt))

    def next(self):
        if self.position.size == 0:
            if self.crossover > 0:
                amount_to_invest = (self.p.order_pct *
                                    self.broker.cash)
                self.size = amount_to_invest / self.data.close
                msg = "*** MKT: {} BUY: {}"
                self.log(msg.format(self.p.market, self.size))
                self.buy(size=self.size)

        if self.position.size > 0:
            # we have an open position or made it to the end of backtest
            last_candle = (self.data.close.buflen() == len(self.data.close) + 1)
            if (self.crossover < 0) or last_candle:
                msg = "*** MKT: {} SELL: {}"
                self.log(msg.format(self.p.market, self.size))
                self.close()


if __name__ == "__main__":
    OUTPUTDIR = 'C:/Users/feder/Desktop/portfolio-allocation/src/testReport'

    # read data
    TESTDATA1 = 'clean_gld.csv'
    TESTDATA2 = 'clean_gspc.csv'
    basedir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(basedir, 'testReport')
    infile1 = os.path.join(datadir, TESTDATA1)
    infile2 = os.path.join(datadir, TESTDATA2)

    ohlc1 = pd.read_csv(infile1, index_col='Date', parse_dates=True)
    ohlc2 = pd.read_csv(infile2, index_col='Date', parse_dates=True)

    # initialize Cerebro engine, extende with report method
    cerebro = Cerebro()
    cerebro.broker.setcash(100000)

    # add data
    feed1 = bt.feeds.PandasData(dataname=ohlc1)
    feed2 = bt.feeds.PandasData(dataname=ohlc2)

    #cerebro.adddata(feed1)
    cerebro.adddata(feed2)

    # add Golden Cross strategy
    params = (('fast', 50),
              ('slow', 200),
              ('order_pct', 0.95),
              ('market', 'BTC/USD')
              )
#    cerebro.addstrategy(strategy=CrossOver, **dict(params))
    cerebro.addstrategy(strategy=uniform, n_assets=1, monthly_cash=0)

    # run backtest with both plotting and reporting
    cerebro.run()
    cerebro.plot(volume=False)
    cerebro.report(OUTPUTDIR,
                   infilename='btc_usd.csv',
                   user='Trading John',
                   memo='a.k.a. Golden Cross',)

"""
# read the data
TESTDATA = 'clean_gld.csv'
basedir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(basedir, 'testReport')
infile = os.path.join(datadir, TESTDATA)
ohlc = pd.read_csv(infile, index_col='Date', parse_dates=True)

# initialize the Cerebro engine, now extended with a report method
cerebro = Cerebro()
cerebro.broker.setcash(100)

# add data
feed = bt.feeds.PandasData(dataname=ohlc)
cerebro.adddata(feed)

# add Golden Cross strategy
params = (('fast', 50),
          ('slow', 200),
          ('order_pct', 0.95),
          ('market', 'Gold')
          )
cerebro.addstrategy(strategy=CrossOver, **dict(params))

cerebro.run()
cerebro.report('C:/Users/feder/Desktop/portfolio-allocation/src/testReport',
               user='Trading John',
               memo='Golden Cross / Gold')
"""
