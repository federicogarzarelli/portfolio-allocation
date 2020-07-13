from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
from utils import *
import datetime

import backtrader as bt
from backtrader.utils.py3 import range


class StFetcher(object):
    _STRATS = []

    @classmethod
    def register(cls, target):
        cls._STRATS.append(target)

    @classmethod
    def COUNT(cls):
        return range(len(cls._STRATS))

    def __new__(cls, *args, **kwargs):
        idx = kwargs.pop('idx')

        obj = cls._STRATS[idx](*args, **kwargs)
        return obj


@StFetcher.register
class St0(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)


@StFetcher.register
class St1(bt.SignalStrategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=10)
        crossover = bt.ind.CrossOver(self.data.close, sma1)
        self.signal_add(bt.SIGNAL_LONG, crossover)


class PandasData2(bt.feed.DataBase):
    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', None),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', None),
    )


def runstrat(pargs=None):
    args = parse_args(pargs)
    startdate = datetime.datetime.strptime("2018-01-01", "%Y-%m-%d")
    enddate = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")

    cerebro = bt.Cerebro()

    # Create a Data Feed
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        # Do not pass values before this date
        fromdate=datetime.datetime(2018, 1, 1),
        # Do not pass values before this date
        todate=datetime.datetime(2020, 2, 1),
        timeframe=bt.TimeFrame.Days,
        # Do not pass values after this date
        reverse=False)

    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.optstrategy(StFetcher, idx=StFetcher.COUNT())
    # results = cerebro.run(maxcpus=args.maxcpus, optreturn=args.optreturn)
    results = cerebro.run(runonce=False)

    strats = [x[0] for x in results]  # flatten the result
    for i, strat in enumerate(strats):
        rets = strat.analyzers.returns.get_analysis()
        print('Strat {} Name {}:\n  - analyzer: {}\n'.format(
            i, strat.__class__.__name__, rets))


def parse_args(pargs=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d")  # string to be used after

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for strategy selection')

    parser.add_argument('--data', required=False,
                        default=r'C:\Users\fega\Desktop\portfolio-allocation\modified_data\clean_fvx.csv',
                        help='Data to be read in')

    parser.add_argument('--maxcpus', required=False, action='store',
                        default=None, type=int,
                        help='Limit the number of CPUs to use')

    parser.add_argument('--optreturn', required=False, action='store_true',
                        help='Return reduced/mocked strategy object')

    parser.add_argument('--system', type=str, default='windows', help='operating system, to deal with different paths')
    parser.add_argument('--leverage', type=int, default=1, help='leverage to consider')

    return parser.parse_args(pargs)


if __name__ == '__main__':
    runstrat()
